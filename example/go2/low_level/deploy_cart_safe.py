"""Hardened cartwheel deployment entry point.

This is intentionally a new sibling of ``deploy_cart.py``.  It reuses that
file's checked observation/controller implementation, but replaces the unsafe
deployment boundaries:

* rsl-rl 2.x and 3.x checkpoint-normalizer loading;
* fail-closed metadata and deterministic trajectory resolution;
* the trained post-clip policy tail and a smooth final-pose handoff;
* phase-aware orientation limits, finite-value checks, and same-tick damping;
* a LowState freshness watchdog and a stable pre-launch readiness gate.

The cart archive's base chain remains Y -> X -> Z (qy * qx * qz).  The original
``deploy_cart.py`` is not modified.  The default safety stack includes training's
orientation-termination wall plus an absolute gravity wall and an upright-reference
fallback.  Evaluation disables divergence termination: the supplied model_7400
MuJoCo rollout crosses the training wall around 3.5 s before recovering.  Loosening
it is therefore an explicit flight-test decision via ``--ori-stance-deg``, never
an implicit "known-good" default.

Examples:
    python3 deploy_cart_safe.py --checkpoint /path/to/model_7400.pt --dry-run
    python3 deploy_cart_safe.py --checkpoint /path/to/model_7400.pt eth0

By default, ``meta.json`` is loaded beside the checkpoint and the policy tail
comes from ``meta['ep_tail']``.  The default orientation hardware cap is 180
degrees, so the effective wall is the training wall (45.8 degrees in stance and
73.3 degrees in planned flight for kdm_v2s_retarget).  Set
``--abort-ori-cap-deg`` lower only as an explicit hardware-safety decision.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import math
from pathlib import Path
import re
import sys
import threading
import time
import types


# ``deploy_cart.py`` imports the Unitree SDK at module import time.  Supplying
# tiny construction-only stubs lets --dry-run validate the checkpoint on a
# workstation without weakening the real hardware path: main refuses to start
# DDS unless the actual SDK was imported.
def _module(name):
    if name in sys.modules:
        return sys.modules[name]
    parent_name, _, child = name.rpartition(".")
    parent = _module(parent_name) if parent_name else None
    mod = types.ModuleType(name)
    if parent is not None:
        setattr(parent, child, mod)
    sys.modules[name] = mod
    return mod


def _install_offline_unitree_stubs():
    class _MotorCmd:
        def __init__(self):
            self.mode = 0
            self.q = 0.0
            self.dq = 0.0
            self.kp = 0.0
            self.kd = 0.0
            self.tau = 0.0

    class _LowCmd:
        def __init__(self):
            self.head = [0, 0]
            self.level_flag = 0
            self.gpio = 0
            self.crc = 0
            self.motor_cmd = [_MotorCmd() for _ in range(20)]

    class _Unavailable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Unitree SDK is unavailable; only --dry-run is supported")

    class _CRC:
        def Crc(self, _msg):
            return 0

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError("Unitree SDK is unavailable; only --dry-run is supported")

    channel = _module("unitree_sdk2py.core.channel")
    channel.ChannelPublisher = _Unavailable
    channel.ChannelSubscriber = _Unavailable
    channel.ChannelFactoryInitialize = _unavailable

    default = _module("unitree_sdk2py.idl.default")
    default.unitree_go_msg_dds__LowCmd_ = _LowCmd

    dds = _module("unitree_sdk2py.idl.unitree_go.msg.dds_")
    dds.LowCmd_ = _LowCmd
    dds.LowState_ = object

    crc = _module("unitree_sdk2py.utils.crc")
    crc.CRC = _CRC
    thread = _module("unitree_sdk2py.utils.thread")
    thread.RecurrentThread = _Unavailable

    motion = _module("unitree_sdk2py.comm.motion_switcher.motion_switcher_client")
    motion.MotionSwitcherClient = _Unavailable
    sport = _module("unitree_sdk2py.go2.sport.sport_client")
    sport.SportClient = _Unavailable

    const = _module("unitree_legged_const")
    const.PosStopF = 2.146e9
    const.VelStopF = 16000.0


SDK_IMPORT_ERROR = None
try:
    # Import a dependency-bearing SDK module, not just its top-level package:
    # find_spec can succeed while transitive CycloneDDS bindings are absent.
    import unitree_sdk2py.core.channel as _sdk_channel_probe  # noqa: F401
    import unitree_legged_const as _sdk_const_probe  # noqa: F401
    SDK_AVAILABLE = True
except Exception as exc:  # noqa: BLE001 - preserve --dry-run with a partial SDK install
    SDK_AVAILABLE = False
    SDK_IMPORT_ERROR = repr(exc)
    for module_name in list(sys.modules):
        if module_name == "unitree_sdk2py" or module_name.startswith("unitree_sdk2py.") \
                or module_name == "unitree_legged_const":
            sys.modules.pop(module_name, None)
    _install_offline_unitree_stubs()

import numpy as np

if __package__:
    from . import deploy_cart as _base
else:
    import deploy_cart as _base


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


STATE_TIMEOUT_S_DEFAULT = 0.10
READY_MAX_Q_ERR_DEFAULT = 0.12
READY_MAX_QD_DEFAULT = 0.50
READY_MAX_GYRO_DEFAULT = 0.50
READY_MAX_TILT_DEG_DEFAULT = 20.0
READY_STABLE_S_DEFAULT = 0.50
PRELAUNCH_ABORT_TILT_DEG = 60.0
TRAIN_FLIGHT_MULT_DEFAULT = 1.6
ABORT_ORI_CAP_DEG_DEFAULT = 180.0
STATE_Q_MARGIN_RAD = 0.35
STATE_QD_ABS_MAX = 50.0
STATE_GYRO_ABS_MAX = 25.0
STATE_ACCEL_ABS_MAX = 300.0


def _finite_scalar(name, value, *, lo=None, hi=None, lo_open=False):
    value = float(value)
    if not math.isfinite(value):
        raise SystemExit(f"ABORT: {name} must be finite, got {value!r}")
    if lo is not None and (value <= lo if lo_open else value < lo):
        op = ">" if lo_open else ">="
        raise SystemExit(f"ABORT: {name} must be {op} {lo}, got {value}")
    if hi is not None and value > hi:
        raise SystemExit(f"ABORT: {name} must be <= {hi}, got {value}")
    return value


def _canonical_file(path, label):
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise SystemExit(f"ABORT: {label} is not a file: {p}")
    return p


def load_meta(ckpt_path, explicit=None):
    ckpt = _canonical_file(ckpt_path, "checkpoint")
    path = (Path(explicit).expanduser().resolve() if explicit is not None
            else ckpt.parent / "meta.json")
    if not path.is_file():
        raise SystemExit(
            f"ABORT: metadata is not a file: {path}\n"
            "Pass --meta explicitly or place meta.json beside the checkpoint."
        )
    try:
        with path.open() as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ABORT: cannot read metadata {path}: {exc}") from exc
    if not isinstance(meta, dict):
        raise SystemExit(f"ABORT: metadata root must be an object: {path}")
    logging.info("meta: %s", path)
    return meta, str(path), str(ckpt)


def resolve_traj_path(override, meta, meta_path, ckpt_path):
    """Resolve explicit paths literally; allow safe basename fallbacks for stale meta."""
    script_dir = Path(__file__).resolve().parent
    ckpt_dir = Path(ckpt_path).resolve().parent
    meta_dir = Path(meta_path).resolve().parent if meta_path else ckpt_dir

    if override is not None:
        raw = Path(override).expanduser()
        path = (raw if raw.is_absolute() else Path.cwd() / raw).resolve()
        if not path.is_file():
            raise SystemExit(f"ABORT: --traj is not a file: {path}")
        return str(path)

    raw_value = meta.get("traj_path")
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise SystemExit("ABORT: meta has no non-empty traj_path (pass --traj)")
    raw = Path(raw_value).expanduser()
    literal = raw if raw.is_absolute() else meta_dir / raw
    candidates = [literal]
    if not raw.is_absolute():
        candidates.extend([ckpt_dir / raw, Path.cwd() / raw])
    candidates.extend([
        meta_dir / raw.name,
        ckpt_dir / raw.name,
        script_dir / "hopscotch_utils" / raw.name,
    ])

    tried = []
    for candidate in candidates:
        path = candidate.resolve()
        if path in tried:
            continue
        tried.append(path)
        if path.is_file():
            if path != literal.resolve():
                logging.warning("traj_path %s missing; using %s", raw, path)
            return str(path)
    raise SystemExit(
        "ABORT: reference not found; tried:\n  "
        + "\n  ".join(str(path) for path in tried)
        + "\nPass --traj explicitly."
    )


# Harden the shared mathematical primitives before HopscotchRef is constructed.
_legacy_euler_chain = _base.euler_chain_to_quat_wxyz


def euler_chain_to_quat_wxyz(e, base_names):
    names = tuple(base_names)
    expected = tuple(sorted(_base._BASE_AXIS))
    if len(names) != 3 or tuple(sorted(names)) != expected:
        raise ValueError(
            f"base joint names must be a permutation of {list(_base._BASE_AXIS)}, "
            f"got {list(names)}"
        )
    values = np.asarray(e, dtype=float)
    if values.ndim not in (1, 2) or values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError(f"base Euler array must be finite with final dimension 3, got {values.shape}")
    return _legacy_euler_chain(values, names)


def geodesic_deg(qa, qb):
    qa = np.asarray(qa, dtype=float)
    qb = np.asarray(qb, dtype=float)
    if qa.shape != (4,) or qb.shape != (4,) or not np.isfinite(qa).all() \
            or not np.isfinite(qb).all():
        raise ValueError("geodesic_deg requires two finite shape-(4,) quaternions")
    na = float(np.linalg.norm(qa))
    nb = float(np.linalg.norm(qb))
    if na <= 1e-8 or nb <= 1e-8:
        raise ValueError("geodesic_deg received a zero-norm quaternion")
    dot = np.clip(abs(float(np.dot(qa, qb))) / (na * nb), 0.0, 1.0)
    return float(np.degrees(2.0 * np.arccos(dot)))


class HopscotchRef(_base.HopscotchRef):
    def _index_from_time32(self, t32):
        dt32 = np.float32(self.dt)
        if not np.isfinite(t32) or not np.isfinite(dt32) or dt32 <= 0:
            raise ValueError(f"invalid reference time/dt: {t32!r}/{self.dt!r}")
        f = t32 / dt32
        i0 = int(np.clip(np.floor(f), 0, self.T_state - 2))
        frac = float(np.clip(f - np.float32(i0), 0.0, 1.0))
        return i0, frac

    def _current_time32(self, t):
        """Rebuild training's ``float32(ref_step) * ref.dt`` policy clock."""
        value = float(t)
        if not math.isfinite(value):
            raise ValueError(f"non-finite reference time {t!r}")
        ref_step = int(round(value / self.dt))
        return np.float32(ref_step) * np.float32(self.dt)

    def _index(self, t):
        return self._index_from_time32(self._current_time32(t))

    def _preview_index(self, t, preview_dt, j):
        # Training first forms the current clock from integer ref_step, then adds
        # a float32 preview offset.  The same numeric time can therefore floor to
        # a different knot when reached as a preview; preserve that exact order.
        query = np.float32(
            self._current_time32(t) + np.float32(j) * np.float32(preview_dt)
        )
        return self._index_from_time32(query)

    def preview(self, t, preview_dt, num_future):
        outs = []
        for j in range(num_future + 1):
            i0, frac = self._preview_index(t, preview_dt, j)
            q = (1.0 - frac) * self.q_ref[i0] + frac * self.q_ref[i0 + 1]
            qd = (1.0 - frac) * self.qd_ref[i0] + frac * self.qd_ref[i0 + 1]
            outs.extend([q, qd, np.zeros(12)])
        return np.concatenate(outs)

    def contact_pack(self, t, preview_dt, num_future):
        masks = []
        for j in range(num_future + 1):
            i0, _ = self._preview_index(t, preview_dt, j)
            masks.append(self.contact[i0].astype(float))
        i0, _ = self._index(t)
        ttc_edge = np.clip(
            self.ttc_edge[i0] / _base.TTC_EDGE_CLIP_S, 0.0, 1.0
        )
        return np.concatenate(masks), ttc_edge


_base.euler_chain_to_quat_wxyz = euler_chain_to_quat_wxyz
_base.geodesic_deg = geodesic_deg
_base.HopscotchRef = HopscotchRef


class Policy:
    """NumPy actor supporting rsl-rl 2.x and 3.x normalizer layouts."""

    def __init__(self, ckpt_path, n_obs, n_act):
        import torch

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001 - convert loader failures to a hard abort
            raise SystemExit(f"ABORT: cannot load checkpoint {ckpt_path}: {exc}") from exc
        state = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else None
        if not isinstance(state, dict):
            raise SystemExit("ABORT: checkpoint has no model_state_dict")

        self.W = []
        self.b = []
        for index in (0, 2, 4, 6):
            wk, bk = f"actor.{index}.weight", f"actor.{index}.bias"
            if wk not in state or bk not in state:
                raise SystemExit(f"ABORT: checkpoint actor is missing {wk!r} or {bk!r}")
            self.W.append(state[wk].detach().cpu().numpy().astype(np.float64))
            self.b.append(state[bk].detach().cpu().numpy().astype(np.float64))

        embedded_mean = state.get("actor_obs_normalizer._mean")
        embedded_std = state.get("actor_obs_normalizer._std")
        legacy_norm = ckpt.get("obs_norm_state_dict")
        if embedded_mean is not None and embedded_std is not None:
            mean_t, std_t = embedded_mean, embedded_std
            layout = "rsl-rl 3.x embedded"
        elif isinstance(legacy_norm, dict) and legacy_norm.get("_mean") is not None \
                and legacy_norm.get("_std") is not None:
            mean_t, std_t = legacy_norm["_mean"], legacy_norm["_std"]
            layout = "rsl-rl 2.x top-level"
        else:
            raise SystemExit("ABORT: checkpoint has no actor observation-normalization statistics")

        self.mean = mean_t.detach().cpu().numpy().reshape(-1).astype(np.float64)
        self.std = std_t.detach().cpu().numpy().reshape(-1).astype(np.float64)
        expected_shapes = [(512, n_obs), (256, 512), (128, 256), (n_act, 128)]
        for i, (weight, bias, expected) in enumerate(zip(self.W, self.b, expected_shapes)):
            if weight.shape != expected or bias.shape != (expected[0],):
                raise SystemExit(
                    f"ABORT: actor layer {i} has W{weight.shape}/b{bias.shape}, "
                    f"expected W{expected}/b{(expected[0],)}"
                )
            if not np.isfinite(weight).all() or not np.isfinite(bias).all():
                raise SystemExit(f"ABORT: actor layer {i} contains non-finite values")
        if self.mean.shape != (n_obs,) or self.std.shape != (n_obs,):
            raise SystemExit(
                f"ABORT: normalizer mean/std are {self.mean.shape}/{self.std.shape}, "
                f"expected {(n_obs,)}"
            )
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() \
                or np.any(self.std < 0):
            raise SystemExit("ABORT: observation normalizer is non-finite or has negative std")
        logging.info(
            "policy loaded: %s (iter %s), %s, obs=%d act=%d",
            ckpt_path, ckpt.get("iter"), layout, n_obs, n_act,
        )

    def __call__(self, obs):
        x = np.asarray(obs, dtype=np.float64)
        if x.shape != self.mean.shape or not np.isfinite(x).all():
            raise FloatingPointError(
                f"policy observation must be finite with shape {self.mean.shape}, got {x.shape}"
            )
        x = (x - self.mean) / (self.std + 1e-2)
        for weight, bias in zip(self.W[:-1], self.b[:-1]):
            x = _base._elu(weight @ x + bias)
        action = self.W[-1] @ x + self.b[-1]
        if action.shape != (self.W[-1].shape[0],) or not np.isfinite(action).all():
            raise FloatingPointError("policy produced a non-finite or malformed action")
        return np.clip(action, -1.0, 1.0)


_base.Policy = Policy


_REQUIRED_META = (
    "control_mode", "compliant_feet", "obs_priv_base", "hold_targets",
    "obs_velest", "ff_damping_comp", "ff_armature_comp", "joint_names",
    "action_space", "observation_space", "kp", "kd", "action_scale",
    "q_home", "traj_path", "kin_tau_ff", "kin_anchor", "kin_anchor_qd",
    "act_lpf_beta", "jlimit_clamp_frac", "obs_tau_mode", "obs_act_hist",
    "obs_o1_scale", "obs_ori_err", "obs_contact_prev", "obs_contact_blind",
    "obs_history_len", "num_future", "preview_steps", "clip_obs",
    "term_ori_err", "max_gravity_z", "ep_tail", "episode_length_s",
)


def _validate_meta(meta):
    missing = [key for key in _REQUIRED_META if key not in meta]
    if missing:
        raise SystemExit(f"ABORT: metadata is missing required keys: {', '.join(missing)}")
    if meta["kin_anchor"] not in ("home", "ref"):
        raise SystemExit(f"ABORT: kin_anchor must be 'home' or 'ref', got {meta['kin_anchor']!r}")
    if float(meta["kin_tau_ff"]) != 0.0:
        raise SystemExit("ABORT: this kinematic runtime requires kin_tau_ff == 0")
    q_home = np.asarray(meta["q_home"], dtype=float)
    if q_home.shape != (12,) or not np.isfinite(q_home).all():
        raise SystemExit(f"ABORT: q_home must be a finite 12-vector, got {q_home.shape}")
    _finite_scalar("meta kp", meta["kp"], lo=0.0, lo_open=True)
    _finite_scalar("meta kd", meta["kd"], lo=0.0)
    _finite_scalar("meta action_scale", meta["action_scale"], lo=0.0, lo_open=True)
    _finite_scalar("meta clip_obs", meta["clip_obs"], lo=0.0, lo_open=True)
    _finite_scalar("meta act_lpf_beta", meta["act_lpf_beta"], lo=0.0, hi=1.0)
    if float(meta["act_lpf_beta"]) >= 1.0:
        raise SystemExit("ABORT: meta act_lpf_beta must be in [0, 1)")
    _finite_scalar("meta jlimit_clamp_frac", meta["jlimit_clamp_frac"],
                   lo=0.0, hi=1.0, lo_open=True)
    _finite_scalar("meta term_ori_err", meta["term_ori_err"],
                   lo=0.0, hi=math.pi, lo_open=True)
    _finite_scalar("meta max_gravity_z", meta["max_gravity_z"], lo=-1.0, hi=1.0)


class Custom(_base.Custom):
    def __init__(
        self,
        ckpt_path,
        meta,
        traj_override=None,
        meta_path=None,
        *,
        policy_tail_s=None,
        abort_ori_cap_deg=ABORT_ORI_CAP_DEG_DEFAULT,
        ori_stance_deg=None,
        flight_ori_mult=None,
        upright_fallback=True,
        state_timeout_s=STATE_TIMEOUT_S_DEFAULT,
        ready_max_q_err=READY_MAX_Q_ERR_DEFAULT,
        ready_max_qd=READY_MAX_QD_DEFAULT,
        ready_max_gyro=READY_MAX_GYRO_DEFAULT,
        ready_max_tilt_deg=READY_MAX_TILT_DEG_DEFAULT,
        ready_stable_s=READY_STABLE_S_DEFAULT,
    ):
        _validate_meta(meta)
        resolved_traj = resolve_traj_path(traj_override, meta, meta_path, ckpt_path)
        super().__init__(ckpt_path, meta, resolved_traj, meta_path=meta_path)

        self.track_ticks = self.n_ticks
        trained_tail_s = _finite_scalar("meta ep_tail", meta.get("ep_tail", 0.0), lo=0.0)
        tail_s = trained_tail_s if policy_tail_s is None else _finite_scalar(
            "--policy-tail-s", policy_tail_s, lo=0.0
        )
        if policy_tail_s is not None and abs(tail_s - trained_tail_s) > 1e-9:
            logging.warning(
                "policy tail override %.3f s differs from trained ep_tail %.3f s",
                tail_s, trained_tail_s,
            )
        self.policy_tail_s = tail_s
        self.policy_tail_ticks = int(round(tail_s / self.dt))
        self.policy_ticks = self.track_ticks + self.policy_tail_ticks
        self.n_ticks = self.policy_ticks
        episode_length_s = _finite_scalar(
            "meta episode_length_s", meta["episode_length_s"], lo=0.0, lo_open=True
        )
        required_episode_s = self.ref.duration + trained_tail_s
        if episode_length_s + 1e-9 < required_episode_s:
            raise SystemExit(
                f"ABORT: episode_length_s {episode_length_s:.3f} is shorter than "
                f"reference + trained tail {required_episode_s:.3f}"
            )

        self.abort_ori_cap_deg = _finite_scalar(
            "--abort-ori-cap-deg", abort_ori_cap_deg, lo=0.0, hi=180.0, lo_open=True
        )
        self.trained_ori_stance_deg = math.degrees(float(meta["term_ori_err"]))
        self.ori_stance_deg = (
            self.trained_ori_stance_deg if ori_stance_deg is None
            else _finite_scalar(
                "--ori-stance-deg", ori_stance_deg,
                lo=0.0, hi=180.0, lo_open=True,
            )
        )
        if ori_stance_deg is not None and abs(
                self.ori_stance_deg - self.trained_ori_stance_deg) > 1e-9:
            logging.warning(
                "ORIENTATION WALL OVERRIDE %.1f deg differs from trained %.1f deg; "
                "the policy was not trained beyond the latter",
                self.ori_stance_deg, self.trained_ori_stance_deg,
            )
        if flight_ori_mult is None:
            if "term_flight_mult" not in meta:
                logging.warning(
                    "meta has no term_flight_mult; using training-environment default %.2f",
                    TRAIN_FLIGHT_MULT_DEFAULT,
                )
            flight_ori_mult = meta.get("term_flight_mult", TRAIN_FLIGHT_MULT_DEFAULT)
        self.flight_ori_mult = _finite_scalar(
            "flight orientation multiplier", flight_ori_mult, lo=1.0
        )
        self.max_gravity_z = float(meta["max_gravity_z"])
        self.upright_fallback = bool(upright_fallback)
        self._current_ori_limit_deg = min(self.ori_stance_deg, self.abort_ori_cap_deg)
        self.abort_ori_deg = self._current_ori_limit_deg

        self.state_timeout_s = _finite_scalar(
            "--state-timeout-s", state_timeout_s, lo=0.0, lo_open=True
        )
        self.ready_max_q_err = _finite_scalar(
            "--ready-max-q-err", ready_max_q_err, lo=0.0, lo_open=True
        )
        self.ready_max_qd = _finite_scalar(
            "--ready-max-qd", ready_max_qd, lo=0.0, lo_open=True
        )
        self.ready_max_gyro = _finite_scalar(
            "--ready-max-gyro", ready_max_gyro, lo=0.0, lo_open=True
        )
        ready_tilt = _finite_scalar(
            "--ready-max-tilt-deg", ready_max_tilt_deg, lo=0.0, hi=60.0,
            lo_open=True,
        )
        self.ready_grav_z_max = -math.cos(math.radians(ready_tilt))
        stable_s = _finite_scalar("--ready-stable-s", ready_stable_s, lo=0.0, lo_open=True)
        self.ready_stable_ticks = max(1, int(math.ceil(stable_s / self.dt)))
        self.prelaunch_abort_grav_z = -math.cos(math.radians(PRELAUNCH_ABORT_TILT_DEG))

        self._low_state_packet = None
        self._pinned_state_received_at = None
        self._pinned_state_progressed_at = None
        self._command_lock = threading.RLock()
        self.abort_reason = None
        self.ready_to_launch = False
        self._ready_ticks = 0
        self._safe_armed_logged = False
        self._ready_last_diag = "waiting for LowState"
        self._post_blend_start = None
        self._post_blend_tick = 0
        self.writer_failed = False
        self.writer_failure = None
        self.trace["ori_err_deg"] = []
        self.trace["handoff_tau"] = []

        # Suppress the base class's elapsed-time-only ARMED message.  This class
        # emits ARMED only after the stable posture/motion gate has passed.
        self._armed_logged = True

        logging.info(
            "policy horizon: %d trajectory + %d trained-tail = %d ticks "
            "(tail %.2f s)",
            self.track_ticks, self.policy_tail_ticks, self.policy_ticks,
            self.policy_tail_s,
        )
        logging.info(
            "orientation wall: trained base %.1f deg; configured %.1f stance / %.1f "
            "planned flight; hardware cap %.1f -> effective %.1f / %.1f; "
            "global grav_z wall %.2f",
            self.trained_ori_stance_deg,
            self.ori_stance_deg, self.ori_stance_deg * self.flight_ori_mult,
            self.abort_ori_cap_deg,
            min(self.ori_stance_deg, self.abort_ori_cap_deg),
            min(self.ori_stance_deg * self.flight_ori_mult, self.abort_ori_cap_deg),
            self.max_gravity_z,
        )
        if self.upright_fallback:
            logging.info(
                "extra upright-reference fallback: grav_b_z > -0.40 while "
                "ref grav_z < %.2f",
                _base.REF_UPRIGHT_GRAV_Z,
            )
        else:
            logging.warning(
                "UPRIGHT-REFERENCE FALLBACK DISABLED explicitly; only geodesic and "
                "global gravity walls remain"
            )
        if ori_stance_deg is None and Path(ckpt_path).stem == "model_7400":
            logging.warning(
                "training-wall safety is stricter than evaluation: model_7400's "
                "stored nominal MuJoCo rollout crosses it around 3.5 s before recovery"
            )
        elif ori_stance_deg is None:
            logging.warning(
                "training-wall safety is stricter than the evaluation protocol, which "
                "disables divergence termination; inspect this checkpoint's trace"
            )
        logging.info(
            "stand integrator handoff: up to %.1f Nm fades over the first %d policy "
            "ticks and is recorded as trace['handoff_tau']",
            self.tau_i_max, self.handoff_fade_ticks,
        )

    # ----------------------------- hardware setup / state acquisition
    def Init(self):
        if not SDK_AVAILABLE:
            raise RuntimeError("Unitree SDK is unavailable; use --dry-run on this machine")
        self.InitLowCmd()
        self.lowcmd_publisher = _base.ChannelPublisher("rt/lowcmd", _base.LowCmd_)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = _base.ChannelSubscriber("rt/lowstate", _base.LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)

        self.sc = _base.SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()
        self.msc = _base.MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()
        for _attempt in range(10):
            status, result = self.msc.CheckMode()
            if status not in (0, None) or not isinstance(result, dict):
                raise RuntimeError(f"CheckMode failed: status={status!r}, result={result!r}")
            if not result.get("name"):
                break
            logging.warning("releasing active high-level mode %r", result.get("name"))
            self.sc.StandDown()
            self.msc.ReleaseMode()
            time.sleep(1.0)
        else:
            raise RuntimeError("could not release the active high-level mode after 10 attempts")

    def LowStateMessageHandler(self, msg):
        received_at = time.perf_counter()
        tick = getattr(msg, "tick", None)
        tick = (int(tick) & 0xFFFFFFFF) if tick is not None else None
        previous = self._low_state_packet
        if tick is None or previous is None or len(previous) < 4 or tick != previous[3]:
            progressed_at = received_at
        else:
            progressed_at = previous[2]
        # One atomic assignment gives the control thread a coherent state, arrival
        # time, and source-progress time.  Repeated DDS copies of one frozen tick
        # cannot keep the watchdog alive.
        self._low_state_packet = (msg, received_at, progressed_at, tick)
        if self.hf_on and len(self.hf_t) < self.HF_MAX:
            motor_state = msg.motor_state
            self.hf_t.append(received_at - self.hf_t0)
            self.hf_tau.append([
                motor_state[m].tau_est for m in _base.MOTOR_FROM_ISO
            ])

    @staticmethod
    def _normalized_quat(quat):
        quat = np.asarray(quat, dtype=float)
        if quat.shape != (4,) or not np.isfinite(quat).all():
            raise ValueError(f"IMU quaternion must be a finite 4-vector, got {quat}")
        norm = float(np.linalg.norm(quat))
        if norm < 0.5 or norm > 1.5:
            raise ValueError(f"IMU quaternion norm {norm:.3g} is outside [0.5, 1.5]")
        return quat / norm

    def _imu_quat(self):
        return self._normalized_quat(self.low_state.imu_state.quaternion)

    def _abort_now(self, reason):
        with self._command_lock:
            first = not self.aborted
            if self.abort_reason is None:
                self.abort_reason = str(reason)
            self.aborted = True
            self.hf_on = False
            self._damped_stop()
        if first:
            logging.error("SAFETY ABORT: %s -> damping now", reason)

    def _pin_and_validate_state(self):
        packet = self._low_state_packet
        if packet is None:
            return False
        state, received_at = packet[:2]
        progressed_at = packet[2] if len(packet) >= 3 else received_at
        age = time.perf_counter() - received_at
        progress_age = time.perf_counter() - progressed_at
        if not math.isfinite(age) or age < 0.0 or age > self.state_timeout_s:
            self._abort_now(
                f"LowState stale for {age * 1000.0:.0f} ms "
                f"(limit {self.state_timeout_s * 1000.0:.0f} ms)"
            )
            return False
        if (not math.isfinite(progress_age) or progress_age < 0.0
                or progress_age > self.state_timeout_s):
            self._abort_now(
                f"LowState source tick has not advanced for {progress_age * 1000.0:.0f} ms "
                f"(limit {self.state_timeout_s * 1000.0:.0f} ms)"
            )
            return False
        self.low_state = state  # pin one callback snapshot for this entire control tick
        self._pinned_state_received_at = received_at
        self._pinned_state_progressed_at = progressed_at
        try:
            q = np.asarray([
                state.motor_state[m].q for m in _base.MOTOR_FROM_ISO
            ], dtype=float)
            qd = np.asarray([
                state.motor_state[m].dq for m in _base.MOTOR_FROM_ISO
            ], dtype=float)
            gyro = np.asarray(state.imu_state.gyroscope, dtype=float)
            accel = np.asarray(state.imu_state.accelerometer, dtype=float)
            foot = np.asarray([
                state.foot_force[j] for j in _base.FOOTFORCE_FROM_ISO_FOOT
            ], dtype=float)
            self._normalized_quat(state.imu_state.quaternion)
        except Exception as exc:  # noqa: BLE001 - malformed DDS data must fail closed
            self._abort_now(f"malformed LowState: {exc}")
            return False
        for name, values, shape in (
            ("joint positions", q, (12,)),
            ("joint velocities", qd, (12,)),
            ("gyroscope", gyro, (3,)),
            ("accelerometer", accel, (3,)),
            ("foot forces", foot, (4,)),
        ):
            if values.shape != shape or not np.isfinite(values).all():
                self._abort_now(f"LowState {name} are non-finite or shape {values.shape}")
                return False
        if np.any(q < _base.JOINT_RANGE_ISO[:, 0] - STATE_Q_MARGIN_RAD) \
                or np.any(q > _base.JOINT_RANGE_ISO[:, 1] + STATE_Q_MARGIN_RAD):
            self._abort_now("LowState joint position is outside the physical range plus margin")
            return False
        if np.max(np.abs(qd)) > STATE_QD_ABS_MAX:
            self._abort_now(f"LowState |qd| exceeds {STATE_QD_ABS_MAX:g} rad/s")
            return False
        if np.max(np.abs(gyro)) > STATE_GYRO_ABS_MAX:
            self._abort_now(f"LowState |gyro| exceeds {STATE_GYRO_ABS_MAX:g} rad/s")
            return False
        if np.max(np.abs(accel)) > STATE_ACCEL_ABS_MAX:
            self._abort_now(f"LowState |accel| exceeds {STATE_ACCEL_ABS_MAX:g} m/s^2")
            return False
        return True

    # ----------------------------------------------- readiness / safety gates
    def _instant_ready(self):
        q = np.asarray([self.low_state.motor_state[m].q for m in range(12)], dtype=float)
        qd = np.asarray([self.low_state.motor_state[m].dq for m in range(12)], dtype=float)
        gyro = np.asarray(self.low_state.imu_state.gyroscope, dtype=float)
        quat = self._imu_quat()
        grav_z = float((_base.rotmat_from_quat_wxyz(quat).T
                        @ np.array([0.0, 0.0, -1.0]))[2])
        q_err = float(np.max(np.abs(q - self.q0_motor)))
        qd_max = float(np.max(np.abs(qd)))
        gyro_max = float(np.max(np.abs(gyro)))
        okay = (
            q_err <= self.ready_max_q_err
            and qd_max <= self.ready_max_qd
            and gyro_max <= self.ready_max_gyro
            and grav_z <= self.ready_grav_z_max
        )
        diag = (
            f"qerr={q_err:.3f}/{self.ready_max_q_err:.3f}, "
            f"|qd|={qd_max:.3f}/{self.ready_max_qd:.3f}, "
            f"|gyro|={gyro_max:.3f}/{self.ready_max_gyro:.3f}, "
            f"grav_z={grav_z:.3f}/{self.ready_grav_z_max:.3f}"
        )
        return okay, diag

    def _update_readiness(self):
        if self.hold_percent < 1.0 or self.start_policy or self.handoff_done:
            return
        okay, diag = self._instant_ready()
        self._ready_last_diag = diag
        if okay:
            self._ready_ticks += 1
        else:
            if self.ready_to_launch:
                logging.warning("launch readiness lost: %s", diag)
            self._ready_ticks = 0
            self.ready_to_launch = False
        if self._ready_ticks >= self.ready_stable_ticks:
            self.ready_to_launch = True
            if not self._safe_armed_logged:
                logging.info(
                    "ARMED: readiness stable for %d ticks (%s); waiting for Enter",
                    self.ready_stable_ticks, diag,
                )
                self._safe_armed_logged = True
        elif self.motiontime % 25 == 0:
            logging.info(
                "holding q0; readiness %d/%d ticks (%s)",
                self._ready_ticks, self.ready_stable_ticks, diag,
            )

    def request_launch(self):
        """Atomically acknowledge Enter and revalidate the newest state."""
        with self._command_lock:
            if self.aborted:
                return False
            if not self._pin_and_validate_state():
                return False
            ready_now, diag = self._instant_ready()
            if not self.ready_to_launch or not ready_now:
                self._abort_now(f"launch requested after readiness was lost: {diag}")
                return False
            self.start_policy = True
            return True

    def _assert_finite_command(self):
        for motor in range(12):
            command = self.low_cmd.motor_cmd[motor]
            values = np.asarray([
                command.q, command.dq, command.kp, command.kd, command.tau
            ], dtype=float)
            if not np.isfinite(values).all():
                raise FloatingPointError(
                    f"motor {motor} command contains non-finite values {values}"
                )

    def _assert_pinned_state_fresh_at_publish(self):
        if self._pinned_state_received_at is None:
            raise RuntimeError("no LowState snapshot was pinned for this command")
        now = time.perf_counter()
        arrival_age = now - self._pinned_state_received_at
        progress_age = now - self._pinned_state_progressed_at
        if arrival_age > self.state_timeout_s or progress_age > self.state_timeout_s:
            raise RuntimeError(
                f"pinned LowState became stale during the tick "
                f"(arrival {arrival_age * 1000.0:.0f} ms, "
                f"source {progress_age * 1000.0:.0f} ms, "
                f"limit {self.state_timeout_s * 1000.0:.0f} ms)"
            )

    # ------------------------------------------------------- policy / phases
    def _policy_tick(self):
        t = self.ii * self.dt
        ref_index, _ = self.ref._index(t)
        planned_air = bool(self.ref.airborne[ref_index])
        training_limit = self.ori_stance_deg * (
            self.flight_ori_mult if planned_air else 1.0
        )
        self._current_ori_limit_deg = min(training_limit, self.abort_ori_cap_deg)
        self.abort_ori_deg = self._current_ori_limit_deg

        quat = self._aligned_quat()
        _, quat_ref = self.ref.base_ref_at(t)
        ori_err = geodesic_deg(quat, quat_ref)
        self._ori_err_deg = ori_err  # make the periodic base log current, not one tick stale
        grav_b_z = float((_base.rotmat_from_quat_wxyz(quat).T
                          @ np.array([0.0, 0.0, -1.0]))[2])
        grav_ref_z = float((_base.rotmat_from_quat_wxyz(quat_ref).T
                            @ np.array([0.0, 0.0, -1.0]))[2])
        trace_len = len(self.trace["t"])
        handoff_fade = max(0.0, 1.0 - self.ii / self.handoff_fade_ticks)
        handoff_tau = handoff_fade * self.tau_i[_base.MOTOR_FROM_ISO]

        saved_upright_threshold = _base.REF_UPRIGHT_GRAV_Z
        if not self.upright_fallback:
            _base.REF_UPRIGHT_GRAV_Z = -2.0  # make inherited `ref_z < threshold` false
        try:
            super()._policy_tick()
        finally:
            _base.REF_UPRIGHT_GRAV_Z = saved_upright_threshold
        if len(self.trace["t"]) == trace_len + 1:
            self.trace["ori_err_deg"].append(ori_err)
            self.trace["handoff_tau"].append(handoff_tau.copy())

        if grav_b_z > self.max_gravity_z:
            self._abort_now(
                f"grav_z {grav_b_z:.3f} exceeds global wall {self.max_gravity_z:.3f} "
                f"at t={t:.2f}"
            )
        elif self.aborted:
            if ori_err > self._current_ori_limit_deg:
                reason = (
                    f"orientation error {ori_err:.1f} deg exceeds "
                    f"{self._current_ori_limit_deg:.1f} deg "
                    f"({'flight' if planned_air else 'stance'}) at t={t:.2f}"
                )
            else:
                reason = (
                    f"upright-reference tilt guard at t={t:.2f} "
                    f"(grav_z={grav_b_z:.3f}, ref={grav_ref_z:.3f})"
                )
            # The base method has already recorded the diagnostic row.  Replace
            # the just-written policy targets before LowCmdWrite publishes them.
            self._abort_now(reason)

    def _post_policy_attitude_fault(self):
        """Final-reference guard retained throughout the blend and final hold."""
        t = max(0.0, (self.n_ticks - 1) * self.dt)
        quat = self._aligned_quat()
        _, quat_ref = self.ref.base_ref_at(t)
        ori_err = geodesic_deg(quat, quat_ref)
        grav_b_z = float((_base.rotmat_from_quat_wxyz(quat).T
                          @ np.array([0.0, 0.0, -1.0]))[2])
        grav_ref_z = float((_base.rotmat_from_quat_wxyz(quat_ref).T
                            @ np.array([0.0, 0.0, -1.0]))[2])
        limit = min(self.ori_stance_deg, self.abort_ori_cap_deg)
        if grav_b_z > self.max_gravity_z:
            return f"post-policy grav_z {grav_b_z:.3f} exceeds {self.max_gravity_z:.3f}"
        if ori_err > limit:
            return f"post-policy orientation error {ori_err:.1f} deg exceeds {limit:.1f} deg"
        if self.upright_fallback and grav_ref_z < _base.REF_UPRIGHT_GRAV_Z \
                and grav_b_z > -0.4:
            return (
                f"post-policy upright tilt guard (grav_z={grav_b_z:.3f}, "
                f"ref={grav_ref_z:.3f})"
            )
        return None

    def _capture_post_blend_start(self):
        self._post_blend_start = {
            name: np.asarray([
                getattr(self.low_cmd.motor_cmd[m], name) for m in range(12)
            ], dtype=float)
            for name in ("q", "dq", "kp", "kd", "tau")
        }

    def _apply_post_blend(self):
        if self._post_blend_start is None:
            raise RuntimeError("post-policy blend was not initialized")
        count = max(1, int(self.settle_duration))
        x = 1.0 if count == 1 else self._post_blend_tick / float(count - 1)
        x = float(np.clip(x, 0.0, 1.0))
        alpha = x * x * (3.0 - 2.0 * x)  # smoothstep, zero slope at both ends
        finish = {
            "q": self.qf_motor,
            "dq": np.zeros(12),
            "kp": np.full(12, self.Kp_stand),
            "kd": np.full(12, self.Kd_stand),
            "tau": np.zeros(12),
        }
        for name, target in finish.items():
            values = (1.0 - alpha) * self._post_blend_start[name] + alpha * target
            for motor, value in enumerate(values):
                setattr(self.low_cmd.motor_cmd[motor], name, float(value))
        self._post_blend_tick += 1

    def _tick(self):
        if self.aborted:
            self._damped_stop()
            return True
        if self._low_state_packet is None:
            return False
        if not self._pin_and_validate_state():
            return True

        if not self.handoff_done:
            quat = self._imu_quat()
            grav_z = float((_base.rotmat_from_quat_wxyz(quat).T
                            @ np.array([0.0, 0.0, -1.0]))[2])
            if grav_z > self.prelaunch_abort_grav_z:
                self._abort_now(
                    f"pre-launch body tilt exceeds {PRELAUNCH_ABORT_TILT_DEG:.0f} deg "
                    f"(grav_z={grav_z:.3f})"
                )
                return True

        if self.start_policy and self.hold_percent >= 1.0 and not self.handoff_done:
            ready_now, diag = self._instant_ready()
            if not self.ready_to_launch or not ready_now:
                self._abort_now(f"launch requested after readiness was lost: {diag}")
                return True

        if self.handoff_done and self.ii >= self.n_ticks:
            fault = self._post_policy_attitude_fault()
            if fault is not None:
                self._abort_now(fault)
                return True

        entering_post_blend = (
            self.handoff_done and self.ii >= self.n_ticks
            and self.settle_percent < 1.0 and not self.aborted
        )
        if entering_post_blend and self._post_blend_start is None:
            self._capture_post_blend_start()

        published = super()._tick()
        if entering_post_blend and not self.aborted:
            self._apply_post_blend()
        if not self.aborted:
            self._update_readiness()
        return published

    def LowCmdWrite(self):
        """Fail closed at the final DDS boundary for every controller phase."""
        with self._command_lock:
            try:
                if not self._tick():
                    return
                # A main-thread e-stop request can arrive during a tick.  Re-arm
                # damping at the final boundary so no later phase write wins the race.
                if self.aborted:
                    self._damped_stop()
                else:
                    self._assert_pinned_state_fresh_at_publish()
                self._assert_finite_command()
                self.low_cmd.crc = self.crc.Crc(self.low_cmd)
                self.lowcmd_publisher.Write(self.low_cmd)
                return
            except Exception as exc:  # noqa: BLE001 - this is the last safety boundary
                logging.exception("control tick or DDS write failed")
                self._abort_now(f"control/write exception: {type(exc).__name__}: {exc}")

            # Make one best-effort damping write.  A DDS failure cannot be fixed in
            # software, but main is told explicitly instead of assuming the writer
            # thread is still healthy.
            try:
                self._assert_finite_command()
                self.low_cmd.crc = self.crc.Crc(self.low_cmd)
                self.lowcmd_publisher.Write(self.low_cmd)
            except Exception as retry_exc:  # noqa: BLE001
                self.writer_failed = True
                self.writer_failure = repr(retry_exc)
                logging.critical(
                    "DAMPING WRITE FAILED (%r): use the physical e-stop immediately",
                    retry_exc,
                )
                raise

    def stop_writer(self, timeout=1.0):
        """Stop the recurrent publisher after damping has been sent for several ticks."""
        thread = self.lowCmdWriteThreadPtr
        if thread is not None:
            # Unitree RecurrentThread.Wait() discards its superclass return value.
            # Query the Future state after requesting stop instead of bool(None).
            thread.Wait(timeout)
            result = thread.GetResult(0.0)
            stopped = getattr(result, "code", None) == 0
            if not stopped:
                logging.critical(
                    "command writer did not stop within %.1f s; use the physical e-stop",
                    timeout,
                )
            return bool(stopped)
        return True

    # -------------------------------------------------------------- reporting
    def save_trace(self, outdir="runs"):
        # The inherited plots remain useful.  Temporarily expose the full policy
        # horizon so its text summary does not call the trained tail an overrun.
        reference_duration = self.ref.duration
        self.ref.duration = (self.n_ticks - 1) * self.dt
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                stem = super().save_trace(outdir)
        finally:
            self.ref.duration = reference_duration
        if stem is None:
            output = captured.getvalue()
            if output:
                print(output, end="")
            return None

        anchor = "ref" if self.anchor_ref else "home"
        reason = self.abort_reason or ("unspecified safety guard" if self.aborted else "none")
        geo = np.asarray(self.trace["ori_err_deg"], dtype=float)
        geo_line = (
            f"geodesic ori error: RMS {np.sqrt(np.mean(geo ** 2)):.2f} deg, "
            f"max {np.max(geo):.2f} deg\n" if len(geo) else ""
        )

        def correct(text):
            text = text.replace("kinematic ref-anchor", f"kinematic {anchor}-anchor")
            text = text.replace("ABORTED (tilt)", f"ABORTED ({reason})")
            text = text.replace("|resid|", "|cmd-ref|")
            text = text.replace("max|resid|", "max|cmd-ref|")
            worst_grav_z = max(self.trace["tilt"]) if self.trace["tilt"] else float("nan")
            text = re.sub(
                r"worst tilt \(grav_z\) : [^\n]+",
                f"worst tilt (grav_z) : {worst_grav_z:+.3f}\n"
                f"orientation walls    : {self.ori_stance_deg:.1f} deg stance / "
                f"{self.ori_stance_deg * self.flight_ori_mult:.1f} deg flight, "
                f"cap {self.abort_ori_cap_deg:.1f} deg; grav_z {self.max_gravity_z:.2f}; "
                f"upright fallback {'on' if self.upright_fallback else 'OFF'}",
                text,
            )
            if geo_line and "contact-blind:" in text:
                text = text.replace("contact-blind:", geo_line + "contact-blind:", 1)
            return text

        output = correct(captured.getvalue())
        if output:
            print(output, end="")
        report_path = Path(stem + ".txt")
        if report_path.is_file():
            report_path.write_text(correct(report_path.read_text()))
        logging.info("abort reason: %s", reason)
        return stem


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True,
        help="cartwheel kinematic checkpoint; meta.json is discovered beside it",
    )
    parser.add_argument("--meta", default=None, help="override sibling meta.json")
    parser.add_argument(
        "--traj", default=None,
        help="explicit trajectory override; relative paths are resolved from cwd",
    )
    parser.add_argument(
        "--policy-tail-s", type=float, default=None,
        help="override meta ep_tail; default is the metadata's nominal trained tail",
    )
    parser.add_argument(
        "--abort-ori-cap-deg", "--abort_ori_deg", dest="abort_ori_cap_deg",
        type=float, default=ABORT_ORI_CAP_DEG_DEFAULT,
        help="hardware cap on the phase-aware training orientation wall (0,180]",
    )
    parser.add_argument(
        "--ori-stance-deg", type=float, default=None,
        help="explicitly override meta term_ori_err; values above training allow "
             "untrained divergence states",
    )
    parser.add_argument(
        "--flight-ori-mult", type=float, default=None,
        help="override meta/default planned-flight multiplier (default 1.6 if absent)",
    )
    parser.add_argument(
        "--disable-upright-fallback", action="store_true",
        help="explicitly disable the extra grav_z>-0.40 guard when reference is upright",
    )
    parser.add_argument(
        "--state-timeout-s", type=float, default=STATE_TIMEOUT_S_DEFAULT,
        help="damp if the newest LowState is older than this",
    )
    parser.add_argument("--ready-max-q-err", type=float, default=READY_MAX_Q_ERR_DEFAULT)
    parser.add_argument("--ready-max-qd", type=float, default=READY_MAX_QD_DEFAULT)
    parser.add_argument("--ready-max-gyro", type=float, default=READY_MAX_GYRO_DEFAULT)
    parser.add_argument(
        "--ready-max-tilt-deg", type=float, default=READY_MAX_TILT_DEG_DEFAULT,
    )
    parser.add_argument("--ready-stable-s", type=float, default=READY_STABLE_S_DEFAULT)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="load and validate without initializing DDS (Unitree SDK not required)",
    )
    parser.add_argument("--out", default="data", help="parent trace directory")
    parser.add_argument("iface", nargs="?", default=None)
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    meta, meta_path, checkpoint = load_meta(args.checkpoint, args.meta)
    custom = Custom(
        checkpoint,
        meta,
        args.traj,
        meta_path=meta_path,
        policy_tail_s=args.policy_tail_s,
        abort_ori_cap_deg=args.abort_ori_cap_deg,
        ori_stance_deg=args.ori_stance_deg,
        flight_ori_mult=args.flight_ori_mult,
        upright_fallback=not args.disable_upright_fallback,
        state_timeout_s=args.state_timeout_s,
        ready_max_q_err=args.ready_max_q_err,
        ready_max_qd=args.ready_max_qd,
        ready_max_gyro=args.ready_max_gyro,
        ready_max_tilt_deg=args.ready_max_tilt_deg,
        ready_stable_s=args.ready_stable_s,
    )

    if args.dry_run:
        logging.info(
            "DRY RUN OK: obs=%d, policy=%d+%d=%d ticks, blind=%s, anchor=%s, "
            "qd=%s, lpf=%g",
            custom.n_obs, custom.track_ticks, custom.policy_tail_ticks,
            custom.policy_ticks, custom.blind,
            "ref" if custom.anchor_ref else "home", custom.anchor_qd,
            custom.LPF_BETA,
        )
        logging.info(
            "base Euler chain: %s | state watchdog %.0f ms | artifacts: %s/%s_<stamp>/",
            "->".join(name[-2:] for name in custom.ref.base_euler_chain),
            custom.state_timeout_s * 1000.0, args.out, custom.run_tag,
        )
        return 0

    if not SDK_AVAILABLE:
        raise SystemExit(
            "ABORT: Unitree SDK is unavailable on this machine; use --dry-run. "
            f"Import error: {SDK_IMPORT_ERROR}"
        )

    print("WARNING: harness, mats, and a clear robot workspace are required.")
    input("Press Enter to initialize low-level control...")
    if args.iface:
        _base.ChannelFactoryInitialize(0, args.iface)
    else:
        _base.ChannelFactoryInitialize(0)
    custom.Init()
    custom.Start()

    launched = False
    saved = False
    try:
        while True:
            if custom.aborted:
                if not saved:
                    saved = True
                    try:
                        custom.save_trace(args.out)
                    except Exception:  # noqa: BLE001 - robot remains in damping
                        logging.exception("save_trace failed (robot is still damping)")
                if custom.writer_failed:
                    print(
                        "CRITICAL: DDS damping write failed "
                        f"({custom.writer_failure}); use the physical e-stop immediately."
                    )
                else:
                    print(
                        f"Aborted ({custom.abort_reason}); robot is damping. "
                        "Ctrl-C when secured."
                    )
                time.sleep(10.0)
                continue

            if not launched and custom.ready_to_launch:
                input(
                    "Robot is stable, fresh, upright, and holding q0. "
                    "Press Enter to LAUNCH..."
                )
                if custom.aborted:
                    continue
                launched = custom.request_launch()

            if custom.settle_percent >= 1.0:
                time.sleep(1.0)
                if not saved:
                    saved = True
                    custom.save_trace(args.out)
                input(
                    "Trajectory complete; robot is holding qf. Secure/support it, "
                    "then press Enter to DAMP and exit..."
                )
                custom._abort_now("normal completion: operator requested damping")
                time.sleep(0.25)
                stopped = custom.stop_writer()
                if custom.writer_failed or not stopped:
                    raise RuntimeError(
                        "normal-completion damping/writer shutdown was not confirmed; "
                        "use the physical e-stop"
                    )
                print("Done; damping was published before the writer stopped.")
                return 0
            time.sleep(0.25)
    except BaseException as exc:
        reason = ("Ctrl-C soft e-stop" if isinstance(exc, KeyboardInterrupt)
                  else f"main-thread exception: {type(exc).__name__}: {exc}")
        custom._abort_now(reason)
        time.sleep(0.25)
        if isinstance(exc, KeyboardInterrupt):
            logging.warning("Ctrl-C -> damped stop requested; saving trace")
        else:
            logging.exception("main thread failed -> damped stop requested")
        if launched and not saved:
            try:
                custom.save_trace(args.out)
            except Exception:  # noqa: BLE001
                logging.exception("save_trace failed (robot is damping)")
        custom.stop_writer()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
