"""Go2 hardware deployment for the KINEMATIC (chill_out / dm_v*, kin_v4+) family.

Sibling of deploy_meta.py, which flies the residual hendeca/matrix line and is kept
untouched for those checkpoints. THIS script implements the kin_v4 contract the dm
family was trained on ("control_mode": "kinematic" in meta.json):

  * REF ANCHOR, LATCH-HELD (mode B): q* = q_ref(t_latch) + a * action_scale, with
    q_ref sampled ONCE per 50 Hz tick on the NOMINAL reference clock and held for
    the whole tick (publishing at 50 Hz holds it for us). Zero action = kinematic
    playback. q* is hard-clamped to jlimit_clamp_frac of the joint range.
  * qd* = qd_ref (latch-held) when meta kin_anchor_qd, else 0.  NO tau_ff — the
    only feedforward this script ever writes is the decaying stand-up integrator.
  * obs frames are 58-wide (obs_act_hist): [measured30 | contact4 | tau12 | act12],
    where act12 is the most recent EMITTED action (post-clip, PRE-latency — the
    policy's own fresh output, never the delayed plant-side one).
  * obs_tau_mode "cmd": the tau channel is the PRE-envelope commanded torque
    computed from the held targets against the frame's own (1-step stale)
    measured q/qd — deploy-exact by construction, no motor model involved.
  * NO velocity estimator / odometry: the dm family runs obs_velest=False. A meta
    asking for velest aborts at load (fly those ckpts with deploy_meta.py).

Everything this script and training must agree on is read from meta.json and
asserted at load, before the robot moves: kp/kd/action_scale, obs layout flags
(obs_act_hist/obs_ori_err/obs_contact_prev/obs_contact_blind/obs_o1_scale,
obs_tau_mode), history/preview geometry, act_lpf_beta, jlimit_clamp_frac,
kin_anchor/kin_anchor_qd, observation_space/action_space.

Plumbing (SDK channels, stand-up fold/align/hold + integrator handoff, tilt abort,
50 Hz RecurrentThread, trace/report artifacts, high-rate tau_est log) is ported
from deploy_meta.py. Removed relative to it: ff-comp bake (kin refs carry no
torques; nonzero comps in meta abort), resid_scale (the action IS the control
here, not a residual to attenuate), velest odometry.

HW sim2real readout (dm_v7/v8 verdict): watch the tap force on the drift-side
front foot during diag stances — ~40-80 N means the plant model is good; 150 N+
means the sysid has drifted. The contact-profile artifact shows exactly this.

Usage:
    python3 deploy_dm.py --checkpoint runs/dm_v11_json/model_1000.pt [iface]
    # meta.json is auto-found next to the checkpoint; override with --meta
"""
import time
import sys
import os
import json
import logging


import numpy as np


from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
import unitree_legged_const as go2
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


# --------------------------------------------------------------------- orders
ISO_NAMES = [f"{leg}_{part}_joint" for part in ("hip", "thigh", "calf")
             for leg in ("FL", "FR", "RL", "RR")]
MOTOR_FROM_ISO = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
FOOTFORCE_FROM_ISO_FOOT = [1, 0, 3, 2]           # FL FR RL RR <- (FR FL RR RL)
FOOT_NAMES = ["FL", "FR", "RL", "RR"]


# --------------------------------------------- fixed control-loop constants
CTRL_DT = 0.02
OBS_QD_SCALE = 1.0 / 15.0
OBS_TAU_SCALE = 1.0 / 45.0
OBS_ACC_SCALE = 1.0 / 9.81
BODY_WEIGHT = 150.0                              # N, load-channel normalizer
TTC_EDGE_CLIP_S = 0.3
CONTACT_FORCE_THRESHOLD_HW = 20.0                # raw units, binary contacts. TUNABLE
FOOT_FORCE_TO_N = 1.0                            # raw foot_force -> Newtons. CALIBRATE
# (both are obs-NO-OPS for a contact-blind checkpoint; they still feed the logs)


# GO2 joint ranges, ISO order (hip x4, thigh x4, calf x4). Source: mujoco_menagerie
# unitree_go2 go2.xml (same limits the Isaac USD exposes to joint_pos_limits, which
# is what training clamped against). NOTE rear thighs have a DIFFERENT range from
# the front ones — a single per-part triple would clamp RL/RR thigh wrongly.
JOINT_RANGE_ISO = np.array([
    [-1.0472, 1.0472], [-1.0472, 1.0472], [-1.0472, 1.0472], [-1.0472, 1.0472],
    [-1.5708, 3.4907], [-1.5708, 3.4907], [-0.5236, 4.5379], [-0.5236, 4.5379],
    [-2.7227, -0.83776], [-2.7227, -0.83776], [-2.7227, -0.83776], [-2.7227, -0.83776]])


MDC_VBAT, MDC_PBAT = 28.8, 1728.0
MDC_GR, MDC_KT, MDC_R = 6.33, 0.26, 0.66
MDC_ALPHA = MDC_R / (MDC_KT * MDC_GR)
MDC_BETA = MDC_GR * MDC_KT
TAU_MAX_ISO = np.array([23.7] * 8 + [45.43] * 4)


def mdc_apply(tau_des, qd):
    """Voltage/power envelope estimate — LOGGING ONLY in this script (obs_tau_mode
    "mdc" is a legacy approximation; the dm family trains and flies "cmd")."""
    V = np.clip(MDC_ALPHA * tau_des + MDC_BETA * qd, -MDC_VBAT, MDC_VBAT)
    tau_v = (V - MDC_BETA * qd) / MDC_ALPHA
    A = float(np.sum(MDC_R * tau_v * tau_v / (MDC_KT * MDC_GR) ** 2))
    B = float(np.sum(tau_v * qd))
    if A + B > MDC_PBAT:
        eta = 2.0 * MDC_PBAT / (B + np.sqrt(B * B + 4.0 * A * MDC_PBAT) + 1e-8)
        tau_v = tau_v * np.clip(eta, 0.0, 1.0)
    return np.clip(np.nan_to_num(tau_v), -TAU_MAX_ISO, TAU_MAX_ISO)


# ------------------------------------------------------------ quaternion math
def euler_xyz_to_quat_wxyz(e):
    def axis_quat(a, ax):
        q = np.zeros((len(a), 4))
        q[:, 0] = np.cos(a / 2.0)
        q[:, 1 + ax] = np.sin(a / 2.0)
        return q

    def qmul_batch(a, b):
        aw, ax, ay, az = a.T
        bw, bx, by, bz = b.T
        return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                         aw * bx + ax * bw + ay * bz - az * by,
                         aw * by - ax * bz + ay * bw + az * bx,
                         aw * bz + ax * by - ay * bx + az * bw], axis=1)
    e = np.atleast_2d(np.asarray(e, float))
    return qmul_batch(qmul_batch(axis_quat(e[:, 0], 0), axis_quat(e[:, 1], 1)),
                      axis_quat(e[:, 2], 2))


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def yaw_from_quat_wxyz(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rotmat_from_quat_wxyz(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_about_z(yaw):
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def _wrap_pi(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def quat_to_euler_xyz(q):
    """Inverse of euler_xyz_to_quat_wxyz (R = Rx Ry Rz) — must match the reference's
    convention or measured and reference RPY are not comparable."""
    R = rotmat_from_quat_wxyz(q)
    ry = np.arcsin(np.clip(R[0, 2], -1.0, 1.0))
    rx = np.arctan2(-R[1, 2], R[2, 2])
    rz = np.arctan2(-R[0, 1], R[0, 0])
    return np.array([rx, ry, rz])


def build_gather_map(src_joint_names, target_joint_names):
    """Indices reordering a src-ordered 12-vec into target order (q_t = q_s[gather]),
    matched on the leg+part token. NEVER resolve these positionally."""
    src_index = {n.replace("_joint", ""): i for i, n in enumerate(src_joint_names)}
    gather = []
    for name in target_joint_names:
        token = name.replace("_joint", "")
        if token not in src_index:
            raise KeyError(f"joint '{name}' not in reference joint set {list(src_index)}")
        gather.append(src_index[token])
    return np.asarray(gather, dtype=np.int64)


# ----------------------------------------------------------------- reference
class HopscotchRef:
    """Numpy port of HopscotchReferenceManager's load path, both formats. The
    KINEMATIC family never commands reference torques, so `u` is not kept and no
    ff comp is baked (nonzero comps in meta abort in Custom)."""

    def __init__(self, path):
        if str(path).endswith(".npz"):
            q, v, contact, force_ref, src_names, dt = self._load_npz(path)
            fmt = "gridded npz"
        else:
            q, v, contact, force_ref, src_names, dt = self._load_json(path)
            fmt = "modes json (half-open)"
        self.dt = dt
        self.force_ref = force_ref                 # (T,4) planned |F| per foot, N

        gather = build_gather_map(src_names, ISO_NAMES)
        self.q_ref = q[:, 6:18][:, gather]
        self.qd_ref = v[:, 6:18][:, gather]

        self.base_pos = q[:, 0:3].copy()           # z offset irrelevant (only xy used)
        self.base_quat = euler_xyz_to_quat_wxyz(q[:, 3:6])
        self.contact = contact
        self.airborne = (~contact).all(axis=1)

        self.T_state = self.q_ref.shape[0]
        self.duration = (self.T_state - 1) * self.dt

        which = np.zeros(self.T_state, dtype=np.int64)
        i = jcount = 0
        while i < self.T_state:
            if self.airborne[i]:
                j = i
                while j < self.T_state and self.airborne[j]:
                    j += 1
                if (j - i) >= 50:
                    jcount += 1
                    which[i:] = jcount
                i = j
            else:
                i += 1
        self.which_jump = which
        self.n_jumps = jcount

        # per-foot knots until that foot's planned contact bit next flips
        edge = np.full((self.T_state, 4), self.T_state, dtype=np.float64)
        nxt_flip = np.full(4, 2.0 * self.T_state)
        for i in range(self.T_state - 2, -1, -1):
            flip = contact[i + 1] != contact[i]
            nxt_flip[flip] = i + 1
            edge[i] = nxt_flip - i
        self.ttc_edge = edge * self.dt             # (T,4) seconds
        logging.info("reference [%s]: %d knots @ %.4f s, %.2f s, %d jumps",
                     fmt, self.T_state, self.dt, self.duration, self.n_jumps)

    # ------------------------------------------------------------- loaders
    @staticmethod
    def _load_json(path):
        """Legacy list-of-modes JSON. HALF-OPEN concat: drop each non-final mode's
        terminal row so the boundary knot keeps the NEXT mode's post-impact state."""
        modes = json.load(open(path))
        qs, vs, cs, fs = [], [], [], []
        for i, m in enumerate(modes):
            q = np.asarray(m["q"], float)
            v = np.asarray(m["v"], float)
            lam = np.asarray(m["lam"], float)      # (steps, 3*n_active) active feet only
            cm = np.zeros((len(q), 4), bool)
            fm = np.zeros((len(q), 4))             # planned |F| per foot (N)
            for ci, foot in enumerate(m["contacts"]):
                idx = FOOT_NAMES.index(foot.split("_")[0])
                cm[:, idx] = True
                fm[:, idx] = np.linalg.norm(lam[:, 3 * ci:3 * ci + 3], axis=1)
            if i < len(modes) - 1:                 # HALF-OPEN: drop non-final LAST row
                q, v, cm, fm = q[:-1], v[:-1], cm[:-1], fm[:-1]
            qs.append(q); vs.append(v); cs.append(cm); fs.append(fm)
        src = list(modes[0]["joint_names"][6:])
        return (np.concatenate(qs), np.concatenate(vs), np.concatenate(cs),
                np.concatenate(fs), src, float(modes[0]["dt"]))

    @staticmethod
    def _load_npz(path):
        """PRE-GRIDDED npz: one row per 1 kHz knot, already concatenated. NO half-open
        drop here — the impact row already carries the post-impact state."""
        d = np.load(path, allow_pickle=True)
        q = np.asarray(d["q"], float)
        v = np.asarray(d["v"], float)
        cin = np.asarray(d["contact"]).astype(bool)
        lam = np.asarray(d["lam"], float).reshape(len(q), 4, 3)
        contact = np.zeros((len(q), 4), bool)
        force = np.zeros((len(q), 4))
        for ci, foot in enumerate(str(x) for x in d["feet"]):
            idx = FOOT_NAMES.index(foot.split("_")[0])                      # BY NAME
            contact[:, idx] = cin[:, ci]
            force[:, idx] = np.linalg.norm(lam[:, ci], axis=1)
        src = [str(x) for x in d["joint_names"][6:18]]
        return q, v, contact, force, src, 1.0 / float(d["rate"])

    # -------------------------------------------------------------- sampling
    def _index(self, t):
        f = t / self.dt
        i0 = int(np.clip(np.floor(f), 0, self.T_state - 2))
        frac = float(np.clip(f - i0, 0.0, 1.0))
        return i0, frac

    def ref_at(self, t):
        """(q_ref, qd_ref) lerped at t — the 50 Hz latch samples exactly this once
        per tick; holding it for the tick is what mode B means on hardware."""
        i0, fr = self._index(t)
        q = (1 - fr) * self.q_ref[i0] + fr * self.q_ref[i0 + 1]
        qd = (1 - fr) * self.qd_ref[i0] + fr * self.qd_ref[i0 + 1]
        return q, qd

    def preview(self, t, preview_dt, num_future):
        """[q12 qd12 tau12(=0)] per point. The kin family's preview tau_ff slots are
        ZEROED in training (mocap has no torques) — mirrored here as literal zeros."""
        outs = []
        for j in range(num_future + 1):
            q, qd = self.ref_at(t + j * preview_dt)
            outs += [q, qd, np.zeros(12)]
        return np.concatenate(outs)                # 36*(1+num_future)

    def phase_info(self, t):
        i0, _ = self._index(t)
        phase = float(np.clip(t / self.duration, 0.0, 1.0))
        return np.concatenate([[phase], self.contact[i0].astype(float),
                               [float(self.airborne[i0])],
                               [self.which_jump[i0] / max(1, self.n_jumps)]])  # 7

    def contact_pack(self, t, preview_dt, num_future):
        """(masks 4*(1+num_future), ttc_edge_norm 4) — the plan-derived half."""
        masks = []
        for j in range(num_future + 1):
            i0, _ = self._index(t + j * preview_dt)
            masks.append(self.contact[i0].astype(float))
        i0, _ = self._index(t)
        ttc_e = np.clip(self.ttc_edge[i0] / TTC_EDGE_CLIP_S, 0.0, 1.0)
        return np.concatenate(masks), ttc_e

    def base_ref_at(self, t):
        """(pos xy (2), quat wxyz (4)) — lerped like the training manager."""
        i0, fr = self._index(t)
        pos = (1 - fr) * self.base_pos[i0] + fr * self.base_pos[i0 + 1]
        q0, q1 = self.base_quat[i0], self.base_quat[i0 + 1]
        if np.dot(q0, q1) < 0:
            q1 = -q1
        q = (1 - fr) * q0 + fr * q1
        return pos[:2], q / max(np.linalg.norm(q), 1e-8)


# ----------------------------------------------------------------- policy
def _elu(x):
    return np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))


class Policy:
    """Checkpoint -> numpy: EmpiricalNormalization + actor MLP [512,256,128].
    The dm family has no estimator head (obs_velest is refused upstream)."""

    def __init__(self, ckpt_path, n_obs, n_act):
        import torch
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck["model_state_dict"]
        self.W = [sd[f"actor.{i}.weight"].numpy().astype(np.float64) for i in (0, 2, 4, 6)]
        self.b = [sd[f"actor.{i}.bias"].numpy().astype(np.float64) for i in (0, 2, 4, 6)]
        on = ck["obs_norm_state_dict"]
        self.mean = on["_mean"].numpy().ravel().astype(np.float64)
        self.std = on["_std"].numpy().ravel().astype(np.float64)
        if self.W[0].shape[1] != n_obs:
            raise SystemExit(f"ABORT: actor input {self.W[0].shape[1]} != meta "
                             f"observation_space {n_obs}")
        if self.W[-1].shape[0] != n_act:
            raise SystemExit(f"ABORT: actor output {self.W[-1].shape[0]} != meta "
                             f"action_space {n_act}")
        if self.mean.shape != (n_obs,):
            raise SystemExit(f"ABORT: obs_norm {self.mean.shape} != ({n_obs},)")
        logging.info("policy loaded: %s (iter %s) obs=%d act=%d",
                     ckpt_path, ck.get("iter"), n_obs, n_act)

    def __call__(self, obs):
        x = (obs - self.mean) / (self.std + 1e-2)
        for W, b in zip(self.W[:-1], self.b[:-1]):
            x = _elu(W @ x + b)
        return np.clip(self.W[-1] @ x + self.b[-1], -1.0, 1.0)


def load_meta(ckpt_path, explicit=None):
    p = explicit or os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "meta.json")
    if not os.path.exists(p):
        raise SystemExit(f"ABORT: no meta.json at {p} (pass --meta). This script is "
                         f"meta-driven by design; it will not guess a model's contract.")
    with open(p) as f:
        meta = json.load(f)
    logging.info("meta: %s", p)
    return meta, p


def _stem(path):
    s = os.path.splitext(os.path.basename(str(path)))[0]
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in s) or "unnamed"


class Custom:
    def __init__(self, ckpt_path, meta, traj_override=None, meta_path=None):
        self.dt = CTRL_DT
        self.motiontime = 0

        # ---------------- contract checks: fail here, not on the robot ----------
        def need(key, want, why=""):
            got = meta.get(key, want)
            if isinstance(want, float) and isinstance(got, (int, float)):
                ok = abs(float(got) - want) < 1e-9
            else:
                ok = got == want
            if not ok:
                raise SystemExit(f"ABORT: meta['{key}']={got!r}, this script implements "
                                 f"{want!r}. {why}")
        need("control_mode", "kinematic",
             "Residual/hendeca checkpoints fly with deploy_meta.py, not this script.")
        need("compliant_feet", False, "Compliant feet are a sim-only plant axis.")
        need("obs_priv_base", False, "Teacher run: privileged base obs, not deployable.")
        need("hold_targets", True, "Mode B (latch-held targets) is the trained plant; a "
                                   "mode-A (streamed) ckpt cannot be flown by this tick.")
        need("obs_velest", False, "No odometry in this runtime; velest ckpts are the "
                                  "hendeca line -> deploy_meta.py.")
        need("ff_damping_comp", 0.0, "Kinematic contract has no feedforward to bake.")
        need("ff_armature_comp", 0.0, "Kinematic contract has no feedforward to bake.")
        if list(meta.get("joint_names", ISO_NAMES)) != ISO_NAMES:
            raise SystemExit("ABORT: meta joint_names differ from ISO_NAMES; the motor "
                             "remap would be wrong.")
        n_act = int(meta.get("action_space", 12))
        if n_act != 12:
            raise SystemExit(f"ABORT: action_space={n_act}. The 13-action phase head "
                             f"(v14ph) is not part of the kinematic family.")

        self.KP = float(meta["kp"])
        self.KD = float(meta["kd"])
        self.ACTION_SCALE = float(meta["action_scale"])
        self.H = int(meta.get("obs_history_len", 10))
        self.NUM_FUTURE = int(meta.get("num_future", 2))
        self.CLIP_OBS = float(meta.get("clip_obs", 10.0))
        self.PREVIEW_DT = int(meta.get("preview_steps", 5)) * CTRL_DT
        self.use_ori_err = bool(meta.get("obs_ori_err", False))
        self.use_cprev = bool(meta.get("obs_contact_prev", False))
        self.blind = bool(meta.get("obs_contact_blind", False))
        self.o1_scale = bool(meta.get("obs_o1_scale", False))
        self.act_hist = bool(meta.get("obs_act_hist", False))
        self.anchor_ref = str(meta.get("kin_anchor", "home")) == "ref"
        self.anchor_qd = bool(meta.get("kin_anchor_qd", False))
        self.LPF_BETA = float(meta.get("act_lpf_beta", 0.0))
        self.tau_mode = str(meta.get("obs_tau_mode", "mdc"))
        if self.tau_mode not in ("cmd", "mdc", "zero"):
            raise SystemExit(f"ABORT: obs_tau_mode {self.tau_mode!r} not implemented.")
        if self.tau_mode == "mdc":
            logging.warning("obs_tau_mode 'mdc': HW approximates the applied-torque "
                            "channel with the MDC model on held cmds (dm family "
                            "trains 'cmd', which is exact here).")

        # hard q* clamp at jlimit_clamp_frac of the joint range (training's clamp)
        frac = float(meta.get("jlimit_clamp_frac", 0.97))
        mid = (JOINT_RANGE_ISO[:, 0] + JOINT_RANGE_ISO[:, 1]) / 2.0
        half = (JOINT_RANGE_ISO[:, 1] - JOINT_RANGE_ISO[:, 0]) / 2.0
        self.q_clamp_lo = mid - frac * half
        self.q_clamp_hi = mid + frac * half

        # q_home: the meta's canonical stand pose — the anchor only when kin_anchor
        # is "home" (legacy kin_v0-v3); the dm family anchors on the reference.
        self.q_home = np.asarray(meta["q_home"], float)

        mdc = meta.get("mdc", {})
        if mdc and abs(float(mdc.get("Vbat", MDC_VBAT)) - MDC_VBAT) > 1e-6:
            raise SystemExit(f"ABORT: meta MDC Vbat {mdc.get('Vbat')} != {MDC_VBAT}")

        # ---------------- reference (meta-driven, format auto-dispatch) ---------
        traj = traj_override or meta["traj_path"]
        if not os.path.exists(traj):
            local = os.path.join("hopscotch_utils", os.path.basename(traj))
            if os.path.exists(local):
                logging.warning("traj_path %s missing; using local copy %s", traj, local)
                traj = local
            else:
                raise SystemExit(f"ABORT: reference not found: {traj} (nor {local}). "
                                 f"Copy it next to the checkpoint or pass --traj.")
        self.traj_path = traj
        self.ckpt_path = ckpt_path
        self.meta_path = meta_path
        self.ref = HopscotchRef(traj)
        self.n_ticks = int(np.ceil(self.ref.duration / self.dt))

        # ---------------- obs layout, derived then asserted ---------------------
        frame_dim = 58 if self.act_hist else 46    # measured30+contact4+tau12[+act12]
        tail = 7 + 5                               # phase + attitude
        tail += 3 if self.use_ori_err else 0
        tail += (4 * (1 + self.NUM_FUTURE) + 4 + 4) if self.use_cprev else 0
        n_obs = frame_dim * self.H + 36 * (1 + self.NUM_FUTURE) + tail
        meta_obs = int(meta.get("observation_space", n_obs))
        if n_obs != meta_obs:
            raise SystemExit(f"ABORT: this script builds a {n_obs}-dim obs but meta says "
                             f"{meta_obs}. Obs-flag combination is not supported.")
        self.frame_dim = frame_dim

        # O(1) conditioning (meta-keyed). Frame: [gyro3 acc3 q12 qd12 | contact4 |
        # tau12 | act12]; act slots stay 1.0 (clipped actions are O(1) natively).
        # Preview per point: [q12 qd12 tau12] — tau slots ALWAYS zeroed for kin,
        # independent of o1_scale (the env zeroes them via its scale vector; the
        # preview above already writes literal zeros, the 0 here is belt+braces).
        fs = np.ones(frame_dim)
        pv = np.ones(36)
        if self.o1_scale:
            fs[3:6] = OBS_ACC_SCALE
            fs[18:30] = OBS_QD_SCALE
            fs[34:46] = OBS_TAU_SCALE
            pv[12:24] = OBS_QD_SCALE
        pv[24:36] = 0.0
        self.frame_scale = fs
        self.actor_scale = np.concatenate([np.tile(fs, self.H),
                                           np.tile(pv, 1 + self.NUM_FUTURE),
                                           np.ones(tail)])
        assert self.actor_scale.shape == (n_obs,), self.actor_scale.shape
        self.n_obs = n_obs

        self.policy = Policy(ckpt_path, n_obs, n_act)
        logging.info("KINEMATIC: q* = %s + a*%g rad (clamp %g of range), qd* = %s, "
                     "tau_ff = 0, lpf_beta = %g | act_hist %s (frame %d) | "
                     "obs %d | zero action = %s",
                     "q_ref(latch)" if self.anchor_ref else "q_home",
                     self.ACTION_SCALE, frac,
                     "qd_ref" if self.anchor_qd else "0", self.LPF_BETA,
                     "IN" if self.act_hist else "out", frame_dim, n_obs,
                     "KINEMATIC PLAYBACK" if self.anchor_ref else "STAND AT HOME")
        logging.info("CONTACT-BLIND: %s -- history contact bits + load block %s",
                     self.blind, "ZEROED (foot-force calibration is a no-op)" if self.blind
                     else "live from foot_force")

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None

        # ---- stand-up (proven recipe), Unitree MOTOR order ----
        self.Kp_stand, self.Kd_stand = 60.0, 5.0
        self.foldPos = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
                        -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]
        self.q0_motor = np.zeros(12)
        self.q0_motor[MOTOR_FROM_ISO] = self.ref.q_ref[0]
        self.qf_motor = np.zeros(12)
        self.qf_motor[MOTOR_FROM_ISO] = self.ref.q_ref[-1]
        self.startPos = [0.0] * 12
        self.fold_duration, self.fold_percent = 50, 0
        self.align_duration, self.align_percent = 50, 0
        self.hold_duration, self.hold_percent = 150, 0
        self.Ki, self.tau_i_max = 100.0, 15.0
        self.tau_i = np.zeros(12)
        self.handoff_fade_ticks = 25

        # ---- policy-phase state ----
        self.ii = 0
        self.start_policy = False             # armed gate: main thread's Enter releases it
        self._armed_logged = False
        self.handoff_done = False
        self.prev_action = np.zeros(12)       # a_{t-1}: the 1-step-latency ACTUATED action
        self.last_emitted = np.zeros(12)      # a_t: freshest output, the act-hist channel
        self.prev_measured = None
        self.held_targets = None              # (q_target, qd_target) of the LAST tick
        self.lpf_prev = None                  # one-pole target-filter state
        self.history = None
        self.q_align = None                   # yaw-alignment quat (IMU -> ref world)
        self.settle_percent = 0
        self.settle_duration = 50
        self.aborted = False

        # ---- floating-base trace (appended once per policy tick) ----
        self.trace = {k: [] for k in
                      ("t", "rpy", "rpy_ref", "ori_err", "ref_xy", "tilt", "action",
                       "act_emitted", "contacts", "forces", "q", "qd", "q_ref",
                       "qd_ref", "q_target", "tau_obs", "tau_p", "tau_d", "tau_cmd")}
        self.run_tag = "_".join([_stem(ckpt_path),
                                 _stem(meta_path) if meta_path else "nometa",
                                 "kin",
                                 _stem(traj)])

        # ---- high-rate measured-torque log (DDS callback thread) ----
        self.hf_on = False
        self.hf_t0 = 0.0
        self.hf_t = []
        self.hf_tau = []
        self.HF_MAX = 400000

        self._loop_hz = 0.0
        self._rate_t0 = None
        self._rate_n = 0

        self.firstRun = True
        self.lowCmdWriteThreadPtr = None
        self.crc = CRC()

    # ---------------------------------------------------------------- public
    def Init(self):
        self.InitLowCmd()
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)

        self.sc = SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()
        status, result = self.msc.CheckMode()
        while result['name']:
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.dt, target=self.LowCmdWrite, name="writebasiccmd")
        self.lowCmdWriteThreadPtr.Start()

    # --------------------------------------------------------------- private
    def InitLowCmd(self):
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].q = go2.PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = go2.VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def LowStateMessageHandler(self, msg: LowState_):
        self.low_state = msg
        if self.hf_on and len(self.hf_t) < self.HF_MAX:
            ms = msg.motor_state
            self.hf_t.append(time.perf_counter() - self.hf_t0)
            self.hf_tau.append([ms[m].tau_est for m in MOTOR_FROM_ISO])

    # ------------------------------------------------------------ obs pieces
    def _imu_quat(self):
        return np.asarray(self.low_state.imu_state.quaternion, float)

    def _read_iso(self):
        q = np.array([self.low_state.motor_state[m].q for m in MOTOR_FROM_ISO])
        qd = np.array([self.low_state.motor_state[m].dq for m in MOTOR_FROM_ISO])
        return q, qd

    def _read_foot_forces_n(self):
        raw = np.array([self.low_state.foot_force[j] for j in FOOTFORCE_FROM_ISO_FOOT],
                       dtype=float)
        return raw * FOOT_FORCE_TO_N

    def _aligned_quat(self):
        return qmul(self.q_align, self._imu_quat())

    def _seed_policy_state(self):
        """History seeded EXACTLY like the env's _reset_idx at t0=0: rest specific-
        force +1g, start-phase joints, reference contact mask (zero when blind),
        zero torque, zero act slots (= on-ref)."""
        sensor_init = np.zeros(30)
        sensor_init[5] = 9.81
        sensor_init[6:18] = self.ref.q_ref[0]
        cont0 = self.ref.phase_info(0.0)[1:5]
        if self.blind:
            cont0 = np.zeros_like(cont0)
        frame_init = np.concatenate(
            [sensor_init, cont0, np.zeros(12)]
            + ([np.zeros(12)] if self.act_hist else []))
        self.history = np.tile(frame_init, (self.H, 1))
        self.prev_measured = sensor_init
        self.prev_action = np.zeros(12)
        self.last_emitted = np.zeros(12)
        self.held_targets = None
        self.lpf_prev = None
        yaw_ref0 = yaw_from_quat_wxyz(self.ref.base_quat[0])
        d_yaw = yaw_ref0 - yaw_from_quat_wxyz(self._imu_quat())
        self.q_align = quat_about_z(d_yaw)
        self.handoff_done = True
        self.hf_t0 = time.perf_counter()
        self.hf_on = True
        logging.info("HANDOFF: policy takes over (yaw align %+.1f deg)", np.degrees(d_yaw))

    def _policy_tick(self):
        t = self.ii * self.dt
        q, qd = self._read_iso()
        gyro = np.asarray(self.low_state.imu_state.gyroscope, float)
        accel = np.asarray(self.low_state.imu_state.accelerometer, float)
        measured_now = np.concatenate([gyro, accel, q, qd])
        forces_n = self._read_foot_forces_n()
        contacts_true = (forces_n > CONTACT_FORCE_THRESHOLD_HW).astype(float)
        contacts_obs = np.zeros(4) if self.blind else contacts_true

        # Torque channel per obs_tau_mode. "cmd" (the dm family) is the pre-envelope
        # commanded torque from the HELD targets against the frame's OWN measured
        # q/qd (prev_measured = this interval's start) — the exact quantity training
        # computed, no motor model. "mdc" approximates the applied torque instead.
        frame_measured = self.prev_measured
        if self.held_targets is None:
            tau_p = np.zeros(12)
            tau_d = np.zeros(12)
            tau_cmd = np.zeros(12)
            tau_obs = np.zeros(12)
        else:
            qt, qdt = self.held_targets
            tau_p = self.KP * (qt - frame_measured[6:18])
            tau_d = self.KD * (qdt - frame_measured[18:30])
            tau_cmd = tau_p + tau_d
            if self.tau_mode == "cmd":
                tau_obs = tau_cmd
            elif self.tau_mode == "mdc":
                tau_obs = mdc_apply(tau_cmd, frame_measured[18:30])
            else:
                tau_obs = np.zeros(12)

        # frame: [measured30 (1-step stale) | contact4 | tau12 | act12 (last EMITTED)]
        frame_parts = [frame_measured, contacts_obs, tau_obs]
        if self.act_hist:
            frame_parts.append(self.last_emitted)
        frame = np.concatenate(frame_parts)
        self.prev_measured = measured_now
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = frame

        quat = self._aligned_quat()               # body -> ref-world
        R = rotmat_from_quat_wxyz(quat)
        grav_b = R.T @ np.array([0.0, 0.0, -1.0])
        yaw = yaw_from_quat_wxyz(quat)
        pos_ref_xy, quat_ref = self.ref.base_ref_at(t)
        yaw_e = yaw - yaw_from_quat_wxyz(quat_ref)
        attitude = np.concatenate([grav_b, [np.sin(yaw_e)], [np.cos(yaw_e)]])

        blocks = [self.history.reshape(-1),
                  self.ref.preview(t, self.PREVIEW_DT, self.NUM_FUTURE),
                  self.ref.phase_info(t), attitude]

        if self.use_ori_err:
            qc = quat * np.array([1.0, -1.0, -1.0, -1.0])
            qe = qmul(qc, quat_ref)
            blocks.append(2.0 * np.sign(qe[0] if qe[0] != 0 else 1.0) * qe[1:4])

        if self.use_cprev:
            masks, ttc_e = self.ref.contact_pack(t, self.PREVIEW_DT, self.NUM_FUTURE)
            load = (np.zeros(4) if self.blind
                    else np.clip(forces_n / BODY_WEIGHT, 0.0, 2.0))
            blocks.append(np.concatenate([masks, ttc_e, load]))

        obs = np.concatenate(blocks)
        assert obs.shape == (self.n_obs,), (obs.shape, self.n_obs)
        obs = np.clip(obs * self.actor_scale, -self.CLIP_OBS, self.CLIP_OBS)
        action = self.policy(obs)

        # 1-step action latency (training nominal): actuate a_{t-1}, remember a_t.
        # The act-hist channel carries the EMITTED stream (fresh), not the actuated.
        a_cmd = self.prev_action
        self.prev_action = action
        self.last_emitted = action.copy()

        # ---- KINEMATIC composition: anchor + action, clamp, optional one-pole LPF.
        # q_ref/qd_ref sampled ONCE at the latch time t; publishing holds them 20 ms.
        q_ref_t, qd_ref_t = self.ref.ref_at(t)
        anchor = q_ref_t if self.anchor_ref else self.q_home
        q_target = np.clip(anchor + self.ACTION_SCALE * a_cmd,
                           self.q_clamp_lo, self.q_clamp_hi)
        if self.LPF_BETA > 0.0:
            q_target = (self.LPF_BETA * (self.lpf_prev if self.lpf_prev is not None else q)
                        + (1.0 - self.LPF_BETA) * q_target)
        self.lpf_prev = q_target
        qd_target = qd_ref_t if self.anchor_qd else np.zeros(12)

        fade = max(0.0, 1.0 - self.ii / self.handoff_fade_ticks)
        for i in range(12):
            m = MOTOR_FROM_ISO[i]
            self.low_cmd.motor_cmd[m].q = float(q_target[i])
            self.low_cmd.motor_cmd[m].dq = float(qd_target[i])
            self.low_cmd.motor_cmd[m].kp = self.KP
            self.low_cmd.motor_cmd[m].kd = self.KD
            self.low_cmd.motor_cmd[m].tau = float(fade * self.tau_i[m])
        self.held_targets = (q_target, qd_target)

        # ---- trace ----
        _qc = quat * np.array([1.0, -1.0, -1.0, -1.0])
        _qe = qmul(_qc, quat_ref)
        tr = self.trace
        tr["t"].append(t)
        tr["rpy"].append(quat_to_euler_xyz(quat))
        tr["rpy_ref"].append(quat_to_euler_xyz(quat_ref))
        tr["ori_err"].append(2.0 * np.sign(_qe[0] if _qe[0] != 0 else 1.0) * _qe[1:4])
        tr["ref_xy"].append(pos_ref_xy.copy())
        tr["tilt"].append(grav_b[2])
        tr["action"].append(a_cmd.copy())
        tr["act_emitted"].append(action.copy())
        tr["contacts"].append(contacts_true.copy())
        tr["forces"].append(forces_n.copy())
        tr["q"].append(q.copy())
        tr["qd"].append(qd.copy())
        tr["q_ref"].append(q_ref_t.copy())
        tr["qd_ref"].append(qd_ref_t.copy())
        tr["q_target"].append(q_target.copy())
        tr["tau_obs"].append(tau_obs.copy())
        tr["tau_p"].append(tau_p.copy())
        tr["tau_d"].append(tau_d.copy())
        tr["tau_cmd"].append(tau_cmd.copy())

        if self.motiontime % 10 == 0:
            _rp = np.degrees(_wrap_pi(tr["rpy"][-1] - tr["rpy_ref"][-1]))[:2]
            logging.info("t %.2f cont %s%s rp_e [%+.1f %+.1f] yaw_e %+.1f tilt %.2f "
                         "|a| %.2f (loop %.1f Hz)",
                         t, contacts_true.astype(int), " (blinded)" if self.blind else "",
                         *_rp, np.degrees(yaw_e), grav_b[2],
                         np.abs(a_cmd).max(), self._loop_hz)

        if grav_b[2] > -0.4:
            logging.error("TILT ABORT at t %.2f (grav_z %.2f) -> damping", t, grav_b[2])
            self.aborted = True
        self.ii += 1

    # ------------------------------------------------------- trace / reporting
    def save_trace(self, outdir="runs"):
        """Summary metrics + PNGs + the raw npz. Called from the MAIN thread once the
        policy phase has stopped writing. Safe on a short/aborted run."""
        raw = {k: v for k, v in self.trace.items() if len(v)}
        n = min((len(v) for v in raw.values()), default=0)
        T = {k: np.asarray(v[:n], float) for k, v in raw.items()}
        if not T or n < 2:
            logging.warning("no trace captured (run ended before the policy phase)")
            return None
        stamp = time.strftime("%Y%m%d_%H%M%S")
        rundir = os.path.join(outdir, f"{self.run_tag}_{stamp}")
        os.makedirs(rundir, exist_ok=True)
        stem = os.path.join(rundir, f"{self.run_tag}_{stamp}")

        t = T["t"]
        rpy_e = np.degrees(_wrap_pi(T["rpy"] - T["rpy_ref"]))       # (N,3) deg
        aborted = bool(self.aborted)

        # ---- summary ------------------------------------------------------
        lines = [f"run          : {self.run_tag}",
                 f"checkpoint   : {self.ckpt_path}",
                 f"meta         : {self.meta_path}",
                 f"reference    : {self.traj_path}",
                 f"contract     : kinematic ref-anchor, frame {self.frame_dim}, "
                 f"obs {self.n_obs}, tau_obs '{self.tau_mode}', lpf {self.LPF_BETA:g}",
                 f"outcome      : {'ABORTED (tilt)' if aborted else 'completed'}"
                 f"   {len(t)}/{self.n_ticks} ticks, {t[-1]:.2f}/{self.ref.duration:.2f} s",
                 f"contact-blind: {self.blind}",
                 "",
                 f"{'axis':>7}{'RMS err':>10}{'max|err|':>10}{'final':>9}"]
        for i, nm in enumerate(("roll", "pitch", "yaw")):
            e = rpy_e[:, i]
            lines.append(f"{nm:>7}{np.sqrt((e**2).mean()):9.2f}°{np.abs(e).max():9.2f}°"
                         f"{e[-1]:8.2f}°")
        act_diff = np.abs(np.diff(T["act_emitted"], axis=0))
        lines += ["",
                  f"worst tilt (grav_z) : {T['tilt'].max():+.3f}   (abort at > -0.40)",
                  f"|action| mean/max   : {np.abs(T['action']).mean():.3f} / "
                  f"{np.abs(T['action']).max():.3f}",
                  f"action saturation   : {(np.abs(T['action']) > 0.99).mean() * 100:.1f}% of joint-ticks",
                  f"jitter (mrad/tick)  : {act_diff.mean() * 1000 * self.ACTION_SCALE:.1f}"
                  f"   (HW pass line ~25; deploy-stream, same stat as the sim probes)"]
        # ---- contact profile: plan vs measured, per foot ---------------------
        idx = np.array([self.ref._index(tt)[0] for tt in t])
        fref = self.ref.force_ref[idx]                              # (N,4) N
        plan = self.ref.contact[idx]                                # (N,4) bool
        meas = T["forces"] > CONTACT_FORCE_THRESHOLD_HW             # (N,4)
        qref = T["q_ref"]                                           # latch-sampled
        qdref = T["qd_ref"]
        lines += ["", "contact timing vs plan (ms, + = late; n/a = pad never released):",
                  f"{'event':>11}{'plan_t':>8}" + "".join(f"{nm:>7}" for nm in FOOT_NAMES)]
        events = {}                                # (plan_tick, kind) -> {foot: delay|None}
        for f in range(4):
            p = plan[:, f].astype(int); m = meas[:, f].astype(int)
            for k in range(1, len(p)):
                if p[k] == p[k - 1]:
                    continue
                kind = "touchdown" if p[k] else "takeoff"
                if kind == "touchdown" and m[max(0, k - 15):k].all():
                    events.setdefault((k, kind), {})[f] = None
                    continue
                want = p[k]
                cand = [j for j in range(max(1, k - 15), min(len(m), k + 16))
                        if m[j] == want and m[j - 1] != want]
                delay = (min(cand, key=lambda j: abs(j - k)) - k) * 1000 * self.dt \
                    if cand else np.nan
                events.setdefault((k, kind), {})[f] = delay
        for (k, kind), feet in sorted(events.items()):
            row = f"{kind:>11}{t[k]:7.2f}s"
            for f in range(4):
                if f not in feet:
                    row += f"{'-':>7}"
                elif feet[f] is None:
                    row += f"{'n/a':>7}"
                elif np.isnan(feet[f]):
                    row += f"{'?':>7}"
                else:
                    row += f"{feet[f]:+7.0f}"
            lines.append(row)
        # tap-force sim2real readout (dm_v7/v8): swing-foot touches during diag
        # stances. ~40-80 N (calibrated) = plant model good; 150 N+ = sysid drifted.
        is_diag = plan.sum(1) == 2
        swing_touch = (~plan) & meas & is_diag[:, None]
        if swing_touch.any():
            tf = T["forces"][swing_touch]
            lines += ["", f"diag swing-foot touches: {int(swing_touch.sum())} tick-feet, "
                          f"median force {np.median(tf):.0f} max {tf.max():.0f} "
                          f"(raw units x FOOT_FORCE_TO_N={FOOT_FORCE_TO_N:g}) — the "
                          f"live sysid readout: ~40-80 N good, 150 N+ drifted"]
        else:
            lines += ["", "diag swing-foot touches: none detected"]

        # ---- joint tracking --------------------------------------------------
        # resid here = q_target - q_ref(latch) = the policy's kinematic correction
        # (same formula as the residual line since the anchor IS the ref).
        e_plan = T["q"] - qref
        e_cmd = T["q"] - T["q_target"]
        resid = T["q_target"] - qref
        e_qd = T["qd"] - qdref
        lines += ["", "joint tracking (RMS over the run, rad / rad/s):",
                  f"{'':>7}{'|q-ref|':>10}{'|q-cmd|':>10}{'|resid|':>10}"
                  f"{'max|resid|':>12}{'|qd-ref|':>10}"]
        for pi, part in enumerate(("hip", "thigh", "calf")):
            sl = slice(pi * 4, pi * 4 + 4)             # ISO order is part-major
            lines.append(f"{part:>7}{np.sqrt((e_plan[:, sl] ** 2).mean()):10.4f}"
                         f"{np.sqrt((e_cmd[:, sl] ** 2).mean()):10.4f}"
                         f"{np.sqrt((resid[:, sl] ** 2).mean()):10.4f}"
                         f"{np.abs(resid[:, sl]).max():12.4f}"
                         f"{np.sqrt((e_qd[:, sl] ** 2).mean()):10.3f}")
        lines.append(f"{'all':>7}{np.sqrt((e_plan ** 2).mean()):10.4f}"
                     f"{np.sqrt((e_cmd ** 2).mean()):10.4f}"
                     f"{np.sqrt((resid ** 2).mean()):10.4f}"
                     f"{np.abs(resid).max():12.4f}"
                     f"{np.sqrt((e_qd ** 2).mean()):10.3f}")
        worst = int(np.argmax(np.abs(e_cmd).max(0)))
        lines.append(f"worst-tracked joint : {ISO_NAMES[worst]} "
                     f"(max |q-cmd| {np.abs(e_cmd[:, worst]).max():.4f} rad)")

        report = "\n".join(lines)
        print("\n" + "=" * 62 + "\n" + report + "\n" + "=" * 62)
        with open(stem + ".txt", "w") as f:
            f.write(report + "\n")
        np.savez(stem + ".npz", n_ticks=self.n_ticks, aborted=aborted,
                 blind=self.blind, run_tag=self.run_tag,
                 ckpt_path=str(self.ckpt_path), meta_path=str(self.meta_path),
                 traj_path=str(self.traj_path),
                 action_scale=self.ACTION_SCALE, joint_names=np.array(ISO_NAMES),
                 force_ref=fref, contact_plan=plan, **T)

        # ---- torque artifacts -------------------------------------------------
        np.savez(stem + "_torque_commanded.npz",
                 t=t, tau_p=T["tau_p"], tau_d=T["tau_d"], tau_cmd=T["tau_cmd"],
                 tau_obs=T["tau_obs"], tau_max=TAU_MAX_ISO, kp=self.KP, kd=self.KD,
                 rate=1.0 / self.dt, joint_names=np.array(ISO_NAMES),
                 run_tag=self.run_tag)

        n_hf = min(len(self.hf_t), len(self.hf_tau))
        hf_t = np.asarray(self.hf_t[:n_hf], float)
        hf_tau = np.asarray(self.hf_tau[:n_hf], float)
        hf_rate = float("nan")
        if n_hf > 2:
            hf_rate = (n_hf - 1) / (hf_t[-1] - hf_t[0])
            np.savez(stem + "_torque_measured.npz",
                     t=hf_t, tau_est=hf_tau, tau_max=TAU_MAX_ISO, rate=hf_rate,
                     joint_names=np.array(ISO_NAMES), run_tag=self.run_tag)
            logging.info("measured-torque log: %d samples @ %.1f Hz, |tau_est| max %.1f N.m",
                         n_hf, hf_rate, np.abs(hf_tau).max())
        else:
            logging.warning("no high-rate torque samples captured (%d) -- was the "
                            "policy phase reached?", n_hf)

        # ---- plots (lazy import: a headless robot may not have matplotlib) ----
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:                                   # noqa: BLE001
            logging.warning("matplotlib unavailable (%s); wrote %s.npz/.txt only",
                            type(e).__name__, stem)
            return stem

        air = self.ref.airborne
        spans, i = [], 0
        while i < len(air):
            if air[i]:
                j = i
                while j < len(air) and air[j]:
                    j += 1
                spans.append((i * self.ref.dt, j * self.ref.dt))
                i = j
            else:
                i += 1

        fig, ax = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
        for a in ax:
            for s0, s1 in spans:
                a.axvspan(s0, s1, color="0.88", lw=0, zorder=0)
        for i, nm in enumerate(("roll", "pitch", "yaw")):
            ax[i].plot(t, np.degrees(T["rpy_ref"][:, i]), "--", c="0.45", lw=1.4,
                       label="reference")
            ax[i].plot(t, np.degrees(T["rpy"][:, i]), c="C0", lw=1.6, label="measured")
            e = rpy_e[:, i]
            ax[i].set_ylabel(f"{nm} [deg]")
            ax[i].set_title(f"{nm}   RMS {np.sqrt((e**2).mean()):.2f}°   "
                            f"max {np.abs(e).max():.2f}°", fontsize=9, loc="left")
            ax[i].legend(fontsize=8, loc="upper left")
        ax[3].plot(t, T["tilt"], c="C3", lw=1.5, label="grav_z")
        ax[3].axhline(-0.4, ls=":", c="r", lw=1.2, label="abort threshold")
        ax[3].plot(t, np.abs(T["action"]).max(1), c="C2", lw=1.2, label="|action| max")
        ax[3].axhline(1.0, ls=":", c="0.6", lw=1.0)
        ax[3].set_ylabel("tilt / |a|")
        ax[3].legend(fontsize=8, loc="lower left", ncol=3)
        for f, nm in enumerate(FOOT_NAMES):
            ax[4].plot(t, T["forces"][:, f], lw=1.3, label=nm)
        ax[4].axhline(CONTACT_FORCE_THRESHOLD_HW, ls=":", c="k", lw=1.2,
                      label="contact thresh")
        ax[4].set_ylabel("foot force [N?]")
        ax[4].set_xlabel("t [s]  (shaded = planned flight)")
        ax[4].legend(fontsize=8, loc="upper left", ncol=5)
        fig.suptitle(f"{self.run_tag}\n{'ABORTED' if aborted else 'completed'}   "
                     f"blind={self.blind}   kinematic ref-anchor", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(stem + ".png", dpi=130)
        plt.close(fig)

        # ---- contact profile PNG ---------------------------------------------
        fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
        for f, nm in enumerate(FOOT_NAMES):
            a = ax[f]
            p = plan[:, f]
            k = 0
            while k < len(p):
                if p[k]:
                    j = k
                    while j < len(p) and p[j]:
                        j += 1
                    a.axvspan(t[k], t[min(j, len(t) - 1)], color="0.90", lw=0, zorder=0)
                    k = j
                else:
                    k += 1
            a.plot(t, T["forces"][:, f], c="C3", lw=1.4, label="measured (raw)")
            a.axhline(CONTACT_FORCE_THRESHOLD_HW, ls=":", c="k", lw=1.0,
                      label="thresh" if f == 0 else None)
            ar = a.twinx()
            ar.plot(t, fref[:, f], "--", c="0.35", lw=1.3, label="plan |F| (N)")
            ar.set_ylabel("plan [N]", fontsize=8, color="0.35")
            ar.tick_params(labelsize=7, colors="0.35")
            a.set_ylabel(f"{nm} raw")
            if f == 0:
                h1, l1 = a.get_legend_handles_labels()
                h2, l2 = ar.get_legend_handles_labels()
                a.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left", ncol=3)
        ax[-1].set_xlabel("t [s]  (shaded = that foot's planned stance)")
        fig.suptitle(f"{self.run_tag}\ncontact profile -- plan (N, dashed) vs measured "
                     f"pad (raw)\nwatch diag swing-foot taps: ~40-80 N = plant good, "
                     f"150 N+ = sysid drifted", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(stem + "_contact.png", dpi=130)
        plt.close(fig)

        # ---- joint tracking PNGs ---------------------------------------------
        # Position shows plan / plan+action (commanded) / measured — the gap between
        # the first two is the policy's correction, between the last two the servo
        # error. Velocity has no action channel: commanded dq IS qd_ref (or 0).
        for what, meas_a, ref_a, cmd_a, unit, sfx in (
                ("joint position", T["q"], qref, T["q_target"], "rad", "_joint_pos"),
                ("joint velocity", T["qd"], qdref, None, "rad/s", "_joint_vel")):
            fig, ax = plt.subplots(4, 3, figsize=(15, 11), sharex=True)
            for li, leg in enumerate(FOOT_NAMES):
                for pi, part in enumerate(("hip", "thigh", "calf")):
                    a = ax[li][pi]
                    j = pi * 4 + li                     # ISO order is part-major
                    for s0, s1 in spans:
                        a.axvspan(s0, s1, color="0.88", lw=0, zorder=0)
                    a.plot(t, ref_a[:, j], "--", c="0.45", lw=1.3,
                           label="plan ref" + ("" if cmd_a is not None
                                               else " (= commanded dq)"))
                    if cmd_a is not None:
                        a.plot(t, cmd_a[:, j], c="C2", lw=1.0, alpha=0.9,
                               label="ref + action (commanded)")
                    a.plot(t, meas_a[:, j], c="C0", lw=1.4, label="measured")
                    base = cmd_a if cmd_a is not None else ref_a
                    e = meas_a[:, j] - base[:, j]
                    a.set_title(f"{leg}_{part}   RMS {np.sqrt((e ** 2).mean()):.4f} "
                                f"max {np.abs(e).max():.4f} {unit}",
                                fontsize=8, loc="left")
                    if li == 3:
                        a.set_xlabel("t [s]  (shaded = planned flight)")
                    if pi == 0:
                        a.set_ylabel(f"[{unit}]")
            ax[0][0].legend(fontsize=7, loc="best")
            fig.suptitle(f"{self.run_tag}\n{what} -- RMS/max quoted against the "
                         f"COMMANDED value" + ("" if cmd_a is not None
                                               else " (dq command carries no action)"),
                         fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            fig.savefig(stem + sfx + ".png", dpi=130)
            plt.close(fig)

        # ---- commanded-torque PNG --------------------------------------------
        # No feedforward exists in this family: tau_cmd = kp(q*-q) + kd(qd*-qd),
        # computed against the frame-stale measurement (the obs quantity).
        fig, ax = plt.subplots(4, 3, figsize=(15, 11), sharex=True)
        for li, leg in enumerate(FOOT_NAMES):
            for pi, part in enumerate(("hip", "thigh", "calf")):
                a = ax[li][pi]
                j = pi * 4 + li
                for s0, s1 in spans:
                    a.axvspan(s0, s1, color="0.88", lw=0, zorder=0)
                a.plot(t, T["tau_p"][:, j], "--", c="0.45", lw=1.0, label="tau_p")
                a.plot(t, T["tau_d"][:, j], ":", c="0.45", lw=1.0, label="tau_d")
                a.plot(t, T["tau_cmd"][:, j], c="C0", lw=1.3, label="tau_cmd = p + d")
                for sgn in (1.0, -1.0):
                    a.axhline(sgn * TAU_MAX_ISO[j], ls=":", c="r", lw=1.0,
                              label="tau_max" if (sgn > 0 and j == 0) else None)
                a.set_title(f"{leg}_{part}   |cmd|max {np.abs(T['tau_cmd'][:, j]).max():.1f}",
                            fontsize=8, loc="left")
                if li == 3:
                    a.set_xlabel("t [s]  (shaded = planned flight)")
                if pi == 0:
                    a.set_ylabel("[N.m]")
        ax[0][0].legend(fontsize=7, loc="best")
        fig.suptitle(f"{self.run_tag}\ncommanded torque -- kp*(q_target-q) + "
                     f"kd*(qd_target-qd), kp={self.KP:g} kd={self.KD:g}, NO tau_ff",
                     fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(stem + "_torque_commanded.png", dpi=130)
        plt.close(fig)

        # ---- measured-torque PNG ---------------------------------------------
        if n_hf > 2:
            fig, ax = plt.subplots(4, 3, figsize=(15, 11), sharex=True)
            for li, leg in enumerate(FOOT_NAMES):
                for pi, part in enumerate(("hip", "thigh", "calf")):
                    a = ax[li][pi]
                    j = pi * 4 + li
                    for s0, s1 in spans:
                        a.axvspan(s0, s1, color="0.88", lw=0, zorder=0)
                    a.plot(hf_t, hf_tau[:, j], c="C3", lw=0.8)
                    for sgn in (1.0, -1.0):
                        a.axhline(sgn * TAU_MAX_ISO[j], ls=":", c="r", lw=1.0)
                    a.set_title(f"{leg}_{part}   |tau_est| max "
                                f"{np.abs(hf_tau[:, j]).max():.1f}   RMS "
                                f"{np.sqrt((hf_tau[:, j] ** 2).mean()):.1f} N.m",
                                fontsize=8, loc="left")
                    if li == 3:
                        a.set_xlabel("t [s]  (shaded = planned flight)")
                    if pi == 0:
                        a.set_ylabel("[N.m]")
            fig.suptitle(f"{self.run_tag}\nmeasured torque (motor_state.tau_est) -- "
                         f"{n_hf} samples @ {hf_rate:.0f} Hz", fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            fig.savefig(stem + "_torque_measured.png", dpi=130)
            plt.close(fig)

        wrote = ["npz", "txt", "png", "_contact.png", "_joint_pos.png", "_joint_vel.png",
                 "_torque_commanded.{png,npz}"]
        if n_hf > 2:
            wrote.append("_torque_measured.{png,npz}")
        logging.info("trace written: %s/  (%s)", rundir, ", ".join(wrote))
        logging.info("  -> copy the Vicon record into that directory")
        return stem

    def _damped_stop(self):
        for m in range(12):
            self.low_cmd.motor_cmd[m].q = 0.0
            self.low_cmd.motor_cmd[m].dq = 0.0
            self.low_cmd.motor_cmd[m].kp = 0.0
            self.low_cmd.motor_cmd[m].kd = 3.0
            self.low_cmd.motor_cmd[m].tau = 0.0

    # ------------------------------------------------------------- main tick
    def LowCmdWrite(self):
        """Publishes every tick. Any exception in the tick body converges to the
        damped stop instead of killing the thread."""
        try:
            if not self._tick():
                return                      # not ready yet (no low_state) - don't publish
        except Exception:
            logging.exception("control tick crashed -> damping")
            self.aborted = True
            self._damped_stop()
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

    def _tick(self):
        if self.low_state is None:
            return False
        if self.firstRun:
            self.startPos = [self.low_state.motor_state[m].q for m in range(12)]
            if np.abs(np.asarray(self.startPos) - self.q0_motor).max() < 0.1:
                self.fold_percent = self.align_percent = 1
                logging.info("already within 0.1 rad of q0 - skipping fold/align")
            self.firstRun = False
        self.motiontime += 1

        now = time.perf_counter()
        if self._rate_t0 is None:
            self._rate_t0 = now
        self._rate_n += 1
        if now - self._rate_t0 >= 1.0:
            self._loop_hz = self._rate_n / (now - self._rate_t0)
            self._rate_t0, self._rate_n = now, 0

        if self.aborted:
            self.hf_on = False
            self._damped_stop()

        elif self.fold_percent < 1:
            self.fold_percent = min(self.fold_percent + 1.0 / self.fold_duration, 1)
            for m in range(12):
                self.low_cmd.motor_cmd[m].q = float((1 - self.fold_percent) * self.startPos[m]
                                                    + self.fold_percent * self.foldPos[m])
                self.low_cmd.motor_cmd[m].dq = 0
                self.low_cmd.motor_cmd[m].kp = self.Kp_stand
                self.low_cmd.motor_cmd[m].kd = self.Kd_stand
                self.low_cmd.motor_cmd[m].tau = 0

        elif self.align_percent < 1:
            self.align_percent = min(self.align_percent + 1.0 / self.align_duration, 1)
            for m in range(12):
                self.low_cmd.motor_cmd[m].q = float((1 - self.align_percent) * self.foldPos[m]
                                                    + self.align_percent * self.q0_motor[m])
                self.low_cmd.motor_cmd[m].dq = 0
                self.low_cmd.motor_cmd[m].kp = self.Kp_stand
                self.low_cmd.motor_cmd[m].kd = self.Kd_stand
                self.low_cmd.motor_cmd[m].tau = 0

        elif (self.hold_percent < 1) or (not self.start_policy):
            # stage 3: hold q0 + integrator; once calibrated, keep holding (armed)
            # until the main thread's Enter sets start_policy
            self.hold_percent = min(self.hold_percent + 1.0 / self.hold_duration, 1)
            for m in range(12):
                err_m = self.q0_motor[m] - self.low_state.motor_state[m].q
                self.tau_i[m] = np.clip(self.tau_i[m] + self.Ki * err_m * self.dt,
                                        -self.tau_i_max, self.tau_i_max)
                self.low_cmd.motor_cmd[m].q = float(self.q0_motor[m])
                self.low_cmd.motor_cmd[m].dq = 0
                self.low_cmd.motor_cmd[m].kp = self.Kp_stand
                self.low_cmd.motor_cmd[m].kd = self.Kd_stand
                self.low_cmd.motor_cmd[m].tau = float(self.tau_i[m])
            if self.hold_percent < 1:
                if self.motiontime % 10 == 0:
                    q_now = np.array([self.low_state.motor_state[m].q for m in range(12)])
                    ff = self._read_foot_forces_n()
                    logging.info("calibrating: hold max|err| %.3f foot_force(N?) %s "
                                 "(loop %.1f Hz)",
                                 np.abs(q_now - self.q0_motor).max(), np.round(ff, 1),
                                 self._loop_hz)
            elif not self._armed_logged:
                q_now = np.array([self.low_state.motor_state[m].q for m in range(12)])
                ff = self._read_foot_forces_n()
                logging.info("ARMED: calibrated (max|err| %.3f, foot_force(N?) %s, "
                             "loop %.1f Hz) + holding q0 -> press Enter at the LAUNCH "
                             "prompt to start the policy",
                             np.abs(q_now - self.q0_motor).max(), np.round(ff, 1),
                             self._loop_hz)
                self._armed_logged = True

        elif self.ii < self.n_ticks:
            if not self.handoff_done:
                self._seed_policy_state()
            self._policy_tick()

        elif self.settle_percent < 1:
            self.hf_on = False                # policy phase over: stop the torque log
            self.settle_percent = min(self.settle_percent + 1.0 / self.settle_duration, 1)
            for m in range(12):
                self.low_cmd.motor_cmd[m].q = float(self.qf_motor[m])
                self.low_cmd.motor_cmd[m].dq = 0
                self.low_cmd.motor_cmd[m].kp = self.Kp_stand
                self.low_cmd.motor_cmd[m].kd = self.Kd_stand
                self.low_cmd.motor_cmd[m].tau = 0

        return True


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default = "dm_utils/v12_1499.pt",required=False,
                    help=".pt of a KINEMATIC-family run (dm_v*, kin_v4+)")
    ap.add_argument("--meta", default="dm_utils/v12_meta.json", help="default: meta.json beside the checkpoint")
    ap.add_argument("--traj", default="traj_hopscotch_friction_6cm_lsq.json", help="override meta's traj_path")
    ap.add_argument("--dry-run", action="store_true",
                    help="load + validate everything, then exit without touching the robot")
    ap.add_argument("--out", default="data",
                    help="parent directory for per-run trace dirs (default: data/)")
    ap.add_argument("iface", nargs="?", default=None)
    args = ap.parse_args()

    meta, meta_path = load_meta(args.checkpoint, args.meta)

    if args.dry_run:
        logging.info("--dry-run: constructing controller (no DDS, no motion)...")
        c = Custom(args.checkpoint, meta, args.traj, meta_path=meta_path)
        logging.info("DRY RUN OK: obs=%d ticks=%d blind=%s anchor=%s qd=%s lpf=%g",
                     c.n_obs, c.n_ticks, c.blind,
                     "ref" if c.anchor_ref else "home", c.anchor_qd, c.LPF_BETA)
        logging.info("artifacts would land in: %s/%s_<stamp>/", args.out, c.run_tag)
        sys.exit(0)

    print("WARNING: Please ensure there are no obstacles around the robot while running.")
    input("Press Enter to continue...")

    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)

    custom = Custom(args.checkpoint, meta, args.traj, meta_path=meta_path)
    custom.Init()
    custom.Start()

    launched = False
    saved = False
    try:
        while True:
            if custom.aborted:
                if not saved:                 # save the partial run: the tail before a
                    saved = True              # fall is the most diagnostic part of it
                    try:
                        custom.save_trace(args.out)
                    except Exception:
                        logging.exception("save_trace failed (robot is still damping)")
                time.sleep(2)
                print("Aborted - robot in damping mode. Ctrl-C when secured.")
                time.sleep(10)
            if (not launched) and (not custom.aborted) and custom.hold_percent >= 1:
                input("Robot calibrated + holding q0. Press Enter to LAUNCH...")
                custom.start_policy = True
                launched = True
            if custom.settle_percent >= 1:
                time.sleep(1)
                if not saved:
                    saved = True
                    custom.save_trace(args.out)
                print("Done!")
                sys.exit(0)
            time.sleep(0.5)
    except KeyboardInterrupt:
        # Ctrl-C after a launched run should still keep the data.
        if launched and not saved:
            custom.save_trace(args.out)
        raise
