"""Meta-driven Go2 hardware deployment — one script for ANY checkpoint of the
hendeca/matrix line. Reads the run's meta.json instead of hardcoding per-model
constants, so a batch of .pt files can be flown without editing source.

Supersedes sysid_rl.py (sysid1 ff comps hardcoded) and trial.py (ff comps 0):
both are the same code path with `ff_damping_comp` / `ff_armature_comp` read from
meta rather than baked in. A model trained without comp simply has 0.0 there.

WHAT IS READ FROM meta.json (and therefore no longer needs editing):
  * ff_damping_comp / ff_armature_comp  -> the tau_ff bake
  * traj_path                           -> reference, .json OR .npz (auto-dispatch)
  * obs_contact_blind                   -> zeroes the three pad-derived obs sites
  * kp / kd / action_scale / clip_obs / odom_clamp / obs_history_len / num_future
  * observation_space / action_space    -> asserted against what this script builds
  * obs_velest / obs_ori_err / obs_contact_prev -> which obs blocks are appended

Anything this script cannot honour raises at LOAD time, before the robot moves.

TWO THINGS THE NAIVE PORT GETS WRONG (both cost a run if missed):

1. HALF-OPEN CONCAT IS JSON-ONLY. Legacy list-of-modes JSON drops each non-final
   mode's LAST row so the boundary knot keeps the next mode's post-impact state
   (the 2026-07-12 fix). The pre-gridded npz ALREADY carries post-impact state at
   the impact row -- applying the same drop there re-creates that bug mirrored.

2. IF THE REFERENCE MODELS THE MOTOR, THE COMP IS A DIFFERENCE. reference_grid_1khz
   declares motor_model "friction_inertia", so its `u` already contains viscous +
   armature torques. Compensating with our absolute values double-counts. The
   correct correction is (OUR plant - THE REFERENCE'S), mirroring
   reference_manager_hopscotch.py. Coulomb/bias are deliberately NOT baked.

Both are handled below; see HopscotchRef.

Everything else (46-dim frames, 1-step sensor+action latency, MDC applied-torque
channel, PD with dq=qd_ref, ISO ordering, stand-up + integrator handoff) is
unchanged from sysid_rl.py -- see its header for the contract rationale.

Usage:
    python3 deploy_meta.py --checkpoint runs/mx_blind_union_lsq/model_400.pt [iface]
    # meta.json is auto-found next to the checkpoint; override with --meta
"""
import time
import sys
import os
import json
import logging


import numpy as np


from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
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
# These are contract, not per-model: every value below is ASSERTED against
# meta.json at load, so a checkpoint trained with different ones aborts here
# rather than flying on the wrong controller.
CTRL_DT = 0.02
OBS_QD_SCALE = 1.0 / 15.0
OBS_TAU_SCALE = 1.0 / 45.0
OBS_ACC_SCALE = 1.0 / 9.81
BODY_WEIGHT = 150.0                              # N, load-channel normalizer
TTC_EDGE_CLIP_S = 0.3
CONTACT_FORCE_THRESHOLD_HW = 20.0                # raw units, binary contacts. TUNABLE
FOOT_FORCE_TO_N = 1.0                            # raw foot_force -> Newtons. CALIBRATE
# (both of the above are NO-OPS for a contact-blind checkpoint -- see _policy_tick)


MDC_VBAT, MDC_PBAT = 28.8, 1728.0
MDC_GR, MDC_KT, MDC_R = 6.33, 0.26, 0.66
MDC_ALPHA = MDC_R / (MDC_KT * MDC_GR)
MDC_BETA = MDC_GR * MDC_KT
TAU_MAX_ISO = np.array([23.7] * 8 + [45.43] * 4)
# The ff-bake envelope literals used by reference_manager_hopscotch.py. Kept as
# literals (not recomputed from MDC_*) so the bake reproduces training exactly;
# they agree with MDC_ALPHA/MDC_BETA to 0.005%.
FF_ENVELOPE = (0.401, 1.646, 28.8, (23.7, 23.7, 45.43))




def mdc_apply(tau_des, qd):
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
    """Inverse of euler_xyz_to_quat_wxyz: recover (rx, ry, rz) with R = Rx Ry Rz.
    MUST match the reference's convention -- the trajectory stores Euler XYZ in
    q[:, 3:6] and base_quat is built from it with exactly that composition, so
    measured and reference RPY are only comparable if extracted the same way."""
    R = rotmat_from_quat_wxyz(q)
    ry = np.arcsin(np.clip(R[0, 2], -1.0, 1.0))
    rx = np.arctan2(-R[1, 2], R[2, 2])
    rz = np.arctan2(-R[0, 1], R[0, 0])
    return np.array([rx, ry, rz])




def build_gather_map(src_joint_names, target_joint_names):
    """Indices reordering a src-ordered 12-vec into target order (q_t = q_s[gather]),
    matched on the leg+part token. NEVER resolve these positionally -- a positional
    read yields a scrambled-but-self-consistent policy (the v11-era remap signature)."""
    src_index = {n.replace("_joint", ""): i for i, n in enumerate(src_joint_names)}
    gather = []
    for name in target_joint_names:
        token = name.replace("_joint", "")
        if token not in src_index:
            raise KeyError(f"joint '{name}' not in reference joint set {list(src_index)}")
        gather.append(src_index[token])
    return np.asarray(gather, dtype=np.int64)




def _pertype(val, names=ISO_NAMES):
    """scalar OR [hip, thigh, calf] -> per-joint 12-vector in `names` order."""
    a = np.asarray(val, dtype=float).ravel()
    if a.size == 1:
        return np.full(len(names), float(a[0]))
    assert a.size == 3, "ff comp must be a scalar or [hip, thigh, calf]"
    return np.array([a[0] if "hip" in n else (a[1] if "thigh" in n else a[2])
                     for n in names], float)




# ----------------------------------------------------------------- reference
class HopscotchRef:
    """Numpy port of HopscotchReferenceManager's load path, for BOTH reference
    formats, with the meta-driven ff comp baked exactly as training baked it."""


    def __init__(self, path, ff_damping_comp=0.0, ff_armature_comp=0.0):
        if str(path).endswith(".npz"):
            q, v, acc, u, contact, force_ref, src_names, dt, ref_motor = self._load_npz(path)
            fmt = "gridded npz"
        else:
            q, v, acc, u, contact, force_ref, src_names, dt, ref_motor = self._load_json(path)
            fmt = "modes json (half-open)"
        self.dt = dt
        self.force_ref = force_ref                 # (T,4) planned |F| per foot, N


        gather = build_gather_map(src_names, ISO_NAMES)
        self.q_ref = q[:, 6:18][:, gather]
        self.qd_ref = v[:, 6:18][:, gather]
        tau = u[:, gather]
        qdd = acc[:, 6:18][:, gather]


        # ---- ff comp: u <- u + d*qd_ref + Ia*qdd_ref, then MDC-envelope clamp.
        d12 = _pertype(ff_damping_comp)
        ia12 = _pertype(ff_armature_comp)
        requested = bool(np.any(d12) or np.any(ia12))
        if requested and ref_motor is not None:
            # The reference's u ALREADY carries its own motor torques -> compensate
            # the DIFFERENCE, not our absolutes. Baking absolutes double-counts and
            # flips the armature term's sign.
            rv = np.asarray(ref_motor.get("viscous", np.zeros(12)), float)[gather]
            ra = np.asarray(ref_motor.get("armature", np.zeros(12)), float)[gather]
            rc = np.asarray(ref_motor.get("coulomb", np.zeros(12)), float)[gather]
            logging.info("ref generator models the motor (%s): compensating the DIFFERENCE "
                         "(viscous max %.4f -> %.4f, armature max %.5f -> %.5f)",
                         ref_motor.get("model"), np.abs(d12).max(), np.abs(d12 - rv).max(),
                         np.abs(ia12).max(), np.abs(ia12 - ra).max())
            d12 = d12 - rv
            ia12 = ia12 - ra
            # Coulomb/bias are NOT baked (sign-dependent, and the two sysids disagree
            # ~7x on the calf). Surface the mismatch instead.
            self.ref_coulomb_max = float(np.abs(rc).max())
        else:
            self.ref_coulomb_max = None


        if requested:
            tau = tau + d12 * self.qd_ref + ia12 * qdd
            alpha, beta, vbat, eff = FF_ENVELOPE
            eff_tgt = np.array([eff[2] if "calf" in n else eff[0] for n in ISO_NAMES], float)
            ceil = np.minimum(np.maximum((vbat - beta * np.abs(self.qd_ref)) / alpha, 0.0),
                              eff_tgt)
            clipped = np.abs(tau) > ceil
            tau = np.clip(tau, -ceil, ceil)
            logging.info("ff comp d=%s Ia=%s -> |u_ff|max %.1f N.m, envelope-clipped %.2f%% of knots",
                         ff_damping_comp, ff_armature_comp, np.abs(tau).max(),
                         clipped.any(1).mean() * 100.0)
        else:
            logging.info("ff comp DISABLED (both comps zero in meta) -- raw trajopt u")
        self.tau_ff = tau


        self.base_pos = q[:, 0:3].copy()           # z offset irrelevant (only xy used)
        self.base_quat = euler_xyz_to_quat_wxyz(q[:, 3:6])
        self.contact = contact
        self.airborne = (~contact).all(axis=1)


        self.T_state = self.q_ref.shape[0]
        self.T_ctrl = self.tau_ff.shape[0]
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
        qs, vs, us, as_, cs, fs = [], [], [], [], [], []
        for i, m in enumerate(modes):
            q = np.asarray(m["q"], float)
            v = np.asarray(m["v"], float)
            u = np.asarray(m["u"], float)
            a = np.asarray(m["a"], float)
            lam = np.asarray(m["lam"], float)      # (steps, 3*n_active) active feet only
            cm = np.zeros((len(q), 4), bool)
            fm = np.zeros((len(q), 4))             # planned |F| per foot (N)
            for ci, foot in enumerate(m["contacts"]):
                idx = FOOT_NAMES.index(foot.split("_")[0])
                cm[:, idx] = True
                fm[:, idx] = np.linalg.norm(lam[:, 3 * ci:3 * ci + 3], axis=1)
            if i < len(modes) - 1:                 # HALF-OPEN: drop non-final LAST row
                q, v, u, a, cm, fm = q[:-1], v[:-1], u[:-1], a[:-1], cm[:-1], fm[:-1]
            qs.append(q); vs.append(v); us.append(u); as_.append(a); cs.append(cm)
            fs.append(fm)
        src = list(modes[0]["joint_names"][6:])
        return (np.concatenate(qs), np.concatenate(vs), np.concatenate(as_),
                np.concatenate(us), np.concatenate(cs), np.concatenate(fs), src,
                float(modes[0]["dt"]), None)       # legacy JSON models no motor


    @staticmethod
    def _load_npz(path):
        """PRE-GRIDDED npz (reference_grid_1khz family): one row per 1 kHz knot,
        already concatenated across segments.

        *** NO HALF-OPEN CONCAT HERE. *** The row stamped at each impact time already
        carries the POST-impact state (mode increments, contacts flip on, lam jumps,
        base_vz steps -- all AT that row). Dropping rows would mirror the old bug."""
        d = np.load(path, allow_pickle=True)
        q = np.asarray(d["q"], float)
        v = np.asarray(d["v"], float)
        acc = np.asarray(d["a"], float)
        u = np.asarray(d["u"], float)
        cin = np.asarray(d["contact"]).astype(bool)
        lam = np.asarray(d["lam"], float).reshape(len(q), 4, 3)
        contact = np.zeros((len(q), 4), bool)
        force = np.zeros((len(q), 4))
        for ci, foot in enumerate(str(x) for x in d["feet"]):
            idx = FOOT_NAMES.index(foot.split("_")[0])                      # BY NAME
            contact[:, idx] = cin[:, ci]
            force[:, idx] = np.linalg.norm(lam[:, ci], axis=1)
        src = [str(x) for x in d["joint_names"][6:18]]
        dt = 1.0 / float(d["rate"])
        # Does the GENERATOR already model the motor? Surfaced from the file's own
        # metadata so a regenerated trajectory cannot silently repeat the mistake.
        ref_motor = None
        try:
            meta = json.loads(str(d["meta_json"]))
            mp = meta.get("robot", {}).get("motor_parameters")
            if mp and meta.get("robot", {}).get("motor_model"):
                ref_motor = {k: np.asarray(mp[k], float) for k in
                             ("coulomb", "viscous", "armature", "bias") if k in mp}
                ref_motor["model"] = meta["robot"]["motor_model"]
        except Exception:
            ref_motor = None
        return q, v, acc, u, contact, force, src, dt, ref_motor


    # -------------------------------------------------------------- sampling
    def _index(self, t):
        f = t / self.dt
        i0 = int(np.clip(np.floor(f), 0, self.T_state - 2))
        frac = float(np.clip(f - i0, 0.0, 1.0))
        return i0, frac


    def ref_at(self, t):
        i0, fr = self._index(t)
        q = (1 - fr) * self.q_ref[i0] + fr * self.q_ref[i0 + 1]
        qd = (1 - fr) * self.qd_ref[i0] + fr * self.qd_ref[i0 + 1]
        tau = self.tau_ff[min(i0, self.T_ctrl - 1)]
        return q, qd, tau


    def preview(self, t, preview_dt, num_future):
        outs = []
        for j in range(num_future + 1):
            q, qd, tau = self.ref_at(t + j * preview_dt)
            outs += [q, qd, tau]
        return np.concatenate(outs)                # 36*(1+num_future)


    def phase_info(self, t):
        i0, _ = self._index(t)
        phase = float(np.clip(t / self.duration, 0.0, 1.0))
        return np.concatenate([[phase], self.contact[i0].astype(float),
                               [float(self.airborne[i0])],
                               [self.which_jump[i0] / max(1, self.n_jumps)]])  # 7


    def contact_pack(self, t, preview_dt, num_future):
        """(masks 4*(1+num_future), ttc_edge_norm 4) -- the plan-derived half."""
        masks = []
        for j in range(num_future + 1):
            i0, _ = self._index(t + j * preview_dt)
            masks.append(self.contact[i0].astype(float))
        i0, _ = self._index(t)
        ttc_e = np.clip(self.ttc_edge[i0] / TTC_EDGE_CLIP_S, 0.0, 1.0)
        return np.concatenate(masks), ttc_e


    def base_ref_at(self, t):
        """(pos xy (2), quat wxyz (4)) -- lerped like the training manager."""
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
    """Checkpoint -> numpy: EmpiricalNormalization + actor MLP, plus the concurrent
    velocity-estimator head (NO normalizer -- it consumes the O(1)-scaled raw
    history, mirroring vel_estimator.py). Shapes are checked against meta."""


    def __init__(self, ckpt_path, n_obs, n_act, want_estimator, est_in):
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
        self.eW = self.eb = None
        if want_estimator:
            es = ck["estimator_state_dict"]
            self.eW = [es[f"net.{i}.weight"].numpy().astype(np.float64) for i in (0, 2, 4)]
            self.eb = [es[f"net.{i}.bias"].numpy().astype(np.float64) for i in (0, 2, 4)]
            if self.eW[0].shape[1] != est_in:
                raise SystemExit(f"ABORT: estimator input {self.eW[0].shape[1]} != "
                                 f"history {est_in}")
        logging.info("policy loaded: %s (iter %s) obs=%d act=%d estimator=%s",
                     ckpt_path, ck.get("iter"), n_obs, n_act, bool(want_estimator))


    def estimate_vel(self, hist_scaled):
        x = hist_scaled
        for W, b in zip(self.eW[:-1], self.eb[:-1]):
            x = _elu(W @ x + b)
        return self.eW[-1] @ x + self.eb[-1]       # (3,) body-frame m/s


    def __call__(self, obs):
        x = (obs - self.mean) / (self.std + 1e-2)
        for W, b in zip(self.W[:-1], self.b[:-1]):
            x = _elu(W @ x + b)
        return np.clip(self.W[-1] @ x + self.b[-1], -1.0, 1.0)




def load_meta(ckpt_path, explicit=None):
    """Returns (meta, path). The PATH is returned too because it names the run: a
    checkpoint flown against the wrong meta is indistinguishable from one flown
    against the right one unless the artifact says which was used."""
    p = explicit or os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "meta.json")
    if not os.path.exists(p):
        raise SystemExit(f"ABORT: no meta.json at {p} (pass --meta). This script is "
                         f"meta-driven by design; it will not guess a model's contract.")
    with open(p) as f:
        meta = json.load(f)
    logging.info("meta: %s", p)
    return meta, p


# ------------------------------------------------------------------ run naming
def _stem(path):
    """basename without extension, reduced to characters that are safe in a filename.
    Stray spaces in a path have bitten this script before (see the 2026-07-30 fix)."""
    s = os.path.splitext(os.path.basename(str(path)))[0]
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in s) or "unnamed"




class Custom:
    def __init__(self, ckpt_path, meta, traj_override=None, resid_scale=1.0,
                 meta_path=None):
        self.dt = CTRL_DT
        self.motiontime = 0
        # residual-authority attenuation (2026-07-31 sim verdict: the +3-5 cm apex /
        # late-TD margin is authority-carried; 0.75x improved every MJ-DR tail with no
        # downside, 0.5x = full nominal launch fix but opens the tilt tail). Applied at
        # the q_target composition ONLY -- obs, tau_ff, PD gains, ref untouched, so the
        # policy's inputs stay contract-exact and scaling toward 0 degrades to pure
        # PD+FF (the vindicated zero-residual baseline).
        self.RESID_SCALE = float(resid_scale)
        if self.RESID_SCALE != 1.0:
            logging.warning("RESIDUAL AUTHORITY %.2fx (trained 1.0x): effective "
                            "action_scale %.3f rad", self.RESID_SCALE,
                            self.RESID_SCALE * float(meta["action_scale"]))


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
        need("compliant_feet", False, "Compliant feet are a sim-only plant axis.")
        need("obs_priv_base", False, "Teacher run: privileged base obs, not deployable.")
        need("hold_targets", True, "The applied-torque channel assumes held targets.")
        if list(meta.get("joint_names", ISO_NAMES)) != ISO_NAMES:
            raise SystemExit("ABORT: meta joint_names differ from ISO_NAMES; the motor "
                             "remap would be wrong.")
        n_act = int(meta.get("action_space", 12))
        if n_act != 12:
            raise SystemExit(f"ABORT: action_space={n_act}. A 13th action is the bounded "
                             f"phase residual (v14ph); this tick does not implement it.")


        self.KP = float(meta["kp"])
        self.KD = float(meta["kd"])
        self.ACTION_SCALE = float(meta["action_scale"])
        self.H = int(meta.get("obs_history_len", 10))
        self.NUM_FUTURE = int(meta.get("num_future", 2))
        self.CLIP_OBS = float(meta.get("clip_obs", 10.0))
        self.ODOM_CLAMP = float(meta.get("odom_clamp", 0.5))
        self.PREVIEW_DT = int(meta.get("preview_steps", 5)) * CTRL_DT
        self.use_velest = bool(meta.get("obs_velest", False))
        self.use_ori_err = bool(meta.get("obs_ori_err", False))
        self.use_cprev = bool(meta.get("obs_contact_prev", False))
        self.blind = bool(meta.get("obs_contact_blind", False))


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
        self.traj_path = traj                            # AFTER the local-copy fallback
        self.ckpt_path = ckpt_path
        self.meta_path = meta_path
        self.ref = HopscotchRef(traj,
                                meta.get("ff_damping_comp", 0.0),
                                meta.get("ff_armature_comp", 0.0))
        self.n_ticks = int(np.ceil(self.ref.duration / self.dt))


        # ---------------- obs layout, derived then asserted ---------------------
        frame_dim = 46                                   # measured30 + contact4 + tau12
        tail = 7 + 5                                     # phase + attitude
        tail += 3 if self.use_ori_err else 0
        tail += 5 if self.use_velest else 0
        tail += (4 * (1 + self.NUM_FUTURE) + 4 + 4) if self.use_cprev else 0
        n_obs = frame_dim * self.H + 36 * (1 + self.NUM_FUTURE) + tail
        meta_obs = int(meta.get("observation_space", n_obs))
        if n_obs != meta_obs:
            raise SystemExit(f"ABORT: this script builds a {n_obs}-dim obs but meta says "
                             f"{meta_obs}. Obs-flag combination is not supported.")


        fs = np.ones(frame_dim)
        fs[3:6] = OBS_ACC_SCALE
        fs[18:30] = OBS_QD_SCALE
        fs[34:46] = OBS_TAU_SCALE
        pv_s = np.ones(36)
        pv_s[12:24] = OBS_QD_SCALE
        pv_s[24:36] = OBS_TAU_SCALE
        self.frame_scale = fs                            # also the estimator-input scaling
        self.actor_scale = np.concatenate([np.tile(fs, self.H),
                                           np.tile(pv_s, 1 + self.NUM_FUTURE),
                                           np.ones(tail)])
        assert self.actor_scale.shape == (n_obs,), self.actor_scale.shape
        self.n_obs = n_obs


        self.policy = Policy(ckpt_path, n_obs, n_act, self.use_velest, frame_dim * self.H)
        logging.info("CONTACT-BLIND: %s -- history contact bits + load block %s",
                     self.blind, "ZEROED (foot-force calibration is a no-op)" if self.blind
                     else "live from foot_force")
        if self.ref.ref_coulomb_max is not None:
            logging.info("reference u compensates its own Coulomb (max %.3f N.m); ours is "
                         "not baked into ff by design", self.ref.ref_coulomb_max)


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
        self.prev_action = np.zeros(12)
        self.prev_measured = None
        self.held_targets = None
        self.history = None
        self.q_align = None                       # yaw-alignment quat (IMU -> ref world)
        self.odom_xy = self.ref.base_pos[0, :2].copy()
        self.settle_percent = 0
        self.settle_duration = 50
        self.aborted = False


        # ---- floating-base trace (appended once per policy tick, ~170 rows) ----
        # Plain list appends: microseconds, safe inside the 50 Hz control thread.
        # The main thread only reads it after the policy phase has stopped writing.
        self.trace = {k: [] for k in
                      ("t", "rpy", "rpy_ref", "ori_err", "odom_xy", "ref_xy", "vhat",
                       "tilt", "action", "contacts", "forces", "q", "qd", "q_target",
                       "tau_applied")}
        # Every artifact this run writes is named for the four things that decide what
        # the robot actually did: which weights, which contract, how much residual
        # authority they were given, and which reference they tracked. A png named for
        # the checkpoint alone cannot be told apart from the same checkpoint flown at a
        # different resid_scale or against a re-cut trajectory.
        self.run_tag = "_".join([_stem(ckpt_path),
                                 _stem(meta_path) if meta_path else "nometa",
                                 f"r{self.RESID_SCALE:.2f}".replace(".", "p"),
                                 _stem(traj)])


        # measured loop-rate meter (should read ~50 Hz)
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
        sensor_init = np.zeros(30)
        sensor_init[5] = 9.81
        sensor_init[6:18] = self.ref.q_ref[0]
        cont0 = self.ref.phase_info(0.0)[1:5]
        if self.blind:
            cont0 = np.zeros_like(cont0)          # BLIND site 3/3: history seed
        frame_init = np.concatenate([sensor_init, cont0, np.zeros(12)])
        self.history = np.tile(frame_init, (self.H, 1))
        self.prev_measured = sensor_init
        self.prev_action = np.zeros(12)
        self.held_targets = None
        self.odom_xy = self.ref.base_pos[0, :2].copy()
        yaw_ref0 = yaw_from_quat_wxyz(self.ref.base_quat[0])
        d_yaw = yaw_ref0 - yaw_from_quat_wxyz(self._imu_quat())
        self.q_align = quat_about_z(d_yaw)
        self.handoff_done = True
        logging.info("HANDOFF: policy takes over (yaw align %+.1f deg)", np.degrees(d_yaw))


    def _policy_tick(self):
        t = self.ii * self.dt
        q, qd = self._read_iso()
        gyro = np.asarray(self.low_state.imu_state.gyroscope, float)
        accel = np.asarray(self.low_state.imu_state.accelerometer, float)
        measured_now = np.concatenate([gyro, accel, q, qd])
        forces_n = self._read_foot_forces_n()
        contacts_true = (forces_n > CONTACT_FORCE_THRESHOLD_HW).astype(float)
        # BLIND site 1/3: the history contact bits. contacts_true stays for logging.
        contacts_obs = np.zeros(4) if self.blind else contacts_true


        if self.held_targets is None:
            tau_applied = np.zeros(12)
        else:
            qt, qdt, tff = self.held_targets
            tau_applied = mdc_apply(self.KP * (qt - q) + self.KD * (qdt - qd) + tff, qd)


        frame = np.concatenate([self.prev_measured, contacts_obs, tau_applied])
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


        vhat = np.zeros(3)
        e_h = np.zeros(2)
        if self.use_velest:
            # head on the O(1)-scaled history (PRE-normalizer), then odometry.
            # No integration on the handoff tick (mirror of the reset-obs gate).
            hist_scaled = (self.history * self.frame_scale).reshape(-1)
            vhat = self.policy.estimate_vel(hist_scaled)
            if self.ii > 0:
                self.odom_xy += (R @ vhat)[:2] * self.dt
            e_w = pos_ref_xy - self.odom_xy
            cy, sy = np.cos(yaw), np.sin(yaw)
            e_h = np.array([cy * e_w[0] + sy * e_w[1], -sy * e_w[0] + cy * e_w[1]])
            blocks.append(np.concatenate([vhat, np.clip(e_h, -self.ODOM_CLAMP,
                                                        self.ODOM_CLAMP) / self.ODOM_CLAMP]))


        if self.use_cprev:
            masks, ttc_e = self.ref.contact_pack(t, self.PREVIEW_DT, self.NUM_FUTURE)
            # BLIND site 2/3: the measured load block. Plan masks + ttc stay live --
            # they are reference-derived, not pad-derived.
            load = (np.zeros(4) if self.blind
                    else np.clip(forces_n / BODY_WEIGHT, 0.0, 2.0))
            blocks.append(np.concatenate([masks, ttc_e, load]))


        obs = np.concatenate(blocks)
        assert obs.shape == (self.n_obs,), (obs.shape, self.n_obs)
        obs = np.clip(obs * self.actor_scale, -self.CLIP_OBS, self.CLIP_OBS)
        action = self.policy(obs)


        a_cmd = self.prev_action                  # 1-step act latency (training nominal)
        self.prev_action = action
        q_ref_t, qd_ref_t, tau_ff_t = self.ref.ref_at(t)
        q_target = q_ref_t + self.ACTION_SCALE * self.RESID_SCALE * a_cmd


        fade = max(0.0, 1.0 - self.ii / self.handoff_fade_ticks)
        for i in range(12):
            m = MOTOR_FROM_ISO[i]
            self.low_cmd.motor_cmd[m].q = float(q_target[i])
            self.low_cmd.motor_cmd[m].dq = float(qd_ref_t[i])
            self.low_cmd.motor_cmd[m].kp = self.KP
            self.low_cmd.motor_cmd[m].kd = self.KD
            self.low_cmd.motor_cmd[m].tau = float(tau_ff_t[i] + fade * self.tau_i[m])
        self.held_targets = (q_target, qd_ref_t, tau_ff_t)


        # ---- trace (floating-base tracking + everything needed to reconstruct) ----
        # ori_err is recomputed here rather than reused: the obs block only exists
        # when the ckpt has obs_ori_err, but the metric is worth having either way.
        _qc = quat * np.array([1.0, -1.0, -1.0, -1.0])
        _qe = qmul(_qc, quat_ref)
        tr = self.trace
        tr["t"].append(t)
        tr["rpy"].append(quat_to_euler_xyz(quat))
        tr["rpy_ref"].append(quat_to_euler_xyz(quat_ref))
        tr["ori_err"].append(2.0 * np.sign(_qe[0] if _qe[0] != 0 else 1.0) * _qe[1:4])
        tr["odom_xy"].append(self.odom_xy.copy())
        tr["ref_xy"].append(pos_ref_xy.copy())
        tr["vhat"].append(vhat.copy())
        tr["tilt"].append(grav_b[2])
        tr["action"].append(a_cmd.copy())
        tr["contacts"].append(contacts_true.copy())
        tr["forces"].append(forces_n.copy())
        tr["q"].append(q.copy())
        tr["qd"].append(qd.copy())
        tr["q_target"].append(q_target.copy())
        tr["tau_applied"].append(tau_applied.copy())


        if self.motiontime % 10 == 0:
            # yaw_e kept verbatim (same quantity as every earlier run's log, and the
            # one the actor's attitude block sees). rp_e is the NEW roll/pitch pair,
            # from the Euler-XYZ decomposition -- a different convention to yaw_e, so
            # never read the two as three components of one vector.
            _rp = np.degrees(_wrap_pi(tr["rpy"][-1] - tr["rpy_ref"][-1]))[:2]
            logging.info("t %.2f cont %s%s vhat [%+.2f %+.2f %+.2f] eh [%+.2f %+.2f] "
                         "rp_e [%+.1f %+.1f] yaw_e %+.1f tilt %.2f |a| %.2f (loop %.1f Hz)",
                         t, contacts_true.astype(int), " (blinded)" if self.blind else "",
                         *vhat, *e_h, *_rp, np.degrees(yaw_e), grav_b[2],
                         np.abs(a_cmd).max(), self._loop_hz)


        if grav_b[2] > -0.4:
            logging.error("TILT ABORT at t %.2f (grav_z %.2f) -> damping", t, grav_b[2])
            self.aborted = True
        self.ii += 1


    # ------------------------------------------------------- trace / reporting
    def save_trace(self, outdir="runs"):
        """Summary metrics + a PNG + the raw npz. Called from the MAIN thread once
        the policy phase has stopped writing (settle done, or aborted). Safe to call
        on a short/aborted run -- it plots whatever was captured."""
        # Truncate to the shortest key: an abort can land between two appends in the
        # control thread, leaving one key a row longer. Losing the partial row beats
        # raising here and taking the whole trace with it.
        raw = {k: v for k, v in self.trace.items() if len(v)}
        n = min((len(v) for v in raw.values()), default=0)
        T = {k: np.asarray(v[:n], float) for k, v in raw.items()}
        if not T or n < 2:
            logging.warning("no trace captured (run ended before the policy phase)")
            return None
        # One directory per run, so the Vicon record can simply be dropped in next to
        # the trace it belongs to. Files inside keep the full name anyway: a png
        # copied OUT of here still says which checkpoint/meta/resid/traj produced it.
        stamp = time.strftime("%Y%m%d_%H%M%S")
        rundir = os.path.join(outdir, f"{self.run_tag}_{stamp}")
        os.makedirs(rundir, exist_ok=True)
        stem = os.path.join(rundir, f"{self.run_tag}_{stamp}")


        t = T["t"]
        rpy_e = np.degrees(_wrap_pi(T["rpy"] - T["rpy_ref"]))       # (N,3) deg
        pos_e = T["odom_xy"] - T["ref_xy"]                          # (N,2) m
        aborted = bool(self.aborted)


        # ---- summary ------------------------------------------------------
        lines = [f"run          : {self.run_tag}",
                 f"checkpoint   : {self.ckpt_path}",
                 f"meta         : {self.meta_path}",
                 f"reference    : {self.traj_path}",
                 f"resid auth   : {self.RESID_SCALE:.2f}x",
                 f"outcome      : {'ABORTED (tilt)' if aborted else 'completed'}"
                 f"   {len(t)}/{self.n_ticks} ticks, {t[-1]:.2f}/{self.ref.duration:.2f} s",
                 f"contact-blind: {self.blind}",
                 "",
                 f"{'axis':>7}{'RMS err':>10}{'max|err|':>10}{'final':>9}"]
        for i, nm in enumerate(("roll", "pitch", "yaw")):
            e = rpy_e[:, i]
            lines.append(f"{nm:>7}{np.sqrt((e**2).mean()):9.2f}°{np.abs(e).max():9.2f}°"
                         f"{e[-1]:8.2f}°")
        for i, nm in enumerate(("odom x", "odom y")):
            e = pos_e[:, i]
            lines.append(f"{nm:>7}{np.sqrt((e**2).mean()):9.3f}m{np.abs(e).max():9.3f}m"
                         f"{e[-1]:8.3f}m")
        lines += ["",
                  f"worst tilt (grav_z) : {T['tilt'].max():+.3f}   (abort at > -0.40)",
                  f"|action| mean/max   : {np.abs(T['action']).mean():.3f} / "
                  f"{np.abs(T['action']).max():.3f}",
                  f"action saturation   : {(np.abs(T['action']) > 0.99).mean() * 100:.1f}% of joint-ticks",
                  f"resid authority     : {self.RESID_SCALE:.2f}x (effective scale "
                  f"{self.ACTION_SCALE * self.RESID_SCALE:.3f} rad)",
                  f"jitter (mrad/tick)  : {np.abs(np.diff(T['action'], axis=0)).mean() * 1000 * self.ACTION_SCALE * self.RESID_SCALE:.1f}"]
        # ---- contact profile: plan vs measured, per foot ---------------------
        # Ref sampled on the policy ticks. fref is in NEWTONS (trajopt lam); measured
        # is RAW PAD units (FOOT_FORCE_TO_N uncalibrated) -- compare TIMING and shape,
        # not amplitude. Sim baseline (matched plants, model_400s): takeoffs +0..40 ms,
        # touchdowns +60..140 ms late via ~3 cm apex overshoot -- the HW question is
        # whether the same signature holds here.
        idx = np.array([self.ref._index(tt)[0] for tt in t])
        fref = self.ref.force_ref[idx]                              # (N,4) N
        plan = self.ref.contact[idx]                                # (N,4) bool
        meas = T["forces"] > CONTACT_FORCE_THRESHOLD_HW             # (N,4)
        qref = self.ref.q_ref[idx]                                  # (N,12) ISO, rad
        qdref = self.ref.qd_ref[idx]                                # (N,12) ISO, rad/s
        lines += ["", "contact timing vs plan (ms, + = late; n/a = pad never released):",
                  f"{'event':>11}{'plan_t':>8}" + "".join(f"{nm:>7}" for nm in FOOT_NAMES)]
        events = {}                                # plan_tick -> {foot: delay_ms|None}
        for f in range(4):
            p = plan[:, f].astype(int); m = meas[:, f].astype(int)
            for k in range(1, len(p)):
                if p[k] == p[k - 1]:
                    continue
                kind = "touchdown" if p[k] else "takeoff"
                # pad-release guard (FL creep): a TD delay needs the pad OFF beforehand
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


        # ---- joint tracking: the three quantities a residual controller lives on --
        #   q_ref    the plan's angles
        #   q_target q_ref + action_scale*resid_scale*action  -- what the PD was TOLD
        #   q        what the joints did
        # |q-cmd| is the servo's own error; |q-ref| is the deviation from the plan;
        # |resid| is how much authority the policy actually spent. A run where
        # |resid| >> |q-cmd| means the policy, not the PD, shaped the motion.
        # NOTE velocity has no residual: _policy_tick commands dq = qd_ref unmodified.
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
                 traj_path=str(self.traj_path), resid_scale=self.RESID_SCALE,
                 action_scale=self.ACTION_SCALE, joint_names=np.array(ISO_NAMES),
                 force_ref=fref, contact_plan=plan, q_ref=qref, qd_ref=qdref, **T)


        # ---- plot (lazy import: a headless robot may not have matplotlib) ----
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:                                   # noqa: BLE001
            logging.warning("matplotlib unavailable (%s); wrote %s.npz/.txt only",
                            type(e).__name__, stem)
            return stem


        # planned flight windows, for shading -- makes landings readable at a glance
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


        fig, ax = plt.subplots(6, 1, figsize=(11, 15), sharex=True)
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
        ax[3].plot(t, T["ref_xy"][:, 0], "--", c="0.45", lw=1.4, label="ref x")
        ax[3].plot(t, T["odom_xy"][:, 0], c="C0", lw=1.6, label="odom x")
        ax[3].plot(t, T["ref_xy"][:, 1], "--", c="0.7", lw=1.4, label="ref y")
        ax[3].plot(t, T["odom_xy"][:, 1], c="C1", lw=1.6, label="odom y")
        ax[3].set_ylabel("base xy [m]")
        ax[3].set_title(f"position (velest odometry, NOT ground truth)   "
                        f"final dx {pos_e[-1, 0]:+.3f} m  dy {pos_e[-1, 1]:+.3f} m",
                        fontsize=9, loc="left")
        ax[3].legend(fontsize=8, loc="upper left", ncol=2)
        ax[4].plot(t, T["tilt"], c="C3", lw=1.5, label="grav_z")
        ax[4].axhline(-0.4, ls=":", c="r", lw=1.2, label="abort threshold")
        ax[4].plot(t, np.abs(T["action"]).max(1), c="C2", lw=1.2, label="|action| max")
        ax[4].axhline(1.0, ls=":", c="0.6", lw=1.0)
        ax[4].set_ylabel("tilt / |a|")
        ax[4].legend(fontsize=8, loc="lower left", ncol=3)
        for f, nm in enumerate(FOOT_NAMES):
            ax[5].plot(t, T["forces"][:, f], lw=1.3, label=nm)
        ax[5].axhline(CONTACT_FORCE_THRESHOLD_HW, ls=":", c="k", lw=1.2,
                      label="contact thresh")
        ax[5].set_ylabel("foot force [N?]")
        ax[5].set_xlabel("t [s]  (shaded = planned flight)")
        ax[5].legend(fontsize=8, loc="upper left", ncol=5)
        fig.suptitle(f"{self.run_tag}\n{'ABORTED' if aborted else 'completed'}   "
                     f"blind={self.blind}   resid {self.RESID_SCALE:.2f}x", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(stem + ".png", dpi=130)
        plt.close(fig)


        # ---- contact profile PNG: per-foot measured vs plan ------------------
        # Measured pad force (raw units, left axis) vs planned |F| (N, right axis,
        # grey dashed) + that foot's planned-stance shading. Amplitudes live in
        # different units until FOOT_FORCE_TO_N is calibrated -- read TIMING/shape.
        fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
        for f, nm in enumerate(FOOT_NAMES):
            a = ax[f]
            p = plan[:, f]
            k = 0
            while k < len(p):                     # this foot's planned stance spans
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
        fig.suptitle(f"{self.run_tag}\ncontact profile -- plan (N, dashed) vs "
                     f"measured pad (raw)", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(stem + "_contact.png", dpi=130)
        plt.close(fig)
        # ---- joint tracking PNGs: 4 legs x 3 joints ---------------------------
        # Position gets three curves because the controller is RESIDUAL: the plan, the
        # plan plus what the policy added, and the measurement. The gap between the
        # first two is the policy's contribution; the gap between the second and third
        # is the servo's error. Reading only "ref vs actual" conflates them.
        # Velocity gets two: qd_ref IS the commanded dq, untouched by the residual.
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
                               label="ref + RL residual (commanded)")
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
                                               else " (no residual on velocity)"),
                         fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            fig.savefig(stem + sfx + ".png", dpi=130)
            plt.close(fig)


        logging.info("trace written: %s/  (png, _contact.png, _joint_pos.png, "
                     "_joint_vel.png, npz, txt)", rundir)
        logging.info("  -> copy the Vicon record into that directory, then: "
                     "python3 plot_base_pos.py --dir %s", rundir)
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
        damped stop instead of killing the thread (which would freeze the command
        stream with the motors holding the last stiff command)."""
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


        # measured loop rate (updated once per second)
        now = time.perf_counter()
        if self._rate_t0 is None:
            self._rate_t0 = now
        self._rate_n += 1
        if now - self._rate_t0 >= 1.0:
            self._loop_hz = self._rate_n / (now - self._rate_t0)
            self._rate_t0, self._rate_n = now, 0


        if self.aborted:
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
    ap.add_argument("--checkpoint", default="hopscotch_utils/pz3_999.pt")
    ap.add_argument("--meta", default="hopscotch_utils/pz3_meta.json", help="default: meta.json beside the checkpoint")
    ap.add_argument("--traj", default="traj_hopscotch_friction_6cm_lsq.json", help="override meta's traj_path")
    ap.add_argument("--resid_scale", type=float, default=0.50,
                    help="residual authority attenuation (1.0 = trained). 0.75 = the "
                         "2026-07-31 sim pick (every MJ-DR tail improved, partial apex/"
                         "timing fix, ~15 mrad); 0.5 = full nominal launch fix but opens "
                         "the tilt tail on the worst plants. Applied at q_target only; "
                         "obs/ff/gains/ref untouched.")
    ap.add_argument("--dry-run", action="store_true",
                    help="load + validate everything, then exit without touching the robot")
    ap.add_argument("--out", default="data",
                    help="parent directory for per-run trace dirs. Each run gets its "
                         "own data/<run_tag>_<stamp>/ holding png/_contact.png/npz/txt; "
                         "drop the Vicon record in beside them. default: data/")
    ap.add_argument("iface", nargs="?", default=None)
    args = ap.parse_args()


    meta, meta_path = load_meta(args.checkpoint, args.meta)


    if args.dry_run:
        # Construction does all the validation (contract checks, reference load, ff
        # bake, obs layout, checkpoint shapes) and touches neither DDS nor the motors.
        logging.info("--dry-run: constructing controller (no DDS, no motion)...")
        c = Custom(args.checkpoint, meta, args.traj, resid_scale=args.resid_scale,
                   meta_path=meta_path)
        logging.info("DRY RUN OK: obs=%d ticks=%d blind=%s ff_d=%s ff_Ia=%s",
                     c.n_obs, c.n_ticks, c.blind,
                     meta.get("ff_damping_comp"), meta.get("ff_armature_comp"))
        logging.info("artifacts would land in: %s/%s_<stamp>/ "
                     "(png, _contact.png, _joint_pos.png, _joint_vel.png, npz, txt)",
                     args.out, c.run_tag)
        sys.exit(0)


    print("WARNING: Please ensure there are no obstacles around the robot while running.")
    input("Press Enter to continue...")


    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)


    custom = Custom(args.checkpoint, meta, args.traj, resid_scale=args.resid_scale,
                    meta_path=meta_path)
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
