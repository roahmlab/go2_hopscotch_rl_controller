"""Go2 hardware deployment of the hendeca-v18_cprev residual RL policy (velest +
contact-preview line).

Policy provenance: /mnt/ws-frb/users/vansht/resrl/go2_hopscotch/hendeca_v18_cprev
(trained from the 2026-07 working tree: v13 recipe + obs_ori_err + obs_velest +
obs_contact_prev, w_imp_tan 0.25/0.20/0.20). Verified against the CURRENT
isaac_port sources (go2_mujoco_vec_env.py deployment mirror, vel_estimator.py,
reference_manager_hopscotch.py) and the checkpoint tensors (actor 608-512-256-128-12
ELU, estimator 460-256-128-3 ELU, obs_norm (1,608)).

DIFFERENCES vs the v3 controller (go2_hopscotch_hendeca.py):
  * obs 580 -> 608: appends [ori_err 3 | velest 5 | contact pack 20] after attitude.
      ori_err: 2*sign(w)*vec(conj(q_true) x q_ref(t))  (body-frame rotvec, onboard
               IMU-estimator quat yaw-aligned to the ref world at handoff)
      velest:  vhat(3) = EstimatorHead(O(1)-scaled 460 history — PRE-normalizer!);
               odom_xy += R(q_true) vhat * dt  (world XY, seeded to ref start, no
               integration on the handoff tick); obs = [vhat,
               clamp(ref_xy(t) - odom_xy, +-0.5)/0.5 rotated into heading frame]
      pack:    planned contact masks at t, t+0.1, t+0.2 (12) + per-foot
               time-to-next-contact-EDGE /0.3 s clamped [0,1] (4) +
               measured foot load |F|/150 clamped [0,2] (4)
  * reference loader is the FIXED half-open concat (non-final modes drop their LAST
    row) — v18 trained post-bugfix; the v3 controller's pre-fix loader would be wrong.
  * empirical normalization stats are (1,608) from this checkpoint.
Everything else (46-dim frames, 1-step sensor+action latency, MDC applied-torque
channel, PD 100/2.5 with dq=qd_ref, ISO ordering, stand-up + integrator handoff)
is identical — see the v3 controller's header for the full contract rationale.

DR corruptions (velest/ori_err/load noise+bias) are training-only robustness axes
(noise_gate-scaled); the hardware supplies real noise, so clean signals here.

HW calibration knobs: CONTACT_FORCE_THRESHOLD_HW (binary contacts) and
FOOT_FORCE_TO_N (the load channel needs foot_force in ~Newtons; standing should
read load ~ 0.25/foot. Log foot_force during the hold and set the scale).
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

# ------------------------------------------------------- obs / control consts
KP, KD = 100.0, 2.5
ACTION_SCALE = 0.2
CTRL_DT = 0.02
OBS_HISTORY_LEN = 10
PREVIEW_DT = 0.1
NUM_FUTURE = 2
CLIP_OBS = 10.0
OBS_QD_SCALE = 1.0 / 15.0
OBS_TAU_SCALE = 1.0 / 45.0
OBS_ACC_SCALE = 1.0 / 9.81
ODOM_CLAMP = 0.5                                 # meta.json odom_clamp
BODY_WEIGHT = 150.0                              # N, load-channel normalizer
TTC_EDGE_CLIP_S = 0.3
CONTACT_FORCE_THRESHOLD_HW = 20.0                # raw units, binary contacts. TUNABLE
FOOT_FORCE_TO_N = 1.0                            # raw foot_force -> Newtons. CALIBRATE

MDC_VBAT, MDC_PBAT = 28.8, 1728.0
MDC_GR, MDC_KT, MDC_R = 6.33, 0.26, 0.66
MDC_ALPHA = MDC_R / (MDC_KT * MDC_GR)
MDC_BETA = MDC_GR * MDC_KT
TAU_MAX_ISO = np.array([23.7] * 8 + [45.43] * 4)


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


# ------------------------------------------------- reference (FIXED loader!)
class HopscotchRef:
    """Numpy port of the CURRENT HopscotchReferenceManager (post half-open-bugfix,
    the loader v18 trained against): non-final modes drop their LAST row so each
    boundary knot keeps the next mode's initial (post-impact) state. Includes the
    v18 ttc_edge precompute + contact pack."""

    def __init__(self, json_path):
        modes = json.load(open(json_path))
        self.dt = float(modes[0]["dt"])
        qs, vs, us, cs = [], [], [], []
        for i, m in enumerate(modes):
            q = np.asarray(m["q"], float)
            v = np.asarray(m["v"], float)
            u = np.asarray(m["u"], float)
            cm = np.zeros((len(q), 4), bool)
            for foot in m["contacts"]:
                cm[:, FOOT_NAMES.index(foot.split("_")[0])] = True
            if i < len(modes) - 1:               # HALF-OPEN: drop non-final LAST row
                q, v, u, cm = q[:-1], v[:-1], u[:-1], cm[:-1]
            qs.append(q); vs.append(v); us.append(u); cs.append(cm)
        q = np.concatenate(qs); v = np.concatenate(vs)
        u = np.concatenate(us); contact = np.concatenate(cs)

        src = [n.replace("_joint", "") for n in modes[0]["joint_names"][6:]]
        gather = [src.index(n.replace("_joint", "")) for n in ISO_NAMES]
        self.q_ref = q[:, 6:18][:, gather]
        self.qd_ref = v[:, 6:18][:, gather]
        self.tau_ff = u[:, gather]                # ff comps 0.0
        self.base_pos = q[:, 0:3].copy()          # z offset irrelevant (only xy used)
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

        # v18: per-foot knots until that foot's planned contact bit next flips
        edge = np.full((self.T_state, 4), self.T_state, dtype=np.float64)
        nxt_flip = np.full(4, 2.0 * self.T_state)
        for i in range(self.T_state - 2, -1, -1):
            flip = contact[i + 1] != contact[i]
            nxt_flip[flip] = i + 1
            edge[i] = nxt_flip - i
        self.ttc_edge = edge * self.dt            # (T,4) seconds

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

    def preview(self, t):
        outs = []
        for j in range(NUM_FUTURE + 1):
            q, qd, tau = self.ref_at(t + j * PREVIEW_DT)
            outs += [q, qd, tau]
        return np.concatenate(outs)               # 108

    def phase_info(self, t):
        i0, _ = self._index(t)
        phase = float(np.clip(t / self.duration, 0.0, 1.0))
        return np.concatenate([[phase], self.contact[i0].astype(float),
                               [float(self.airborne[i0])],
                               [self.which_jump[i0] / max(1, self.n_jumps)]])  # 7

    def contact_pack(self, t):
        """(masks 12, ttc_edge_norm 4) — plan-derived half of the v18 pack."""
        masks = []
        for j in range(NUM_FUTURE + 1):
            i0, _ = self._index(t + j * PREVIEW_DT)
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


class HendecaV18Policy:
    """Checkpoint -> numpy: EmpiricalNormalization + actor MLP (608->...->12) and
    the concurrent velocity-estimator head (460->256->128->3, ELU, NO normalizer —
    it consumes the O(1)-scaled raw history, mirroring vel_estimator.py)."""

    def __init__(self, ckpt_path):
        import torch
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck["model_state_dict"]
        self.W = [sd[f"actor.{i}.weight"].numpy().astype(np.float64) for i in (0, 2, 4, 6)]
        self.b = [sd[f"actor.{i}.bias"].numpy().astype(np.float64) for i in (0, 2, 4, 6)]
        on = ck["obs_norm_state_dict"]
        self.mean = on["_mean"].numpy().ravel().astype(np.float64)
        self.std = on["_std"].numpy().ravel().astype(np.float64)
        es = ck["estimator_state_dict"]
        self.eW = [es[f"net.{i}.weight"].numpy().astype(np.float64) for i in (0, 2, 4)]
        self.eb = [es[f"net.{i}.bias"].numpy().astype(np.float64) for i in (0, 2, 4)]
        assert self.W[0].shape == (512, 608), self.W[0].shape
        assert self.eW[0].shape == (256, 460), self.eW[0].shape
        assert self.mean.shape == (608,)
        logging.info("v18 policy loaded: %s (iter %s)", ckpt_path, ck.get("iter"))

    def estimate_vel(self, hist_scaled_460):
        x = hist_scaled_460
        for W, b in zip(self.eW[:-1], self.eb[:-1]):
            x = _elu(W @ x + b)
        return self.eW[-1] @ x + self.eb[-1]      # (3,) body-frame m/s

    def __call__(self, obs):
        x = (obs - self.mean) / (self.std + 1e-2)
        for W, b in zip(self.W[:-1], self.b[:-1]):
            x = _elu(W @ x + b)
        return np.clip(self.W[-1] @ x + self.b[-1], -1.0, 1.0)


class Custom:
    def __init__(self, ckpt_path, traj_path):
        self.dt = CTRL_DT
        self.motiontime = 0

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None

        self.ref = HopscotchRef(traj_path)
        self.policy = HendecaV18Policy(ckpt_path)
        self.n_ticks = int(np.ceil(self.ref.duration / self.dt))
        logging.info("ref: %d knots, %.2f s, %d jumps -> %d policy ticks",
                     self.ref.T_state, self.ref.duration, self.ref.n_jumps, self.n_ticks)

        # O(1) conditioning; extra 28 dims (ori_err 3 + velest 5 + pack 20) scale 1
        fs = np.ones(46)
        fs[3:6] = OBS_ACC_SCALE
        fs[18:30] = OBS_QD_SCALE
        fs[34:46] = OBS_TAU_SCALE
        pv_s = np.ones(36)
        pv_s[12:24] = OBS_QD_SCALE
        pv_s[24:36] = OBS_TAU_SCALE
        self.frame_scale = fs                     # also the estimator-input scaling
        self.actor_scale = np.concatenate([np.tile(fs, OBS_HISTORY_LEN),
                                           np.tile(pv_s, 1 + NUM_FUTURE),
                                           np.ones(7 + 5 + 3 + 5 + 20)])   # phase+att+ori+ve+pack
        assert self.actor_scale.shape == (608,), self.actor_scale.shape

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
        """Onboard attitude-estimator quaternion (wxyz, body->world). Roll/pitch are
        filter-tight; yaw drifts, but only yaw-SINCE-HANDOFF enters the obs/odometry
        and the training DR modeled exactly this drift (yaw_bias 0.1 rad in flight)."""
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
        """IMU quat composed with the fixed z-rotation that maps the handoff yaw
        onto the reference world's yaw(0) — the training spawn convention."""
        return qmul(self.q_align, self._imu_quat())

    def _seed_policy_state(self):
        sensor_init = np.zeros(30)
        sensor_init[5] = 9.81
        sensor_init[6:18] = self.ref.q_ref[0]
        cont0 = self.ref.phase_info(0.0)[1:5]
        frame_init = np.concatenate([sensor_init, cont0, np.zeros(12)])
        self.history = np.tile(frame_init, (OBS_HISTORY_LEN, 1))
        self.prev_measured = sensor_init
        self.prev_action = np.zeros(12)
        self.held_targets = None
        self.odom_xy = self.ref.base_pos[0, :2].copy()
        yaw_ref0 = yaw_from_quat_wxyz(self.ref.base_quat[0])
        d_yaw = yaw_ref0 - yaw_from_quat_wxyz(self._imu_quat())
        self.q_align = quat_about_z(d_yaw)
        self.handoff_done = True
        logging.info("HANDOFF: v18 policy takes over (yaw align %+.1f deg)",
                     np.degrees(d_yaw))

    def _policy_tick(self):
        t = self.ii * self.dt
        q, qd = self._read_iso()
        gyro = np.asarray(self.low_state.imu_state.gyroscope, float)
        accel = np.asarray(self.low_state.imu_state.accelerometer, float)
        measured_now = np.concatenate([gyro, accel, q, qd])
        forces_n = self._read_foot_forces_n()
        contacts = (forces_n > CONTACT_FORCE_THRESHOLD_HW).astype(float)

        if self.held_targets is None:
            tau_applied = np.zeros(12)
        else:
            qt, qdt, tff = self.held_targets
            tau_applied = mdc_apply(KP * (qt - q) + KD * (qdt - qd) + tff, qd)

        frame = np.concatenate([self.prev_measured, contacts, tau_applied])
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

        # ori_err rotvec: 2*sign(w)*vec(conj(q) x q_ref)  (mirror of the env)
        qc = quat * np.array([1.0, -1.0, -1.0, -1.0])
        qe = qmul(qc, quat_ref)
        ori_err = 2.0 * np.sign(qe[0] if qe[0] != 0 else 1.0) * qe[1:4]

        # velest: head on the O(1)-scaled history (PRE-normalizer), then odometry.
        # No integration on the handoff tick (mirror of the reset-obs gate).
        hist_scaled = (self.history * self.frame_scale).reshape(-1)
        vhat = self.policy.estimate_vel(hist_scaled)
        if self.ii > 0:
            self.odom_xy += (R @ vhat)[:2] * self.dt
        e_w = pos_ref_xy - self.odom_xy
        cy, sy = np.cos(yaw), np.sin(yaw)
        e_h = np.array([cy * e_w[0] + sy * e_w[1], -sy * e_w[0] + cy * e_w[1]])
        velest_block = np.concatenate([vhat, np.clip(e_h, -ODOM_CLAMP, ODOM_CLAMP) / ODOM_CLAMP])

        # v18 contact pack: planned masks + ttc-edge (deploy-exact) + measured load
        masks, ttc_e = self.ref.contact_pack(t)
        load = np.clip(forces_n / BODY_WEIGHT, 0.0, 2.0)
        pack = np.concatenate([masks, ttc_e, load])

        obs = np.concatenate([self.history.reshape(-1), self.ref.preview(t),
                              self.ref.phase_info(t), attitude, ori_err,
                              velest_block, pack])
        assert obs.shape == (608,)
        obs = np.clip(obs * self.actor_scale, -CLIP_OBS, CLIP_OBS)
        action = self.policy(obs)

        a_cmd = self.prev_action                  # 1-step act latency (training nominal)
        self.prev_action = action
        q_ref_t, qd_ref_t, tau_ff_t = self.ref.ref_at(t)
        q_target = q_ref_t + ACTION_SCALE * a_cmd

        fade = max(0.0, 1.0 - self.ii / self.handoff_fade_ticks)
        for i in range(12):
            m = MOTOR_FROM_ISO[i]
            self.low_cmd.motor_cmd[m].q = float(q_target[i])
            self.low_cmd.motor_cmd[m].dq = float(qd_ref_t[i])
            self.low_cmd.motor_cmd[m].kp = KP
            self.low_cmd.motor_cmd[m].kd = KD
            self.low_cmd.motor_cmd[m].tau = float(tau_ff_t[i] + fade * self.tau_i[m])
        self.held_targets = (q_target, qd_ref_t, tau_ff_t)

        if self.motiontime % 10 == 0:
            logging.info("t %.2f cont %s vhat [%+.2f %+.2f %+.2f] eh [%+.2f %+.2f] "
                         "yaw_e %+.1f tilt %.2f |a| %.2f",
                         t, contacts.astype(int), *vhat, *e_h,
                         np.degrees(yaw_e), grav_b[2], np.abs(a_cmd).max())

        if grav_b[2] > -0.4:
            logging.error("TILT ABORT at t %.2f (grav_z %.2f) -> damping", t, grav_b[2])
            self.aborted = True
        self.ii += 1

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
            if self.motiontime % 10 == 0:
                q_now = np.array([self.low_state.motor_state[m].q for m in range(12)])
                ff = self._read_foot_forces_n()
                logging.info("hold max|err| %.3f  foot_force(N?) %s",
                             np.abs(q_now - self.q0_motor).max(), np.round(ff, 1))
            if self.hold_percent >= 1 and not self._armed_logged:
                logging.info("ARMED: calibrated + holding q0 - waiting for Enter")
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
    ap.add_argument("--checkpoint", default="hopscotch_utils/model_1499.pt")
    ap.add_argument("--traj", default="hopscotch_utils/traj_hopscotch_friction.json")
    ap.add_argument("iface", nargs="?", default=None)
    args = ap.parse_args()

    print("WARNING: Please ensure there are no obstacles around the robot while running.")
    input("Press Enter to continue...")

    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)

    custom = Custom(args.checkpoint, args.traj)
    custom.Init()
    custom.Start()

    launched = False
    while True:
        if custom.aborted:
            time.sleep(2)
            print("Aborted - robot in damping mode. Ctrl-C when secured.")
            time.sleep(10)
        if (not launched) and (not custom.aborted) and custom.hold_percent >= 1:
            input("Robot calibrated + holding q0. Press Enter to LAUNCH...")
            custom.start_policy = True
            launched = True
        if custom.settle_percent >= 1:
            time.sleep(1)
            print("Done!")
            sys.exit(0)
        time.sleep(0.5)
