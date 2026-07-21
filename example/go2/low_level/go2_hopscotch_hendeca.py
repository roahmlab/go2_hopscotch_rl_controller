"""Go2 hardware deployment of the hendeca-v3 residual RL policy (rsl-rl / PyTorch).

Policy provenance: /mnt/ws-frb/users/vansht/resrl/go2_hopscotch/hopscotch_hendeca_v3
(trained at repo commit 534dd1d). Every obs/action convention below was verified
against that commit's go2_hopscotch_env.py / go2_mujoco_vec_env.py /
reference_manager_hopscotch.py / domain_rand.py / motor_model.py and the
checkpoint's own tensors (actor 580-512-256-128-12 ELU, obs_norm _mean/_std).

OBS CONTRACT (580, assembled at 50 Hz, ISO = Isaac type-grouped joint order):
  [0:460]   10-frame history, oldest->newest, of 46-dim frames:
              [gyro 3, accel(specific force) 3, q 12, qd 12,   <- 1-CONTROL-STEP DELAYED
               foot contact bool 4 (FL FR RL RR),              <- fresh
               applied torque post-MDC 12]                     <- prev targets @ fresh q,qd
  [460:568] reference preview at t, t+0.1 s, t+0.2 s: [q_ref, qd_ref, tau_ff] each 12
  [568:575] phase block: t/duration 1, ref contacts 4, ref airborne 1, which-jump 1
  [575:580] attitude: projected gravity 3 (body frame), sin/cos yaw error vs ref 2
  Then: * O(1) scale (accel /9.81, qd /15, tau /45), clip +-10,
        empirical normalization (x-mean)/(std+0.01), actor MLP, clip actions +-1.

TRAINING-NOMINAL LATENCIES (domain_rand SCHED never samples below 1 control step):
  sensors: the 30 measured dims in the newest frame are the PREVIOUS tick's reading.
  actions: the command sent at tick k uses the action computed at tick k-1.
Both are replicated in software here; the real pipeline's few extra ms sit inside
the DR ranges (obs 1-3 steps, act 1-2 steps at level 1.0).

ACTION -> COMMAND (ISO -> Unitree motor order at the boundary):
  q_target = q_ref(t) + 0.2 * a_prev ; dq = qd_ref(t) ; kp=100 kd=2.5 ; tau = tau_ff(t)
(the firmware's 1 kHz PD loop stands in for the sim's 200 Hz/1 kHz inner loop; the
MDC motor model is only replicated for the applied-torque OBS channel, the real
motors enforce the physical version themselves).

Not replicated (deliberately): sensor/attitude noise, coulomb obs term (<=0.5 N*m,
i.e. <=0.011 after the /45 obs scale), kp/kd/mass/friction DR - all zero-mean
robustness axes the hardware provides for real.

Stand-up: proven fold -> align -> 3 s hold with integral gravity calibration
(go2_standup_q0.py results: max|err| 0.013 rad), then handoff with the hold
torque faded over 0.5 s.
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
# ISO (Isaac type-grouped): [FL FR RL RR]_hip, [FL FR RL RR]_thigh, [FL FR RL RR]_calf
ISO_NAMES = [f"{leg}_{part}_joint" for part in ("hip", "thigh", "calf")
             for leg in ("FL", "FR", "RL", "RR")]
# Unitree motor index for each ISO slot (motors: FR0-2, FL3-5, RR6-8, RL9-11)
MOTOR_FROM_ISO = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
# low_state.foot_force order is FR FL RR RL; obs contact order is FL FR RL RR
FOOTFORCE_FROM_ISO_FOOT = [1, 0, 3, 2]
FOOT_NAMES = ["FL", "FR", "RL", "RR"]

# ------------------------------------------------------- obs / control consts
KP, KD = 100.0, 2.5                      # meta.json (training PD, firmware-side here)
ACTION_SCALE = 0.2
CTRL_DT = 0.02                           # 50 Hz
OBS_HISTORY_LEN = 10
PREVIEW_DT = 0.1                         # preview_steps(5) * step_dt
NUM_FUTURE = 2
CLIP_OBS = 10.0
OBS_QD_SCALE = 1.0 / 15.0
OBS_TAU_SCALE = 1.0 / 45.0
OBS_ACC_SCALE = 1.0 / 9.81
# HW foot-force threshold for the binary contact obs. Sim used 1 N on true GRF;
# the Go2 sole sensors idle ~0-10 raw units unloaded, so gate above that. TUNABLE.
CONTACT_FORCE_THRESHOLD_HW = 20.0

# MDC nominal (meta.json / GO2_MDC; SCHED nominal vbat/pbat = 28.8 V / 1728 W)
MDC_VBAT, MDC_PBAT = 28.8, 1728.0
MDC_GR, MDC_KT, MDC_R = 6.33, 0.26, 0.66
MDC_ALPHA = MDC_R / (MDC_KT * MDC_GR)    # V / Nm
MDC_BETA = MDC_GR * MDC_KT               # V*s/rad (Ke = Kt)
TAU_MAX_ISO = np.array([23.7] * 8 + [45.43] * 4)


def mdc_apply(tau_des, qd):
    """Nominal single-robot port of Go1MotorModel.apply_constraints (ISO order)."""
    V = np.clip(MDC_ALPHA * tau_des + MDC_BETA * qd, -MDC_VBAT, MDC_VBAT)
    tau_v = (V - MDC_BETA * qd) / MDC_ALPHA
    A = float(np.sum(MDC_R * tau_v * tau_v / (MDC_KT * MDC_GR) ** 2))   # copper loss
    B = float(np.sum(tau_v * qd))                                        # mech (Kt*Kv=1)
    if A + B > MDC_PBAT:
        eta = 2.0 * MDC_PBAT / (B + np.sqrt(B * B + 4.0 * A * MDC_PBAT) + 1e-8)
        tau_v = tau_v * np.clip(eta, 0.0, 1.0)
    return np.clip(np.nan_to_num(tau_v), -TAU_MAX_ISO, TAU_MAX_ISO)


# ------------------------------------------------------------ quaternion math
def euler_xyz_to_quat_wxyz(e):
    """Intrinsic XYZ euler -> wxyz quats, (N,3)->(N,4). Mirror of the ref manager."""
    def axis_quat(a, ax):
        q = np.zeros((len(a), 4))
        q[:, 0] = np.cos(a / 2.0)
        q[:, 1 + ax] = np.sin(a / 2.0)
        return q

    def qmul(a, b):
        aw, ax, ay, az = a.T
        bw, bx, by, bz = b.T
        return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                         aw * bx + ax * bw + ay * bz - az * by,
                         aw * by - ax * bz + ay * bw + az * bx,
                         aw * bz + ax * by - ay * bx + az * bw], axis=1)
    e = np.atleast_2d(np.asarray(e, float))
    return qmul(qmul(axis_quat(e[:, 0], 0), axis_quat(e[:, 1], 1)), axis_quat(e[:, 2], 2))


def yaw_from_quat_wxyz(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rotmat_from_quat_wxyz(q):
    """body -> world rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# ----------------------------------------------------------- reference (v3!)
class HopscotchRef:
    """Numpy port of HopscotchReferenceManager @ commit 534dd1d (the exact loader
    the v3 policy trained against, INCLUDING its boundary-knot concat semantics —
    the later half-open-interval bugfix must NOT be applied here)."""

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
            if i > 0:                       # drop duplicated boundary knot (v3 semantics)
                q, v, u, cm = q[1:], v[1:], u[1:], cm[1:]
            qs.append(q); vs.append(v); us.append(u); cs.append(cm)
        q = np.concatenate(qs); v = np.concatenate(vs)
        u = np.concatenate(us); contact = np.concatenate(cs)

        src = [n.replace("_joint", "") for n in modes[0]["joint_names"][6:]]
        gather = [src.index(n.replace("_joint", "")) for n in ISO_NAMES]
        self.q_ref = q[:, 6:18][:, gather]           # (T,12) ISO
        self.qd_ref = v[:, 6:18][:, gather]
        self.tau_ff = u[:, gather]                    # ff comps are 0.0 for v3
        self.base_quat = euler_xyz_to_quat_wxyz(q[:, 3:6])
        self.contact = contact                        # (T,4) FL FR RL RR
        self.airborne = (~contact).all(axis=1)

        self.T_state = self.q_ref.shape[0]
        self.T_ctrl = self.tau_ff.shape[0]
        self.duration = (self.T_state - 1) * self.dt

        # which-jump segmentation (min 50 knots ~ 50 ms of flight), mirror of v3
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

    def _index(self, t):
        f = t / self.dt
        i0 = int(np.clip(np.floor(f), 0, self.T_state - 2))
        frac = float(np.clip(f - i0, 0.0, 1.0))
        return i0, frac

    def ref_at(self, t):
        """(q_ref, qd_ref, tau_ff) at time t: q/qd lerped, tau floor-indexed."""
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
        return np.concatenate(outs)                   # 108

    def phase_info(self, t):
        i0, _ = self._index(t)
        phase = float(np.clip(t / self.duration, 0.0, 1.0))
        return np.concatenate([[phase], self.contact[i0].astype(float),
                               [float(self.airborne[i0])],
                               [self.which_jump[i0] / max(1, self.n_jumps)]])  # 7

    def yaw_at(self, t):
        i0, fr = self._index(t)
        q0, q1 = self.base_quat[i0], self.base_quat[i0 + 1]
        if np.dot(q0, q1) < 0:
            q1 = -q1
        q = (1 - fr) * q0 + fr * q1
        return yaw_from_quat_wxyz(q / max(np.linalg.norm(q), 1e-8))


# ----------------------------------------------------------------- policy
class HendecaPolicy:
    """rsl-rl checkpoint -> numpy inference: EmpiricalNormalization then
    580-512-256-128-12 ELU MLP, actions clipped to [-1,1]."""

    def __init__(self, ckpt_path):
        import torch                                   # needed only to unpickle
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck["model_state_dict"]
        self.W = [sd[f"actor.{i}.weight"].numpy().astype(np.float64) for i in (0, 2, 4, 6)]
        self.b = [sd[f"actor.{i}.bias"].numpy().astype(np.float64) for i in (0, 2, 4, 6)]
        on = ck["obs_norm_state_dict"]
        self.mean = on["_mean"].numpy().ravel().astype(np.float64)
        self.std = on["_std"].numpy().ravel().astype(np.float64)
        assert self.W[0].shape == (512, 580) and self.W[-1].shape == (12, 128)
        logging.info("policy loaded: %s (iter %s)", ckpt_path, ck.get("iter"))

    def __call__(self, obs):
        x = (obs - self.mean) / (self.std + 1e-2)      # rsl_rl eps
        for W, b in zip(self.W[:-1], self.b[:-1]):
            x = W @ x + b
            x = np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))   # ELU(alpha=1)
        return np.clip(self.W[-1] @ x + self.b[-1], -1.0, 1.0)


class Custom:
    def __init__(self, ckpt_path, traj_path):
        self.dt = CTRL_DT
        self.motiontime = 0

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None

        self.ref = HopscotchRef(traj_path)
        self.policy = HendecaPolicy(ckpt_path)
        self.n_ticks = int(np.ceil(self.ref.duration / self.dt))
        logging.info("ref: %d knots @ %.0f Hz, %.2f s, %d jumps -> %d policy ticks",
                     self.ref.T_state, 1 / self.ref.dt, self.ref.duration,
                     self.ref.n_jumps, self.n_ticks)

        # O(1) obs conditioning (obs_o1_scale=True in meta.json)
        fs = np.ones(46)
        fs[3:6] = OBS_ACC_SCALE
        fs[18:30] = OBS_QD_SCALE
        fs[34:46] = OBS_TAU_SCALE
        pv_s = np.ones(36)
        pv_s[12:24] = OBS_QD_SCALE
        pv_s[24:36] = OBS_TAU_SCALE
        self.actor_scale = np.concatenate([np.tile(fs, OBS_HISTORY_LEN),
                                           np.tile(pv_s, 1 + NUM_FUTURE),
                                           np.ones(7 + 5)])
        assert self.actor_scale.shape == (580,)

        # ---- stand-up (proven fold/align/hold recipe), Unitree MOTOR order ----
        self.Kp_stand, self.Kd_stand = 60.0, 5.0
        self.foldPos = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65,       # FR FL RR RL
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
        self.tau_i = np.zeros(12)                 # motor order
        self.handoff_fade_ticks = 25

        # ---- policy-phase state ----
        self.ii = 0
        self.start_policy = False             # armed gate: main thread's Enter releases it
        self._armed_logged = False
        self.handoff_done = False
        self.prev_action = np.zeros(12)           # 1-step ACTION latency (training nominal)
        self.prev_measured = None                 # 1-step SENSOR latency
        self.held_targets = None                  # (q_target, qd_target, tau_ff) ISO
        self.history = None                       # (10, 46) oldest->newest
        self.yaw_imu0 = 0.0
        self.yaw_ref0 = self.ref.yaw_at(0.0)
        self.settle_percent = 0
        self.settle_duration = 50
        self.aborted = False

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
        """Onboard attitude-estimator quaternion (wxyz, body->world). Roll/pitch are
        filter-tight; yaw drifts, but only yaw-SINCE-HANDOFF enters the obs and the
        training DR modeled exactly this drift (yaw_bias 0.1 rad in flight)."""
        return np.asarray(self.low_state.imu_state.quaternion, float)

    def _read_iso(self):
        q = np.array([self.low_state.motor_state[m].q for m in MOTOR_FROM_ISO])
        qd = np.array([self.low_state.motor_state[m].dq for m in MOTOR_FROM_ISO])
        return q, qd

    def _read_contacts(self):
        ff = np.array([self.low_state.foot_force[j] for j in FOOTFORCE_FROM_ISO_FOOT],
                      dtype=float)
        return (ff > CONTACT_FORCE_THRESHOLD_HW).astype(float)

    def _seed_policy_state(self):
        """Mirror of the training reset seeding: idealized stationary sensor frame
        (gyro 0, accel [0,0,9.81], q = q_ref(0), qd = 0), ref contacts, zero torque."""
        sensor_init = np.zeros(30)
        sensor_init[5] = 9.81
        sensor_init[6:18] = self.ref.q_ref[0]
        cont0 = self.ref.phase_info(0.0)[1:5]
        frame_init = np.concatenate([sensor_init, cont0, np.zeros(12)])
        self.history = np.tile(frame_init, (OBS_HISTORY_LEN, 1))
        self.prev_measured = sensor_init
        self.prev_action = np.zeros(12)
        self.held_targets = None
        self.yaw_imu0 = yaw_from_quat_wxyz(self._imu_quat())
        self.handoff_done = True
        logging.info("HANDOFF: policy takes over (imu yaw0 %.1f deg)",
                     np.degrees(self.yaw_imu0))

    def _policy_tick(self):
        t = self.ii * self.dt
        q, qd = self._read_iso()
        gyro = np.asarray(self.low_state.imu_state.gyroscope, float)
        accel = np.asarray(self.low_state.imu_state.accelerometer, float)
        measured_now = np.concatenate([gyro, accel, q, qd])
        contacts = self._read_contacts()

        # applied-torque obs channel: previous tick's held targets at fresh q/qd,
        # through the nominal MDC (mirror of the sim's post-MDC _applied_torque)
        if self.held_targets is None:
            tau_applied = np.zeros(12)
        else:
            qt, qdt, tff = self.held_targets
            tau_applied = mdc_apply(KP * (qt - q) + KD * (qdt - qd) + tff, qd)

        frame = np.concatenate([self.prev_measured, contacts, tau_applied])   # 46
        self.prev_measured = measured_now
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = frame

        # attitude from the onboard estimator: projected gravity + yaw error
        quat = self._imu_quat()
        grav_b = rotmat_from_quat_wxyz(quat).T @ np.array([0.0, 0.0, -1.0])
        yaw_true = self.yaw_ref0 + (yaw_from_quat_wxyz(quat) - self.yaw_imu0)
        yaw_e = yaw_true - self.ref.yaw_at(t)
        attitude = np.concatenate([grav_b, [np.sin(yaw_e)], [np.cos(yaw_e)]])

        obs = np.concatenate([self.history.reshape(-1), self.ref.preview(t),
                              self.ref.phase_info(t), attitude])
        obs = np.clip(obs * self.actor_scale, -CLIP_OBS, CLIP_OBS)
        action = self.policy(obs)

        # command uses the PREVIOUS action (1-step act latency, training nominal)
        a_cmd = self.prev_action
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
            logging.info("t %.2f phase %.2f cont %s yaw_e %+.1f deg tilt %.2f |a| %.2f "
                         "(loop %.1f Hz)",
                         t, t / self.ref.duration, contacts.astype(int),
                         np.degrees(yaw_e), grav_b[2], np.abs(a_cmd).max(), self._loop_hz)

        # safety: training terminates on tilt; on HW go limp-damped instead of fighting
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
            # stage 1: lie -> fold (feet tucked under hips, unloaded)
            self.fold_percent = min(self.fold_percent + 1.0 / self.fold_duration, 1)
            for m in range(12):
                self.low_cmd.motor_cmd[m].q = float((1 - self.fold_percent) * self.startPos[m]
                                                    + self.fold_percent * self.foldPos[m])
                self.low_cmd.motor_cmd[m].dq = 0
                self.low_cmd.motor_cmd[m].kp = self.Kp_stand
                self.low_cmd.motor_cmd[m].kd = self.Kd_stand
                self.low_cmd.motor_cmd[m].tau = 0

        elif self.align_percent < 1:
            # stage 2: fold -> q0 (vertical push-up)
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
                # still calibrating: report progress + real loop rate
                if self.motiontime % 10 == 0:
                    q_now = np.array([self.low_state.motor_state[m].q for m in range(12)])
                    logging.info("calibrating: hold max|err| %.3f (loop %.1f Hz)",
                                 np.abs(q_now - self.q0_motor).max(), self._loop_hz)
            elif not self._armed_logged:
                # calibrated + armed: log once, then go quiet so the launch
                # prompt in the main thread stays readable (was drowned in spam)
                q_now = np.array([self.low_state.motor_state[m].q for m in range(12)])
                logging.info("ARMED: calibrated (max|err| %.3f, loop %.1f Hz) + holding q0 "
                             "-> press Enter at the LAUNCH prompt to start the policy",
                             np.abs(q_now - self.q0_motor).max(), self._loop_hz)
                self._armed_logged = True

        elif self.ii < self.n_ticks:
            if not self.handoff_done:
                self._seed_policy_state()
            self._policy_tick()

        elif self.settle_percent < 1:
            # settle: hold the reference's final pose
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
    ap.add_argument("--checkpoint", default="hopscotch_utils/model_1499_v3.pt")
    ap.add_argument("--traj", default="hopscotch_utils/traj_hopscotch_friction.json")
    ap.add_argument("iface", nargs="?", default=None, help="network interface")
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
