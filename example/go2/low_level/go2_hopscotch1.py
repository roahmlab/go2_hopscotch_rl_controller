import time
import sys
import os
import pickle

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

import numpy as np


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat2eul(q):
    w, x, y, z = q
    r02 = 2 * (x * z + w * y); r12 = 2 * (y * z - w * x); r22 = 1 - 2 * (x * x + y * y)
    r01 = 2 * (x * y - w * z); r00 = 1 - 2 * (y * y + z * z)
    return np.array([np.arctan2(-r12, r22), np.arcsin(np.clip(r02, -1, 1)), np.arctan2(-r01, r00)])


class Custom:
    def __init__(self):
        self.Kp = 40.0
        self.Kd = 2.0

        # stand-up gains (fold/align/hold only; policy phase uses Kp/Kd above)
        self.Kp_stand = 60.0
        self.Kd_stand = 5.0

        self.dt = 0.005
        self.stride = 5
        self.traj_end = 3400


        self.ii = 0

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None
        self.publish = True

        self.startPos = [0.0] * 12

        # folded pose (feet tucked under hips, unloaded), MuJoCo order [FL, FR, RL, RR]
        self.foldPos = np.array([0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
                                 0.2, 1.36, -2.65, -0.2, 1.36, -2.65])
        self.fold_duration = 200
        self.fold_percent = 0

        self.alignment_duration = 200
        self.alignment_percent = 0

        # hold at q0 with integral action: converges to the static holding torque
        self.hold_duration = 1000
        self.hold_percent = 0
        self.Ki = 100.0
        self.tau_i_max = 15.0
        self.tau_i = np.zeros(12)
        self.handoff_fade_ticks = 100

        self.settle_duration = 400
        self.settle_percent = 0

        base_dir = os.path.join(os.path.dirname(__file__), ".")

        # thread handling
        self.lowCmdWriteThreadPtr = None

        self.crc = CRC()

        # timing stats
        self.tick_prev = None
        self.tick_sum = 0.0
        self.tick_max = 0.0
        self.inf_sum = 0.0
        self.inf_max = 0.0
        self.n_tick = 0
        self.log_tick = []
        self.log_align = []
        self.log_inf = []
        self.log_state = []
        self.log_sat = []
        self.sat_peak = 0.0
        self.sat_cnt = 0

        # Load reference trajectory
        f = np.load(os.path.join(base_dir, "hopscotch_utils", "trajectories.npz"))
        x_ref, u_ref = f["x_refs"], f["u_refs"]
        if x_ref.ndim == 3:
            x_ref, u_ref = x_ref[0], u_ref[0]
        self.x_ref = np.asarray(x_ref, dtype=np.float64)
        self.u_ref = np.asarray(u_ref, dtype=np.float64)
        self.traj_length = self.u_ref.shape[0]
        self.eul_ref = np.stack([quat2eul(self.x_ref[i, 3:7]) for i in range(len(self.x_ref))])
        self.ref_feat = np.concatenate(
            [self.eul_ref, self.x_ref[:, 7:19], self.x_ref[:, 22:25], self.x_ref[:, 25:37]],
            axis=1)

        # Load the actor; checkpoint metadata selects architecture, gains and observation layout.
        self.blind = "actor_blind1.pkl"
        with open(os.path.join(base_dir, "hopscotch_utils", self.blind), 'rb') as file:
            ck = pickle.load(file)
        self.actor = ck["actor"]
        self.arch = str(np.asarray(ck.get("arch", "gru")))
        self.Kp = float(np.asarray(ck.get("kp", ck.get("pd_kp", self.Kp))))
        self.Kd = float(np.asarray(ck.get("kd", ck.get("pd_kd", self.Kd))))
        self.preview_offsets = tuple(int(o) for o in ck["preview"])
        self.use_accelerometer = bool(np.asarray(ck.get("accelerometer", False)).item())
        self.accelerometer_gravity = float(
            np.asarray(ck.get("accelerometer_gravity", 9.81)).item())
        self.accelerometer_clip_g = float(
            np.asarray(ck.get("accelerometer_clip_g", 16.0)).item())
        obs_width = 73 + 3 * self.use_accelerometer + 30 * len(self.preview_offsets)
        if self.arch == "gru":
            self.h_offs = np.cumsum([0] + [uz.shape[0] for _, uz, *_ in self.actor[0]])
            self.h = np.zeros(self.h_offs[-1], dtype=np.float32)
            self.nf = self.actor[0][0][0].shape[0]
            self.policy(np.zeros(self.nf, dtype=np.float32))
            self.h[:] = 0.0
        else:
            self.setup_transformer(ck)
        assert self.nf == obs_width, f"obs width {obs_width} != actor NF {self.nf}"
        sizes = ([int(np.asarray(s)) for s in ck["gru_sizes"]] if self.arch == "gru"
                 else [int(np.asarray(ck["tf"]["d"])), int(np.asarray(ck["tf"]["k_obs"]))])
        print(f"{self.blind}: arch {self.arch} {sizes} iter {ck.get('iter')} "
              f"kp {self.Kp:g} kd {self.Kd:g} preview {self.preview_offsets}", flush=True)

        # Get q0 and qf
        self.q0 = self.x_ref[0][7:19]
        self.qf = self.x_ref[self.traj_end][7:19]

        # Record initial orientation offset
        self.firstRun = True
        self.record_odom = True
        self.q_off = None
        self.last_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.fault = False
        self.gyro_alpha = 1.0
        self.gyro_f = None

        self.tau_limit = np.array([23.7, 23.7, 45.43] * 4)

        # MuJoCo: [FL, FR, RL, RR]
        # Unitree Go2: [FR, FL, RR, RL]
        self.JOINT_REORDERING = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])


    def setup_transformer(self, ck):
        """Builds the jitted transformer forward; jax is imported only for transformer checkpoints."""
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        import jax
        import jax.numpy as jnp
        apj = jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float32), ck["actor"])
        tf = ck["tf"]
        K = int(np.asarray(tf["k_obs"]))
        D = int(np.asarray(tf["d"]))
        HD = int(np.asarray(tf["heads"]))
        dh = D // HD
        self.k_obs = K
        self.nf = ck["actor"][0].shape[0]
        self.buf = None
        self.jnp = jnp

        def ln(z, gm, bt):
            return (z - z.mean(-1, keepdims=True)) / jnp.sqrt(z.var(-1, keepdims=True) + 1e-6) * gm + bt

        def tf_apply(buf):
            we, be, pos, blocks, (gf, bf), (wh, bh) = apj
            x = buf @ we + be + pos
            for (g1, b1, wq, bq, wk, bk, wv, bv, wu, bu, g2, b2, w1, c1, w2, c2) in blocks:
                y = ln(x, g1, b1)
                q = (y @ wq + bq).reshape(K, HD, dh)
                kk = (y @ wk + bk).reshape(K, HD, dh)
                vv = (y @ wv + bv).reshape(K, HD, dh)
                at = jax.nn.softmax(jnp.einsum("qhd,khd->hqk", q, kk) / jnp.sqrt(dh), axis=-1)
                x = x + jnp.einsum("hqk,khd->qhd", at, vv).reshape(K, D) @ wu + bu
                y = ln(x, g2, b2)
                x = x + jax.nn.gelu(y @ w1 + c1) @ w2 + c2
            x = ln(x, gf, bf)
            return x[-1] @ wh + bh

        self.policy_fn = jax.jit(tf_apply)
        print("compiling transformer ...", flush=True)
        t0 = time.perf_counter()
        self.policy_fn(jnp.zeros((K, self.nf), jnp.float32)).block_until_ready()
        print(f"compiled in {time.perf_counter() - t0:.1f}s", flush=True)

    def warm_policy(self):
        """Runs a throwaway actor tick during stand-up so clocks and caches are hot at the handoff."""
        self.policy(np.zeros(self.nf, dtype=np.float32))
        if self.arch == "gru":
            self.h[:] = 0.0
        else:
            self.buf = None

    def policy(self, obs):
        """Runs one actor tick: numpy GRU state update, or the jitted transformer over a rolling buffer."""
        if self.arch == "gru":
            o = obs
            hs = []
            for j, (wz, uz, bz, wr, ur, br, wh, uh, bh) in enumerate(self.actor[0]):
                hl = self.h[self.h_offs[j]:self.h_offs[j + 1]]
                z = 1.0 / (1.0 + np.exp(-(o @ wz + hl @ uz + bz)))
                r = 1.0 / (1.0 + np.exp(-(o @ wr + hl @ ur + br)))
                n = np.tanh(o @ wh + (r * hl) @ uh + bh)
                o = (1.0 - z) * n + z * hl
                hs.append(o)
            self.h = np.concatenate(hs)
            wo, bo = self.actor[1]
            return o @ wo + bo
        self.buf = (np.tile(obs, (self.k_obs, 1)) if self.buf is None
                    else np.concatenate([self.buf[1:], obs[None]], 0))
        return np.asarray(self.policy_fn(self.jnp.asarray(self.buf)))

    # Public methods
    def Init(self):
        self.InitLowCmd()

        # create publisher #
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        # create subscriber #
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
            interval=self.dt, target=self.LowCmdWrite, name="writebasiccmd"
        )
        self.lowCmdWriteThreadPtr.Start()

    # Private methods
    def InitLowCmd(self):
        self.low_cmd.head[0]=0xFE
        self.low_cmd.head[1]=0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q= go2.PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = go2.VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def LowStateMessageHandler(self, msg: LowState_):
        self.low_state = msg

    def LowCmdWrite(self):

        now = time.perf_counter()
        if self.tick_prev is not None:
            dtick = now - self.tick_prev
            self.tick_sum += dtick
            self.tick_max = max(self.tick_max, dtick)
            self.log_tick.append(dtick)
            if self.hold_percent < 1:
                self.log_align.append(dtick)
            self.n_tick += 1
            if self.n_tick % 200 == 0:
                print(f"tick avg {1e3 * self.tick_sum / 200:.2f} max {1e3 * self.tick_max:.2f} ms | "
                      f"inference avg {1e3 * self.inf_sum / 200:.2f} max {1e3 * self.inf_max:.2f} ms | "
                      f"sat {self.sat_cnt}/200 peak {self.sat_peak:.2f}x", flush=True)
                self.tick_sum = self.tick_max = self.inf_sum = self.inf_max = 0.0
                self.sat_peak = 0.0
                self.sat_cnt = 0
        self.tick_prev = now

        if self.low_state is None:
            return

        if self.fault:
            for i in range(12):
                idx = self.JOINT_REORDERING[i]
                self.low_cmd.motor_cmd[idx].q = 0.0
                self.low_cmd.motor_cmd[idx].dq = 0.0
                self.low_cmd.motor_cmd[idx].kp = 0.0
                self.low_cmd.motor_cmd[idx].kd = 5.0
                self.low_cmd.motor_cmd[idx].tau = 0.0
            if self.publish:
                self.low_cmd.crc = self.crc.Crc(self.low_cmd)
                self.lowcmd_publisher.Write(self.low_cmd)
            return

        if self.hold_percent < 1:
            self.warm_policy()

        if self.firstRun:
            self.startPos = [self.low_state.motor_state[i].q for i in self.JOINT_REORDERING]
            self.firstRun = False

        if self.fold_percent < 1:
            # stage 1: lie -> fold (feet tucked under hips, unloaded)
            self.fold_percent += 1.0 / self.fold_duration
            self.fold_percent = min(self.fold_percent, 1)

            for i in range(12):
                idx = self.JOINT_REORDERING[i]
                self.low_cmd.motor_cmd[idx].q = float((1 - self.fold_percent) * self.startPos[i] + self.fold_percent * self.foldPos[i])
                self.low_cmd.motor_cmd[idx].dq = 0
                self.low_cmd.motor_cmd[idx].kp = self.Kp_stand
                self.low_cmd.motor_cmd[idx].kd = self.Kd_stand
                self.low_cmd.motor_cmd[idx].tau = 0

        elif self.alignment_percent < 1:
            # stage 2: fold -> q0 (vertical push-up)
            self.alignment_percent += 1.0 / self.alignment_duration
            self.alignment_percent = min(self.alignment_percent, 1)

            for i in range(12):
                idx = self.JOINT_REORDERING[i]
                self.low_cmd.motor_cmd[idx].q = float((1 - self.alignment_percent) * self.foldPos[i] + self.alignment_percent * self.q0[i])
                self.low_cmd.motor_cmd[idx].dq = 0
                self.low_cmd.motor_cmd[idx].kp = self.Kp_stand
                self.low_cmd.motor_cmd[idx].kd = self.Kd_stand
                self.low_cmd.motor_cmd[idx].tau = 0

        elif self.hold_percent < 1:
            # stage 3: hold q0, integrate out the static holding torque
            self.hold_percent += 1.0 / self.hold_duration
            self.hold_percent = min(self.hold_percent, 1)

            for i in range(12):
                idx = self.JOINT_REORDERING[i]
                err_i = float(self.q0[i]) - self.low_state.motor_state[idx].q
                self.tau_i[i] = np.clip(self.tau_i[i] + self.Ki * err_i * self.dt,
                                        -self.tau_i_max, self.tau_i_max)
                self.low_cmd.motor_cmd[idx].q = float(self.q0[i])
                self.low_cmd.motor_cmd[idx].dq = 0
                self.low_cmd.motor_cmd[idx].kp = self.Kp_stand
                self.low_cmd.motor_cmd[idx].kd = self.Kd_stand
                self.low_cmd.motor_cmd[idx].tau = float(self.tau_i[i])

            if self.n_tick % 40 == 0 or self.hold_percent >= 1:
                q = np.array([self.low_state.motor_state[self.JOINT_REORDERING[i]].q for i in range(12)])
                err = q - self.q0
                tag = "final alignment" if self.hold_percent >= 1 else "hold"
                print(f"{tag} err vs q0: " + np.array2string(err, precision=3, suppress_small=True)
                      + f"  max|err|: {np.max(np.abs(err)):.3f}", flush=True)

        elif (self.hold_percent >= 1) and (self.ii < self.traj_end):

            imu_quat = np.array(self.low_state.imu_state.quaternion)
            nq = np.linalg.norm(imu_quat)
            if nq > 1e-6:
                imu_quat = imu_quat / nq
                self.last_quat = imu_quat
            else:
                imu_quat = self.last_quat

            if self.record_odom:
                self.q_off = quat_mul(self.x_ref[0][3:7], quat_inv(imu_quat))
                self.record_odom = False

            inf_t0 = time.perf_counter()

            # current
            dof_pos = np.array([self.low_state.motor_state[i].q for i in range(12)])
            dof_vel = np.array([self.low_state.motor_state[i].dq for i in range(12)])
            dof_pos = dof_pos[self.JOINT_REORDERING]
            dof_vel = dof_vel[self.JOINT_REORDERING]

            base_quat = quat_mul(self.q_off, imu_quat)
            base_quat = base_quat / np.linalg.norm(base_quat)
            gyro_raw = np.array(self.low_state.imu_state.gyroscope)
            if self.gyro_f is None:
                self.gyro_f = gyro_raw
            self.gyro_f = self.gyro_f + self.gyro_alpha * (gyro_raw - self.gyro_f)
            ang_vel_body = self.gyro_f

            x = np.zeros(37)
            x[3:7] = base_quat
            x[7:19] = dof_pos
            x[22:25] = ang_vel_body
            x[25:37] = dof_vel

            eul = quat2eul(base_quat)
            xr = self.x_ref[self.ii]
            st = np.concatenate([eul, dof_pos, ang_vel_body, dof_vel])
            rf = self.ref_feat[self.ii]
            upd = self.u_ref[self.ii] + self.Kp * (xr[7:19] - dof_pos) + self.Kd * (xr[25:37] - dof_vel)
            obs = [st]
            if self.use_accelerometer:
                acceleration = np.asarray(self.low_state.imu_state.accelerometer)
                limit = self.accelerometer_clip_g * self.accelerometer_gravity
                obs.append(np.clip(np.nan_to_num(acceleration), -limit, limit))
            obs.extend([rf - st, upd, [self.ii / self.traj_length]])
            for off in self.preview_offsets:
                obs.append(self.ref_feat[min(self.ii + off, self.traj_length)])
            obs = np.clip(np.nan_to_num(np.concatenate(obs)), -1e4, 1e4).astype(np.float32)

            v = self.policy(obs)

            tau = np.clip(self.u_ref[self.ii] + v, -self.tau_limit, self.tau_limit)
            if not np.isfinite(tau).all():
                self.fault = True
                print("FAULT: non-finite command, entering damping mode", flush=True)
                return
            q_des = self.x_ref[self.ii][7:19]
            dq_des = self.x_ref[self.ii][25:37]

            # torque-saturation check: full firmware torque = PD + tau vs motor limit
            total = self.Kp * (q_des - dof_pos) + self.Kd * (dq_des - dof_vel) + tau
            ratio = np.abs(total) / self.tau_limit
            self.sat_peak = max(self.sat_peak, float(ratio.max()))
            self.sat_cnt += int((ratio >= 1.0).any())
            self.log_sat.append([self.ii, int((ratio >= 1.0).sum()), float(ratio.max())])

            # set joint commands
            for i in range(12):
                idx = self.JOINT_REORDERING[i]
                self.low_cmd.motor_cmd[idx].q = float(q_des[i])
                self.low_cmd.motor_cmd[idx].dq = float(dq_des[i])
                self.low_cmd.motor_cmd[idx].kp = self.Kp
                self.low_cmd.motor_cmd[idx].kd = self.Kd
                self.low_cmd.motor_cmd[idx].tau = float(tau[i])

            inf = time.perf_counter() - inf_t0
            self.inf_sum += inf
            self.inf_max = max(self.inf_max, inf)
            self.log_inf.append(inf)
            self.log_state.append(np.concatenate([[self.ii], dof_pos, dof_vel, base_quat, ang_vel_body, v]))

            self.ii += self.stride

        elif (self.hold_percent >= 1) and (self.ii >= self.traj_end) and (self.settle_percent < 1):

            self.settle_percent += 1.0 / self.settle_duration
            self.settle_percent = min(self.settle_percent, 1)

            for i in range(12):
                idx = self.JOINT_REORDERING[i]
                self.low_cmd.motor_cmd[idx].q = float(self.qf[i])
                self.low_cmd.motor_cmd[idx].dq = 0
                self.low_cmd.motor_cmd[idx].kp = self.Kp
                self.low_cmd.motor_cmd[idx].kd = self.Kd
                self.low_cmd.motor_cmd[idx].tau = 0

        if self.publish:
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.lowcmd_publisher.Write(self.low_cmd)



if __name__ == '__main__':

    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        custom = Custom()
        custom.publish = False
        custom.InitLowCmd()
        custom.low_state = unitree_go_msg_dds__LowState_()
        custom.fold_percent = 1
        custom.alignment_percent = 1
        custom.hold_percent = 1
        custom.Start()
        time.sleep(8)
        sys.exit(0)

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init()
    custom.Start()

    while True:
        if custom.settle_percent >= 1:
           time.sleep(1)
           np.savez("run_log.npz", tick=np.array(custom.log_tick),
                    align=np.array(custom.log_align), inf=np.array(custom.log_inf),
                    state=np.array(custom.log_state), sat=np.array(custom.log_sat))
           for nm, a in (("tick", custom.log_tick), ("align", custom.log_align), ("inf", custom.log_inf)):
               a = 1e3 * np.array(a)
               print(f"{nm}: n={len(a)} med {np.median(a):.2f} p90 {np.percentile(a, 90):.2f} "
                     f"p99 {np.percentile(a, 99):.2f} max {a.max():.2f} ms")
           S = np.array(custom.log_state)
           if len(S):
               ti = S[:, 0].astype(int)
               qerr = S[:, 1:13] - custom.x_ref[ti, 7:19]
               querr = S[:, 25:29] - custom.x_ref[ti, 3:7]
               print(f"tracking: joint err rms {np.sqrt((qerr ** 2).mean()):.4f} max {np.abs(qerr).max():.4f} rad, "
                     f"quat err max {np.abs(querr).max():.4f}")
           print("Done!")
           sys.exit(-1)
        time.sleep(1)
