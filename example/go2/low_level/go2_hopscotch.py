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


class Custom:
    def __init__(self):
        self.Kp = 40.0
        self.Kd = 10.0

        self.dt = 0.005
        self.stride = 5
        self.traj_end = 1000

        self.preview_offsets = (25, 50, 100)
        self.obs_idx = np.r_[3:19, 22:37]

        self.ii = 0

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None

        self.startPos = [0.0] * 12
        self.alignment_duration = 400
        self.alignment_percent = 0

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

        # Load reference trajectory
        f = np.load(os.path.join(base_dir, "hopscotch_utils", "trajectories.npz"))
        x_ref, u_ref = f["x_refs"], f["u_refs"]
        if x_ref.ndim == 3:
            x_ref, u_ref = x_ref[0], u_ref[0]
        self.x_ref = np.asarray(x_ref, dtype=np.float64)
        self.u_ref = np.asarray(u_ref, dtype=np.float64)
        self.traj_length = self.u_ref.shape[0]

        # Load blind GRU actor (from residual-controller shac2)
        with open(os.path.join(base_dir, "hopscotch_utils", "actor_blind.pkl"), 'rb') as file:
            ck = pickle.load(file)
        self.actor = ck["actor"]
        self.h_offs = np.cumsum([0] + [uz.shape[0] for _, uz, *_ in self.actor[0]])
        self.h = np.zeros(self.h_offs[-1])

        # warm up inference
        o = np.zeros(self.actor[0][0][0].shape[0])
        for wz, uz, *_ in self.actor[0]:
            o = 1.0 / (1.0 + np.exp(-(o @ wz + self.h[:uz.shape[0]] @ uz)))
        o = o @ self.actor[1][0]

        # Get q0 and qf
        self.q0 = self.x_ref[0][7:19]
        self.qf = self.x_ref[self.traj_end][7:19]

        # Record initial orientation offset
        self.firstRun = True
        self.record_odom = True
        self.q_off = None

        self.tau_limit = np.array([23.7, 23.7, 45.43] * 4)

        # MuJoCo: [FL, FR, RL, RR]
        # Unitree Go2: [FR, FL, RR, RL]
        self.JOINT_REORDERING = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])


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
            if self.alignment_percent < 1:
                self.log_align.append(dtick)
            self.n_tick += 1
            if self.n_tick % 200 == 0:
                print(f"tick avg {1e3 * self.tick_sum / 200:.2f} max {1e3 * self.tick_max:.2f} ms | "
                      f"inference avg {1e3 * self.inf_sum / 200:.2f} max {1e3 * self.inf_max:.2f} ms", flush=True)
                self.tick_sum = self.tick_max = self.inf_sum = self.inf_max = 0.0
        self.tick_prev = now

        if self.low_state is None:
            return

        if self.firstRun:
            self.startPos = [self.low_state.motor_state[i].q for i in self.JOINT_REORDERING]
            self.firstRun = False

        if self.alignment_percent < 1:

            self.alignment_percent += 1.0 / self.alignment_duration
            self.alignment_percent = min(self.alignment_percent, 1)

            for i in range(12):
                idx = self.JOINT_REORDERING[i]
                self.low_cmd.motor_cmd[idx].q = float((1 - self.alignment_percent) * self.startPos[i] + self.alignment_percent * self.q0[i])
                self.low_cmd.motor_cmd[idx].dq = 0
                self.low_cmd.motor_cmd[idx].kp = self.Kp
                self.low_cmd.motor_cmd[idx].kd = self.Kd
                self.low_cmd.motor_cmd[idx].tau = 0

        elif (self.alignment_percent >= 1) and (self.ii < self.traj_end):

            imu_quat = np.array(self.low_state.imu_state.quaternion)

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
            ang_vel_body = np.array(self.low_state.imu_state.gyroscope)

            x = np.zeros(37)
            x[3:7] = base_quat
            x[7:19] = dof_pos
            x[22:25] = ang_vel_body
            x[25:37] = dof_vel

            obs = [x[self.obs_idx], (self.x_ref[self.ii] - x)[self.obs_idx],
                   self.u_ref[self.ii], [self.ii / self.traj_length]]
            for off in self.preview_offsets:
                obs.append(self.x_ref[min(self.ii + off, self.traj_length)][7:19])
            obs = np.concatenate(obs)

            # policy inference
            o = obs
            hs = []
            for j, (wz, uz, bz, wr, ur, br, wh, uh, bh) in enumerate(self.actor[0]):
                hl = self.h[self.h_offs[j]:self.h_offs[j + 1]]
                z = 1.0 / (1.0 + np.exp(-(o @ wz + hl @ uz + bz)))
                r = 1.0 / (1.0 + np.exp(-(o @ wr + hl @ ur + br)))
                n = np.tanh(o @ wh + (r * hl) @ uh + bh)
                o = (1.0 - z) * n + z * hl
                hs.append(o)
            wo, bo = self.actor[1]
            self.h = np.concatenate(hs)
            v = o @ wo + bo

            tau = np.clip(self.u_ref[self.ii] + v, -self.tau_limit, self.tau_limit)
            q_des = self.x_ref[self.ii][7:19]
            dq_des = self.x_ref[self.ii][25:37]

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

        elif (self.alignment_percent >= 1) and (self.ii >= self.traj_end) and (self.settle_percent < 1):

            self.settle_percent += 1.0 / self.settle_duration
            self.settle_percent = min(self.settle_percent, 1)

            for i in range(12):
                idx = self.JOINT_REORDERING[i]
                self.low_cmd.motor_cmd[idx].q = float(self.qf[i])
                self.low_cmd.motor_cmd[idx].dq = 0
                self.low_cmd.motor_cmd[idx].kp = self.Kp
                self.low_cmd.motor_cmd[idx].kd = self.Kd
                self.low_cmd.motor_cmd[idx].tau = 0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)



if __name__ == '__main__':

    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        ChannelFactoryInitialize(0)
        custom = Custom()
        custom.InitLowCmd()
        custom.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        custom.lowcmd_publisher.Init()
        custom.low_state = unitree_go_msg_dds__LowState_()
        custom.alignment_percent = 1
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
                    state=np.array(custom.log_state))
           for nm, a in (("tick", custom.log_tick), ("align", custom.log_align), ("inf", custom.log_inf)):
               a = 1e3 * np.array(a)
               print(f"{nm}: n={len(a)} avg {a.mean():.2f} p99 {np.percentile(a, 99):.2f} max {a.max():.2f} ms")
           print("Done!")
           sys.exit(-1)
        time.sleep(1)
