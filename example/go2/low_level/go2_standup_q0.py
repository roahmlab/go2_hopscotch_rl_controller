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

class Custom:
    def __init__(self):
        self.Kp = 60.0
        self.Kd = 5.0
        self.motiontime = 0
        self.dt = 0.002

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None

        # Unitree motor order: [FR, FL, RR, RL]
        # fold pose (legs tucked under body), from go2_stand_example.py
        self._targetPos_1 = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
                             -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]
        # lie-down pose, from go2_stand_example.py
        self._targetPos_3 = [-0.35, 1.36, -2.65, 0.35, 1.36, -2.65,
                             -0.5, 1.36, -2.65, 0.5, 1.36, -2.65]

        # MuJoCo: [FL, FR, RL, RR]
        # Unitree Go2: [FR, FL, RR, RL]
        self.JOINT_REORDERING = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

        # stand target = first frame of the hopscotch reference trajectory
        base_dir = os.path.join(os.path.dirname(__file__), ".")
        data_fp = os.path.join(base_dir, "hopscotch_utils", "traj_hopscotch_friction_6cm_lsq.json")
        with open(data_fp, 'r') as file:
            data = json.load(file)

        self.q0_mj = np.array(data[0]['q'][0][6:])  # MuJoCo order
        self.q0_motor = [0.0] * 12                  # Unitree motor order
        for i in range(12):
            self.q0_motor[self.JOINT_REORDERING[i]] = float(self.q0_mj[i])

        # integral action during the hold: converges to the static holding
        # torque (gravity + stance geometry) with time constant Kp/Ki ~ 0.6 s
        self.Ki = 100.0
        self.tau_i_max = 15.0
        self.tau_i = np.zeros(12)  # motor order

        self.startPos = [0.0] * 12
        self.duration_1 = 500   # lie -> fold        (1.0 s)
        self.duration_2 = 500   # fold -> q0         (1.0 s)
        self.duration_3 = 1500  # hold q0            (3.0 s)
        self.duration_4 = 900   # q0 -> lie back down (1.8 s)
        self.percent_1 = 0
        self.percent_2 = 0
        self.percent_3 = 0
        self.percent_4 = 0

        self.firstRun = True

        # thread handling
        self.lowCmdWriteThreadPtr = None

        self.crc = CRC()

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

    def LogError(self):
        # per-joint error vs q0, MuJoCo order [FL, FR, RL, RR]
        if self.motiontime % 100 == 0:  # every 0.2 s
            q = np.array([self.low_state.motor_state[self.JOINT_REORDERING[i]].q for i in range(12)])
            err = q - self.q0_mj
            tau_i_mj = np.array([self.tau_i[self.JOINT_REORDERING[i]] for i in range(12)])
            logging.info("err vs q0 (MJ order): " + np.array2string(err, precision=3, suppress_small=True)
                         + "  max|err|: %.3f" % np.max(np.abs(err)))
            logging.info("tau_i    (MJ order): " + np.array2string(tau_i_mj, precision=2, suppress_small=True))

    def LowCmdWrite(self):

        if self.low_state is None:
            return

        if self.firstRun:
            for i in range(12):
                self.startPos[i] = self.low_state.motor_state[i].q
            self.firstRun = False

        self.motiontime += 1

        if self.percent_1 < 1:
            # stage 1: lie -> fold (feet tucked under hips, unloaded)
            self.percent_1 += 1.0 / self.duration_1
            self.percent_1 = min(self.percent_1, 1)
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = (1 - self.percent_1) * self.startPos[i] + self.percent_1 * self._targetPos_1[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0

        elif self.percent_2 < 1:
            # stage 2: fold -> q0 (vertical push-up)
            self.percent_2 += 1.0 / self.duration_2
            self.percent_2 = min(self.percent_2, 1)
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = (1 - self.percent_2) * self._targetPos_1[i] + self.percent_2 * self.q0_motor[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0
            self.LogError()

        elif self.percent_3 < 1:
            # stage 3: hold q0, integrate out the static holding torque
            self.percent_3 += 1.0 / self.duration_3
            self.percent_3 = min(self.percent_3, 1)
            for i in range(12):
                err_i = self.q0_motor[i] - self.low_state.motor_state[i].q
                self.tau_i[i] = np.clip(self.tau_i[i] + self.Ki * err_i * self.dt,
                                        -self.tau_i_max, self.tau_i_max)
                self.low_cmd.motor_cmd[i].q = self.q0_motor[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = float(self.tau_i[i])
            self.LogError()
            if self.percent_3 >= 1:
                tau_i_mj = np.array([self.tau_i[self.JOINT_REORDERING[i]] for i in range(12)])
                logging.info("CALIBRATED holding torque (MJ order): "
                             + np.array2string(tau_i_mj, precision=2, suppress_small=True))

        elif self.percent_4 <= 1:
            # stage 4: q0 -> lie back down, fading the integral torque out with the ramp
            self.percent_4 += 1.0 / self.duration_4
            self.percent_4 = min(self.percent_4, 1)
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = (1 - self.percent_4) * self.q0_motor[i] + self.percent_4 * self._targetPos_3[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = float((1 - self.percent_4) * self.tau_i[i])

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)


if __name__ == '__main__':

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
        if custom.percent_4 == 1.0:
            time.sleep(1)
            print("Done!")
            sys.exit(-1)
        time.sleep(1)
