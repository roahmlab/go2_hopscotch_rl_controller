import time
import sys
import os
import json

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

import jax
import numpy as np
from brax import math
from brax.io import model as brax_model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from jax import numpy as jp
from scipy.spatial.transform import Rotation

import pyvicon_datastream as pv

class Custom:
    def __init__(self):
        self.Kp = 50.0
        self.Kd = 0.5

        self.dt = 0.02
        self.stride = 20

        self.preview_offsets = jp.arange(1, 5) * 2

        self.init_history = True
        self.history = None
        self.last_action = jp.zeros(12)

        self.ii = 0

        self.low_cmd = unitree_go_msg_dds__LowCmd_()  
        self.low_state = None  

        # fb info
        self.fb_pos = None
        self.fb_prev = None
        self.fb_quat = None
        self.mocap_dt = 0.01

        self.startPos = [0.0] * 12
        self.alignment_duration = 250
        self.alignment_percent = 0

        self.settle_duration = 250
        self.settle_percent = 0

        base_dir = os.path.join(os.path.dirname(__file__), ".")
        rc = json.load(open(os.path.join(base_dir, "hopscotch_utils", "run_config.json")))

        # thread handling
        self.lowCmdWriteThreadPtr = None
        self.mocapThreadPtr = None

        self.crc = CRC()

        # Load reference trajectory
        data_fp = os.path.join(base_dir, "hopscotch_utils", "traj_hopscotch_friction_6cm_lsq.json")
        with open(data_fp, 'r') as file:
            data = json.load(file)

        self.onehots = jp.zeros((0, 3))
        self.contacts = jp.zeros((0, 4))
        self.q_ref = jp.zeros((0, 18))
        self.v_ref = jp.zeros((0, 18))
        self.a_ref = jp.zeros((0, 18))
        self.u_ref = jp.zeros((0, 12))

        jj = 0
        for mode in data:
            mode_T = int(mode['T'] / mode['dt'])

            if mode['name'] == "4_stance":
                oh = jp.array([1, 0, 0])
            elif mode['name'] == "flying":
                oh = jp.array([0, 1, 0])
            elif mode['name'] == "diag_stance":
                oh = jp.array([0, 0, 1])
            else:
                raise Exception("Unrecognized Phase: " + mode['name'])
            
            contacts = jp.array(["FL_foot" in mode['contacts'], "FR_foot" in mode['contacts'], "RL_foot" in mode['contacts'], "RR_foot" in mode['contacts']])

            while jj < mode_T:
                self.onehots = jp.vstack((self.onehots, oh))
                self.contacts = jp.vstack((self.contacts, contacts))
                self.q_ref = jp.vstack((self.q_ref, jp.array(mode['q'][jj])))
                self.v_ref = jp.vstack((self.v_ref, jp.array(mode['v'][jj])))
                self.a_ref = jp.vstack((self.a_ref, jp.array(mode['a'][jj])))
                self.u_ref = jp.vstack((self.u_ref, jp.array(mode['u'][jj])))

                jj += self.stride
            
            jj = jj % mode_T

        self.q_ref = self.q_ref.at[:, 2].set(self.q_ref[:, 2] + rc.get('config')['reftrack']['ref_z_offset'])
        self.u_ref = self.u_ref + rc.get('config')['reftrack']['ff_damping_comp'] * self.v_ref[:, 6:18] + rc.get('config')['reftrack']['ff_armature_comp'] * self.a_ref[:, 6:18]

        # Load RL policy (from Cesar RL repo)
        params_path = os.path.join(base_dir, "hopscotch_utils", "params_final.pkl")

        nf_kwargs = dict(
            policy_hidden_layer_sizes=tuple(rc["policy_hidden"]),
            value_hidden_layer_sizes=tuple(rc["value_hidden"]))
        if rc.get("asymmetric_obs"):
            nf_kwargs.update(policy_obs_key="state",
                            value_obs_key="privileged_state")
        normalize = (running_statistics.normalize
                    if rc.get("normalize_observations", True) else (lambda x, y: x))
        net = ppo_networks.make_ppo_networks(
            rc.get("obs_size"), rc.get("action_size"),
            preprocess_observations_fn=normalize, **nf_kwargs)
        params = brax_model.load_params(params_path)
        self.policy = ppo_networks.make_inference_fn(net)(params, deterministic=True)

        # Get q0 and qf
        self.q0 = jp.array(data[0]['q'][0][6:])
        self.qf = jp.array(data[-1]['q'][-1][6:])

        self.traj_length = self.q_ref.shape[0]

        # Record initial odometry
        self.firstRun = True
        self.record_odom = True
        self.init_xyz = None
        self.init_quat_inv = None

        self.tau_limit = jp.array(np.tile(np.asarray(rc.get("config")['env']['torque_limit']), 4))
        self.tau_ff_clip = self.tau_limit * rc.get("config")['env']['ff_clip_frac']

        # run config parameters
        self.action_scale = rc.get("config")['env']['action_scale']

        # MuJoCo: [FL, FR, RL, RR]
        # Unitree Go2: [FR, FL, RR, RL]
        self.JOINT_REORDERING = jp.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])


    # Public methods
    def Init(self):
        self.InitLowCmd()

        # create publisher #
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        # create subscriber # 
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)

        # start vicon
        self.TARGET = "go2"
        self.vicon = pv.PyViconDatastream()
        self.vicon.set_stream_mode(pv.StreamMode.ClientPullPreFetch)
        self.vicon.connect("192.168.0.149")
        self.vicon.enable_segment_data()

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
        self.mocapThreadPtr = RecurrentThread(
            interval=self.mocap_dt, target=self.UpdateMocap, name="updatemocap"
        )
        self.mocapThreadPtr.Start()
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
        # print("FR_0 motor state: ", msg.motor_state[go2.LegID["FR_0"]])
        # print("IMU state: ", msg.imu_state)
        # print("Battery state: voltage: ", msg.power_v, "current: ", msg.power_a)

    def LowCmdWrite(self):

        if self.firstRun:
            self.startPos = jp.array([self.low_state.motor_state[i].q for i in range(12)])
            self.startPos = self.startPos.at[self.JOINT_REORDERING].get()
            self.firstRun = False

        if self.low_state is None or self.fb_pos is None or self.fb_prev is None:
            return

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

        if (self.alignment_percent >= 1) and (self.ii < self.traj_length):

            if self.record_odom:
                self.init_xyz = jp.array(self.fb_pos)
                self.init_xyz = self.init_xyz.at[2].set(0)
                self.init_quat_inv = math.quat_inv(self.fb_quat)
                self.record_odom = False

            # current
            dof_pos = jp.array([self.low_state.motor_state[i].q for i in range(12)])
            dof_vel = jp.array([self.low_state.motor_state[i].dq for i in range(12)])
            dof_pos = dof_pos.at[self.JOINT_REORDERING].get()
            dof_vel = dof_vel.at[self.JOINT_REORDERING].get()
        
            world_quat = jp.array(self.fb_quat)
            base_pos = math.rotate(self.fb_pos - self.init_xyz, self.init_quat_inv)
            base_quat = math.quat_mul(world_quat, self.init_quat_inv)
            world_to_body = math.quat_inv(world_quat)
            base_vel = (self.fb_pos - self.fb_prev) / self.mocap_dt
            lin_vel_body = math.rotate(base_vel, world_to_body)
            ang_vel_body = jp.array(self.low_state.imu_state.gyroscope)
            proj_gravity = math.rotate(jp.array([0.0, 0.0, -1.0]), world_to_body)

            q_ref = self.q_ref[self.ii]
            v_ref = self.v_ref[self.ii]
            a_ref = self.a_ref[self.ii]
            u_ref = self.u_ref[self.ii]
            onehot = self.onehots[self.ii]

            base_quat_ref = Rotation.from_euler("XYZ", q_ref[3:6]).as_quat(scalar_first=True)
            q_rel = math.quat_mul(math.quat_inv(base_quat), base_quat_ref)
            ori_err = 2.0 * jp.sign(q_rel[0] + 1e-8) * q_rel[1:4]

            cur = [proj_gravity, ang_vel_body, lin_vel_body, dof_pos - q_ref[6:18], dof_vel - v_ref[6:18], base_pos - q_ref[0:3], ori_err, onehot, self.last_action]
            current = jp.concatenate(cur)


            # future
            def prev(off):
                j = (self.ii + off) % self.traj_length
                return jp.concatenate([self.q_ref[j, 6:18] - dof_pos, self.v_ref[j, 6:18],
                                    self.contacts[j].astype(jp.float32), self.onehots[j]])
            preview = jax.vmap(prev)(self.preview_offsets).reshape(-1)


            # ff
            ff = [a_ref[6:18], u_ref / self.tau_limit]
            ff = jp.concatenate(ff)


            # past
            if self.init_history:
                tau_applied = self.Kp * (self.q0 - dof_pos) - self.Kd * dof_vel
                self.history = np.tile(np.concatenate((dof_pos, dof_vel, base_pos, ang_vel_body, tau_applied)), (16, 1))
                self.init_history = False

            hist = self.history[::2, :].reshape(-1)


            # policy inference
            obs = jp.concatenate([current, preview, ff, hist])
            obs = jp.clip(jp.nan_to_num(obs), -100.0, 100.0)

            action, _ = self.policy(obs, jax.random.PRNGKey(0))
            action = np.clip(action, -1.0, 1.0)

            q_des = q_ref[6:18] + self.action_scale * action

            tau_ff = np.clip(u_ref, -self.tau_ff_clip, self.tau_ff_clip)

            # set joint commands
            for i in range(12):
                idx = self.JOINT_REORDERING[i]
                self.low_cmd.motor_cmd[idx].q = float(q_des[i])
                self.low_cmd.motor_cmd[idx].dq = 0
                self.low_cmd.motor_cmd[idx].kp = self.Kp
                self.low_cmd.motor_cmd[idx].kd = self.Kd
                self.low_cmd.motor_cmd[idx].tau = float(tau_ff[i])


            # update history
            tau_applied = tau_ff + self.Kp * (q_des - dof_pos) - self.Kd * dof_vel
            self.history[:-1, :] = self.history[1:, :]
            self.history[-1, :] = np.concatenate((dof_pos, dof_vel, base_pos, ang_vel_body, tau_applied))

            self.last_action = action.copy()

            self.ii += 1

        if (self.alignment_percent >= 1) and (self.ii == self.traj_length) and (self.settle_percent < 1):

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

    def UpdateMocap(self):
        if self.vicon.get_frame() == pv.Result.Success:
            seg = self.vicon.get_subject_root_segment_name(self.TARGET)
            pos = 0.001 * self.vicon.get_segment_global_translation(self.TARGET, seg)
            rot = np.roll(self.vicon.get_segment_global_quaternion(self.TARGET, seg), 1)

            self.fb_prev = self.fb_pos
            self.fb_pos = jp.array(pos)
            self.fb_quat = jp.array(rot)



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
        if custom.settle_percent >= 1:
           time.sleep(1)
           print("Done!")
           sys.exit(-1)     
        time.sleep(1)