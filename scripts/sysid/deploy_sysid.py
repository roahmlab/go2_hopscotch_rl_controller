import sys
import numpy as np
import os
import time 

from unitree_sdk2py.core.channel import ChannelPublisher,ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

# --- Sysid config --- #

LEG_WIRE_INDEX = {"FR": 0, "FL": 3, "RR": 6, "RL": 9}
ACTIVE_LEG = "FR"
ACTIVE_IDX = LEG_WIRE_INDEX[ACTIVE_LEG]

KP_ACTIVE = 60.0   # sim used 60 — revisit if tracking is too soft
KD_ACTIVE = 3.0    # sim used 3  — revisit if tracking is too soft

CONTROL_DT = 0.002  # 500 Hz low-level control loop

POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0

def load_exciting_trajectory(csv_path):
   """Load FR-only exciting trajectory: columns [t, q(3), qd(3), qdd(3), tau_ff(3)]."""
   data = np.loadtxt(csv_path)
   t = data[:, 0]
   q_ref = data[:, 1:4]
   qd_ref = data[:, 4:7]
   qdd_ref = data[:, 7:10]   # not used for FF+PD control, kept for reference/validation
   tau_ff = data[:, 10:13]
   return t, q_ref, qd_ref, qdd_ref, tau_ff


class Go2SysidRobot:
    def __init__(self):
        
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None
        self.crc = CRC()

        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q = POS_STOP_F
            self.low_cmd.motor_cmd[i].dq = VEL_STOP_F
            self.low_cmd.motor_cmd[i].kp = 0.0
            self.low_cmd.motor_cmd[i].kd = 0.0
            self.low_cmd.motor_cmd[i].tau = 0.0 

    def init(self):
       self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
       self.lowcmd_publisher.Init()

       self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
       self.lowstate_subscriber.Init(self._low_state_handler, 10) 

       self.sc = SportClient()
       self.sc.SetTimeout(5.0)
       self.sc.Init()

       self.msc = MotionSwitcherClient()
       self.msc.SetTimeout(5.0)
       self.msc.Init()

        # Sport mode must be released before direct low-level motor_cmd is honored.
       status, result = self.msc.CheckMode()
       while result['name']:
           self.sc.StandDown()
           self.msc.ReleaseMode()
           status, result = self.msc.CheckMode()
           time.sleep(1)

       while self.low_state is None:
           time.sleep(0.01)

    def _low_state_handler(self, msg: LowState_):
       self.low_state = msg

    def publish_active_leg_command(self, q3, qd3, tau3):
       """Command the active leg's 3 joints; all other joints stay uncommanded (free)."""
       for j in range(3):
           m = self.low_cmd.motor_cmd[ACTIVE_IDX + j]
           m.q = q3[j]
           m.dq = qd3[j]
           m.tau = tau3[j]
           m.kp = KP_ACTIVE
           m.kd = KD_ACTIVE

       self.low_cmd.crc = self.crc.Crc(self.low_cmd)
       self.lowcmd_publisher.Write(self.low_cmd)

    def get_dof_pos(self):
       return np.array([self.low_state.motor_state[ACTIVE_IDX + j].q for j in range(3)])

    def get_dof_vel(self):
       return np.array([self.low_state.motor_state[ACTIVE_IDX + j].dq for j in range(3)])

    def get_tau_est(self):
       return np.array([self.low_state.motor_state[ACTIVE_IDX + j].tau_est for j in range(3)])

def run_sysid(csv_path="../../data/go2/FR/exciting-trajectory-1.csv", iface=None):
    
    if iface is not None:
       ChannelFactoryInitialize(0, iface)
    else:
       ChannelFactoryInitialize(0)

    robot = Go2SysidRobot() ### start the robot with low level commands
    robot.init()


    

    # load trajectory_data
    t_csv, q_ref_csv, qd_ref_csv, qdd_ref_csv, tau_ff_csv = load_exciting_trajectory(csv_path)
    print(f"Loaded {len(t_csv)} samples, {t_csv[-1]:.2f}s trajectory for leg {ACTIVE_LEG}")

    input("Press Enter to start (Ctrl+C to abort)...")

    log = {'t': [], 'q_actual': [], 'qd_actual': [], 'tau_est': [],
          'q_ref': [], 'qd_ref': [], 'tau_ff': [], 'tau_fb': []}
    
    t_start = time.time()
    

    try:
        while True:
            tick_start = time.time()
            t_elapsed = tick_start - t_start
            if t_elapsed > t_csv[-1]:
                break

            q3 = np.array([np.interp(t_elapsed, t_csv, q_ref_csv[:, j]) for j in range(3)])
            qd3 = np.array([np.interp(t_elapsed, t_csv, qd_ref_csv[:, j]) for j in range(3)])
            tau3 = np.array([np.interp(t_elapsed, t_csv, tau_ff_csv[:, j]) for j in range(3)])

            robot.publish_active_leg_command(q3, qd3, tau3) ## PD or PD+FF?

            q_actual = robot.get_dof_pos() ## from the encoder
            qd_actual = robot.get_dof_vel() ## from the encoder
            tau_estimated=robot.get_tau_est() ## estimated, not directly from a sensor iirc
        
            tau_fb = KP_ACTIVE * (q3 - q_actual) + KD_ACTIVE * (qd3 - qd_actual) ### calculated for logging estimates, go2 directly does this in low level

            log['t'].append(t_elapsed)
            log['q_actual'].append(q_actual)
            log['qd_actual'].append(qd_actual)
            log['tau_est'].append(tau_estimated)
            log['q_ref'].append(q3)
            log['qd_ref'].append(qd3)
            log['tau_ff'].append(tau3)
            log['tau_fb'].append(tau_fb)

            time.sleep(max(CONTROL_DT - (time.time() - tick_start), 0))

    finally:
        out_dir = f"../../data/go2/{ACTIVE_LEG}"
        os.makedirs(out_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        t_col = np.array(log['t']).reshape(-1, 1)
        
        # Primary file: exact [t, q, qd, tau] format the inertial ID pipeline expects
        traj_path = f"{out_dir}/traj_data_{timestamp}.csv"
        main_cols = np.hstack([np.array(log[k]) for k in ('q_actual', 'qd_actual', 'tau_est')])
        np.savetxt(traj_path, np.hstack([t_col, main_cols]))

        # Secondary file: ref/feedforward/feedback diagnostics, not consumed by the pipeline
        debug_path = f"{out_dir}/traj_data_{timestamp}_debug.csv"
        debug_cols = np.hstack([np.array(log[k]) for k in ('q_ref', 'qd_ref', 'tau_ff', 'tau_fb')])
        np.savetxt(debug_path, np.hstack([t_col, debug_cols]))

        print(f"Saved {len(log['t'])} samples to {traj_path} (+ debug: {debug_path})")

if __name__ == '__main__':

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    iface = sys.argv[1] if len(sys.argv) > 1 else None
    run_sysid(iface=iface)