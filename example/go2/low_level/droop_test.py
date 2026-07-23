"""Single-joint torque-speed droop test: fires constant-torque bursts on one unloaded joint (robot ON ITS BACK), logs commanded vs estimated torque across speed, fits the ceiling curve tau_max(w) = a*(1 - s*w)."""
import sys
import time
import threading

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- config ----------------
LEG = "FL"                                 # FL / FR / RL / RR
JOINT = 1                                  # 0 abd, 1 thigh, 2 calf (thigh = widest travel)
TAU_LEVELS = (4.0, 8.0, 12.0, 16.0, 20.0)  # Nm, each fired in both directions
TRAVEL = (-0.9, 2.7)                       # rad, test travel (front thigh range -1.57..3.49)
DQ_CAP = 24.0                              # rad/s, abort burst above this
BURST_TIMEOUT = 0.8                        # s
DT = 0.002
HOLD_KP, HOLD_KD = 40.0, 2.0
CATCH_KD = 2.0
START_POSE = np.array([0.0, 1.36, -2.65] * 4)   # MuJoCo order, calves tucked
TAU_LIMIT = 23.7                           # abd/thigh spec stall
OUT = "droop_test"
# -----------------------------------------

JOINT_REORDERING = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]   # MuJoCo -> firmware
LEG_IDX = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}
MJ = LEG_IDX[LEG] * 3 + JOINT
FW = JOINT_REORDERING[MJ]


def build_trials():
    trials = []
    for tau in TAU_LEVELS:
        trials.append((+tau, TRAVEL[0], TRAVEL[1]))
        trials.append((-tau, TRAVEL[1], TRAVEL[0]))
    return trials


def fit_and_plot(npz_path):
    f = np.load(npz_path)
    dq, tau_est, tau_cmd, volt = f["dq"], f["tau_est"], f["tau_cmd"], f["volt"]
    sgn = np.sign(tau_cmd)
    v = dq * sgn                                        # speed in the driving direction
    te = tau_est * sgn
    drive = v > 2.0
    sag = drive & (te < 0.9 * np.abs(tau_cmd))          # points on the ceiling
    print(f"{len(dq)} samples, {int(drive.sum())} driving, {int(sag.sum())} on ceiling, "
          f"battery {volt.min():.1f}-{volt.max():.1f} V")
    if sag.sum() >= 20:
        A = np.stack([np.ones(int(sag.sum())), -v[sag]], axis=1)
        (a, b), *_ = np.linalg.lstsq(A, te[sag], rcond=None)
        s = b / a
        print(f"ceiling fit: tau_max(w) = {a:.1f} * (1 - {s:.4f}*w)   (stall {a:.1f} Nm vs spec {TAU_LIMIT})")
        print(f"sim TORQUE_SPEED_SLOPE equivalent ~= {s:.3f}")
    else:
        a = s = None
        print("no significant droop detected in this speed range "
              f"(max driving speed {v[drive].max() if drive.any() else 0:.1f} rad/s)")
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(v[drive], te[drive], c=np.abs(tau_cmd[drive]), s=6, cmap="viridis")
    plt.colorbar(sc, label="|tau_cmd| [Nm]")
    if a is not None:
        vv = np.linspace(0, max(v[drive].max(), 1), 50)
        ax.plot(vv, a * (1 - s * vv), "r-", lw=2, label=f"ceiling {a:.1f}*(1-{s:.3f}w)")
        ax.legend()
    ax.set_xlabel("joint speed in driving direction [rad/s]")
    ax.set_ylabel("tau_est in driving direction [Nm]")
    ax.set_title(f"{LEG} joint {JOINT} torque-speed  (color = commanded level)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT + ".png", dpi=150)
    print(f"saved {OUT}.png")


class DroopTest:
    def __init__(self):
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None
        self.crc = CRC()
        self.trials = build_trials()
        self.trial = -1
        self.phase = "align"
        self.timer = 0
        self.align_from = None
        self.pre_from = None
        self.log = []
        self.done = threading.Event()
        self.estop = threading.Event()

    def Init(self):
        for i in range(20):
            self.low_cmd.head[0] = 0xFE
            self.low_cmd.head[1] = 0xEF
            self.low_cmd.level_flag = 0xFF
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].q = go2.PosStopF
            self.low_cmd.motor_cmd[i].dq = go2.VelStopF
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self.OnLowState, 10)
        sc = SportClient(); sc.SetTimeout(5.0); sc.Init()
        msc = MotionSwitcherClient(); msc.SetTimeout(5.0); msc.Init()
        status, result = msc.CheckMode()
        while result["name"]:
            sc.StandDown()
            msc.ReleaseMode()
            status, result = msc.CheckMode()
            time.sleep(1)

    def OnLowState(self, msg):
        self.low_state = msg
        if self.phase == "burst" and 0 <= self.trial < len(self.trials):
            m = msg.motor_state[FW]
            self.log.append((time.time(), m.q, m.dq, m.tau_est,
                             self.trials[self.trial][0], self.trial, msg.power_v))

    def set_joint(self, mj, q, dq, kp, kd, tau):
        c = self.low_cmd.motor_cmd[JOINT_REORDERING[mj]]
        c.q, c.dq, c.kp, c.kd, c.tau = float(q), float(dq), float(kp), float(kd), float(tau)

    def Step(self):
        if self.low_state is None:
            return
        if self.estop.is_set():
            for i in range(12):
                self.set_joint(i, 0, 0, 0, 5.0, 0)
            self.publish()
            return
        qs = np.array([self.low_state.motor_state[JOINT_REORDERING[i]].q for i in range(12)])
        dq_test = self.low_state.motor_state[FW].dq
        q_test = self.low_state.motor_state[FW].q
        self.timer += 1
        t = self.timer * DT

        if self.phase == "align":
            if self.align_from is None:
                self.align_from = qs.copy()
            r = min(t / 3.0, 1.0)
            tgt = (1 - r) * self.align_from + r * START_POSE
            for i in range(12):
                self.set_joint(i, tgt[i], 0, HOLD_KP, HOLD_KD, 0)
            if r >= 1.0:
                self.next_trial()
        elif self.phase == "pre":
            tau, q0, q1 = self.trials[self.trial]
            if self.pre_from is None:
                self.pre_from = q_test
            r = min(t / 2.0, 1.0)
            self.hold_others()
            self.set_joint(MJ, (1 - r) * self.pre_from + r * q0, 0, HOLD_KP, HOLD_KD, 0)
            if r >= 1.0:
                self.phase, self.timer = "settle", 0
        elif self.phase == "settle":
            tau, q0, q1 = self.trials[self.trial]
            self.hold_others()
            self.set_joint(MJ, q0, 0, HOLD_KP, HOLD_KD, 0)
            if t >= 0.5:
                self.phase, self.timer = "burst", 0
                print(f"trial {self.trial + 1}/{len(self.trials)}: tau {tau:+.0f} Nm", flush=True)
        elif self.phase == "burst":
            tau, q0, q1 = self.trials[self.trial]
            self.hold_others()
            self.set_joint(MJ, 0, 0, 0, 0, tau)
            past = q_test > q1 if tau > 0 else q_test < q1
            if past or abs(dq_test) > DQ_CAP or t > BURST_TIMEOUT:
                self.phase, self.timer = "catch", 0
        elif self.phase == "catch":
            self.hold_others()
            self.set_joint(MJ, 0, 0, 0, CATCH_KD, 0)
            if t > 0.3 and abs(dq_test) < 0.5:
                self.next_trial()
        self.publish()

    def hold_others(self):
        for i in range(12):
            if i != MJ:
                self.set_joint(i, START_POSE[i], 0, HOLD_KP, HOLD_KD, 0)

    def next_trial(self):
        self.trial += 1
        self.pre_from = None
        self.timer = 0
        if self.trial >= len(self.trials):
            self.phase = "damp"
            for i in range(12):
                self.set_joint(i, 0, 0, 0, 2.0, 0)
            self.done.set()
        else:
            self.phase = "pre"

    def publish(self):
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)


def main():
    print("WARNING: robot MUST be on its back (or fully suspended) with legs free to swing.")
    print(f"Test joint: {LEG} joint {JOINT} (firmware idx {FW}), torque levels {TAU_LEVELS} Nm both directions.")
    input("Press Enter to start...")
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)
    dt_obj = DroopTest()
    dt_obj.Init()
    thread = RecurrentThread(interval=DT, target=dt_obj.Step, name="droop")
    threading.Thread(target=lambda: (input(), dt_obj.estop.set()), daemon=True).start()
    thread.Start()
    while not dt_obj.done.wait(0.5):
        if dt_obj.estop.is_set():
            print("E-STOP: damping mode.")
            time.sleep(2)
            return
    time.sleep(1)
    log = np.array(dt_obj.log)
    np.savez(OUT + ".npz", t=log[:, 0], q=log[:, 1], dq=log[:, 2], tau_est=log[:, 3],
             tau_cmd=log[:, 4], trial=log[:, 5], volt=log[:, 6])
    print(f"saved {OUT}.npz ({len(log)} samples)")
    fit_and_plot(OUT + ".npz")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "fit":
        OUT = sys.argv[2].rsplit(".", 1)[0]
        fit_and_plot(sys.argv[2])
        sys.exit(0)
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC
    from unitree_sdk2py.utils.thread import RecurrentThread
    import unitree_legged_const as go2
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    main()
