#!/usr/bin/env python3
"""Reconstruct the Go2 base trajectory (world xyz) from a deploy log WITHOUT mocap.

Stance: dead-reckoning on the loaded feet (a loaded foot does not move, so the base moves by
minus the change of the foot's body-frame position rotated into the world).  Height during
stance is absolute: the lowest loaded foot is on the floor.  Flight (no loaded foot): ballistic
from the take-off velocity.  Orientation: the IMU roll/pitch/yaw in the log.

Usage:  base_odom.py <run_dir> [--exclude RR] [--thr 40] [--plan cart_opt2.npz] [--no-plot]
Writes <run_dir>/odom_base_pos.{npz,png}.  Also validates itself on an Isaac single-env trace
when given a *_traces.npz (compares against the simulator's ground-truth base position).
"""
import argparse, glob, os, sys
import numpy as np

FEET = ("FL", "FR", "RL", "RR")
HIP = {"FL": (0.1934, 0.0465, 0.0), "FR": (0.1934, -0.0465, 0.0),
       "RL": (-0.1934, 0.0465, 0.0), "RR": (-0.1934, -0.0465, 0.0)}
THIGH_Y = {"FL": 0.0955, "FR": -0.0955, "RL": 0.0955, "RR": -0.0955}
L_THIGH, L_CALF, FOOT_X, FOOT_R = 0.213, 0.213, -0.002, 0.022      # menagerie go2.xml


def Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
def R_rpy(r, p, y): return Rz(y) @ Ry(p) @ Rx(r)


def foot_body(foot, hip, thigh, calf):
    """Foot-centre position in the trunk frame for one leg (MuJoCo joint conventions)."""
    p = np.array(HIP[foot]); R = Rx(hip)
    p = p + R @ np.array([0.0, THIGH_Y[foot], 0.0]); R = R @ Ry(thigh)
    p = p + R @ np.array([0.0, 0.0, -L_THIGH]); R = R @ Ry(calf)
    return p + R @ np.array([FOOT_X, 0.0, -L_CALF])


def feet_body(q, names):
    """(T,4,3) foot centres in the trunk frame from (T,12) joint angles and their names."""
    idx = {n: i for i, n in enumerate(names)}
    out = np.zeros((len(q), 4, 3))
    for j, f in enumerate(FEET):
        h, t, c = (idx[f + "_hip_joint"], idx[f + "_thigh_joint"], idx[f + "_calf_joint"])
        for k in range(len(q)):
            out[k, j] = foot_body(f, q[k, h], q[k, t], q[k, c])
    return out


def rot_from_euler(rpy, conv):
    """conv 'xyz': deploy_cart's R = Rx Ry Rz (quat_to_euler_xyz); 'zyx': Isaac's R = Rz Ry Rx."""
    r, p, y = rpy
    return (Rx(r) @ Ry(p) @ Rz(y)) if conv == "xyz" else (Rz(y) @ Ry(p) @ Rx(r))


def odometry(t, rpy, q, names, loaded, g=9.81, bridge_s=0.06, conv="xyz"):
    """Returns base position (T,3), per-step mode (0 flight, 1 stance), take-off events."""
    T = len(t); dt = np.diff(t, prepend=t[0] - (t[1] - t[0]))
    pb = feet_body(q, names)                                   # (T,4,3) body frame
    R = np.array([rot_from_euler(rpy[k], conv) for k in range(T)])
    n0 = max(3, int(0.2 / max(dt[1], 1e-3)))                                   # heading frame: world yaw of the
    psi0 = np.arctan2(np.median(R[:n0, 1, 0]), np.median(R[:n0, 0, 0]))        # body x-axis over the first 0.2 s
    R = np.einsum("ij,kjl->kil", Rz(-psi0), R)
    pw_rel = np.einsum("kij,kfj->kfi", R, pb)                 # foot relative to base, world axes
    # pad flicker: a no-contact gap shorter than `bridge_s` is not a flight; carry the last stance
    # velocity through it (no gravity) instead of integrating a ballistic arc from noisy take-off data
    # The only true full flight of the cartwheel is the launch (planned 2.86-3.34 s): every other
    # no-contact gap (kick-up hop, handstand pad flicker, landing bounces) keeps at least one foot
    # near the floor, so gaps outside [2.7, 3.6] s are bridged whatever their length (up to 0.5 s).
    anyload = loaded.any(1).copy(); hold = np.zeros(T, bool); k = 0
    while k < T:
        if not anyload[k]:
            j = k
            while j < T and not anyload[j]: j += 1
            gap = t[min(j, T - 1)] - t[k]; in_launch = 2.7 <= t[k] <= 3.6
            if 0 < k and j < T and (gap < bridge_s or (not in_launch and gap < 0.5)): hold[k:j] = True
            k = j
        else: k += 1
    x = np.zeros((T, 3)); v = np.zeros(3); mode = np.zeros(T, int)
    anyload = loaded.any(1) | hold
    def landing_height(k0):
        """height of the base at the next real touchdown after step k0 (absolute, from the legs)."""
        j = k0
        while j < T and not loaded[j].any(): j += 1
        return (j, -pw_rel[j, loaded[j], 2].min() + FOOT_R) if j < T else (None, None)
    def takeoff_velocity(k0):
        """linear fit of the base position over the last 0.1 s of stance before step k0."""
        i0 = max(0, k0 - max(2, int(round(0.1 / max(dt[k0], 1e-3)))))
        tt = t[i0:k0]; return np.array([np.polyfit(tt, x[i0:k0, i], 1)[0] for i in range(3)]) if len(tt) >= 2 else v
    # start: robot standing still on all loaded feet; base height from the lowest foot
    x[0, 2] = -(pw_rel[0, loaded[0], 2].min() if loaded[0].any() else pw_rel[0, :, 2].min()) + FOOT_R
    vhist = []
    for k in range(1, T):
        if hold[k]:                                                          # pad flicker / hop: feet still down
            x[k] = x[k - 1]; mode[k] = 2; continue
        both = loaded[k] & loaded[k - 1]
        if both.any():
            step = -(pw_rel[k, both] - pw_rel[k - 1, both]).mean(0)       # feet fixed -> base moves
            x[k] = x[k - 1] + step
            x[k, 2] = -pw_rel[k, loaded[k], 2].min() + FOOT_R              # absolute height in stance
            v = step / dt[k]; vhist.append(v); vhist = vhist[-5:]; mode[k] = 1
        else:
            if mode[k - 1] in (1, 2):                                       # take-off: launch velocity
                v = np.clip(takeoff_velocity(k), -3.0, 3.0)
                j, z1 = landing_height(k)                                   # vertical: arc through the landing height
                if j is not None and j > k:
                    Tf = t[j] - t[k - 1]; v[2] = (z1 - x[k - 1, 2] + 0.5 * g * Tf * Tf) / Tf
                if os.environ.get("ODOM_DEBUG"): print(f"    take-off at t={t[k]:.2f}: v={v.round(2)}  landing at t={t[j] if j else -1:.2f} z1={z1}")
            v = v + np.array([0.0, 0.0, -g]) * dt[k]
            x[k] = x[k - 1] + v * dt[k]
            if loaded[k].any():                                             # touchdown this step
                x[k, 2] = -pw_rel[k, loaded[k], 2].min() + FOOT_R
    return x, mode, pw_rel


def load_hw(run_dir, exclude, thr):
    f = [p for p in glob.glob(os.path.join(run_dir, "*.npz")) if "torque" not in p and "odom" not in p][0]
    z = np.load(f, allow_pickle=True)
    names = [str(n) for n in z["joint_names"]]; t = z["t"]; q = z["q"]; rpy = z["rpy"]
    F = z["forces"].copy()
    # the pads are compressed (FL reads ~0.5x) and creep: calibrate per foot on the first 0.4 s of
    # quiet standing (all four feet down, ~37 N true per foot) and call a foot loaded above half of
    # that, with a floor of `thr` N. RR is excluded by default (pad broken since 2026-09).
    standing = F[t < 0.4].mean(0)
    loaded = F > np.maximum(thr, 0.5 * standing)
    for f_ in exclude: loaded[:, FEET.index(f_)] = False
    return t, rpy, q, names, loaded, z


def load_sim(path):
    z = np.load(path, allow_pickle=True); dt = float(z["dt"]); n = len(z["q_s"]); t = np.arange(n) * dt
    rpy = z["brpy_s"]; rpy = np.radians(rpy) if np.abs(rpy).max() > 10 else rpy
    names = [f + "_hip_joint" for f in FEET] + [f + "_thigh_joint" for f in FEET] + [f + "_calf_joint" for f in FEET]
    loaded = z["ff_s"] > 30.0
    return t, rpy, z["q_s"], names, loaded, z


def plan_xyz(path, t):
    r = np.load(path, allow_pickle=True); return np.stack([np.interp(t, r["t"], r["q"][:, i]) for i in range(3)], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path"); ap.add_argument("--exclude", nargs="*", default=["RR"], help="pads to ignore (RR is broken)")
    ap.add_argument("--thr", type=float, default=12.0, help="floor (N) of the per-foot loaded threshold (default half the standing load)")
    ap.add_argument("--plan", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "second_one", "cart_opt2.npz"))
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--bridge", type=float, default=0.06, help="no-contact gaps shorter than this (s) are pad flicker, not flight")
    ap.add_argument("--euler", choices=["auto", "xyz", "zyx"], default="auto", help="rpy convention: deploy logs are xyz (R=RxRyRz), Isaac traces zyx")
    a = ap.parse_args()
    sim = a.path.endswith(".npz")
    if sim:
        t, rpy, q, names, loaded, z = load_sim(a.path); out_dir = os.path.dirname(a.path)
    else:
        t, rpy, q, names, loaded, z = load_hw(a.path, a.exclude, a.thr); out_dir = a.path
    conv = a.euler if a.euler != "auto" else ("zyx" if sim else "xyz")
    x, mode, pw_rel = odometry(t, rpy, q, names, loaded, bridge_s=a.bridge, conv=conv)
    print(f"{a.path}\n  {len(t)} steps, stance {100*mode.mean():.0f}% of steps, flight segments: {int(np.sum(np.diff(mode)==-1))}")
    if sim:
        gt = z["bpos_s"][:len(t)]; err = x - (gt - gt[0] + x[0])
        print(f"  SIM CHECK vs ground truth: xyz RMS err {100*np.sqrt((err**2).mean(0))} cm, final err {100*err[-1]} cm")
        rf = z["rf_s"]; n = min(len(rf), len(t)); fk_err = (gt[:n, None, :] + pw_rel[:n]) - rf[:n]
        print(f"  FK check (foot world pos vs sim): RMS {100*np.sqrt((fk_err**2).mean()):.1f} cm")
    xy_travel = x[-1, :2] - x[0, :2]
    print(f"  base travel start->end: x {xy_travel[0]:+.2f} m, y {xy_travel[1]:+.2f} m | max |y| excursion {np.abs(x[:,1]-x[0,1]).max():.2f} m | apex z {x[:,2].max():.2f} m")
    plan = None
    if os.path.exists(a.plan):
        plan = plan_xyz(a.plan, t); plan = plan - plan[0] + x[0]
        print(f"  plan travel: x {plan[-1,0]-plan[0,0]:+.2f}, y {plan[-1,1]-plan[0,1]:+.2f} | odometry minus plan at the end: x {x[-1,0]-plan[-1,0]:+.2f}, y {x[-1,1]-plan[-1,1]:+.2f} m")
    np.savez(os.path.join(out_dir, "odom_base_pos.npz"), t=t, xyz=x, mode=mode, plan=plan if plan is not None else np.zeros(0))
    if not a.no_plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
        for i, lab in enumerate("xyz"):
            ax[i].plot(t, x[:, i], "C0", lw=1.6, label="leg odometry")
            if plan is not None: ax[i].plot(t, plan[:, i], "k--", lw=1, label="plan")
            if sim: ax[i].plot(t, z["bpos_s"][:len(t), i] - z["bpos_s"][0, i] + x[0, i], "C3", lw=1, label="sim truth")
            ax[i].fill_between(t, *ax[i].get_ylim(), where=mode == 0, color="0.85", label="flight" if i == 0 else None)
            ax[i].fill_between(t, *ax[i].get_ylim(), where=mode == 2, color="#ffe8c0", label="pad gap (held)" if i == 0 else None)
            ax[i].set_ylabel(f"base {lab} [m]"); ax[i].grid(alpha=.3)
        ax[0].legend(ncol=4, fontsize=8); ax[-1].set_xlabel("t [s]"); fig.suptitle(os.path.basename(out_dir) + "  base position without mocap")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "odom_base_pos.png"), dpi=110); print("  ->", os.path.join(out_dir, "odom_base_pos.png"))


if __name__ == "__main__":
    main()
