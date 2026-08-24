"""Plot a hardware run against the reference: joint position, joint velocity, base pose, torque, accelerometer, contact."""
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "run_log.npz"
PREFIX = Path(sys.argv[2]) if len(sys.argv) > 2 else LOG.with_suffix("")
JOINTS = [f"{leg} {j}" for leg in ("FL", "FR", "RL", "RR") for j in ("hip", "thigh", "calf")]
TAU_LIMIT = np.array([23.7, 23.7, 45.43] * 4)
# Fallback only; schedule() derives these from TRAJ when it carries the contact data.
FLIGHTS_S = ((0.40, 0.70), (1.10, 1.40), (1.90, 2.20), (2.70, 3.00))
LANDINGS_MS = (700, 1400, 2200, 3000)
BASE_ = BASE
# Stacked like render.py: the LAST uncommented line wins -- comment to switch trajectory.
# TRAJ = BASE_ / "hopscotch_utils" / "trajectories.npz"
# TRAJ = BASE_ / "hopscotch_utils" / "trajectories_long_stride.npz"
# TRAJ = BASE_ / "hopscotch_utils" / "trajectories_two_leg2.npz"
TRAJ = BASE_ / "hopscotch_utils" / "trajectories_new_hopscotch.npz"


def schedule():
    """(landings_ms, flights_s) from TRAJ's contact passthrough; module constants if absent."""
    f = np.load(TRAJ)
    if "impact_times" not in f.files or "contact" not in f.files:
        return LANDINGS_MS, FLIGHTS_S
    land = tuple(int(round(1e3 * float(t))) for t in np.asarray(f["impact_times"]).ravel())
    air = np.flatnonzero(np.asarray(f["contact"]).sum(1) == 0)
    if not air.size:
        return land, ()
    return land, tuple((int(g[0]) / 1e3, (int(g[-1]) + 1) / 1e3)
                       for g in np.split(air, np.flatnonzero(np.diff(air) > 1) + 1))


def contact_plan_npz(path):
    """Planned per-foot contact force (T,4,3) from an npz lam: 4 feet x 6D wrench, force first."""
    lam = np.asarray(np.load(path)["lam"], float)
    return np.stack([lam[:, 6 * i:6 * i + 3] for i in range(4)], axis=1)


def load_contact_plan(path):
    """Planned per-foot contact force (T,4,3) in MuJoCo foot order, from the trajopt modes.

    lam is variable width -- only feet in contact carry force variables, so flight modes
    have none -- and the half-open concat matches the one HopscotchRef uses.
    """
    modes = json.load(open(path))
    feet = ["FL", "FR", "RL", "RR"]
    blocks = []
    for i, m in enumerate(modes):
        lam = np.asarray(m["lam"], float)
        n = len(np.asarray(m["q"]))
        F = np.zeros((n, 4, 3))
        for j, c in enumerate(m["contacts"]):
            F[:, feet.index(c.split("_")[0]), :] = lam[:, 3 * j:3 * j + 3]
        blocks.append(F[:-1] if i < len(modes) - 1 else F)
    return np.concatenate(blocks)


def quat2eul(q):
    """Roll/pitch/yaw in degrees, the convention the trainer and deploy scripts use."""
    w, x, y, z = (q[..., i] for i in range(4))
    return np.degrees(np.stack([
        np.arctan2(-(2 * (y * z - w * x)), 1 - 2 * (x * x + y * y)),
        np.arcsin(np.clip(2 * (x * z + w * y), -1.0, 1.0)),
        np.arctan2(-(2 * (x * y - w * z)), 1 - 2 * (y * y + z * z))], axis=-1))


f = np.load(TRAJ)
X = f["x_refs"]
if X.ndim == 3:
    X = X[0]

LANDINGS_MS, FLIGHTS_S = schedule()

S = np.load(LOG)["state"]
ti = S[:, 0].astype(int)
tt = S[:, 0] * 0.001
q, qd = S[:, 1:13], S[:, 13:25]
quat, gyro, v = S[:, 25:29], S[:, 29:32], S[:, 32:44]
qr, dqr = X[ti, 7:19], X[ti, 25:37]
quat_r, gyro_r = X[ti, 3:7], X[ti, 22:25]
eul, eul_r = quat2eul(quat), quat2eul(quat_r)
# Logs written before the torque/accel fields were added stop at column 44.
extended = S.shape[1] >= 71
tau_meas = S[:, 44:56] if extended else None
tau_cmd = S[:, 56:68] if extended else None
accel = S[:, 68:71] if extended else None
has_foot = S.shape[1] >= 75
foot = S[:, 71:75] if has_foot else None
has_mocap = S.shape[1] >= 83 and np.isfinite(S[:, 75:78]).any()
mocap = S[:, 75:78] if has_mocap else None
mocap_age = S[:, 81] if has_mocap else None
mocap_seq = S[:, 82] if has_mocap else None
pos_r = X[ti, :3]
FEET = ("FL", "FR", "RL", "RR")
if has_foot:
    # The old hopscotch ships a per-mode JSON; newer references carry lam in the npz.
    TRAJ_JSON = BASE / "hopscotch_utils" / "traj_hopscotch_friction_6cm_lsq.json"
    src = np.load(TRAJ)
    if TRAJ.name == "trajectories.npz" and TRAJ_JSON.exists():
        plan_F = load_contact_plan(TRAJ_JSON)[ti]
    elif "lam" in src.files:
        plan_F = contact_plan_npz(TRAJ)[ti]
    else:
        plan_F = None
    plan_c = plan_F[:, :, 2] > 1.0 if plan_F is not None else None
    # foot_force is uncalibrated: threshold at a fraction of this run's own range.
    foot_c = foot > (0.15 * np.nanmax(foot) if np.nanmax(foot) > 0 else np.inf)

qe, de = q - qr, qd - dqr
ee = (eul - eul_r + 180.0) % 360.0 - 180.0
ratio = np.abs(tau_meas) / TAU_LIMIT if extended else None
rate = 1000.0 / np.median(np.diff(ti)) if len(ti) > 1 else float("nan")

head = f"{'joint':>10}{'q rms':>9}{'q max':>9}{'dq rms':>9}{'dq max':>9}{'|v|max':>10}"
if extended:
    head += f"{'|tau|max':>10}{'tau/lim':>9}{'sat':>6}"
out = [f"log   : {LOG.resolve()}",
       f"rows  : {len(ti)}  {tt[0]:.2f}-{tt[-1]:.2f} s @ {rate:.0f} Hz"
       + ("" if extended else f"   (pre-torque/accel log, {S.shape[1]} cols)"), "", head]
for j, name in enumerate(JOINTS):
    row = (f"{name:>10}{np.sqrt((qe[:, j] ** 2).mean()):8.4f} "
           f"{np.abs(qe[:, j]).max():8.4f} "
           f"{np.sqrt((de[:, j] ** 2).mean()):8.3f} "
           f"{np.abs(de[:, j]).max():8.3f} "
           f"{np.abs(v[:, j]).max():9.4f}")
    if extended:
        row += (f"{np.abs(tau_meas[:, j]).max():10.3f}"
                f"{ratio[:, j].max():9.2f}{int((ratio[:, j] >= 1.0).sum()):6d}")
    out.append(row)
out += ["", f"{'axis':>10}{'rms':>11}{'max|err|':>11}{'final':>10}"]
for i, lbl in enumerate(("roll", "pitch", "yaw")):
    out.append(f"{lbl:>10}{np.sqrt((ee[:, i] ** 2).mean()):10.3f}°"
               f"{np.abs(ee[:, i]).max():10.3f}°{ee[-1, i]:9.3f}°")
we = gyro - gyro_r
for i, lbl in enumerate(("wx", "wy", "wz")):
    out.append(f"{lbl:>10}{np.sqrt((we[:, i] ** 2).mean()):9.3f}r/s"
               f"{np.abs(we[:, i]).max():9.3f}r/s{we[-1, i]:8.3f}r/s")
tail = (f"\noverall   q rms {np.sqrt((qe ** 2).mean()):.4f} rad   "
        f"dq rms {np.sqrt((de ** 2).mean()):.3f} rad/s   "
        f"|v| mean {np.abs(v).mean():.3f} max {np.abs(v).max():.3f} Nm")
if extended:
    tail += (f"\n          peak tau/limit {ratio.max():.2f}   "
             f"ticks with any joint saturated {int((ratio >= 1.0).any(axis=1).sum())}/{len(ti)}   "
             f"cmd-meas rms {np.sqrt(((tau_cmd - tau_meas) ** 2).mean()):.2f} Nm")
    tail += (f"\n          accel |.| mean {np.linalg.norm(accel, axis=1).mean():.2f} "
             f"max {np.linalg.norm(accel, axis=1).max():.2f} m/s^2")
if has_foot and plan_c is not None:
    agree = (foot_c == plan_c).mean(axis=0)
    tail += ("\n          contact schedule agreement  "
             + "  ".join(f"{f} {100 * a:.0f}%" for f, a in zip(FEET, agree))
             + f"   (all {100 * (foot_c == plan_c).mean():.0f}%)")
    # Scale check: if foot_force really is Newtons these ratios sit near 1.
    fz = plan_F[:, :, 2]
    tail += ("\n          stance Fz measured/planned  "
             + "  ".join(f"{f} {foot[plan_c[:, i], i].mean() / max(fz[plan_c[:, i], i].mean(), 1e-6):.2f}"
                         for i, f in enumerate(FEET))
             + f"   rms err {np.sqrt(((foot - fz)[plan_c] ** 2).mean()):.1f} N")
if has_mocap:
    pe = mocap - pos_r
    tail += ("\n          mocap pos err rms "
             + "  ".join(f"{lbl} {np.sqrt((pe[:, i] ** 2).mean()):.4f}m"
                         for i, lbl in enumerate("xyz"))
             + f"   age med {1e3 * np.median(mocap_age):.1f} max {1e3 * mocap_age.max():.1f} ms"
             + f"   frames dropped {int(np.clip(np.diff(mocap_seq) - 1, 0, None).sum())}")
out.append(tail)
report = "\n".join(out)
print(report)
with open(f"{PREFIX}_summary.txt", "w", encoding="utf-8") as fh:
    fh.write(report + "\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def grid(rows, cols, title, size):
    fig, axes = plt.subplots(rows, cols, figsize=size, sharex=True, squeeze=False)
    fig.suptitle(title, y=0.995)
    for ax in axes.flat:
        ax.grid(alpha=0.3)
        for a, b in FLIGHTS_S:
            ax.axvspan(a, b, color="0.9", lw=0, zorder=0)
    for ax in axes[-1]:
        ax.set_xlabel("time [s]   (shaded = planned flight)")
    return fig, axes


fig, axes = grid(4, 3, "go2 joint position vs reference (solid = measured, dashed = ref)", (16, 11))
for i, ax in enumerate(axes.flat):
    ax.plot(tt, qr[:, i], "--", c="0.35", lw=1.3, label="ref")
    ax.plot(tt, q[:, i], c="C1", lw=1.3, label="measured")
    ax.set_ylabel(JOINTS[i] + " [rad]")
axes.flat[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{PREFIX}_joint_pos.png", dpi=150)

fig, axes = grid(4, 3, "go2 joint velocity vs reference (solid = measured, dashed = ref)", (16, 11))
for i, ax in enumerate(axes.flat):
    ax.plot(tt, dqr[:, i], "--", c="0.35", lw=1.3, label="ref")
    ax.plot(tt, qd[:, i], c="C1", lw=1.3, label="measured")
    ax.set_ylabel(JOINTS[i] + " [rad/s]")
axes.flat[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{PREFIX}_joint_vel.png", dpi=150)

fig, axes = grid(3, 2, "go2 base pose vs reference (solid = measured, dashed = ref)", (14, 10))
for i, lbl in enumerate(("roll", "pitch", "yaw")):
    axes[i, 0].plot(tt, eul_r[:, i], "--", c="0.35", lw=1.3, label="ref")
    axes[i, 0].plot(tt, eul[:, i], c="C1", lw=1.3, label="measured")
    axes[i, 0].set_ylabel(f"{lbl} [deg]")
for i, lbl in enumerate(("wx", "wy", "wz")):
    axes[i, 1].plot(tt, gyro_r[:, i], "--", c="0.35", lw=1.3)
    axes[i, 1].plot(tt, gyro[:, i], c="C1", lw=1.3)
    axes[i, 1].set_ylabel(f"{lbl} [rad/s]")
axes[0, 0].set_title("orientation")
axes[0, 1].set_title("angular velocity")
axes[0, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{PREFIX}_base.png", dpi=150)
print(f"\nsaved {PREFIX}_joint_pos.png, _joint_vel.png, _base.png, _summary.txt")

if extended:
    fig, axes = grid(4, 3, "go2 joint torque: commanded vs measured (dotted = motor limit)", (16, 11))
    for i, ax in enumerate(axes.flat):
        ax.axhline(TAU_LIMIT[i], c="C3", lw=0.8, ls=":")
        ax.axhline(-TAU_LIMIT[i], c="C3", lw=0.8, ls=":")
        ax.plot(tt, tau_cmd[:, i], c="0.55", lw=0.9, label="commanded")
        ax.plot(tt, tau_meas[:, i], c="C0", lw=1.2, label="measured")
        ax.plot(tt, v[:, i], c="C4", lw=0.8, alpha=0.7, label="residual v")
        ax.set_ylabel(JOINTS[i] + " [Nm]")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{PREFIX}_torque.png", dpi=150)

    fig, axes = grid(4, 1, "go2 IMU accelerometer (the 3 dims the actor consumes)", (11, 10))
    for k, lb in enumerate("xyz"):
        axes[k, 0].plot(tt, accel[:, k], c="C1", lw=1.1)
        axes[k, 0].set_ylabel(f"accel {lb} [m/s^2]")
    axes[3, 0].plot(tt, np.linalg.norm(accel, axis=1), c="C0", lw=1.1)
    axes[3, 0].axhline(9.81, c="0.5", lw=0.8, ls=":")
    axes[3, 0].set_ylabel("|accel| [m/s^2]   (dotted = 1 g)")
    fig.tight_layout()
    fig.savefig(f"{PREFIX}_accel.png", dpi=150)
    print(f"saved {PREFIX}_torque.png, _accel.png")

if has_mocap:
    fig, axes = grid(4, 1, "go2 mocap base position vs reference (solid = mocap, dashed = ref)", (12, 10))
    for i, lbl in enumerate("xyz"):
        axes[i, 0].plot(tt, pos_r[:, i], "--", c="0.35", lw=1.3, label="ref")
        axes[i, 0].plot(tt, mocap[:, i], c="C1", lw=1.3, label="mocap (as seen by actor)")
        axes[i, 0].set_ylabel(f"{lbl} [m]")
    axes[3, 0].plot(tt, 1e3 * mocap_age, c="C0", lw=1.1)
    axes[3, 0].set_ylabel("mocap age [ms]")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{PREFIX}_mocap.png", dpi=150)
    print(f"saved {PREFIX}_mocap.png")

if has_foot:
    fig, axes = grid(4, 1, "go2 foot contact: measured sensor vs planned normal force", (12, 11))
    for i, f in enumerate(FEET):
        ax = axes[i, 0]
        if plan_c is not None:
            ax.fill_between(tt, 0, 1, where=plan_c[:, i], transform=ax.get_xaxis_transform(),
                            color="C0", alpha=0.12, lw=0, zorder=0, label="planned stance")
            ax.plot(tt, plan_F[:, i, 2], "--", c="0.35", lw=1.3, label="planned Fz [N]")
        ax.plot(tt, foot[:, i], c="C1", lw=1.2, label="measured Fz [N]")
        ax.set_ylabel(f"{f}  Fz [N]")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{PREFIX}_contact.png", dpi=150)
    print(f"saved {PREFIX}_contact.png")
