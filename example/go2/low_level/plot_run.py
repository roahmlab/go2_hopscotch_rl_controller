"""Plot a hardware run's joint tracking, torque and accelerometer against the reference."""
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "run_log.npz"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else LOG.with_suffix(".png")
JOINTS = [f"{leg} {j}" for leg in ("FL", "FR", "RL", "RR") for j in ("hip", "thigh", "calf")]
TAU_LIMIT = np.array([23.7, 23.7, 45.43] * 4)

f = np.load(BASE / "hopscotch_utils" / "trajectories.npz")
X = f["x_refs"]
if X.ndim == 3:
    X = X[0]

D = np.load(LOG)
S = D["state"]
ti = S[:, 0].astype(int)
tt = S[:, 0] * 0.001
q, qd = S[:, 1:13], S[:, 13:25]
quat, gyro, v = S[:, 25:29], S[:, 29:32], S[:, 32:44]
qr, dqr = X[ti, 7:19], X[ti, 25:37]
quat_r, gyro_r = X[ti, 3:7], X[ti, 22:25]
# Logs written before the torque/accel fields were added stop at column 44.
extended = S.shape[1] >= 71
tau_meas = S[:, 44:56] if extended else None
tau_cmd = S[:, 56:68] if extended else None
accel = S[:, 68:71] if extended else None

qe, de = q - qr, qd - dqr
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
tail = (f"\noverall   q rms {np.sqrt((qe ** 2).mean()):.4f} rad   "
        f"dq rms {np.sqrt((de ** 2).mean()):.3f} rad/s   "
        f"|v| mean {np.abs(v).mean():.3f} max {np.abs(v).max():.3f} Nm")
if extended:
    tail += (f"\n          peak tau/limit {ratio.max():.2f}   "
             f"ticks with any joint saturated {int((ratio >= 1.0).any(axis=1).sum())}/{len(ti)}   "
             f"cmd-meas rms {np.sqrt(((tau_cmd - tau_meas) ** 2).mean()):.2f} Nm")
    tail += (f"\n          accel |.| mean {np.linalg.norm(accel, axis=1).mean():.2f} "
             f"max {np.linalg.norm(accel, axis=1).max():.2f} m/s^2")
tail += f"\n          quat err max {np.abs(quat - quat_r).max():.4f}"
out.append(tail)
report = "\n".join(out)
print(report)
SUMMARY = OUT.with_name(OUT.stem + "_summary.txt")
with open(SUMMARY, "w") as fh:
    fh.write(report + "\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(5, 3, figsize=(16, 15), sharex=True)
ax = axes.flat[0]
for k in range(3):
    ax.plot(tt, gyro[:, k], lw=1.0)
    ax.plot(tt, gyro_r[:, k], lw=0.8, ls="--", color=ax.lines[-1].get_color())
ax.set_ylabel("ang vel [rad/s]")
ax.grid(alpha=0.3)
ax = axes.flat[1]
ax.plot(tt, np.linalg.norm(qe, axis=1), lw=1.0)
ax.set_ylabel("joint err norm [rad]")
ax.grid(alpha=0.3)
for i in range(12):
    ax = axes.flat[2 + i]
    ax.plot(tt, q[:, i], lw=1.0, label="meas")
    ax.plot(tt, qr[:, i], lw=0.9, ls="--", label="ref")
    ax.set_ylabel(JOINTS[i] + " [rad]")
    ax.grid(alpha=0.3)
axes.flat[2].legend(fontsize=7)
for ax in axes[-1]:
    ax.set_xlabel("time [s]")
for ax in axes.flat[14:]:
    ax.axis("off")
fig.suptitle("go2 hardware run vs reference (solid = measured, dashed = ref)", y=0.995)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"\nsaved {OUT}, {SUMMARY}")

if extended:
    out_t = OUT.with_name(OUT.stem + "_torque.png")
    fig, axes = plt.subplots(4, 3, figsize=(16, 11), sharex=True)
    for i in range(12):
        ax = axes.flat[i]
        ax.axhline(TAU_LIMIT[i], c="C3", lw=0.8, ls=":")
        ax.axhline(-TAU_LIMIT[i], c="C3", lw=0.8, ls=":")
        ax.plot(tt, tau_cmd[:, i], c="0.55", lw=0.9, label="commanded")
        ax.plot(tt, tau_meas[:, i], c="C0", lw=1.2, label="measured")
        ax.plot(tt, v[:, i], c="C4", lw=0.8, alpha=0.7, label="residual v")
        ax.set_ylabel(JOINTS[i] + " [Nm]")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("time [s]")
    fig.suptitle("go2 joint torque: commanded vs measured (dotted = motor limit)", y=0.995)
    fig.tight_layout()
    fig.savefig(out_t, dpi=150)

    out_a = OUT.with_name(OUT.stem + "_accel.png")
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for k, lb in enumerate("xyz"):
        axes[k].plot(tt, accel[:, k], c="C1", lw=1.1)
        axes[k].set_ylabel(f"accel {lb} [m/s^2]")
        axes[k].grid(alpha=0.3)
    axes[3].plot(tt, np.linalg.norm(accel, axis=1), c="C0", lw=1.1)
    axes[3].axhline(9.81, c="0.5", lw=0.8, ls=":")
    axes[3].set_ylabel("|accel| [m/s^2]")
    axes[3].set_xlabel("time [s]   (dotted = 1 g)")
    axes[3].grid(alpha=0.3)
    fig.suptitle("go2 IMU accelerometer (the 3 dims the actor consumes)", y=0.995)
    fig.tight_layout()
    fig.savefig(out_a, dpi=150)
    print(f"saved {out_t}, {out_a}")
