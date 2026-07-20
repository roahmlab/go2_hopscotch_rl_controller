"""Plot a hardware run's joint and orientation tracking against the reference."""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

BASE = Path(__file__).resolve().parent
LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "run_log.npz"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else LOG.with_suffix(".png")
JOINTS = [f"{leg}_{j}" for leg in ("FL", "FR", "RL", "RR") for j in ("abd", "hip", "knee")]

f = np.load(BASE / "hopscotch_utils" / "trajectories.npz")
X = f["x_refs"]
if X.ndim == 3:
    X = X[0]

S = np.load(LOG)["state"]
ti = S[:, 0].astype(int)
tt = S[:, 0] * 0.001
q = S[:, 1:13]
qd = S[:, 13:25]
quat = S[:, 25:29]
gyro = S[:, 29:32]
v = S[:, 32:44]
qr = X[ti, 7:19]
quat_r = X[ti, 3:7]
gyro_r = X[ti, 22:25]
eul = Rotation.from_quat(quat, scalar_first=True).as_euler("XYZ")
eul_r = Rotation.from_quat(quat_r, scalar_first=True).as_euler("XYZ")

print(f"{len(ti)} ticks ({tt[0]:.3f}-{tt[-1]:.3f}s)")
print(f"joint err rms {np.sqrt(np.mean((q - qr) ** 2)):.4f} rad  max {np.abs(q - qr).max():.4f}")
print(f"quat err max {np.abs(quat - quat_r).max():.4f}  |v| mean {np.abs(v).mean():.3f} max {np.abs(v).max():.3f} Nm")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(10, 3, figsize=(16, 26), sharex=True)
ax = axes.flat[0]
for k, lb in enumerate("wxyz"):
    ax.plot(tt, quat[:, k], lw=1.0, label=lb)
    ax.plot(tt, quat_r[:, k], lw=0.8, ls="--", color=ax.lines[-1].get_color())
ax.set_ylabel("base quat")
ax.legend(fontsize=7)
ax = axes.flat[1]
for k, lb in enumerate(("roll", "pitch", "yaw")):
    ax.plot(tt, eul[:, k], lw=1.0, label=lb)
    ax.plot(tt, eul_r[:, k], lw=0.8, ls="--", color=ax.lines[-1].get_color())
ax.set_ylabel("base euler [rad]")
ax.legend(fontsize=7)
ax = axes.flat[2]
for k in range(3):
    ax.plot(tt, gyro[:, k], lw=1.0)
    ax.plot(tt, gyro_r[:, k], lw=0.8, ls="--", color=ax.lines[-1].get_color())
ax.set_ylabel("ang vel [rad/s]")
ax = axes.flat[3]
ax.plot(tt, np.linalg.norm(q - qr, axis=1), lw=1.0)
ax.set_ylabel("joint err norm [rad]")
for i in range(12):
    ax = axes.flat[4 + i]
    ax.plot(tt, q[:, i], lw=1.0, label="meas")
    ax.plot(tt, qr[:, i], lw=0.9, ls="--", label="ref")
    ax.set_ylabel(JOINTS[i] + " [rad]")
    ax.grid(alpha=0.3)
for ax in axes[-1]:
    ax.set_xlabel("time [s]")
axes.flat[4].legend(fontsize=7)
for i in range(12):
    ax = axes.flat[16 + i]
    ax.plot(tt, v[:, i], lw=1.0, color="tab:red")
    ax.set_ylabel("v " + JOINTS[i] + " [Nm]")
    ax.grid(alpha=0.3)
for ax in axes.flat[28:]:
    ax.axis("off")
fig.suptitle("go2 hardware run vs reference (solid = measured, dashed = ref)", y=0.995)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"saved {OUT}")
