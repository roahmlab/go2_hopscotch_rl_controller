"""Overlays measured foot clearance (mocap markers vs base+joint FK) on the reference, from a go2_hopscotch4 run_log.npz."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "run_log.npz"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else LOG.with_name(LOG.stem + "_feet.png")
FEET = ("FL", "FR", "RL", "RR")
MIN_CLEARANCE = 0.045
# Fallbacks only; schedule() derives these from TRAJ when it carries the contact data.
LANDINGS_MS = (700, 1400, 2200, 3000)
TAKEOFFS_MS = (400, 1100, 1900, 2700)
FLIGHTS_S = tuple((a / 1e3, b / 1e3) for a, b in zip(TAKEOFFS_MS, LANDINGS_MS))
BASE_ = HERE
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


LANDINGS_MS, _FL = schedule()
TAKEOFFS_MS = tuple(int(round(1e3 * a)) for a, _ in _FL)

HIP_XYZ = np.array([[0.1934, 0.0465, 0.0], [0.1934, -0.0465, 0.0],
                    [-0.1934, 0.0465, 0.0], [-0.1934, -0.0465, 0.0]])
THIGH_XYZ = np.array([[0.0, 0.0955, 0.0], [0.0, -0.0955, 0.0],
                      [0.0, 0.0955, 0.0], [0.0, -0.0955, 0.0]])
CALF_XYZ = np.array([0.0, 0.0, -0.213])
FOOT_XYZ = np.array([-0.002, 0.0, -0.213])


def quat_rotate(q, v):
    t = 2.0 * np.cross(q[1:], v)
    return v + q[0] * t + np.cross(q[1:], t)


def feet_in_base(q):
    out = np.empty((4, 3))
    for i in range(4):
        h, t, k = q[3 * i], q[3 * i + 1], q[3 * i + 2]
        ck, sk = np.cos(k), np.sin(k)
        p = CALF_XYZ + np.array([ck * FOOT_XYZ[0] + sk * FOOT_XYZ[2], 0.0,
                                 -sk * FOOT_XYZ[0] + ck * FOOT_XYZ[2]])
        ct, st = np.cos(t), np.sin(t)
        p = THIGH_XYZ[i] + np.array([ct * p[0] + st * p[2], p[1], -st * p[0] + ct * p[2]])
        ch, sh = np.cos(h), np.sin(h)
        out[i] = HIP_XYZ[i] + np.array([p[0], ch * p[1] - sh * p[2], sh * p[1] + ch * p[2]])
    return out


def reference_clearance():
    """Clearance of the reference itself; same identity the controller uses (radius cancels)."""
    f = np.load(TRAJ)
    X = f["x_refs"]
    X = X[0] if X.ndim == 3 else X
    out = np.empty((len(X), 4))
    for t in range(len(X)):
        pf = feet_in_base(X[t, 7:19])
        out[t] = [X[t, 2] + quat_rotate(X[t, 3:7], pf[i])[2] for i in range(4)]
    return out


def main():
    z = np.load(LOG, allow_pickle=True)
    if "feet" not in z or not len(z["feet"]):
        raise SystemExit(f"{LOG.name} has no 'feet' array -- run go2_hopscotch4.py")
    F = np.asarray(z["feet"], float)
    ii = F[:, 0].astype(int)
    used, fk, moc, src = F[:, 1:5], F[:, 5:9], F[:, 9:13], F[:, 13:17]
    ref = reference_clearance()
    ms = ii  # x_ref is the 1 kHz grid, so the index is milliseconds

    have_mocap = np.isfinite(moc).any()
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(ms, 1e3 * ref[np.clip(ii, 0, len(ref) - 1), i], color="0.6", lw=1.4,
                label="reference")
        ax.plot(ms, 1e3 * fk[:, i], color="tab:orange", lw=1.0, alpha=0.9, label="FK (base+joints)")
        if have_mocap:
            ax.plot(ms, 1e3 * moc[:, i], color="tab:blue", lw=1.0, label="mocap marker")
            miss = src[:, i] < 0.5
            if miss.any():
                ax.plot(ms[miss], 1e3 * used[miss, i], ".", color="tab:red", ms=2.5,
                        label="FK fallback")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.axhline(1e3 * MIN_CLEARANCE, color="tab:green", ls=":", lw=1.0,
                   label=f"swing min {1e3 * MIN_CLEARANCE:.0f} mm")
        for t in LANDINGS_MS:
            ax.axvline(t, color="tab:red", ls="--", lw=0.7, alpha=0.5)
        for t in TAKEOFFS_MS:
            ax.axvline(t, color="tab:blue", ls="--", lw=0.7, alpha=0.4)
        ax.set_ylabel(f"{FEET[i]}  [mm]")
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(ncol=5, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("reference time [ms]   (blue dash = takeoff, red dash = landing)")
    fig.suptitle(f"foot clearance -- {LOG.name}", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"saved {OUT}")

    print(f"{'foot':5s} {'mocap%':>7s} {'rms(mocap-FK)':>14s} {'max|d|':>8s} "
          f"{'min ref':>8s} {'min FK':>8s} {'min mocap':>10s}   [mm]")
    for i, nm in enumerate(FEET):
        d = moc[:, i] - fk[:, i]
        ok = np.isfinite(d)
        r = ref[np.clip(ii, 0, len(ref) - 1), i]
        print(f"{nm:5s} {100.0 * src[:, i].mean():7.1f} "
              f"{1e3 * np.sqrt((d[ok] ** 2).mean()) if ok.any() else np.nan:14.2f} "
              f"{1e3 * np.abs(d[ok]).max() if ok.any() else np.nan:8.2f} "
              f"{1e3 * r.min():8.2f} {1e3 * np.nanmin(fk[:, i]):8.2f} "
              f"{1e3 * np.nanmin(moc[:, i]) if ok.any() else np.nan:10.2f}")


if __name__ == "__main__":
    main()
