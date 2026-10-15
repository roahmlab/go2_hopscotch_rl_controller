"""Vicon base position against the hopscotch reference, cropped to the run.

Differs from plot_base_pos.py in three ways: the reference is trajectories.npz (the
same file the controller flies, rather than the modes JSON), the overlay is the only
figure, and the idle lead-in and tail are cut away -- the plot spans the motion plus
--pad seconds, not the whole recording.

z is compared as a DELTA: the marker cluster sits above the body origin the plan
integrates, so both traces are zeroed on their pre-launch mean. Height gained is
comparable, absolute height is not.

Usage:
    python3 plot_vicon_base.py [xyz_go2_v5.npz] [output_prefix]
    python3 plot_vicon_base.py xyz_go2_v5.npz --t0 12.4 --yaw 90
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
# Fallback only; schedule() derives these from TRAJ when it carries the contact data.
FLIGHTS_S = ((0.40, 0.70), (1.10, 1.40), (1.90, 2.20), (2.70, 3.00))
LANDINGS_MS = (700, 1400, 2200, 3000)
BASE_ = BASE
# Stacked like render.py: the LAST uncommented line wins -- comment to switch trajectory.
# TRAJ = BASE_ / "hopscotch_utils" / "trajectories.npz"
# TRAJ = BASE_ / "hopscotch_utils" / "trajectories_long_stride.npz"
# TRAJ = BASE_ / "hopscotch_utils" / "trajectories_two_leg2.npz"
# TRAJ = BASE_ / "hopscotch_utils" / "trajectories_new_hopscotch.npz"
TRAJ = BASE_ / "hopscotch_utils" / "trajectories_cartwheel_2.npz"


def schedule(path=None):
    """(landings_ms, flights_s) from the reference's contact passthrough; module constants if absent."""
    f = np.load(path or TRAJ)
    if "impact_times" not in f.files or "contact" not in f.files:
        return LANDINGS_MS, FLIGHTS_S
    land = tuple(int(round(1e3 * float(t))) for t in np.asarray(f["impact_times"]).ravel())
    air = np.flatnonzero(np.asarray(f["contact"]).sum(1) == 0)
    if not air.size:
        return land, ()
    return land, tuple((int(g[0]) / 1e3, (int(g[-1]) + 1) / 1e3)
                       for g in np.split(air, np.flatnonzero(np.diff(air) > 1) + 1))


REF = TRAJ
LANDINGS_MS, FLIGHTS_S = schedule()


def quat_mul(a, b):
    """Hamilton product, wxyz, broadcasting over leading axes."""
    aw, ax, ay, az = (a[..., i] for i in range(4))
    bw, bx, by, bz = (b[..., i] for i in range(4))
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def quat_inv(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat2eul(q):
    """Roll/pitch/yaw, matching the convention the trainer and deploy scripts use."""
    w, x, y, z = (q[..., i] for i in range(4))
    return np.stack([np.arctan2(-(2 * (y * z - w * x)), 1 - 2 * (x * x + y * y)),
                     np.arcsin(np.clip(2 * (x * z + w * y), -1.0, 1.0)),
                     np.arctan2(-(2 * (x * y - w * z)), 1 - 2 * (y * y + z * z))], axis=-1)


def mean_quat(q):
    """Sign-aligned normalized mean; fine for the small spread of a settle window."""
    q = q * np.sign(np.sum(q * q[0], axis=-1))[:, None]
    m = np.nanmean(q, axis=0)
    return m / np.linalg.norm(m)


def load_reference(path):
    """(t, base_xyz, base_quat wxyz) at the trajectory's native 1 kHz."""
    d = np.load(path)
    x = d["x_refs"]
    x = x[0] if x.ndim == 3 else x
    return (np.arange(len(x)) * 1e-3, np.asarray(x[:, 0:3], float),
            np.asarray(x[:, 3:7], float))


def load_mocap(path):
    """(t from record start, xyz metres, quat as stored); occlusion NaNs are kept, not filled.

    The recorder names the array quat_xyzw but stores the SDK value unmodified, and on
    this rig that is w-FIRST. The order is not guessed here: a level robot facing 180 deg
    in the Vicon frame has its largest component in z, which fooled a largest-component
    guess, so pick_layout() settles it against the reference once the run is synced.
    """
    d = np.load(path, allow_pickle=True)
    xyz = (np.asarray(d["xyz"], float) if "xyz" in d.files
           else np.stack([np.asarray(d[k], float) for k in "xyz"], axis=1))
    if (str(d["units"]) == "mm" if "units" in d.files
            else np.nanmedian(np.abs(xyz[:, 2])) > 20):
        xyz = xyz * 1e-3
    q = None
    for key in ("quat_xyzw", "quat_wxyz", "quat"):
        if key in d.files:
            q = np.asarray(d[key], float)
            break
    t = np.asarray(d["t"], float)
    t = t - t[0]
    print(f"mocap: {len(t)} samples, {t[-1]:.2f} s, {1 / np.median(np.diff(t)):.0f} Hz, "
          f"{int(np.isnan(xyz).any(axis=1).sum())} occluded")
    return t, xyz, q


def find_t0(t_m, z_m, t_r, z_r):
    """Recording time where plan t=0 lands, by sliding the plan's z profile (normalized correlation)."""
    dt = float(np.median(np.diff(t_m)))
    ok = np.isfinite(z_m)
    g = np.arange(t_m[0], t_m[-1], dt)
    zm = np.interp(g, t_m[ok], z_m[ok])
    zr = np.interp(np.arange(0.0, t_r[-1], dt), t_r, z_r)
    m = len(zr)
    if len(zm) < m:
        raise SystemExit(f"ABORT: recording ({len(zm) * dt:.1f} s) is shorter than the "
                         f"plan ({m * dt:.1f} s)")
    b = zr - zr.mean()
    dot = np.correlate(zm, b, mode="valid")
    c1 = np.concatenate([[0.0], np.cumsum(zm)])
    c2 = np.concatenate([[0.0], np.cumsum(zm * zm)])
    s1, s2 = c1[m:] - c1[:-m], c2[m:] - c2[:-m]
    na = np.sqrt(np.maximum(s2 - s1 * s1 / m, 0.0))
    r = dot / np.maximum(na * np.linalg.norm(b), 1e-12)
    # A motionless window scores on its own noise; require real z swing to be eligible.
    r = np.where(na >= 0.25 * np.linalg.norm(b), r, -np.inf)
    if not np.isfinite(r).any():
        raise SystemExit("ABORT: no window in this recording moves in z like the plan "
                         "-- wrong record, or the robot never launched")
    k = int(np.argmax(r))
    return float(g[k]), float(r[k])


def aligned_attitude(q, s_m, t_r, ref_q, settle):
    """Mocap quaternions rotated by the settle-window offset onto the reference, and that offset."""
    q = q * np.sign(q[:, :1] + (q[:, :1] == 0))
    q_off = quat_mul(mean_quat(ref_q[t_r <= settle]), quat_inv(mean_quat(q[s_m])))
    return quat_mul(np.broadcast_to(q_off, q.shape), q), q_off


def geodesic_deg(q_al, t_al, win, t_r, ref_q):
    """Attitude error (deg) between aligned mocap and the reference interpolated onto the mocap samples."""
    qi = np.stack([np.interp(t_al[win], t_r, ref_q[:, i]) for i in range(4)], axis=1)
    qi = qi / np.linalg.norm(qi, axis=1, keepdims=True)
    return np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(q_al[win] * qi, axis=1)), 0, 1)))


# Identity and 180 deg about the body x, y, z axes: the marker cluster's symmetries, under
# which Vicon jumps between equally valid solutions from one frame to the next.
SYMMETRIES = np.eye(4)


def unflip(q, start):
    """Undoes Vicon's marker-symmetry flips by keeping each sample the symmetric reading closest to the previous sample."""
    out = q.copy()
    prev, last, flips = None, start, 0
    for k in np.flatnonzero(np.isfinite(q).all(axis=1)):
        cands = quat_mul(np.broadcast_to(q[k] / np.linalg.norm(q[k]), (4, 4)), SYMMETRIES)
        pick = start if prev is None else int(np.argmax(np.abs(cands @ prev)))
        flips += int(pick != last)
        out[k], prev, last = cands[pick], cands[pick], pick
    return out, flips


def pick_layout(q_read, s_m, t_al, win, t_r, ref_q, settle):
    """Chooses the stored quaternion order and the Vicon flip state by which combination tracks the reference over the run."""
    best = None
    for name, order in (("w-first", [0, 1, 2, 3]), ("w-last", [3, 0, 1, 2])):
        for start in range(4):
            q_fix, n_flip = unflip(q_read[:, order], start)
            q_al, q_off = aligned_attitude(q_fix, s_m, t_r, ref_q, settle)
            okq = win & np.isfinite(q_al).all(axis=1)
            err = float(np.median(geodesic_deg(q_al, t_al, okq, t_r, ref_q)))
            print(f"       quaternion read {name}, starting symmetry {start}: median attitude error {err:.1f} deg "
                  f"({n_flip} Vicon orientation flips undone)")
            if best is None or err < best[0]:
                best = (err, f"{name}, starting symmetry {start}", q_al, q_off)
    print(f"       -> using {best[1]}")
    return best[2], best[3]


def fit_yaw(xy_m, xy_r):
    """Best-fit planar rotation taking the mocap horizontal path onto the plan's."""
    a, b = xy_m - xy_m.mean(0), xy_r - xy_r.mean(0)
    return np.arctan2(float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])),
                      float(np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mocap", nargs="?", default=str(BASE / "xyz_go2.npz"))
    ap.add_argument("prefix", nargs="?", default=None)
    ap.add_argument("--ref", default=str(REF))
    ap.add_argument("--t0", type=float, default=None, help="seconds into the recording where plan t=0 lands")
    ap.add_argument("--yaw", type=float, default=None, help="mocap->plan yaw in degrees")
    ap.add_argument("--settle", type=float, default=0.2, help="seconds at t=0 averaged to zero both traces")
    ap.add_argument("--pad", type=float, default=0.3, help="seconds of margin kept either side of the motion")
    ap.add_argument("--path-yaw", action="store_true",
                    help="align the Vicon heading by fitting the horizontal path, not the settle orientation")
    args = ap.parse_args()
    mocap = Path(args.mocap)
    prefix = Path(args.prefix) if args.prefix else mocap.with_suffix("")

    _, flights_s = schedule(args.ref)
    t_r, ref, ref_q = load_reference(args.ref)
    dur = t_r[-1]
    t_m, moc_raw, moc_q = load_mocap(mocap)
    print(f"plan : {len(t_r)} knots, {dur:.2f} s, z {ref[:, 2].min():.3f}-{ref[:, 2].max():.3f} m")

    if args.t0 is not None:
        t0, r = args.t0, float("nan")
        print(f"sync : t0 = {t0:.3f} s (manual)")
    else:
        t0, r = find_t0(t_m, moc_raw[:, 2], t_r, ref[:, 2])
        print(f"sync : t0 = {t0:.3f} s into the recording, r = {r:.3f}"
              + ("   WEAK -- check the overlay, or pass --t0" if r < 0.5 else ""))
    t_al = t_m - t0

    win = (t_al >= 0.0) & (t_al <= dur) & np.isfinite(moc_raw).all(axis=1)
    if win.sum() < 10:
        raise SystemExit("ABORT: fewer than 10 valid mocap samples inside the plan window")
    ref_on_m = np.stack([np.interp(t_al[win], t_r, ref[:, i]) for i in range(3)], axis=1)

    s_m = win & (t_al <= args.settle)
    if s_m.sum() < 2:
        s_m = win & (t_al <= t_al[win][0] + args.settle)

    # Orientation: the marker cluster is mounted at an unknown fixed rotation to the
    # body the plan integrates, so the settle window supplies a constant offset -- the
    # rotational analogue of the position zeroing below.
    eul_m = eul_r = q_al = q_off = None
    if moc_q is not None:
        q_al, q_off = pick_layout(moc_q, s_m, t_al, win, t_r, ref_q, args.settle)
        eul_m, eul_r = np.degrees(quat2eul(q_al)), np.degrees(quat2eul(ref_q))

    travel = float(np.linalg.norm(moc_raw[win][-1, :2] - moc_raw[win][0, :2]))
    if args.yaw is not None:
        yaw = np.radians(args.yaw)
        print(f"frame: yaw = {np.degrees(yaw):+.1f} deg (manual)")
    elif q_off is not None and not args.path_yaw:
        # The heading at the settle pose; a path fit is dragged by any drift or fall.
        yaw = float(quat2eul(q_off[None])[0, 2])
        print(f"frame: yaw = {np.degrees(yaw):+.1f} deg from the settle orientation "
              f"(marker cluster assumed body-aligned; --path-yaw fits the path instead)")
    elif travel < 0.05:
        yaw = 0.0
        print(f"frame: only {travel:.3f} m of horizontal travel -- yaw fit would be noise, using 0")
    else:
        yaw = fit_yaw(moc_raw[win][:, :2], ref_on_m[:, :2])
        print(f"frame: yaw = {np.degrees(yaw):+.1f} deg (best fit over {travel:.3f} m)")
        if q_off is not None:
            print(f"       orientation offset from settle: yaw {np.degrees(quat2eul(q_off[None])[0, 2]):+.1f} deg "
                  f"(a large gap from the path fit means the marker cluster is mounted rotated)")
    c, s = np.cos(yaw), np.sin(yaw)
    moc = moc_raw.copy()
    moc[:, 0] = c * moc_raw[:, 0] - s * moc_raw[:, 1]
    moc[:, 1] = s * moc_raw[:, 0] + c * moc_raw[:, 1]
    moc = moc - moc[s_m].mean(0) + ref[t_r <= args.settle].mean(0)

    crop = (t_al >= -args.pad) & (t_al <= dur + args.pad)
    rows = 4 if q_al is not None else 3
    fig, axes = plt.subplots(rows, 1, sharex=True, figsize=(11, 3 * rows))
    for i, lbl in enumerate("xyz"):
        axes[i].plot(t_r, ref[:, i], "--", c="0.35", lw=1.5, label="reference")
        axes[i].plot(t_al[crop], moc[crop, i], c="C1", lw=1.6, label="vicon")
        axes[i].set_ylabel(f"{lbl} [m]")
    if q_al is not None:
        okc = crop & np.isfinite(q_al).all(axis=1) & (t_al >= 0.0) & (t_al <= dur)
        att = np.full(len(t_al), np.nan)
        att[okc] = geodesic_deg(q_al, t_al, okc, t_r, ref_q)
        axes[3].plot(t_al[crop], att[crop], c="C3", lw=1.4)
        axes[3].set_ylabel("attitude error [deg]")
    for ax in axes:
        for a, b in flights_s:
            ax.axvspan(a, b, color="0.9", lw=0, zorder=0)
        ax.grid(alpha=0.3)
        ax.set_xlim(-args.pad, dur + args.pad)
    axes[0].legend(fontsize=9, loc="upper left")
    axes[-1].set_xlabel("t [s]   (shaded = planned flight)")
    fig.suptitle(f"Base position: reference vs Vicon -- {mocap.name}"
                 + (f"   (sync r={r:.2f}, yaw {np.degrees(yaw):+.0f} deg)" if np.isfinite(r) else ""))
    fig.tight_layout()
    fig.savefig(f"{prefix}_base_overlay.png", dpi=130)

    mz, rz = moc[win, 2], ref[:, 2]
    err = moc[win] - ref_on_m
    out = [f"mocap    : {mocap.resolve()}",
           f"reference: {Path(args.ref).resolve()}",
           f"sync     : t0 {t0:.3f} s" + (f", r {r:.3f}" if np.isfinite(r) else " (manual)")
           + f", yaw {np.degrees(yaw):+.1f} deg, zeroed on first {args.settle:.1f} s", "",
           f"{'':>16}{'apex z':>10}{'gain':>10}{'fwd travel':>12}",
           f"{'reference':>16}{rz.max():9.4f}m{rz.max() - rz[0]:9.4f}m{ref[-1, 0] - ref[0, 0]:11.4f}m",
           f"{'vicon':>16}{mz.max():9.4f}m{mz.max() - mz[0]:9.4f}m"
           f"{moc[win][-1, 0] - moc[win][0, 0]:11.4f}m", "",
           f"{'flight':>16}{'ref apex':>12}{'vicon apex':>12}{'delta':>10}"]
    for a, b in flights_s:
        rm, mm = (t_r >= a) & (t_r <= b), win & (t_al >= a) & (t_al <= b)
        if not mm.any():
            continue
        pa, va = rz[rm].max() - rz[0], moc[mm, 2].max() - mz[0]
        out.append(f"{f'{a:.2f}-{b:.2f}s':>16}{pa:11.4f}m{va:11.4f}m{va - pa:+9.4f}m")
    out += ["", f"{'axis':>7}{'RMS err':>11}{'max|err|':>11}{'final':>10}"]
    for i, lbl in enumerate("xyz"):
        e = err[:, i]
        out.append(f"{lbl:>7}{np.sqrt((e ** 2).mean()):10.4f}m{np.abs(e).max():10.4f}m{e[-1]:9.4f}m")
    if eul_m is not None:
        eul_on_r = np.stack([np.interp(t_al[win], t_r, eul_r[:, i]) for i in range(3)], axis=1)
        ee = (eul_m[win] - eul_on_r + 180.0) % 360.0 - 180.0
        out += [""]
        for i, lbl in enumerate(("roll", "pitch", "yaw")):
            out.append(f"{lbl:>7}{np.sqrt((ee[:, i] ** 2).mean()):10.3f}°"
                       f"{np.abs(ee[:, i]).max():10.3f}°{ee[-1, i]:9.3f}°")
        okq = win & np.isfinite(q_al).all(axis=1)
        ang = geodesic_deg(q_al, t_al, okq, t_r, ref_q)
        out.append(f"{'total':>7}{np.sqrt((ang ** 2).mean()):10.3f}°{ang.max():10.3f}°"
                   f"{ang[-1]:9.3f}°   (geodesic attitude error)")
        t_q = t_al[okq]
        out += ["", f"{'touchdown':>16}{'attitude error':>16}"]
        for a, b in flights_s:
            if (t_q >= b).any():
                out.append(f"{f'{b:.2f}s':>16}{ang[np.argmax(t_q >= b)]:15.1f}°")
    report = "\n".join(out)
    print("\n" + report)
    with open(f"{prefix}_base_overlay.txt", "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\nwrote {prefix}_base_overlay.png, {prefix}_base_overlay.txt")


if __name__ == "__main__":
    main()
