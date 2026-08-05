"""Offline floating-base comparison for a Go2 hopscotch run:
Vicon ground truth (record_floating_base.py) vs the trajopt plan, with
deploy_meta.py's velest odometry overlaid where it exists.

Answers the question deploy_meta.py structurally cannot: did the base actually go
where the plan said? The on-robot trace only has `odom_xy` (dead-reckoned from the
velocity-estimator head, xy only, no z) and `ref_xy` -- so apex height and true
forward travel are only visible here.

WHAT CHANGED FROM THE Go1 VERSION:

1. THE PLAN COMES FROM THE TRAJECTORY, NOT A `trial_*.npz`. The Go1 script read a
   croc/spirit dump with keys `xs`/`dt`. The Go2 plan is the same reference
   deploy_meta.py flies -- traj_hopscotch_friction_6cm_lsq.json (modes, HALF-OPEN
   concat) or a pre-gridded reference_grid_1khz.npz. Both are loaded here exactly as
   HopscotchRef does, INCLUDING the half-open rule, so plan time here is the same
   clock the robot ran on.

2. SYNC BY SLIDING THE PLAN OVER THE WHOLE RECORD. The old script trimmed on a 1 cm
   displacement threshold, then cross-correlated. On a Go2 record that threshold
   fires on the STAND-UP, not the launch -- fold/align/hold moves the base far more
   than 1 cm, minutes before the policy starts. Instead the 3.4 s plan z-profile is
   slid over the entire recording and scored by normalized correlation, which is
   indifferent to how much lead-in there is. `--t0` overrides it by hand.

3. YAW IS ESTIMATED, NOT ASSUMED. Vicon world x is not the plan's forward axis, so
   a per-axis start-subtraction (what the Go1 script did) smears the 0.61 m of
   forward travel across mocap x AND y. The best-fit planar rotation between the two
   horizontal paths is solved for and reported.

4. NO ASSUMED 120 Hz. The recorder stores wall-clock `t`; the actual median rate is
   used, and occlusion NaNs are interpolated across for the sync only.

The z comparison is a DELTA comparison: the Vicon marker cluster sits above the body
origin the plan integrates, so both are zeroed on their pre-launch mean. Apex
*height gained* is comparable; absolute z is not.

Usage:
    # normal path -- deploy_meta wrote data/<run_tag>_<stamp>/, you copied the Vicon
    # record in beside the trace, and the plots land back in that same directory:
    python3 plot_base_pos.py --dir data/pz3_400_pz3_meta_r0p75_traj_..._20260731_183632

    python3 plot_base_pos.py                        # loose: xyz_go2.npz + newest run
    python3 plot_base_pos.py --t0 42.7 --yaw 90     # manual sync / frame override
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


DEFAULT_TRAJ = "hopscotch_utils/traj_hopscotch_friction_6cm_lsq.json"
FOOT_NAMES = ["FL", "FR", "RL", "RR"]


# --------------------------------------------------------------------- loaders
def load_reference(path):
    """(t, base_xyz (T,3), airborne (T,)) for BOTH reference formats.

    The half-open concat is JSON-ONLY -- see deploy_meta.HopscotchRef. The gridded
    npz already carries post-impact state at the impact row; dropping rows there
    would mirror the bug the half-open rule exists to fix."""
    if str(path).endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        q = np.asarray(d["q"], float)
        contact = np.asarray(d["contact"]).astype(bool)
        dt = 1.0 / float(d["rate"])
    else:
        modes = json.load(open(path))
        qs, cs = [], []
        for i, m in enumerate(modes):
            q = np.asarray(m["q"], float)
            c = np.zeros((len(q), 4), bool)
            for foot in m["contacts"]:
                c[:, FOOT_NAMES.index(foot.split("_")[0])] = True
            if i < len(modes) - 1:                 # HALF-OPEN: drop non-final LAST row
                q, c = q[:-1], c[:-1]
            qs.append(q)
            cs.append(c)
        q = np.concatenate(qs)
        contact = np.concatenate(cs)
        dt = float(modes[0]["dt"])
    t = np.arange(len(q)) * dt
    return t, q[:, 0:3].copy(), (~contact).all(axis=1)


def load_mocap(path):
    """(t seconds from record start, xyz (N,3) metres). Handles the legacy Go1 file
    (millimetres, no `units` key) so old records still plot correctly."""
    d = np.load(path, allow_pickle=True)
    if "xyz" in d.files:
        xyz = np.asarray(d["xyz"], float)
    else:
        xyz = np.stack([np.asarray(d[k], float) for k in ("x", "y", "z")], axis=1)
    units = str(d["units"]) if "units" in d.files else None
    if units is None:
        # Legacy Go1 record: raw Vicon millimetres. A Go2 base sits at ~0.23 m, so a
        # median |z| in the hundreds can only be mm.
        zmed = np.nanmedian(np.abs(xyz[:, 2]))
        units = "mm" if zmed > 20.0 else "m"
        print(f"mocap file has no units key; inferred {units} (median |z| {zmed:.1f})")
    if units == "mm":
        xyz = xyz * 1e-3
    t = np.asarray(d["t"], float) if "t" in d.files else None
    if t is None:                                  # very old record: nominal 120 Hz
        print("mocap file has no timestamps; assuming 120 Hz")
        t = np.arange(len(xyz)) / 120.0
    t = t - t[0]
    n_bad = int(np.isnan(xyz).any(axis=1).sum())
    fps = 1.0 / np.median(np.diff(t))
    print(f"mocap: {len(t)} samples, {t[-1]:.2f} s, {fps:.1f} Hz median, "
          f"{n_bad} occluded")
    return t, xyz


def is_deploy_npz(path):
    """deploy_meta trace vs Vicon record, told apart by content rather than by
    filename -- the two live side by side in a run directory once the mocap file has
    been copied in, and nothing enforces what it is called."""
    try:
        with np.load(path, allow_pickle=True) as d:
            return "odom_xy" in d.files
    except Exception:                                  # noqa: BLE001
        return False


def newest_run(outdir="data"):
    """Newest deploy trace, looking both at outdir/*.npz (the old flat layout) and
    outdir/<run>/*.npz (the per-run directories deploy_meta writes now)."""
    files = glob.glob(os.path.join(outdir, "*.npz")) + \
        glob.glob(os.path.join(outdir, "*", "*.npz"))
    files = [f for f in sorted(files, key=os.path.getmtime) if is_deploy_npz(f)]
    return files[-1] if files else None


def split_rundir(d):
    """(deploy npz, mocap npz) inside a run directory. Either may be None."""
    npzs = sorted(glob.glob(os.path.join(d, "*.npz")), key=os.path.getmtime)
    dep = [f for f in npzs if is_deploy_npz(f)]
    moc = [f for f in npzs if not is_deploy_npz(f)]
    if len(moc) > 1:
        print(f"  note: {len(moc)} non-deploy npz files in {d}; using the newest "
              f"({os.path.basename(moc[-1])}) -- pass --mocap to pick another")
    return (dep[-1] if dep else None), (moc[-1] if moc else None)


# ------------------------------------------------------------------- alignment
def _uniform(t, v, dt):
    """Resample onto a uniform grid, interpolating across occlusion NaNs. Only the
    sync uses this -- the plotted mocap keeps its gaps."""
    grid = np.arange(t[0], t[-1], dt)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        raise SystemExit("ABORT: mocap z is entirely occluded")
    return grid, np.interp(grid, t[ok], v[ok])


MIN_AMP_FRAC = 0.25            # window must carry >=25% of the plan's z variability


def find_t0(t_moc, z_moc, t_ref, z_ref):
    """Wall-clock time in the recording at which plan t=0 lands, by sliding the plan's
    z-profile over the whole record and scoring normalized correlation.

    Returns (t0, r, candidates). r is the Pearson coefficient at the winning offset --
    read it: a hopscotch launch against a clean record scores > 0.8, and anything under
    ~0.5 means the sync is guesswork (wrong subject? robot never launched? pass --t0).

    NORMALIZED CORRELATION ALONE IS NOT ENOUGH. Dividing by the window's own deviation
    means a MOTIONLESS window is scored on its noise, and sensor noise can out-correlate
    a real-but-imperfect hop -- so on a record with no hopscotch in it (a drive-around,
    say) the sync happily locks onto the robot lying still. Candidate offsets whose z
    variability is a small fraction of the plan's are therefore rejected outright: a
    window that never moved cannot be the window the robot jumped in."""
    dt = float(np.median(np.diff(t_moc)))
    grid, zm = _uniform(t_moc, z_moc, dt)
    zr = np.interp(np.arange(t_ref[0], t_ref[-1], dt), t_ref, z_ref)
    n, m = len(zm), len(zr)
    if n < m:
        raise SystemExit(f"ABORT: recording ({n * dt:.1f} s) is shorter than the plan "
                         f"({m * dt:.1f} s) -- nothing to sync against")

    b = zr - zr.mean()
    nb = float(np.linalg.norm(b))
    dot = np.correlate(zm, b, mode="valid")        # zero-mean template => mean-free
    c1 = np.concatenate([[0.0], np.cumsum(zm)])
    c2 = np.concatenate([[0.0], np.cumsum(zm * zm)])
    s1 = c1[m:] - c1[:-m]
    s2 = c2[m:] - c2[:-m]
    na = np.sqrt(np.maximum(s2 - s1 * s1 / m, 0.0))
    r = dot / np.maximum(na * nb, 1e-12)

    live = na >= MIN_AMP_FRAC * nb
    if live.any():
        r_eff = np.where(live, r, -np.inf)
    else:
        print(f"  WARNING: NO {len(grid) * dt:.0f}s window in this recording moves in z "
              f"like the plan does (best is {100.0 * na.max() / nb:.0f}% of the plan's "
              f"{np.ptp(zr):.3f} m swing). This record probably does not contain a "
              f"hopscotch run at all -- scoring on noise.")
        r_eff = r

    k = int(np.argmax(r_eff))
    # well-separated runners-up: what you pass to --t0 when the winner is wrong
    cands, order = [], np.argsort(r_eff)[::-1]
    for i in order[:5000]:
        if not np.isfinite(r_eff[i]):
            break
        if all(abs(grid[i] - g) > 1.0 for g, _, _ in cands):
            cands.append((float(grid[i]), float(r[i]), float(na[i] / nb)))
        if len(cands) == 3:
            break
    return float(grid[k]), float(r[k]), cands


def fit_yaw(xy_moc, xy_ref):
    """Best-fit planar rotation taking mocap horizontal path -> plan horizontal path
    (Procrustes, rotation only). Both inputs must already be on a common time grid."""
    a = xy_moc - xy_moc.mean(0)
    b = xy_ref - xy_ref.mean(0)
    num = float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))
    den = float(np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]))
    return np.arctan2(num, den)


def rot_z(xy, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]], axis=1)


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None,
                    help="a deploy_meta run directory with the Vicon record copied in: "
                         "reads both npz files from it and writes the plots back into "
                         "it. Overrides --mocap/--run/--prefix unless those are given.")
    ap.add_argument("--mocap", default=None, help="record_floating_base.py npz "
                                                  "(default: xyz_go2.npz, or from --dir)")
    ap.add_argument("--traj", default=DEFAULT_TRAJ, help="reference (.json modes or .npz)")
    ap.add_argument("--run", default=None,
                    help="deploy_meta.py run npz for the odometry overlay "
                         "(default: newest under --outdir, or from --dir)")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--t0", type=float, default=None,
                    help="seconds into the recording where plan t=0 lands "
                         "(default: auto by z cross-correlation)")
    ap.add_argument("--yaw", type=float, default=None,
                    help="mocap->plan yaw in DEGREES (default: best fit)")
    ap.add_argument("--settle", type=float, default=0.2,
                    help="seconds at plan t=0 averaged to zero both signals")
    ap.add_argument("--prefix", default="", help="output png prefix")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    # --dir: everything for this run lives in one directory. Explicit flags still win,
    # so --dir X --mocap Y is a valid way to plot a record kept somewhere else.
    if args.dir:
        if not os.path.isdir(args.dir):
            raise SystemExit(f"ABORT: {args.dir} is not a directory")
        dep, moc = split_rundir(args.dir)
        if moc is None and args.mocap is None:
            raise SystemExit(f"ABORT: no Vicon record in {args.dir} -- copy the "
                             f"record_floating_base.py npz in there first "
                             f"(a deploy trace alone has no ground-truth base pose)")
        args.mocap = args.mocap or moc
        args.run = args.run or dep
        args.prefix = args.prefix or os.path.join(args.dir, "")
        print(f"run dir: {args.dir}")
    if args.mocap is None:
        args.mocap = "xyz_go2.npz"

    t_ref, ref_xyz, airborne = load_reference(args.traj)
    dur = t_ref[-1]
    print(f"plan: {len(t_ref)} knots @ {t_ref[1] - t_ref[0]:.4f} s, {dur:.2f} s")

    t_moc, moc_xyz = load_mocap(args.mocap)

    run = None
    run_path = args.run or newest_run(args.outdir)
    if run_path and os.path.exists(run_path):
        run = np.load(run_path, allow_pickle=True)
        print(f"run trace: {run_path} ({len(run['t'])} ticks, "
              f"{'ABORTED' if bool(run['aborted']) else 'completed'})")
        # The outputs belong WITH the trace they describe. Without this the plots land
        # in the cwd and the next run silently overwrites them, which is how a haul of
        # runs ends up with exactly one set of plots.
        if not args.prefix:
            args.prefix = os.path.join(os.path.dirname(run_path), "")
            print(f"  -> writing plots + summary into {args.prefix}")
    else:
        print("no deploy_meta run trace found -- plotting plan vs mocap only")

    # ---- time sync -------------------------------------------------------
    if args.t0 is not None:
        t0, r = args.t0, float("nan")
        print(f"sync: t0 = {t0:.3f} s (manual)")
    else:
        t0, r, cands = find_t0(t_moc, moc_xyz[:, 2], t_ref, ref_xyz[:, 2])
        print(f"sync: t0 = {t0:.3f} s into the recording, correlation r = {r:.3f}")
        if r < 0.5:
            print("  WARNING: weak sync -- the window below is probably NOT the run.")
            print("  best candidates (t0, r, z-amplitude vs plan):")
            for g, rr, amp in cands:
                print(f"    --t0 {g:8.3f}   r {rr:+.3f}   amp {100 * amp:5.0f}%")
            print("  If none look right, the recording likely does not contain a "
                  "deploy_meta run.")
    t_al = t_moc - t0                              # mocap on the plan's clock

    # ---- overlap window + frame alignment --------------------------------
    win = (t_al >= 0.0) & (t_al <= dur) & np.isfinite(moc_xyz).all(axis=1)
    if win.sum() < 10:
        raise SystemExit("ABORT: fewer than 10 valid mocap samples inside the plan "
                         "window -- the sync or the record is wrong")
    ref_on_moc = np.stack([np.interp(t_al[win], t_ref, ref_xyz[:, i]) for i in range(3)],
                          axis=1)

    travel = np.linalg.norm(moc_xyz[win][-1, :2] - moc_xyz[win][0, :2])
    if args.yaw is not None:
        yaw = np.radians(args.yaw)
        print(f"frame: yaw = {np.degrees(yaw):+.1f} deg (manual)")
    elif travel < 0.05:
        yaw = 0.0
        print(f"frame: horizontal travel only {travel:.3f} m -- yaw fit would be "
              f"noise, using 0 deg")
    else:
        yaw = fit_yaw(moc_xyz[win][:, :2], ref_on_moc[:, :2])
        print(f"frame: yaw = {np.degrees(yaw):+.1f} deg (best fit, {travel:.3f} m of "
              f"horizontal travel)")

    moc = np.empty_like(moc_xyz)
    moc[:, :2] = rot_z(moc_xyz[:, :2], yaw)
    moc[:, 2] = moc_xyz[:, 2]

    # Zero both on the pre-launch settle: the marker cluster sits above the body
    # origin the plan integrates, so only DELTAS are comparable.
    s_moc = win & (t_al <= args.settle)
    s_ref = t_ref <= args.settle
    if s_moc.sum() < 2:
        s_moc = win & (t_al <= t_al[win][0] + args.settle)
    moc = moc - moc[s_moc].mean(0) + ref_xyz[s_ref].mean(0)

    # ---- plots -----------------------------------------------------------
    spans = []
    i = 0
    while i < len(airborne):                       # planned flight windows, for shading
        if airborne[i]:
            j = i
            while j < len(airborne) and airborne[j]:
                j += 1
            spans.append((t_ref[i], t_ref[min(j, len(t_ref) - 1)]))
            i = j
        else:
            i += 1

    def shade(ax):
        for s0, s1 in spans:
            ax.axvspan(s0, s1, color="0.88", lw=0, zorder=0)

    p = args.prefix
    fig1, ax1 = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    for i, lbl in enumerate("xyz"):
        shade(ax1[i])
        ax1[i].plot(t_ref, ref_xyz[:, i], c="C0", lw=1.5)
        ax1[i].set_ylabel(f"{lbl} [m]")
    ax1[2].set_xlabel("t [s]  (shaded = planned flight)")
    fig1.suptitle(f"Plan — base position   ({os.path.basename(args.traj)})")
    fig1.tight_layout()
    fig1.savefig(p + "plan_base_pos.png", dpi=130)

    # The FULL record, on the recording clock, with the synced plan window marked.
    # Deliberately not cropped to the window: when the sync is wrong, a cropped view
    # shows a flat line and reads as "the mocap is dead", when what actually happened
    # is that the window landed on a motionless stretch. Here you see both at once.
    fig2, ax2 = plt.subplots(3, 1, sharex=True, figsize=(11, 8))
    for i, lbl in enumerate("xyz"):
        ax2[i].axvspan(t0, t0 + dur, color="C0", alpha=0.15, lw=0, zorder=0,
                       label="synced plan window" if i == 0 else None)
        ax2[i].plot(t_moc, moc[:, i], c="C1", lw=1.2)
        ax2[i].set_ylabel(f"{lbl} [m]")
    ax2[0].legend(fontsize=8, loc="upper left")
    ax2[2].set_xlabel("t [s]  (recording clock)")
    fig2.suptitle(f"Vicon — full record (yaw {np.degrees(yaw):+.1f}°, "
                  f"plan window {t0:.2f}–{t0 + dur:.2f} s"
                  + (f", sync r={r:.2f}" if np.isfinite(r) else "") + ")")
    fig2.tight_layout()
    fig2.savefig(p + "mocap_base_pos.png", dpi=130)

    fig3, ax3 = plt.subplots(3, 1, sharex=True, figsize=(11, 9))
    for i, lbl in enumerate("xyz"):
        shade(ax3[i])
        ax3[i].plot(t_ref, ref_xyz[:, i], "--", c="0.35", lw=1.5, label="plan")
        ax3[i].plot(t_al, moc[:, i], c="C1", lw=1.6, label="vicon")
        if run is not None and i < 2:
            ax3[i].plot(run["t"], run["odom_xy"][:, i], c="C2", lw=1.2, alpha=0.85,
                        label="velest odom")
        ax3[i].set_ylabel(f"{lbl} [m]")
        ax3[i].set_xlim(-0.2, dur + 0.2)
    ax3[0].legend(fontsize=8, loc="upper left", ncol=3)
    ax3[2].set_xlabel("t [s]  (shaded = planned flight)")
    tag = os.path.basename(run_path) if run is not None else os.path.basename(args.mocap)
    fig3.suptitle(f"Plan vs Vicon — base position   {tag}"
                  + (f"   (sync r={r:.2f})" if np.isfinite(r) else ""))
    fig3.tight_layout()
    fig3.savefig(p + "overlay_base_pos.png", dpi=130)

    # ---- numbers ---------------------------------------------------------
    # Accumulated rather than printed straight out, so the same text lands in the run
    # directory: a png you cannot reread the numbers off is half an artifact.
    mz = moc[win, 2]
    rz = ref_xyz[:, 2]
    out = [f"mocap        : {os.path.abspath(args.mocap)}",
           f"run trace    : {os.path.abspath(run_path) if run is not None else '(none)'}",
           f"plan         : {os.path.abspath(args.traj)}",
           f"sync         : t0 {t0:.3f} s into the recording"
           + (f", correlation r {r:.3f}" if np.isfinite(r) else " (manual)"),
           f"frame        : yaw {np.degrees(yaw):+.1f} deg, "
           f"zeroed on the first {args.settle:.1f} s",
           "",
           f"{'':>16}{'apex z':>10}{'gain':>10}{'fwd travel':>12}",
           f"{'plan':>16}{rz.max():9.4f}m{rz.max() - rz[0]:9.4f}m"
           f"{ref_xyz[-1, 0] - ref_xyz[0, 0]:11.4f}m",
           f"{'vicon':>16}{mz.max():9.4f}m{mz.max() - mz[0]:9.4f}m"
           f"{moc[win][-1, 0] - moc[win][0, 0]:11.4f}m",
           "",
           # per-flight apex: whether each hop cleared what it planned to
           f"{'flight window':>16}{'plan apex':>12}{'vicon apex':>12}{'delta':>10}"]
    for s0, s1 in spans:
        if s1 - s0 < 0.05:
            continue
        rm = (t_ref >= s0) & (t_ref <= s1)
        mm = win & (t_al >= s0) & (t_al <= s1)
        if not mm.any():
            continue
        pa = rz[rm].max() - rz[0]
        va = moc[mm, 2].max() - mz[0]
        out.append(f"{f'{s0:.2f}-{s1:.2f}s':>16}{pa:11.4f}m{va:11.4f}m{va - pa:+9.4f}m")

    err = moc[win] - ref_on_moc                    # moc is already rotated + zeroed
    out += ["", f"{'axis':>7}{'RMS err':>11}{'max|err|':>11}{'final':>10}"]
    for i, lbl in enumerate("xyz"):
        e = err[:, i]
        out.append(f"{lbl:>7}{np.sqrt((e ** 2).mean()):10.4f}m{np.abs(e).max():10.4f}m"
                   f"{e[-1]:9.4f}m")

    report = "\n".join(out)
    print("\n" + report)
    with open(p + "vicon_base_pos.txt", "w") as f:
        f.write(report + "\n")

    print(f"\nwrote {p}plan_base_pos.png, {p}mocap_base_pos.png, "
          f"{p}overlay_base_pos.png, {p}vicon_base_pos.txt")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
