"""Vicon floating-base recorder and live UDP bridge for Go2 hopscotch runs.

Streams the Go2 subject's root-segment pose off the Vicon datastream and writes an
npz that plot_base_pos.py syncs against a deploy_meta.py run trace. This is the ONLY
ground-truth base measurement in the loop -- deploy_meta's `odom_xy` is velest
odometry (dead-reckoned, xy only) and its z is not observed at all.

With --udp it ALSO forwards every fresh frame as a datagram to go2_hopscotch1.py.
That process must never call into this SDK itself: get_frame() blocks ~9 ms per call
(measured 109-115 calls/s against a 120 Hz camera) whichever stream mode is set, and
in-process that starves the 100 Hz control thread through the GIL. Here the block is
harmless -- separate process, separate GIL -- and the controller only ever does a
non-blocking recvfrom. One run of this script does both jobs: live feed + full-rate
recording that is frame-number aligned with what the controller consumed.

WHAT CHANGED FROM THE Go1 VERSION (each one silently corrupted a Go2 record):

1. OCCLUSION IS NOT AN ERROR. A hopscotch flight phase routinely drops markers.
   The old loop appended whatever the SDK returned; a None turns the whole array
   into dtype=object and np.savez then writes garbage. Occlusions are now stored as
   NaN so the SAMPLE TIMING survives the dropout, and the count is reported.

2. PREFETCH REPEATS FRAMES. ClientPullPreFetch hands back the newest frame whenever
   you ask, and this loop spins far faster than the camera rate -- so the old record
   held every sample 5-20x over, which quietly biased the resampling in the plotter.
   Frames are now de-duplicated by the SDK's frame number.

3. WALL-CLOCK, NOT AN ASSUMED RATE. The plotter used to hardcode 120 Hz. `t` is
   authoritative here; nothing downstream assumes a nominal rate.

4. UNITS ARE METRES ON DISK. Vicon streams mm. The old file stored mm and relied on
   the plotter to remember to divide -- a silent 1000x if either side drifted.

5. THE DATA SURVIVES A Ctrl-C. Saving happens in a finally block, so an aborted run
   (the case you most want to look at) still lands on disk.

Usage:
    python3 record_floating_base.py                       # subject "go2", default host
    python3 record_floating_base.py --subject go2_body --out runs/mocap_cshape999.npz
"""
import argparse
import os
import socket
import struct
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pyvicon_datastream as pv


DEFAULT_HOST = "192.168.0.149"
DEFAULT_SUBJECT = "go2"
MM_TO_M = 1e-3
POLL_SLEEP = 5e-4                # idle-poll pacing; see the de-dup block in main()
# Wire format shared with go2_hopscotch1.py: magic, frame no., monotonic send stamp,
# xyz metres, quaternion as Tracker delivers it (the consumer resolves the ordering).
UDP_MAGIC = 0x56424731
UDP_STRUCT = struct.Struct("<Iqdddddddd")   # magic, frame, t_send, xyz, quat4 -> 76 B
DEFAULT_UDP = "127.0.0.1:9870"


stop_flag = False


def wait_for_key():
    global stop_flag
    input("Recording... press Enter to stop.\n")
    stop_flag = True


def list_subjects(vicon):
    """Best-effort subject enumeration -- the wrapper does not expose it on every
    build, so a missing method degrades to 'unknown' rather than killing the run."""
    try:
        return [str(vicon.get_subject_name(i)) for i in range(vicon.get_subject_count())]
    except Exception:                                        # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST, help="Vicon Tracker host")
    ap.add_argument("--subject", default=DEFAULT_SUBJECT,
                    help="subject name in Tracker (the Go1 rig used a different one)")
    ap.add_argument("--out", default="xyz_go2.npz", help="output npz")
    ap.add_argument("--duration", type=float, default=None,
                    help="auto-stop after this many seconds (default: run until Enter)")
    ap.add_argument("--no-plot", action="store_true", help="skip the quicklook png")
    ap.add_argument("--udp", nargs="?", const=DEFAULT_UDP, default=None,
                    metavar="HOST:PORT",
                    help=f"also forward each frame live to the controller (default {DEFAULT_UDP})")
    args = ap.parse_args()

    sock = udp_addr = None
    if args.udp:
        host_s, _, port_s = args.udp.rpartition(":")
        udp_addr = (host_s or "127.0.0.1", int(port_s))
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
        print(f"udp bridge -> {udp_addr[0]}:{udp_addr[1]}")

    vicon = pv.PyViconDatastream()
    # connect() returns Result on some wrapper builds and None/bool on others, so an
    # explicit != Success test is not portable. Treat only an explicit non-Success
    # Result as fatal here; the frame poll below is the real connectivity check.
    res = vicon.connect(args.host)
    if isinstance(res, pv.Result) and res != pv.Result.Success:
        raise SystemExit(f"ABORT: cannot connect to Vicon at {args.host} ({res})")
    vicon.set_stream_mode(pv.StreamMode.ClientPullPreFetch)
    vicon.enable_segment_data()

    # Subject-name check BEFORE the robot moves: a stale name is the Go1->Go2 trap,
    # and it fails as an empty recording 20 s later instead of here.
    for _ in range(300):
        if vicon.get_frame() == pv.Result.Success:
            break
        time.sleep(0.01)
    else:
        raise SystemExit(f"ABORT: connected to {args.host} but no frame in 3 s -- "
                         f"is Tracker streaming?")
    names = list_subjects(vicon)
    if names is not None:
        print(f"vicon subjects: {names}")
        if args.subject not in names:
            raise SystemExit(f"ABORT: subject {args.subject!r} not in {names}. "
                             f"Pass --subject with the name Tracker shows.")
    else:
        print("vicon subject enumeration unavailable; trusting --subject")

    seg = vicon.get_subject_root_segment_name(args.subject)
    if not seg:
        raise SystemExit(f"ABORT: no root segment for subject {args.subject!r}")
    print(f"recording {args.subject}/{seg} from {args.host}")

    t_arr, iso_arr, pos_arr, quat_arr, fnum_arr = [], [], [], [], []
    n_occluded = n_gap = n_sent = n_send_err = 0
    last_fnum = None
    has_fnum = hasattr(vicon, "get_frame_number")

    threading.Thread(target=wait_for_key, daemon=True).start()
    t_start = time.time()

    try:
        while not stop_flag:
            if args.duration is not None and time.time() - t_start >= args.duration:
                break
            if vicon.get_frame() != pv.Result.Success:
                time.sleep(POLL_SLEEP)
                continue

            # De-dup: prefetch re-serves the newest frame on every pull, so without
            # this the record is mostly repeats of a handful of real samples.
            #
            # The POLL_SLEEP on the duplicate path is what keeps this loop off a full
            # core. A prefetch pull costs microseconds, so an unthrottled spin polls
            # ~100k times a second for a ~120 Hz source -- all of it discarded here.
            # At 0.5 ms we still poll ~16x per camera frame (measured capture: 118.7 Hz
            # against a 120 Hz stream), so nothing is missed, and the timestamp is late
            # by at most 0.5 ms -- two orders below what the offline sync resolves.
            if has_fnum:
                try:
                    fnum = int(vicon.get_frame_number())
                except Exception:                            # noqa: BLE001
                    has_fnum, fnum = False, len(t_arr)
                else:
                    if fnum == last_fnum:
                        time.sleep(POLL_SLEEP)
                        continue
                    if last_fnum is not None and fnum > last_fnum + 1:
                        n_gap += fnum - last_fnum - 1
                    last_fnum = fnum
            else:
                fnum = len(t_arr)
                time.sleep(POLL_SLEEP)                       # no de-dup: pace by hand

            pos = vicon.get_segment_global_translation(args.subject, seg)
            rot = vicon.get_segment_global_quaternion(args.subject, seg)
            now = time.time()

            # Occluded -> None from the wrapper, or an exact all-zero triple from
            # Tracker. Both mean "no measurement"; neither means "the base is at 0".
            p = np.full(3, np.nan)
            if pos is not None:
                p = np.asarray(pos, float).ravel()[:3] * MM_TO_M
                if not np.any(p):
                    p = np.full(3, np.nan)
            q = np.full(4, np.nan)
            if rot is not None:
                q = np.asarray(rot, float).ravel()[:4]
            if np.isnan(p).any():
                n_occluded += 1

            # Forward before appending: the live consumer is latency-critical, the
            # recording is not. Occluded frames are recorded but never sent.
            if sock is not None and not np.isnan(p).any():
                qs = np.where(np.isfinite(q), q, 0.0)
                try:
                    sock.sendto(UDP_STRUCT.pack(UDP_MAGIC, int(fnum),
                                                time.monotonic(), *p, *qs), udp_addr)
                    n_sent += 1
                except OSError:
                    n_send_err += 1

            t_arr.append(now)
            iso_arr.append(datetime.fromtimestamp(now, tz=timezone.utc)
                           .astimezone().isoformat())
            pos_arr.append(p)
            quat_arr.append(q)
            fnum_arr.append(fnum)

            if len(t_arr) % 200 == 0:
                rate = len(t_arr) / max(now - t_start, 1e-9)
                print(f"[{now - t_start:6.2f}s] n={len(t_arr):6d} {rate:5.1f} Hz  "
                      f"xyz {np.round(p, 4)}  occluded {n_occluded}"
                      + (f"  sent {n_sent}" + (f" ERR {n_send_err}" if n_send_err else "")
                         if sock is not None else ""))
    except KeyboardInterrupt:
        print("\nCtrl-C -- saving what was captured")
    finally:
        save(args, t_arr, iso_arr, pos_arr, quat_arr, fnum_arr, n_occluded, seg, n_gap)


def save(args, t_arr, iso_arr, pos_arr, quat_arr, fnum_arr, n_occluded, seg, n_gap=0):
    if len(t_arr) < 2:
        print("nothing recorded -- no file written")
        return
    t = np.asarray(t_arr, float)
    xyz = np.asarray(pos_arr, float)                         # (N,3) metres
    quat = np.asarray(quat_arr, float)                       # (N,4) Vicon xyzw
    dt = np.diff(t)
    fps = 1.0 / np.median(dt) if len(dt) else float("nan")

    outdir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(outdir, exist_ok=True)
    np.savez(args.out,
             t=t, x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], xyz=xyz,
             quat_xyzw=quat, frame=np.asarray(fnum_arr, np.int64),
             iso=np.asarray(iso_arr), units="m", subject=args.subject,
             segment=str(seg), host=args.host, fps=fps)

    print(f"\nsaved {args.out}: {len(t)} samples, {t[-1] - t[0]:.2f} s, "
          f"{fps:.1f} Hz median ({n_occluded} occluded)")
    valid = ~np.isnan(xyz[:, 2])
    if valid.any():
        z = xyz[valid, 2]
        print(f"  z: {z.min():.4f} .. {z.max():.4f} m  "
              f"(range {z.max() - z.min():.4f} m)")
        xy = xyz[valid][:, :2]
        print(f"  horizontal travel: {np.linalg.norm(xy[-1] - xy[0]):.4f} m")
    if n_occluded:
        print(f"  WARNING: {100.0 * n_occluded / len(t):.1f}% of samples occluded "
              f"(stored as NaN)")
    if n_gap:
        # Camera frames the poll never saw. A handful is nothing; a large count means
        # the poll is being starved and the record is undersampling the stream.
        print(f"  WARNING: {n_gap} camera frames skipped by the poll "
              f"({100.0 * n_gap / (n_gap + len(t)):.1f}% of the stream)")

    if args.no_plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                   # noqa: BLE001
        print(f"matplotlib unavailable ({type(e).__name__}); npz only")
        return
    tt = t - t[0]
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
    for i, lbl in enumerate("xyz"):
        axs[i].plot(tt, xyz[:, i], lw=1.2)
        axs[i].set_ylabel(f"{lbl} [m]")
    axs[2].set_xlabel("t [s] (wall clock since record start)")
    fig.suptitle(f"Vicon {args.subject}/{seg} -- raw record "
                 f"({fps:.0f} Hz, {n_occluded} occluded)")
    fig.tight_layout()
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=130)
    print(f"  quicklook: {png}")


if __name__ == "__main__":
    sys.exit(main())
