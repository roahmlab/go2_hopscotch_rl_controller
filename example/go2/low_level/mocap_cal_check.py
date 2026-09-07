"""Bench test for mocap calibration: runs go2_cartwheel0b's own bridge and CalibrateMocap code with no robot, and says why it passes or fails."""
import socket
import struct
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).with_name("go2_cartwheel0b.py").read_text(newline="")
ns = {"np": np, "time": time, "socket": socket, "struct": struct}
exec("\n".join(l for l in SRC.splitlines() if l.startswith("MOCAP_")), ns)
exec(SRC[SRC.index("def quat2eul"):SRC.index("class Custom:")], ns)
exec(textwrap.dedent(SRC[SRC.index("    def InitMocap"):SRC.index("    def Start(")]), ns)
exec(textwrap.dedent(SRC[SRC.index("    def _cal_append"):SRC.index("    def MocapPosition")]), ns)
PORT, MAGIC, PKT = ns["MOCAP_UDP_PORT"], ns["MOCAP_MAGIC"], ns["MOCAP_STRUCT"]


class Stub:
    pass


for name in ("InitMocap", "DrainMocap", "_cal_append", "CalibrateMocap"):
    setattr(Stub, name, ns[name])

# --- 0. raw datagram audit: DrainMocap silently drops wrong-size / wrong-magic packets ------
raw = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
raw.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
raw.bind(("0.0.0.0", PORT))
raw.settimeout(3.0)
sizes, magics, n = {}, {}, 0
try:
    while n < 40:
        buf = raw.recv(512)
        n += 1
        sizes[len(buf)] = sizes.get(len(buf), 0) + 1
        if len(buf) >= 4:
            m = struct.unpack("<I", buf[:4])[0]
            magics[m == MAGIC] = magics.get(m == MAGIC, 0) + 1
except socket.timeout:
    pass
raw.close()
print(f"0. datagrams on udp/{PORT}: {n} received in <=3 s")
print(f"   sizes seen {sizes}   expected {PKT.size}   magic ok/bad {magics}")
if n == 0:
    sys.exit("   -> nothing arriving: bridge not running, wrong port, or firewall")
if PKT.size not in sizes or not magics.get(True):
    sys.exit("   -> packets are the WRONG FORMAT: DrainMocap drops every one, so the buffer never fills")

# --- 1. the real InitMocap / DrainMocap / _cal_append, exactly as the controller runs them --
s = Stub()
s.bench, s.mocap_sample, s.mocap_cal = False, None, []
s.mocap_repeat, s.mocap_zero, s.mocap_R, s.mocap_last_fnum = 0, None, None, None
s.InitMocap()
seen = []
t0 = time.perf_counter()
while time.perf_counter() - t0 < 1.5:
    if s.DrainMocap() is not None:
        seen.append(s.mocap_sample)
    s._cal_append()
    time.sleep(0.010)

# --- 2. what the buffer contains -----------------------------------------------------------
fn = np.array([x[1] for x in seen])
pos = np.stack([x[2] for x in seen])
quats = [x[3] for x in seen]
qok = [q for q in quats if q is not None and np.isfinite(np.asarray(q, float)).all()]
print(f"\n1. buffer: {len(s.mocap_cal)} unique-frame samples kept by _cal_append over 1.5 s "
      f"({len(seen)} drains), frame numbers {fn.min()}..{fn.max()}, "
      f"strictly increasing: {bool((np.diff(fn) > 0).all())}, repeats: {int((np.diff(fn) == 0).sum())}")
print(f"   position median {np.round(np.median(pos, 0), 4)} m, spread (std) {np.round(1e3 * pos.std(0), 2)} mm")
print(f"   quaternion: {len(qok)}/{len(quats)} finite; "
      + (f"|q| min {min(np.linalg.norm(q) for q in qok):.3f} max {max(np.linalg.norm(q) for q in qok):.3f}"
         if qok else "NONE -- bridge is not sending orientation"))
if qok:
    qm = np.mean(np.stack(qok), 0)
    nrm = np.linalg.norm(qm)
    if nrm < 0.5:
        print("   !! near-zero quaternion: CalibrateMocap would PASS with a NaN rotation (known bug)")
    qm = qm / max(nrm, 1e-12)
    for label, q in (("wxyz", qm), ("xyzw", np.roll(qm, 1))):
        deg = np.degrees(ns["quat2eul"](q))
        print(f"   as {label}: roll {deg[0]:+7.2f}  pitch {deg[1]:+7.2f}  yaw {deg[2]:+7.2f} deg")
    print("   calibration picks the order that reads more level, then needs |roll|,|pitch| <= 10 and |yaw| <= 15")

# --- 3. the verdict, from the shipped function ---------------------------------------------
print("\n2. CalibrateMocap():")
ok = s.CalibrateMocap()
print(f"   -> {'PASS' if ok else 'FAIL'}")
if ok:
    print(f"   zero {np.round(s.mocap_zero, 4)}   R {np.round(s.mocap_R, 4).tolist()}")
    if not np.isfinite(s.mocap_R).all():
        print("   !! mocap_R is NaN: every p_meas will be NaN and the actor will see a zeroed channel")
