"""Passive foot-force sensor check (no commands sent, safe to run anytime).

Streams the raw foot_force channels at 5 Hz in both raw index order and the
FL/FR/RL/RR order used by go2_hopscotch_est_v18.py (FOOTFORCE_FROM_ISO_FOOT).

Test procedure:
  1. Run this, hold/hang the robot so all feet are off the ground.
     -> healthy channels read a small stable value (~5-15); note each floor.
  2. Press each physical foot firmly, one at a time, and watch which channel
     jumps. Confirms the FL/FR/RL/RR mapping AND the sensor response range
     (healthy: pressing hard should read 80+; a channel that barely moves or
     whose floor wanders by 10+ units is a bad sensor).

Usage: python foot_force_monitor.py [iface]
"""
import sys
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

FOOTFORCE_FROM_ISO_FOOT = [1, 0, 3, 2]           # FL FR RL RR <- (FR FL RR RL)

latest = {"ff": None}


def handler(msg: LowState_):
    latest["ff"] = np.array(msg.foot_force[:4], dtype=float)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(handler, 10)

    print("Streaming raw foot_force (Ctrl-C to stop).")
    print(f"{'raw [0 1 2 3]':>20}   {'FL':>5} {'FR':>5} {'RL':>5} {'RR':>5}")
    ff_min = np.full(4, np.inf)
    ff_max = np.full(4, -np.inf)
    try:
        while True:
            time.sleep(0.2)
            ff = latest["ff"]
            if ff is None:
                print("waiting for rt/lowstate ...")
                continue
            iso = ff[FOOTFORCE_FROM_ISO_FOOT]
            ff_min = np.minimum(ff_min, ff)
            ff_max = np.maximum(ff_max, ff)
            print(f"{np.array2string(ff.astype(int)):>20}   "
                  f"{iso[0]:5.0f} {iso[1]:5.0f} {iso[2]:5.0f} {iso[3]:5.0f}")
    except KeyboardInterrupt:
        iso_min = ff_min[FOOTFORCE_FROM_ISO_FOOT]
        iso_max = ff_max[FOOTFORCE_FROM_ISO_FOOT]
        print("\nsession min (FL FR RL RR):", np.round(iso_min, 1))
        print("session max (FL FR RL RR):", np.round(iso_max, 1))
