import sys
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_


duration = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
if len(sys.argv) > 1:
    ChannelFactoryInitialize(0, sys.argv[1])
else:
    ChannelFactoryInitialize(0)

samples = []


def handle_lowstate(msg: LowState_):
    samples.append((time.perf_counter(),
                    np.asarray(msg.imu_state.accelerometer, dtype=np.float64),
                    np.asarray(msg.imu_state.quaternion, dtype=np.float64)))


subscriber = ChannelSubscriber("rt/lowstate", LowState_)
subscriber.Init(handle_lowstate, 10)

print(f"collecting IMU data for {duration:g} seconds; keep the robot stationary ...", flush=True)
time.sleep(duration)

if not samples:
    raise RuntimeError("no rt/lowstate messages received")

captured = list(samples)
times = np.asarray([sample[0] for sample in captured])
accel = np.stack([sample[1] for sample in captured])
quat = np.stack([sample[2] for sample in captured])
valid = np.isfinite(accel).all(axis=1) & np.isfinite(quat).all(axis=1)
valid &= np.linalg.norm(quat, axis=1) > 1e-6
accel = accel[valid]
quat = quat[valid]

if not len(accel):
    raise RuntimeError("no finite IMU samples received")

quat /= np.linalg.norm(quat, axis=1, keepdims=True)
w, x, y, z = quat.T
expected = 9.81 * np.column_stack([
    2.0 * (x * z - w * y),
    2.0 * (y * z + w * x),
    1.0 - 2.0 * (x * x + y * y),
])
error = accel - expected
rate = (len(captured) - 1) / (times[-1] - times[0]) if len(captured) > 1 else 0.0

np.set_printoptions(precision=4, suppress=True)
print(f"samples: {len(captured)} ({rate:.1f} Hz), finite: {len(accel)}")
print(f"accel mean:       {accel.mean(axis=0)} m/s^2")
print(f"accel std:        {accel.std(axis=0)} m/s^2")
print(f"mean magnitude:   {np.linalg.norm(accel, axis=1).mean():.4f} m/s^2")
print(f"expected mean:    {expected.mean(axis=0)} m/s^2")
print(f"expected RMSE:    {np.sqrt(np.mean(error ** 2, axis=0))} m/s^2")
