"""deploy_cmd.py -- meta-driven HW deploy for the CMD-family checkpoints
(obs_tau_mode "cmd"/"zero", 2026-08-07 line: mx_AtauCmd / mx_A2tauCmd / mx_Afree /
mx_A2free / mx_tauzero). deploy_meta.py stays as-is for the legacy mdc family;
this file subclasses it and changes exactly one obs channel, plus adds electrical
state logging (battery voltage/current/SOC + motor temperatures) for the
day-effect / Vbat_eff-calibration arc.

THE ONE OBS CHANGE. The actor's 12 torque dims per history frame:
  mdc  (deploy_meta.py): tau_applied = MDC(PD+ff) at fixed nominal Vbat -- the
       channel that LIES whenever the pack isn't fresh (2026-08-06 day effect).
  cmd  (this file): pre-MDC commanded torque at nominal gains, computed from the
       FRAME'S OWN measured q/qd -- i.e. prev_measured, NOT this tick's fresh
       q/qd. The sim computes this channel from the latency-delayed measured
       vector that shares the frame (env obs_tau_mode "cmd"); at swing speeds
       (14 rad/s) the 20 ms difference is 0.28 rad x kp100 = 28 N*m, so using
       fresh q here would be a train/deploy break, not a nicety. Deploy-exact by
       construction: no MDC, no battery model, no tau_est anywhere near the obs.
  zero (this file): channel zeroed (the ablation family).
The tau_p/tau_d/tau_cmd/tau_applied DIAGNOSTICS (trace + plots) are unchanged --
they use fresh q/qd exactly as deploy_meta logs them; only the obs differs.

ELECTRICAL LOGGING (all read-only, never enters the obs):
  500 Hz (LowState handler): power_v, power_a  -> <stem>_electrical.npz
  50 Hz  (policy tick):      power_v, power_a, BMS SOC, 12 motor temperatures
  Summary printed at save: V start/end/min (sag), A peak, SOC delta, temp rise.
  Purpose: calibrate the model-Vbat <-> bus-voltage mapping (inverter utilization
  means they are NOT equal), catch the fresh-pack OOD-high regime (train DR tops
  out at 28.8 model-V), and watch the thermal-drift channel (R/Kt vs temp).

Contract: refuses ckpts whose meta lacks obs_tau_mode "cmd"/"zero" (mdc/absent ->
use deploy_meta.py). All deploy_meta contract checks (hold_targets, blind sites,
joint names, action_space 12, MDC constants) run unchanged via the parent.
"""
import glob
import logging
import os
import sys
import time

import numpy as np

from deploy_meta import (Custom, load_meta, mdc_apply, qmul, quat_about_z,
                         quat_to_euler_xyz, rotmat_from_quat_wxyz,
                         yaw_from_quat_wxyz, _wrap_pi, MOTOR_FROM_ISO,
                         CONTACT_FORCE_THRESHOLD_HW, BODY_WEIGHT)


class CustomCmd(Custom):
    def __init__(self, ckpt_path, meta, traj_override=None, resid_scale=1.0,
                 meta_path=None):
        super().__init__(ckpt_path, meta, traj_override, resid_scale=resid_scale,
                         meta_path=meta_path)
        mode = meta.get("obs_tau_mode", "mdc")
        if mode not in ("cmd", "zero"):
            raise SystemExit(
                f"ABORT: meta['obs_tau_mode']={mode!r}. This script implements the "
                f"cmd/zero torque-obs families; mdc-family ckpts deploy with "
                f"deploy_meta.py (feeding this ckpt the mdc channel would hit "
                f"normalizer slots trained on different stats -- the obs_contact_"
                f"blind restore bug class).")
        self.TAU_MODE = mode
        logging.info("TORQUE-OBS MODE: %s (%s)", mode,
                     "pre-MDC nominal-gain cmd torque, frame-delayed q/qd"
                     if mode == "cmd" else "channel zeroed")
        # electrical logs -- separate from self.trace so the parent's save_trace
        # npz schema stays byte-identical to deploy_meta's.
        self.hf_v = []                        # 500 Hz: (t, power_v, power_a)
        self.elec50 = {k: [] for k in ("t", "v", "a", "soc", "temp")}

    # ------------------------------------------------------------ electrical
    @staticmethod
    def _read_power(msg):
        try:
            return float(getattr(msg, "power_v", np.nan)), \
                   float(getattr(msg, "power_a", np.nan))
        except Exception:
            return np.nan, np.nan

    def LowStateMessageHandler(self, msg):
        super().LowStateMessageHandler(msg)   # low_state + 500 Hz tau_est
        if self.hf_on and len(self.hf_v) < self.HF_MAX:
            v, a = self._read_power(msg)
            self.hf_v.append((time.perf_counter() - self.hf_t0, v, a))

    def _elec_tick(self, t):
        msg = self.low_state
        v, a = self._read_power(msg)
        try:
            soc = float(msg.bms_state.soc)
        except Exception:
            soc = np.nan
        try:
            temp = [float(msg.motor_state[m].temperature) for m in MOTOR_FROM_ISO]
        except Exception:
            temp = [np.nan] * 12
        e = self.elec50
        e["t"].append(t); e["v"].append(v); e["a"].append(a)
        e["soc"].append(soc); e["temp"].append(temp)

    # ------------------------------------------------------------ policy tick
    # Copied from deploy_meta.Custom._policy_tick (2026-08-06 state); the marked
    # TAU-OBS block and the one _elec_tick call are the only changes. If the
    # parent tick changes, re-copy -- drift here is a silent obs bug.
    def _policy_tick(self):
        t = self.ii * self.dt
        q, qd = self._read_iso()
        gyro = np.asarray(self.low_state.imu_state.gyroscope, float)
        accel = np.asarray(self.low_state.imu_state.accelerometer, float)
        measured_now = np.concatenate([gyro, accel, q, qd])
        forces_n = self._read_foot_forces_n()
        contacts_true = (forces_n > CONTACT_FORCE_THRESHOLD_HW).astype(float)
        contacts_obs = np.zeros(4) if self.blind else contacts_true

        # diagnostics: unchanged from deploy_meta (fresh q/qd, held targets)
        if self.held_targets is None:
            tau_p = np.zeros(12); tau_d = np.zeros(12)
            tau_ff_h = np.zeros(12); tau_cmd = np.zeros(12)
            tau_applied = np.zeros(12)
        else:
            qt, qdt, tff = self.held_targets
            tau_p = self.KP * (qt - q)
            tau_d = self.KD * (qdt - qd)
            tau_ff_h = tff
            tau_cmd = tau_p + tau_d + tff
            tau_applied = mdc_apply(tau_cmd, qd)

        # ---- TAU-OBS (the one change vs deploy_meta) --------------------------
        # cmd: current held targets vs the FRAME's measured q/qd (prev_measured =
        # 1 tick old, inside the trained obs-latency DR 1-3). NOT tau_cmd above.
        if self.TAU_MODE == "cmd" and self.held_targets is not None:
            pm = self.prev_measured
            qt, qdt, tff = self.held_targets
            tau_obs = self.KP * (qt - pm[6:18]) + self.KD * (qdt - pm[18:30]) + tff
        else:                                  # "zero", or pre-first-command
            tau_obs = np.zeros(12)
        frame = np.concatenate([self.prev_measured, contacts_obs, tau_obs])
        # -----------------------------------------------------------------------
        self.prev_measured = measured_now
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = frame

        quat = self._aligned_quat()
        R = rotmat_from_quat_wxyz(quat)
        grav_b = R.T @ np.array([0.0, 0.0, -1.0])
        yaw = yaw_from_quat_wxyz(quat)
        pos_ref_xy, quat_ref = self.ref.base_ref_at(t)
        yaw_e = yaw - yaw_from_quat_wxyz(quat_ref)
        attitude = np.concatenate([grav_b, [np.sin(yaw_e)], [np.cos(yaw_e)]])

        blocks = [self.history.reshape(-1),
                  self.ref.preview(t, self.PREVIEW_DT, self.NUM_FUTURE),
                  self.ref.phase_info(t), attitude]

        if self.use_ori_err:
            qc = quat * np.array([1.0, -1.0, -1.0, -1.0])
            qe = qmul(qc, quat_ref)
            blocks.append(2.0 * np.sign(qe[0] if qe[0] != 0 else 1.0) * qe[1:4])

        vhat = np.zeros(3)
        e_h = np.zeros(2)
        if self.use_velest:
            hist_scaled = (self.history * self.frame_scale).reshape(-1)
            vhat = self.policy.estimate_vel(hist_scaled)
            if self.ii > 0:
                self.odom_xy += (R @ vhat)[:2] * self.dt
            e_w = pos_ref_xy - self.odom_xy
            cy, sy = np.cos(yaw), np.sin(yaw)
            e_h = np.array([cy * e_w[0] + sy * e_w[1], -sy * e_w[0] + cy * e_w[1]])
            blocks.append(np.concatenate([vhat, np.clip(e_h, -self.ODOM_CLAMP,
                                                        self.ODOM_CLAMP) / self.ODOM_CLAMP]))

        if self.use_cprev:
            masks, ttc_e = self.ref.contact_pack(t, self.PREVIEW_DT, self.NUM_FUTURE)
            load = (np.zeros(4) if self.blind
                    else np.clip(forces_n / BODY_WEIGHT, 0.0, 2.0))
            blocks.append(np.concatenate([masks, ttc_e, load]))

        obs = np.concatenate(blocks)
        assert obs.shape == (self.n_obs,), (obs.shape, self.n_obs)
        obs = np.clip(obs * self.actor_scale, -self.CLIP_OBS, self.CLIP_OBS)
        action = self.policy(obs)

        a_cmd = self.prev_action
        self.prev_action = action
        q_ref_t, qd_ref_t, tau_ff_t = self.ref.ref_at(t)
        q_target = q_ref_t + self.ACTION_SCALE * self.RESID_SCALE * a_cmd

        fade = max(0.0, 1.0 - self.ii / self.handoff_fade_ticks)
        for i in range(12):
            m = MOTOR_FROM_ISO[i]
            self.low_cmd.motor_cmd[m].q = float(q_target[i])
            self.low_cmd.motor_cmd[m].dq = float(qd_ref_t[i])
            self.low_cmd.motor_cmd[m].kp = self.KP
            self.low_cmd.motor_cmd[m].kd = self.KD
            self.low_cmd.motor_cmd[m].tau = float(tau_ff_t[i] + fade * self.tau_i[m])
        self.held_targets = (q_target, qd_ref_t, tau_ff_t)

        _qc = quat * np.array([1.0, -1.0, -1.0, -1.0])
        _qe = qmul(_qc, quat_ref)
        tr = self.trace
        tr["t"].append(t)
        tr["rpy"].append(quat_to_euler_xyz(quat))
        tr["rpy_ref"].append(quat_to_euler_xyz(quat_ref))
        tr["ori_err"].append(2.0 * np.sign(_qe[0] if _qe[0] != 0 else 1.0) * _qe[1:4])
        tr["odom_xy"].append(self.odom_xy.copy())
        tr["ref_xy"].append(pos_ref_xy.copy())
        tr["vhat"].append(vhat.copy())
        tr["tilt"].append(grav_b[2])
        tr["action"].append(a_cmd.copy())
        tr["contacts"].append(contacts_true.copy())
        tr["forces"].append(forces_n.copy())
        tr["q"].append(q.copy())
        tr["qd"].append(qd.copy())
        tr["q_target"].append(q_target.copy())
        tr["tau_applied"].append(tau_applied.copy())
        tr["tau_p"].append(tau_p.copy())
        tr["tau_d"].append(tau_d.copy())
        tr["tau_ff"].append(tau_ff_h.copy())
        tr["tau_cmd"].append(tau_cmd.copy())
        self._elec_tick(t)

        if self.motiontime % 10 == 0:
            _rp = np.degrees(_wrap_pi(tr["rpy"][-1] - tr["rpy_ref"][-1]))[:2]
            logging.info("t %.2f cont %s%s vhat [%+.2f %+.2f %+.2f] eh [%+.2f %+.2f] "
                         "rp_e [%+.1f %+.1f] yaw_e %+.1f tilt %.2f |a| %.2f V %.1f (loop %.1f Hz)",
                         t, contacts_true.astype(int), " (blinded)" if self.blind else "",
                         *vhat, *e_h, *_rp, np.degrees(yaw_e), grav_b[2],
                         np.abs(a_cmd).max(),
                         self.elec50["v"][-1] if self.elec50["v"] else float("nan"),
                         self._loop_hz)

        if grav_b[2] > -0.4:
            logging.error("TILT ABORT at t %.2f (grav_z %.2f) -> damping", t, grav_b[2])
            self.aborted = True
        self.ii += 1

    # ------------------------------------------------------------ save
    def save_trace(self, outdir="runs"):
        ret = super().save_trace(outdir)
        try:
            self._save_electrical(outdir)
        except Exception:
            logging.exception("electrical log save failed (trace itself is saved)")
        return ret

    def _save_electrical(self, outdir):
        if not self.elec50["t"]:
            logging.warning("no electrical samples captured")
            return
        dirs = sorted(glob.glob(os.path.join(outdir, f"{self.run_tag}_*")),
                      key=os.path.getmtime)
        rundir = dirs[-1] if dirs else outdir
        hf = np.asarray(self.hf_v, float) if self.hf_v else np.zeros((0, 3))
        e = {k: np.asarray(v, float) for k, v in self.elec50.items()}
        path = os.path.join(rundir, f"{os.path.basename(rundir)}_electrical.npz")
        np.savez(path, t50=e["t"], v50=e["v"], a50=e["a"], soc50=e["soc"],
                 temp50=e["temp"], t500=hf[:, 0] if len(hf) else np.zeros(0),
                 v500=hf[:, 1] if len(hf) else np.zeros(0),
                 a500=hf[:, 2] if len(hf) else np.zeros(0),
                 tau_obs_mode=self.TAU_MODE)
        v, a, soc, temp = e["v"], e["a"], e["soc"], e["temp"]
        vv = v[np.isfinite(v)]
        aa = a[np.isfinite(a)]
        tt = temp[np.isfinite(temp).all(axis=1)] if len(temp) else np.zeros((0, 12))
        lines = ["electrical summary (never enters the obs):"]
        if len(vv):
            lines.append(f"  bus V   start {vv[0]:.2f}  end {vv[-1]:.2f}  "
                         f"min {vv.min():.2f} (sag {vv[0]-vv.min():.2f})  "
                         f"[model-Vbat DR ceiling 28.8 -- see day-effect arc]")
        if len(aa):
            lines.append(f"  bus A   mean {np.nanmean(aa):.1f}  peak {np.nanmax(aa):.1f}")
        if np.isfinite(soc).any():
            s = soc[np.isfinite(soc)]
            lines.append(f"  BMS SOC start {s[0]:.0f}%%  end {s[-1]:.0f}%%")
        if len(tt):
            rise = tt[-1] - tt[0]
            lines.append(f"  motor T start max {tt[0].max():.0f}C  end max "
                         f"{tt[-1].max():.0f}C  worst rise {rise.max():+.0f}C")
        for ln in lines:
            logging.info(ln)
        with open(os.path.join(rundir, "electrical.txt"), "w") as f:
            f.write("\n".join(lines).replace("%%", "%") + "\n")
        logging.info("electrical log -> %s", path)


if __name__ == '__main__':
    import argparse
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="hopscotch_utils/a2cmd_400.pt")
    ap.add_argument("--meta", default="hopscotch_utils/a2cmd_meta.json", help="default: meta.json beside the checkpoint")
    ap.add_argument("--traj", default="traj_hopscotch_friction_6cm_lsq.json",
                    help="override meta's traj_path")
    ap.add_argument("--resid_scale", type=float, default=1,
                    help="residual authority attenuation. NOTE: unlike the pz3 "
                         "family, wall-family ckpts are NOT blanket-0.75x -- Afree/"
                         "AtauCmd sim picks are full-auth; check the run's eval "
                         "ledger before attenuating (2026-08-06 authority-flip).")
    ap.add_argument("--dry-run", action="store_true",
                    help="load + validate everything, then exit without touching the robot")
    ap.add_argument("--out", default="data",
                    help="parent directory for per-run trace dirs (same layout as "
                         "deploy_meta + <stem>_electrical.npz + electrical.txt)")
    ap.add_argument("iface", nargs="?", default=None)
    args = ap.parse_args()

    meta, meta_path = load_meta(args.checkpoint, args.meta)

    if args.dry_run:
        logging.info("--dry-run: constructing controller (no DDS, no motion)...")
        c = CustomCmd(args.checkpoint, meta, args.traj, resid_scale=args.resid_scale,
                      meta_path=meta_path)
        logging.info("DRY RUN OK: obs=%d ticks=%d blind=%s tau_mode=%s",
                     c.n_obs, c.n_ticks, c.blind, c.TAU_MODE)
        sys.exit(0)

    print("WARNING: Please ensure there are no obstacles around the robot while running.")
    input("Press Enter to continue...")

    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)

    custom = CustomCmd(args.checkpoint, meta, args.traj, resid_scale=args.resid_scale,
                       meta_path=meta_path)
    custom.Init()
    custom.Start()

    launched = False
    saved = False
    try:
        while True:
            if custom.aborted:
                if not saved:
                    saved = True
                    try:
                        custom.save_trace(args.out)
                    except Exception:
                        logging.exception("save_trace failed (robot is still damping)")
                time.sleep(2)
                print("Aborted - robot in damping mode. Ctrl-C when secured.")
                time.sleep(10)
            if (not launched) and (not custom.aborted) and custom.hold_percent >= 1:
                input("Robot calibrated + holding q0. Press Enter to LAUNCH...")
                custom.start_policy = True
                launched = True
            if custom.settle_percent >= 1:
                time.sleep(1)
                if not saved:
                    saved = True
                    custom.save_trace(args.out)
                print("Done!")
                sys.exit(0)
            time.sleep(0.5)
    except KeyboardInterrupt:
        if launched and not saved:
            custom.save_trace(args.out)
        raise
