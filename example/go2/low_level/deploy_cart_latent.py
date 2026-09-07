"""Latent-PPO cartwheel deployment entry point (LAT branch checkpoints).

This is a new sibling of ``deploy_cart_safe.py`` and reuses every one of its
safety boundaries unchanged (checkpoint/meta validation, orientation walls,
LowState watchdog, readiness gate, fail-closed DDS writes).  The single thing it
adds is the policy architecture of ``rl-res/isaac_port/latent_policy.py``:

    z    = LayerNorm(Linear(flatten(TCN(history[L x F]))))         # 64-dim latent
    a    = MLP([z, newest ``actor_frames`` raw frames, rest]) # 512/256/128 ELU

where the observation vector is exactly the one ``deploy_cart.py`` already
builds (``history (L*F) | preview | phase | attitude | ori_err | contact pack``),
normalized by the checkpoint's empirical statistics first, as in training
(``LatentActorCritic.encode``).  The predictor / privileged heads are
training-only and are ignored here.

The architecture is read from ``meta['latent']`` and cross-checked against the
checkpoint tensors and the observation layout, so a stock (non-latent)
checkpoint, a mismatched history length, or a non-TCN encoder aborts before DDS
is touched.  Stock checkpoints should keep using ``deploy_cart_safe.py``.

Examples:
    python3 deploy_cart_latent.py --checkpoint /path/to/lat_rev_v0c/model_15700.pt --dry-run
    python3 deploy_cart_latent.py --checkpoint /path/to/lat_rev_v0c/model_15700.pt eth0
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

import numpy as np

if __package__:
    from . import deploy_cart_safe as _safe
else:
    import deploy_cart_safe as _safe

_base = _safe._base

# Mirrors latent_policy.TCNEncoder.SPEC: (kernel, stride) per conv layer; a layer is
# skipped when the remaining time axis is shorter than its kernel.
TCN_SPEC = ((5, 2), (3, 2), (3, 1))
LAYERNORM_EPS = 1e-5          # torch.nn.LayerNorm default
NORMALIZER_EPS = 1e-2         # rsl_rl EmpiricalNormalization default (same as the stock Policy)
ACTOR_HIDDEN = (512, 256, 128)

# Set by main() from the validated metadata before Custom builds the policy.  The
# base class constructs ``Policy(ckpt_path, n_obs, n_act)`` with no meta argument.
_LATENT_META = None


def _elu(x):
    return np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))


def _latent_cfg(meta, ckpt_path):
    """Validate meta['latent'] against the observation layout; return a plain dict."""
    lat = meta.get("latent")
    if not isinstance(lat, dict):
        raise SystemExit(
            "ABORT: meta has no 'latent' block; this checkpoint is not a latent-PPO "
            "policy (use deploy_cart_safe.py for stock actors)"
        )
    required = ("dim", "enc", "channels", "hist_len", "frame_dim", "actor_frames")
    missing = [key for key in required if key not in lat]
    if missing:
        raise SystemExit(f"ABORT: meta['latent'] is missing {', '.join(missing)}")
    if lat["enc"] != "tcn":
        raise SystemExit(f"ABORT: only the 'tcn' encoder is deployable here, meta says {lat['enc']!r}")
    hist_len = int(lat["hist_len"])
    frame_dim = int(lat["frame_dim"])
    expected_frame = 58 if bool(meta.get("obs_act_hist", False)) else 46
    if hist_len != int(meta["obs_history_len"]):
        raise SystemExit(
            f"ABORT: latent hist_len {hist_len} != meta obs_history_len {meta['obs_history_len']}"
        )
    if frame_dim != expected_frame:
        raise SystemExit(
            f"ABORT: latent frame_dim {frame_dim} != {expected_frame} implied by "
            f"obs_act_hist={bool(meta.get('obs_act_hist', False))}"
        )
    actor_frames = int(lat["actor_frames"])
    if not 0 <= actor_frames <= hist_len:
        raise SystemExit(f"ABORT: latent actor_frames {actor_frames} not in [0, {hist_len}]")
    dim = int(lat["dim"])
    channels = int(lat["channels"])
    if dim <= 0 or channels <= 0:
        raise SystemExit(f"ABORT: latent dim/channels must be positive, got {dim}/{channels}")
    logging.info(
        "latent policy: %s | enc=%s z=%d channels=%d L=%d F=%d actor_frames=%d",
        Path(ckpt_path).name, lat["enc"], dim, channels, hist_len, frame_dim, actor_frames,
    )
    return {"dim": dim, "channels": channels, "hist_len": hist_len,
            "frame_dim": frame_dim, "actor_frames": actor_frames}


class LatentPolicy:
    """NumPy port of LatentActorCritic.act_inference for the TCN encoder."""

    def __init__(self, ckpt_path, n_obs, n_act):
        import torch

        meta = _LATENT_META
        if meta is None:
            meta_path = Path(ckpt_path).resolve().parent / "meta.json"
            try:
                with meta_path.open() as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                raise SystemExit(f"ABORT: latent policy needs meta.json beside the checkpoint: {exc}") from exc
        lat = _latent_cfg(meta, ckpt_path)

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001 - convert loader failures to a hard abort
            raise SystemExit(f"ABORT: cannot load checkpoint {ckpt_path}: {exc}") from exc
        state = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else None
        if not isinstance(state, dict):
            raise SystemExit("ABORT: checkpoint has no model_state_dict")

        def tensor(key, shape):
            if key not in state:
                raise SystemExit(f"ABORT: checkpoint is missing {key!r}")
            value = state[key].detach().cpu().numpy().astype(np.float64)
            if value.shape != tuple(shape):
                raise SystemExit(f"ABORT: {key} has shape {value.shape}, expected {tuple(shape)}")
            if not np.isfinite(value).all():
                raise SystemExit(f"ABORT: {key} contains non-finite values")
            return value

        # ---- observation split (LatentActorCritic.__init__) ----
        self.L, self.F = lat["hist_len"], lat["frame_dim"]
        self.n_hist = self.L * self.F
        self.n_raw = lat["actor_frames"] * self.F
        self.n_rest = n_obs - self.n_hist
        if self.n_rest < 0:
            raise SystemExit(f"ABORT: obs {n_obs} is shorter than the history block {self.n_hist}")
        self.dim = lat["dim"]
        C = lat["channels"]

        # ---- TCN encoder (latent_policy.TCNEncoder) ----
        self.conv = []                    # (W (Cout, Cin, k), b, stride)
        cin, t, index = self.F, self.L, 0
        for k, s in TCN_SPEC:
            if t < k:
                break
            self.conv.append((tensor(f"encoder.conv.{index}.weight", (C, cin, k)),
                              tensor(f"encoder.conv.{index}.bias", (C,)), s))
            cin, t = C, (t - k) // s + 1
            index += 2                    # Conv1d, ELU pairs in the Sequential
        if any(key.startswith(f"encoder.conv.{index}.") for key in state):
            raise SystemExit("ABORT: checkpoint encoder has more conv layers than TCN_SPEC implies")
        self.flat_dim = cin * t
        self.enc_W = tensor("encoder.out.0.weight", (self.dim, self.flat_dim))
        self.enc_b = tensor("encoder.out.0.bias", (self.dim,))
        self.ln_w = tensor("encoder.out.1.weight", (self.dim,))
        self.ln_b = tensor("encoder.out.1.bias", (self.dim,))

        # ---- actor MLP on [z, newest raw frames, rest] ----
        n_in = self.dim + self.n_raw + self.n_rest
        widths = (n_in,) + ACTOR_HIDDEN + (n_act,)
        self.W, self.b = [], []
        for i, index in enumerate((0, 2, 4, 6)):
            self.W.append(tensor(f"actor.{index}.weight", (widths[i + 1], widths[i])))
            self.b.append(tensor(f"actor.{index}.bias", (widths[i + 1],)))

        # ---- observation normalizer (rsl-rl 3.x embedded / 2.x top-level) ----
        embedded_mean = state.get("actor_obs_normalizer._mean")
        embedded_std = state.get("actor_obs_normalizer._std")
        legacy_norm = ckpt.get("obs_norm_state_dict")
        if embedded_mean is not None and embedded_std is not None:
            mean_t, std_t, layout = embedded_mean, embedded_std, "rsl-rl 3.x embedded"
        elif isinstance(legacy_norm, dict) and legacy_norm.get("_mean") is not None \
                and legacy_norm.get("_std") is not None:
            mean_t, std_t, layout = legacy_norm["_mean"], legacy_norm["_std"], "rsl-rl 2.x top-level"
        else:
            raise SystemExit("ABORT: checkpoint has no actor observation-normalization statistics")
        self.mean = mean_t.detach().cpu().numpy().reshape(-1).astype(np.float64)
        self.std = std_t.detach().cpu().numpy().reshape(-1).astype(np.float64)
        if self.mean.shape != (n_obs,) or self.std.shape != (n_obs,):
            raise SystemExit(
                f"ABORT: normalizer mean/std are {self.mean.shape}/{self.std.shape}, expected {(n_obs,)}"
            )
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() \
                or np.any(self.std < 0):
            raise SystemExit("ABORT: observation normalizer is non-finite or has negative std")

        logging.info(
            "latent policy loaded: %s (iter %s), %s, obs=%d = hist %dx%d + rest %d | "
            "TCN %d convs -> %d -> z %d | actor in %d act=%d",
            ckpt_path, ckpt.get("iter"), layout, n_obs, self.L, self.F, self.n_rest,
            len(self.conv), self.flat_dim, self.dim, n_in, n_act,
        )

    # ---- forward ----
    @staticmethod
    def _conv1d(x, W, b, stride):
        """x (Cin, T) -> (Cout, T_out), PyTorch Conv1d semantics (no padding)."""
        k = W.shape[2]
        t_out = (x.shape[1] - k) // stride + 1
        out = np.empty((W.shape[0], t_out))
        for j in range(t_out):
            window = x[:, j * stride:j * stride + k]           # (Cin, k)
            out[:, j] = np.tensordot(W, window, axes=([1, 2], [0, 1])) + b
        return out

    def encode(self, hist):
        x = hist.reshape(self.L, self.F).T                     # (F, L): channels x time
        for W, b, s in self.conv:
            x = _elu(self._conv1d(x, W, b, s))
        z = self.enc_W @ x.reshape(-1) + self.enc_b            # flatten(1) is C-major, then time
        mu = z.mean()
        var = ((z - mu) ** 2).mean()
        return (z - mu) / np.sqrt(var + LAYERNORM_EPS) * self.ln_w + self.ln_b

    def __call__(self, obs):
        x = np.asarray(obs, dtype=np.float64)
        if x.shape != self.mean.shape or not np.isfinite(x).all():
            raise FloatingPointError(
                f"policy observation must be finite with shape {self.mean.shape}, got {x.shape}"
            )
        x = (x - self.mean) / (self.std + NORMALIZER_EPS)
        hist, rest = x[:self.n_hist], x[self.n_hist:]
        parts = [self.encode(hist)]
        if self.n_raw > 0:
            parts.append(hist[self.n_hist - self.n_raw:])      # newest frames (history is oldest -> newest)
        parts.append(rest)
        h = np.concatenate(parts)
        for W, b in zip(self.W[:-1], self.b[:-1]):
            h = _elu(W @ h + b)
        action = self.W[-1] @ h + self.b[-1]
        if action.shape != (self.W[-1].shape[0],) or not np.isfinite(action).all():
            raise FloatingPointError("policy produced a non-finite or malformed action")
        return np.clip(action, -1.0, 1.0)


_base.Policy = LatentPolicy
_safe_build_parser = _safe._build_parser      # captured before main() re-points the safe module


def _build_parser():
    parser = _safe_build_parser()
    # The safe script defaults to a specific stock checkpoint/meta/trajectory; a latent
    # deployment must name its checkpoint, with meta.json and traj_path resolved from it.
    parser.set_defaults(checkpoint="lat_utils/v0c_17598.pt", meta="lat_utils/v0c_meta.json", traj="hopscotch_utils/cart.npz")
    for action in parser._actions:
        if action.dest == "checkpoint":
            action.required = False
            action.help = "latent-PPO cartwheel checkpoint; meta.json is discovered beside it"
    return parser


def main(argv=None):
    global _LATENT_META
    args = _build_parser().parse_args(argv)
    meta, meta_path, checkpoint = _safe.load_meta(args.checkpoint, args.meta)
    _latent_cfg(meta, checkpoint)           # fail closed before any hardware setup
    _LATENT_META = meta
    # Everything below is deploy_cart_safe.main.  It re-parses argv through the
    # module-global _build_parser, so point that at ours: the safe parser's own
    # defaults name a STOCK checkpoint, meta and trajectory, which must never be
    # silently substituted for a latent checkpoint's meta.json / traj_path.
    _safe._build_parser = _build_parser
    return _safe.main(argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
