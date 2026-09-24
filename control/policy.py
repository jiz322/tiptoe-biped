"""
policy.py — rsl_rl policy loader and inference wrapper.

Three loading modes (auto-detected from file extension / contents):

1. ONNX (.onnx) — exported via torch.onnx.export. Run with onnxruntime.
   Requires `pip install onnxruntime` (or `onnxruntime-gpu`).

2. JIT export — produced by export_policy.py or by OnPolicyRunner.save().
   The file is a TorchScript module whose forward() is:
   obs (1, N) → action (1, M). Observation normalizer baked in.

3. Checkpoint (.pt dict) — the raw training checkpoint containing
   'model_state_dict' and optionally 'obs_norm_state_dict'. We
   reconstruct the actor MLP + normalizer from the state dict shapes.

Either way, the public API is:
    policy = Policy('path.onnx')      # or .pt / .jit
    action = policy.forward(obs_np)   # obs (36,) → action (6,)
"""

import os
import numpy as np
import torch
import torch.nn as nn

import config as C


# ════════════════════════════════════════════════════════════
# Lightweight actor rebuild (only needed for checkpoint mode)
# ════════════════════════════════════════════════════════════

class _EmpiricalNormalization(nn.Module):
    """Exact twin of rsl_rl's EmpiricalNormalization.forward:
        y = (x - _mean) / (_std + eps),  eps = 1e-2  (added to STD, not var).
    Do NOT change to sqrt(var + eps): for near-constant channels
    (init_h std=0.006, difficulty std=0) the two formulas differ 2-3x.
    """
    def __init__(self, mean, std, eps=1e-2):
        super().__init__()
        self.register_buffer('mean', mean.reshape(-1).clone())
        self.register_buffer('std',  std.reshape(-1).clone())
        self.eps = eps

    def forward(self, x):
        return (x - self.mean) / (self.std + self.eps)


class _ActorMLP(nn.Module):
    """Reconstruct rsl_rl ActorCritic's actor from state dict keys."""
    def __init__(self, state_dict_keys, state_dict):
        super().__init__()
        actor_keys = sorted([k for k in state_dict_keys if k.startswith('actor.')])
        weight_keys = [k for k in actor_keys if k.endswith('.weight')]

        layers = []
        for i, wk in enumerate(weight_keys):
            bk = wk.replace('.weight', '.bias')
            w = state_dict[wk]
            b = state_dict.get(bk)
            linear = nn.Linear(w.shape[1], w.shape[0])
            linear.weight.data = w
            if b is not None:
                linear.bias.data = b
            layers.append(linear)
            if i < len(weight_keys) - 1:
                layers.append(nn.ELU())  # rsl_rl default activation
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ════════════════════════════════════════════════════════════
# Public class
# ════════════════════════════════════════════════════════════

class Policy:
    def __init__(self, path: str = C.POLICY_PATH, device: str = 'cpu'):
        self.device = torch.device(device)
        print(f"[Policy] Loading {path} on {self.device} ...")

        ext = os.path.splitext(path)[1].lower()
        # Provenance + contract metadata (None when the file does not carry it):
        # training iteration, and the actor's obs normalizer mean/std as (D,)
        # numpy arrays — balance_log's startup guard z-scores the hardware
        # constant channels against them, and stamps them into the npz.
        self.iter = None
        self.norm_mean = None
        self.norm_std = None
        self.input_dim = None

        # ── ONNX ─────────────────────────────────────────────
        if ext == '.onnx':
            self._load_onnx(path)
            self._mode = 'onnx'
            print(f"[Policy] Loaded as ONNX model "
                  f"(providers={self._ort_session.get_providers()})")

        # ── TorchScript JIT ──────────────────────────────────
        else:
            try:
                self.model = torch.jit.load(path, map_location=self.device)
                self.model.eval()
                self._mode = 'jit'
                sd = self.model.state_dict()
                first_w = [v for k, v in sd.items() if v.dim() == 2]
                if first_w:
                    self.input_dim = int(first_w[0].shape[1])
                mean_keys = [k for k in sd if k.endswith(('normalizer._mean', 'normalizer.mean'))]
                std_keys = [k for k in sd if k.endswith(('normalizer._std', 'normalizer.std'))]
                if mean_keys and std_keys:
                    self.norm_mean = sd[mean_keys[0]].reshape(-1).cpu().numpy()
                    self.norm_std = sd[std_keys[0]].reshape(-1).cpu().numpy()
                print(f"[Policy] Loaded as TorchScript JIT module")
            except Exception:
                # ── Raw checkpoint ───────────────────────────
                ckpt = torch.load(path, map_location=self.device, weights_only=False)
                self._load_from_checkpoint(ckpt)
                self._mode = 'checkpoint'
                if isinstance(ckpt, dict) and 'iter' in ckpt:
                    self.iter = int(ckpt['iter'])
                print(f"[Policy] Loaded from training checkpoint (rebuilt actor MLP"
                      f"{'' if self.iter is None else f', iter {self.iter}'})")

        if self.input_dim is not None and self.input_dim != C.OBS_DIM:
            raise RuntimeError(
                f"Policy input width {self.input_dim} != config OBS_DIM {C.OBS_DIM} "
                f"(OBS_FRAME_DIM {C.OBS_FRAME_DIM} x OBS_HISTORY_STEPS "
                f"{C.OBS_HISTORY_STEPS}). The checkpoint was trained on a "
                f"different observation layout — do not fly it on this build_obs."
            )

        # Warm up + shape sanity check
        with torch.no_grad():
            test = torch.zeros(1, C.OBS_DIM, device=self.device)
            out = self._infer(test)
        if out.shape != (1, C.ACT_DIM):
            raise RuntimeError(
                f"Policy output shape {tuple(out.shape)} != (1, {C.ACT_DIM}). "
                f"Check OBS_DIM/ACT_DIM in config.py or re-export the policy."
            )
        print(f"[Policy] OK — obs={C.OBS_DIM}, act={C.ACT_DIM}, mode={self._mode}")

    # ─────────────────────────────────────────────────────────
    # Loaders
    # ─────────────────────────────────────────────────────────

    def _load_onnx(self, path: str):
        """Load an ONNX model via onnxruntime."""
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required to load .onnx policies. "
                "Install with: pip install onnxruntime  (or onnxruntime-gpu)"
            ) from e

        # Pick providers based on requested device
        if self.device.type == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1   # deterministic, low-latency
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._ort_session = ort.InferenceSession(path, sess_options=sess_opts,
                                                 providers=providers)
        self._ort_input_name  = self._ort_session.get_inputs()[0].name
        self._ort_output_name = self._ort_session.get_outputs()[0].name

        # Cache expected input dtype (usually float32)
        in_type = self._ort_session.get_inputs()[0].type   # e.g. 'tensor(float)'
        self._ort_input_np_dtype = np.float32 if 'float' in in_type else np.float64

    def _load_from_checkpoint(self, ckpt):
        """Rebuild actor + normalizer from a raw rsl_rl checkpoint dict."""
        if 'model_state_dict' in ckpt:
            model_sd = ckpt['model_state_dict']
        else:
            model_sd = ckpt

        self._actor = _ActorMLP(model_sd.keys(), model_sd).to(self.device)
        self._actor.eval()
        self.input_dim = int(self._actor.net[0].weight.shape[1])

        # rsl_rl 3.x (actor_obs_normalization=True) stores the actor's obs
        # normalizer as actor_obs_normalizer._mean / ._var / ._std / .count.
        # Older rsl_rl / custom exports used running_mean / running_var.
        # Support both; REFUSE to run if normalizer-ish keys exist but none
        # match — silently skipping normalization means running a different
        # function than was trained (see normalizer_gap_check.py).
        self._normalizer = None
        if 'actor_obs_normalizer._mean' in model_sd:
            mean = model_sd['actor_obs_normalizer._mean']
            std = model_sd.get('actor_obs_normalizer._std')
            if std is None:  # very old ckpts: only _var stored
                std = torch.sqrt(model_sd['actor_obs_normalizer._var'])
            self._normalizer = _EmpiricalNormalization(mean, std).to(self.device)
            self.norm_mean = mean.reshape(-1).cpu().numpy()
            self.norm_std = std.reshape(-1).cpu().numpy()
            print(f"[Policy] Loaded obs normalizer (actor_obs_normalizer._mean/_std, "
                  f"eps=1e-2 on std — rsl_rl formula)")
        else:
            norm_mean_key = None
            for k in model_sd:
                if 'running_mean' in k and 'critic' not in k:
                    norm_mean_key = k
                    break
            if norm_mean_key is not None:
                prefix = norm_mean_key.rsplit('running_mean', 1)[0]
                mean = model_sd[norm_mean_key]
                var = model_sd.get(prefix + 'running_var', torch.ones_like(mean))
                # legacy RunningMeanStd convention: y = (x-mean)/sqrt(var+1e-5);
                # fold into the std+eps form as closely as possible
                self._normalizer = _EmpiricalNormalization(
                    mean, torch.sqrt(var + 1e-5), eps=0.0).to(self.device)
                self.norm_mean = mean.reshape(-1).cpu().numpy()
                self.norm_std = torch.sqrt(var + 1e-5).reshape(-1).cpu().numpy()
                print(f"[Policy] Loaded obs normalizer from legacy keys "
                      f"({norm_mean_key})")

        if self._normalizer is None:
            leftover = [k for k in model_sd
                        if ('normalizer' in k or 'running_mean' in k)
                        and 'critic' not in k]
            if leftover:
                raise RuntimeError(
                    f"Checkpoint contains actor obs-normalizer keys {leftover} "
                    f"that this loader does not recognize. REFUSING to run "
                    f"unnormalized — the actor was trained on normalized obs "
                    f"(this exact silent skip caused the pre-2026-08-06 "
                    f"deployment bug). Update _load_from_checkpoint."
                )
            print(f"[Policy] No obs normalizer in checkpoint "
                  f"(actor_obs_normalization=False training) — raw obs is correct")

    # ─────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────

    def _infer(self, obs_t: torch.Tensor) -> torch.Tensor:
        if self._mode == 'jit':
            return self.model(obs_t)
        elif self._mode == 'onnx':
            obs_np = obs_t.detach().cpu().numpy().astype(self._ort_input_np_dtype)
            out_np = self._ort_session.run(
                [self._ort_output_name],
                {self._ort_input_name: obs_np},
            )[0]
            return torch.from_numpy(out_np).to(self.device)
        else:  # checkpoint
            x = obs_t
            if self._normalizer is not None:
                x = self._normalizer(x)
            return self._actor(x)

    @torch.no_grad()
    def forward(self, obs_np: np.ndarray) -> np.ndarray:
        """obs (36,) → action (6,) clipped to [-1, 1]."""
        obs_t = torch.from_numpy(obs_np).float().unsqueeze(0).to(self.device)
        act_t = self._infer(obs_t).squeeze(0)
        act_t = act_t.clamp(-1.0, 1.0)
        return act_t.cpu().numpy().astype(np.float32)