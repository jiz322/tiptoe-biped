"""
export_policy.py — run on your TRAINING machine to export the rsl_rl
policy as a TorchScript JIT file for sim2real deployment.

Three export paths (tries in order, uses whichever works):

    A. rsl_rl OnPolicyRunner.export()        — if you have the full
       training env + runner available.
    B. Reconstruct from checkpoint .pt       — loads model_state_dict,
       rebuilds actor MLP + obs normalizer, traces to JIT.
    C. Direct checkpoint copy                — just copies the .pt;
       the sim2real policy.py can load raw checkpoints too.

Usage:
    # Method A — from training directory with full env
    python export_policy.py --log-dir logs/biped_phase6/YYYY-MM-DD_HH-MM-SS

    # Method B — from any machine with the checkpoint file
    python export_policy.py --checkpoint model_3000.pt

    # Method C — just copy (policy.py handles reconstruction)
    cp logs/.../model_3000.pt  sim2real/policy.pt

Output: policy.pt (copy to Pi alongside sim2real/)
"""

import argparse
import os
import sys
import torch
import torch.nn as nn


# ════════════════════════════════════════════════════════════
# Method A: full rsl_rl runner export
# ════════════════════════════════════════════════════════════

def export_via_runner(log_dir, output_path):
    """Use rsl_rl's built-in JIT export (requires training env)."""
    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError:
        print("[Export] rsl_rl not installed, skipping method A")
        return False

    # Find latest checkpoint
    models_dir = os.path.join(log_dir, 'models')
    if not os.path.isdir(models_dir):
        # Some versions put checkpoints in log_dir directly
        models_dir = log_dir

    pts = sorted([f for f in os.listdir(models_dir) if f.endswith('.pt')])
    if not pts:
        print(f"[Export] No .pt files in {models_dir}")
        return False

    latest = os.path.join(models_dir, pts[-1])
    print(f"[Export] Found checkpoint: {latest}")
    print(f"[Export] Method A requires the full env setup — "
          f"if this fails, use method B with --checkpoint")
    return False  # placeholder — uncomment below when env is available

    # Uncomment and adapt when your training env is importable:
    # ──────────────────────────────────────────────────────────
    # from your_project.env_cfg import RlEnvCfg
    # from your_project.train_cfg import PPORunnerCfg
    # import gymnasium as gym
    #
    # env = gym.make('Isaac-Biped-v0', cfg=RlEnvCfg())
    # runner = OnPolicyRunner(env, PPORunnerCfg(), log_dir=log_dir)
    # runner.load(latest)
    #
    # export_dir = os.path.dirname(output_path)
    # runner.alg.actor_critic.export(export_dir)
    # exported = os.path.join(export_dir, 'policy_1.pt')  # rsl_rl naming
    # os.rename(exported, output_path)
    # print(f"[Export] Method A success → {output_path}")
    # return True


# ════════════════════════════════════════════════════════════
# Method B: reconstruct actor + normalizer, trace to JIT
# ════════════════════════════════════════════════════════════

class _ObsNormalizer(nn.Module):
    """Exact twin of rsl_rl's EmpiricalNormalization.forward:
        y = (x - _mean) / (_std + eps),  eps = 1e-2  (added to STD, not var).
    Do NOT change to sqrt(var + eps): for near-constant channels
    (init_h std=0.006, difficulty std=0) the two formulas differ 2-3x.
    """
    def __init__(self, mean, std, eps=1e-2):
        super().__init__()
        self.register_buffer('mean', mean.reshape(1, -1).clone())
        self.register_buffer('std',  std.reshape(1, -1).clone())
        self.eps = eps

    def forward(self, x):
        return (x - self.mean) / (self.std + self.eps)


class _ExportablePolicy(nn.Module):
    """Wraps normalizer + actor into a single traceable module."""
    def __init__(self, normalizer, actor):
        super().__init__()
        self.normalizer = normalizer
        self.actor = actor

    def forward(self, obs):
        if self.normalizer is not None:
            obs = self.normalizer(obs)
        return self.actor(obs)


def export_from_checkpoint(checkpoint_path, output_path,
                           activation='elu', obs_dim=36, act_dim=6):
    """Rebuild actor MLP from checkpoint state dict and trace to JIT."""
    print(f"[Export] Method B: loading {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    else:
        sd = ckpt

    # ── Find actor layer keys ────────────────────────────────
    actor_weight_keys = sorted([
        k for k in sd if k.startswith('actor.') and k.endswith('.weight')
    ])
    if not actor_weight_keys:
        print("[Export] ERROR: no 'actor.*' keys in state dict.")
        print(f"[Export] Available keys: {list(sd.keys())[:20]}")
        return False

    # ── Build actor Sequential ───────────────────────────────
    activation_fn = nn.ELU() if activation == 'elu' else nn.ReLU()
    layers = []
    for i, wk in enumerate(actor_weight_keys):
        bk = wk.replace('.weight', '.bias')
        w = sd[wk]
        b = sd.get(bk)
        linear = nn.Linear(w.shape[1], w.shape[0])
        linear.weight.data = w
        if b is not None:
            linear.bias.data = b
        layers.append(linear)
        if i < len(actor_weight_keys) - 1:
            layers.append(activation_fn)

    actor = nn.Sequential(*layers)
    actor.eval()

    # Verify shapes
    test_in = torch.zeros(1, obs_dim)
    test_out = actor(test_in)
    print(f"[Export] Actor: {obs_dim} → hidden → {test_out.shape[-1]}")
    if test_out.shape[-1] != act_dim:
        # Maybe there's an obs normalizer that changes dims (unlikely, but check)
        print(f"[Export] WARNING: actor output dim {test_out.shape[-1]} != "
              f"expected {act_dim}")

    # ── Find normalizer ──────────────────────────────────────
    # rsl_rl 3.x (actor_obs_normalization=True) stores the actor's obs
    # normalizer as actor_obs_normalizer._mean / ._var / ._std / .count.
    # Older rsl_rl / custom exports used running_mean / running_var.
    # Support both; REFUSE to export if normalizer-ish keys exist but none
    # match — silently skipping normalization bakes a JIT file that runs a
    # different function than was trained (see control/policy.py, and the
    # pre-2026-08-06 deployment bug this exact pattern caused there).
    normalizer = None
    if 'actor_obs_normalizer._mean' in sd:
        mean = sd['actor_obs_normalizer._mean']
        std = sd.get('actor_obs_normalizer._std')
        if std is None:  # very old ckpts: only _var stored
            std = torch.sqrt(sd['actor_obs_normalizer._var'])
        normalizer = _ObsNormalizer(mean, std)
        print(f"[Export] Found obs normalizer (actor_obs_normalizer._mean/_std, "
              f"eps=1e-2 on std — rsl_rl formula): mean shape {tuple(mean.shape)}")
    else:
        norm_mean_key = None
        for k in sd:
            if 'running_mean' in k and 'critic' not in k:
                norm_mean_key = k
                break
        if norm_mean_key is not None:
            prefix = norm_mean_key.rsplit('running_mean', 1)[0]
            mean = sd[norm_mean_key]
            var = sd.get(prefix + 'running_var', torch.ones_like(mean))
            # legacy RunningMeanStd convention: y = (x-mean)/sqrt(var+1e-5)
            normalizer = _ObsNormalizer(mean, torch.sqrt(var + 1e-5), eps=0.0)
            print(f"[Export] Found obs normalizer from legacy keys ({norm_mean_key})")

    if normalizer is not None:
        # If normalizer exists, actor input is post-normalization
        # Re-test with normalized input
        test_normed = normalizer(test_in)
        test_out = actor(test_normed)
        print(f"[Export] Actor with normalizer: {obs_dim} → norm → "
              f"hidden → {test_out.shape[-1]}")
    else:
        leftover = [k for k in sd
                    if ('normalizer' in k or 'running_mean' in k)
                    and 'critic' not in k]
        if leftover:
            print(f"[Export] ERROR: checkpoint contains actor obs-normalizer "
                  f"keys {leftover} that this exporter does not recognize. "
                  f"Refusing to export unnormalized. Update export_from_checkpoint.")
            return False
        print(f"[Export] No obs normalizer found "
              f"(actor_obs_normalization=False training) — raw obs is correct")

    # ── Wrap and trace ───────────────────────────────────────
    exportable = _ExportablePolicy(normalizer, actor)
    exportable.eval()

    # Trace
    example_input = torch.zeros(1, obs_dim)
    traced = torch.jit.trace(exportable, example_input)

    # Verify traced output matches
    with torch.no_grad():
        orig_out = exportable(example_input)
        traced_out = traced(example_input)
        diff = (orig_out - traced_out).abs().max().item()
        print(f"[Export] Trace verification: max diff = {diff:.2e}")

    # Save
    traced.save(output_path)
    file_size = os.path.getsize(output_path) / 1024
    print(f"[Export] Saved: {output_path} ({file_size:.1f} KB)")
    print(f"[Export] Copy this file to your Pi as 'policy.pt'")
    return True


# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Export rsl_rl policy to TorchScript for sim2real deployment"
    )
    parser.add_argument('--log-dir', type=str, default=None,
                        help="rsl_rl log directory (method A)")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to checkpoint .pt file (method B)")
    parser.add_argument('--output', type=str, default='policy.pt',
                        help="Output JIT file path")
    parser.add_argument('--activation', type=str, default='elu',
                        choices=['elu', 'relu'],
                        help="Activation function used in training")
    parser.add_argument('--obs-dim', type=int, default=36)
    parser.add_argument('--act-dim', type=int, default=6)
    args = parser.parse_args()

    if args.log_dir:
        if export_via_runner(args.log_dir, args.output):
            return

        # Try finding a checkpoint in the log dir
        models_dir = os.path.join(args.log_dir, 'models')
        if not os.path.isdir(models_dir):
            models_dir = args.log_dir
        pts = sorted([f for f in os.listdir(models_dir) if f.endswith('.pt')])
        if pts:
            ckpt_path = os.path.join(models_dir, pts[-1])
            print(f"[Export] Falling back to method B with {ckpt_path}")
            if export_from_checkpoint(ckpt_path, args.output,
                                       args.activation, args.obs_dim, args.act_dim):
                return

    if args.checkpoint:
        if export_from_checkpoint(args.checkpoint, args.output,
                                   args.activation, args.obs_dim, args.act_dim):
            return

    print("\nUsage:")
    print("  python export_policy.py --checkpoint model_3000.pt")
    print("  python export_policy.py --log-dir logs/biped_phase6/run_001")
    sys.exit(1)


if __name__ == '__main__':
    main()