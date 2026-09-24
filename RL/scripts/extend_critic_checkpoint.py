"""
extend_critic_checkpoint.py
===========================
Extends a saved rsl_rl checkpoint's critic input dimension by N new
privileged-obs slots (default 1, for the slope_deg feature).

The new weight columns are zero-initialized, so the critic's output is
*mathematically identical* to before — no retraining needed. The
normalizer stats are extended with mean=0, var=1 (unit-normal prior).

Usage:
    python extend_critic_checkpoint.py model_2362.pt model_2362_ext.pt

Then resume from model_2362_ext.pt with priv_obs_dim bumped by 1 in the cfg.
"""

import sys
import torch
import copy


def extend_checkpoint(in_path: str, out_path: str, n_new: int = 1):
    ckpt = torch.load(in_path, map_location="cpu")

    # ── Locate the state dict ────────────────────────────────────
    # rsl_rl saves either at top level or under a key like
    # 'model_state_dict'. Inspect and adapt if yours differs.
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt  # flat checkpoint

    modified = []

    # ── 1. Extend critic first-layer weight ──────────────────────
    # Typical key: "critic.0.weight" — shape [hidden, critic_input_dim]
    # The new slots are the LAST columns (appended at the end of priv obs).
    for key in list(sd.keys()):
        if "critic" in key and key.endswith(".weight"):
            w = sd[key]
            if w.dim() == 2:
                # Only extend the FIRST linear layer (largest input dim).
                # Later layers have hidden→hidden dims that shouldn't change.
                # Heuristic: it's the first layer if input_dim > output_dim
                # (or close), i.e. the wide one.
                in_dim, out_dim = w.shape[1], w.shape[0]
                # The first layer maps [critic_obs_dim] → [hidden_dim].
                # For a 512/256/128 critic, the first layer is the one
                # where in_dim (245) < out_dim (512) is NOT true, but
                # in_dim (245) is much larger than subsequent layers'
                # in_dim (512, 256). So: pick the first .weight we see.
                new_cols = torch.zeros(out_dim, n_new)
                sd[key] = torch.cat([w, new_cols], dim=1)
                modified.append(f"{key}: {list(w.shape)} → {list(sd[key].shape)}")
                break  # only the first layer

    # ── 2. Extend critic obs normalizer ──────────────────────────
    # Typical keys: "critic_obs_normalizer.running_mean" (and running_var)
    for key in list(sd.keys()):
        if "critic" in key and "running_mean" in key:
            old = sd[key]
            sd[key] = torch.cat([old, torch.zeros(n_new)])
            modified.append(f"{key}: [{old.shape[0]}] → [{sd[key].shape[0]}]")

        if "critic" in key and "running_var" in key:
            old = sd[key]
            sd[key] = torch.cat([old, torch.ones(n_new)])  # var=1 → unit normal
            modified.append(f"{key}: [{old.shape[0]}] → [{sd[key].shape[0]}]")

    # ── 3. Save ──────────────────────────────────────────────────
    if not modified:
        print("⚠️  No keys were modified. Inspect your checkpoint structure:")
        for k, v in sd.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: {list(v.shape)}")
            else:
                print(f"    {k}: {type(v)}")
        sys.exit(1)

    torch.save(ckpt, out_path)
    print(f"✓ Saved extended checkpoint to {out_path}")
    print(f"  Modified {len(modified)} tensors:")
    for m in modified:
        print(f"    {m}")
    print(f"\n  Remember: set priv_obs_dim = {245 - 180 + n_new}"
          f" (old {245 - 180} + {n_new}) in rl_env_cfg_p6.py")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.pt> <output.pt> [n_new_dims=1]")
        sys.exit(1)
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    extend_checkpoint(sys.argv[1], sys.argv[2], n)
