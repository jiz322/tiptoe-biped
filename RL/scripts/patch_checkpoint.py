#!/usr/bin/env python3
"""
patch_checkpoint_p4.py — Expand a phase 3 checkpoint (obs=32) for phase 4 (obs=36).

Adds 4 zero-initialized input dimensions to the actor and critic first layers,
and expands the observation normalizer buffers.

Phase 3 obs (32): [32 base]
Phase 4 obs (36): [32 base] + [com_cmd_dx, com_cmd_dy] + [yaw_err_sin, yaw_err_cos]

The first 32 dims are identical — weights are preserved.
The 4 new dims are zero-initialized:
    dim 32: com_cmd_dx   (COM offset command, forward)
    dim 33: com_cmd_dy   (COM offset command, lateral)
    dim 34: yaw_err_sin  (sin of heading error from spawn)
    dim 35: yaw_err_cos  (cos of heading error from spawn)

Usage:
    python patch_checkpoint_p4.py <input_checkpoint.pt> <output_checkpoint.pt>

Example:
    python patch_checkpoint_p4.py logs/bipedal_phase3_pushrecovery/model_1500.pt \
                                  logs/phase3_patched/model_for_p4_patched.pt
"""

import sys
import os
import torch


OLD_OBS_DIM = 32
NEW_OBS_DIM = 36
EXTRA_DIMS = NEW_OBS_DIM - OLD_OBS_DIM


def patch_linear_weight(state_dict: dict, key: str) -> bool:
    """Expand a linear layer's weight from (out, OLD) → (out, NEW)."""
    if key not in state_dict:
        return False
    w = state_dict[key]
    if w.shape[1] == OLD_OBS_DIM:
        new_w = torch.zeros(w.shape[0], NEW_OBS_DIM, dtype=w.dtype, device=w.device)
        new_w[:, :OLD_OBS_DIM] = w
        state_dict[key] = new_w
        print(f"  Patched {key}: {w.shape} → {new_w.shape}")
        return True
    elif w.shape[1] == NEW_OBS_DIM:
        print(f"  {key} already has {NEW_OBS_DIM} dims, skipping")
        return False
    else:
        print(f"  WARNING: {key} has unexpected shape {w.shape}, skipping")
        return False


def patch_normalizer(state_dict: dict, prefix: str) -> bool:
    """Expand normalizer buffers from OLD → NEW obs dims.

    Handles both shapes: (OLD,) and (1, OLD) — rsl_rl uses (1, N).
    Handles key suffixes: _mean, _var, _std, running_mean, running_var.
    """
    patched = False
    for suffix in ["_mean", "_var", "_std", "running_mean", "running_var"]:
        key = f"{prefix}{suffix}"
        if key not in state_dict:
            continue
        v = state_dict[key]
        if not isinstance(v, torch.Tensor):
            continue

        if v.dim() == 1 and v.shape[0] == OLD_OBS_DIM:
            new_v = torch.zeros(NEW_OBS_DIM, dtype=v.dtype, device=v.device)
            new_v[:OLD_OBS_DIM] = v
            if "var" in suffix:
                new_v[OLD_OBS_DIM:] = 1.0
            if "std" in suffix:
                new_v[OLD_OBS_DIM:] = 1.0
            state_dict[key] = new_v
            print(f"  Patched {key}: {list(v.shape)} → {list(new_v.shape)}")
            patched = True
        elif v.dim() == 2 and v.shape[1] == OLD_OBS_DIM:
            new_v = torch.zeros(v.shape[0], NEW_OBS_DIM, dtype=v.dtype, device=v.device)
            new_v[:, :OLD_OBS_DIM] = v
            if "var" in suffix:
                new_v[:, OLD_OBS_DIM:] = 1.0
            if "std" in suffix:
                new_v[:, OLD_OBS_DIM:] = 1.0
            state_dict[key] = new_v
            print(f"  Patched {key}: {list(v.shape)} → {list(new_v.shape)}")
            patched = True
        elif (v.dim() == 1 and v.shape[0] == NEW_OBS_DIM) or \
             (v.dim() == 2 and v.shape[1] == NEW_OBS_DIM):
            print(f"  {key} already correct {list(v.shape)}, skipping")
        else:
            print(f"  {key} has shape {list(v.shape)}, not matching obs dim — skipping")

    # Also handle count if present
    for count_suffix in ["count", ".count"]:
        count_key = f"{prefix}{count_suffix}"
        if count_key in state_dict:
            print(f"  {count_key} = {state_dict[count_key]}, keeping as-is")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.pt> <output.pt>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Loading checkpoint: {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

    if "model_state_dict" in checkpoint:
        sd = checkpoint["model_state_dict"]
    else:
        sd = checkpoint

    print(f"\nCheckpoint keys: {list(sd.keys())[:20]}...")
    print(f"\nPatching from {OLD_OBS_DIM} → {NEW_OBS_DIM} obs dims ({EXTRA_DIMS} new dims)")
    print(f"  New dims: [com_cmd_dx, com_cmd_dy, yaw_err_sin, yaw_err_cos]\n")

    # ── Patch actor first layer ───────────────────────────────
    print("Actor:")
    actor_patched = patch_linear_weight(sd, "actor.0.weight")
    if not actor_patched:
        for key in list(sd.keys()):
            if "actor" in key and "weight" in key and "0" in key:
                patch_linear_weight(sd, key)
                break

    # ── Patch critic first layer ──────────────────────────────
    print("\nCritic:")
    critic_patched = patch_linear_weight(sd, "critic.0.weight")
    if not critic_patched:
        for key in list(sd.keys()):
            if "critic" in key and "weight" in key and "0" in key:
                patch_linear_weight(sd, key)
                break

    # ── Patch normalizers ─────────────────────────────────────
    print("\nNormalizers:")
    for prefix in [
        "actor_obs_normalizer.",
        "critic_obs_normalizer.",
        "obs_normalizer.",
    ]:
        patch_normalizer(sd, prefix)

    # Catch any remaining obs-dim-sized tensors we missed
    print("\nScanning for any remaining unpatched obs-dim tensors:")
    for key, val in sd.items():
        if not isinstance(val, torch.Tensor):
            continue
        if val.dim() == 1 and val.shape[0] == OLD_OBS_DIM:
            print(f"  WARNING: {key} still has shape {list(val.shape)} — may need patching")
        elif val.dim() == 2 and val.shape[1] == OLD_OBS_DIM:
            print(f"  WARNING: {key} still has shape {list(val.shape)} — may need patching")

    # ── Save ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if "model_state_dict" in checkpoint:
        checkpoint["model_state_dict"] = sd
    else:
        checkpoint = sd

    if "optimizer_state_dict" in checkpoint:
        opt_sd = checkpoint["optimizer_state_dict"]
        if isinstance(opt_sd, dict) and "state" in opt_sd:
            opt_sd["state"] = {}
            print("\nCleared optimizer momentum state (shapes changed, will rebuild)")
        else:
            checkpoint["optimizer_state_dict"] = {"state": {}, "param_groups": []}
            print("\nReset optimizer state to empty")

    torch.save(checkpoint, output_path)
    print(f"\nSaved patched checkpoint: {output_path}")
    print("Done! Update your PPO config to load this checkpoint.")


if __name__ == "__main__":
    main()