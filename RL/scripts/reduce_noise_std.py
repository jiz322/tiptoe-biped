#!/usr/bin/env python3
"""
reduce_noise_std.py — Reduce the learned noise std in a checkpoint.

When the reward landscape doesn't create enough gradient to push noise
std down (due to action clipping killing gradients), this script
directly sets it to a lower value.

The policy means are preserved — only the exploration noise changes.
Run this before resuming training so the policy starts from a tighter
distribution and fine-tunes from there.

Usage:
    python reduce_noise_std.py <input.pt> <output.pt> [target_std]

Example:
    python reduce_noise_std.py model_8998.pt model_lownoise.pt 0.4
"""

import sys
import os
import torch
import math


def find_noise_std_keys(state_dict: dict) -> list[str]:
    """Find all keys that look like they store the noise std."""
    candidates = []
    for key, val in state_dict.items():
        # rsl_rl stores noise as 'std' or 'log_std' in the actor
        if any(s in key.lower() for s in ["log_std", "noise", "std"]):
            candidates.append(key)
        # Also check for scalar parameters that could be log_std
        if val.dim() == 1 and val.shape[0] <= 12 and "actor" in key.lower():
            candidates.append(key)
    return list(set(candidates))


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.pt> <output.pt> [target_std]")
        print(f"  target_std defaults to 0.4")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    target_std = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4

    print(f"Loading: {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

    if "model_state_dict" in checkpoint:
        sd = checkpoint["model_state_dict"]
    else:
        sd = checkpoint

    # ── Find and display current noise std ────────────────────
    print(f"\nAll keys in state dict:")
    for key, val in sd.items():
        shape_str = str(list(val.shape)) if isinstance(val, torch.Tensor) else type(val).__name__
        print(f"  {key}: {shape_str}")

    noise_keys = find_noise_std_keys(sd)
    print(f"\nCandidate noise std keys: {noise_keys}")

    modified = False
    for key in noise_keys:
        val = sd[key]
        if not isinstance(val, torch.Tensor):
            continue

        print(f"\n── {key} ──")
        print(f"  Shape: {list(val.shape)}")
        print(f"  Values: {val.tolist()}")

        # Determine if this is log_std or std
        if "log" in key.lower():
            current_std = torch.exp(val)
            print(f"  Current std (exp): {current_std.tolist()}")
            new_log_std = torch.full_like(val, math.log(target_std))
            sd[key] = new_log_std
            print(f"  → Set to log({target_std}) = {math.log(target_std):.4f}")
            modified = True
        else:
            # Might be raw std
            if val.min() > 0 and val.max() < 5.0:
                print(f"  Looks like raw std values")
                sd[key] = torch.full_like(val, target_std)
                print(f"  → Set to {target_std}")
                modified = True
            else:
                print(f"  Not modifying (doesn't look like std)")

    if not modified:
        print("\n[WARNING] No noise std parameters were modified!")
        print("You may need to inspect the checkpoint manually.")
        print("Common key names: 'std', 'log_std', 'actor.std', 'noise'")

    # ── Save ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if "model_state_dict" in checkpoint:
        checkpoint["model_state_dict"] = sd

    # Keep optimizer state — only noise changed, shapes are fine
    torch.save(checkpoint, output_path)
    print(f"\nSaved: {output_path}")
    print(f"Target noise std: {target_std}")
    print(f"\nResume training from this checkpoint.")
    print(f"The policy will start with tighter actions and fine-tune.")


if __name__ == "__main__":
    main()
