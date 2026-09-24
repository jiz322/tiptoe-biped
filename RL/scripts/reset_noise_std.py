"""Reset action noise std in an RSL-RL checkpoint.

Usage:
    python reset_noise_std.py <checkpoint_path> [--std 0.3] [--output <output_path>]

If --output is not specified, saves to <original_name>_reset.pt
"""

import argparse
import math
import torch


def main():
    parser = argparse.ArgumentParser(description="Reset noise std in RSL-RL checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint file")
    parser.add_argument("--std", type=float, default=0.3, help="New noise std value (default: 0.3)")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: <name>_reset.pt)")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    data = torch.load(args.checkpoint, map_location="cpu")

    # RSL-RL stores model weights in 'model_state_dict'
    if "model_state_dict" in data:
        state_dict = data["model_state_dict"]
    else:
        print(f"Keys in checkpoint: {list(data.keys())}")
        print("ERROR: Could not find 'model_state_dict'")
        return

    # Find the noise std parameter — different RSL-RL versions use different names
    std_keys = [k for k in state_dict.keys() if "std" in k.lower() or "noise" in k.lower()]
    print(f"\nFound std-related keys: {std_keys}")

    modified = False
    for key in state_dict.keys():
        if key in ("std", "log_std", "actor_logstd", "action_log_std"):
            old_val = state_dict[key]
            print(f"\n  Key: '{key}'")
            print(f"  Old shape: {old_val.shape}")
            print(f"  Old values: {old_val.flatten()[:6].tolist()}")

            if "log" in key:
                new_val = torch.full_like(old_val, math.log(args.std))
                print(f"  New value: log({args.std}) = {math.log(args.std):.4f}")
            else:
                new_val = torch.full_like(old_val, args.std)
                print(f"  New value: {args.std}")

            state_dict[key] = new_val
            modified = True
            print(f"  ✅ Updated!")

    if not modified:
        # Try searching more broadly
        print("\nStandard keys not found. Searching all keys...")
        for key, val in state_dict.items():
            if isinstance(val, torch.Tensor) and val.numel() <= 12 and val.numel() >= 1:
                # Small tensors that could be noise params
                if val.min() > 0 and val.max() < 1.0 and val.min() > 0.01:
                    print(f"  Candidate: '{key}' shape={val.shape} values={val.flatten().tolist()}")

        print("\nNo automatic match. Printing all keys for manual inspection:")
        for key, val in state_dict.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: shape={val.shape}")
            else:
                print(f"  {key}: {type(val)}")
        return

    # Save
    output_path = args.output
    if output_path is None:
        output_path = args.checkpoint.replace(".pt", f"_std{args.std}.pt")

    data["model_state_dict"] = state_dict
    torch.save(data, output_path)
    print(f"\nSaved to: {output_path}")
    print(f"Use this as your load_checkpoint in the PPO config.")


if __name__ == "__main__":
    main()
