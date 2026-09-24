"""Print exploration sigma (policy noise std) for the P13 R-line eval candidates.

Usage (any python with torch; CPU only):
    python RL/scripts/rsl_rl/print_sigma_p13r.py
"""
import os
import torch

RL = os.path.join(os.path.dirname(__file__), "..", "..")
CKPTS = {
    "R1_3110":  "logs/rsl_rl/p9_replant/2026-08-25_14-54-09_R1_robust/model_3110.pt",
    "R3_3680":  "logs/rsl_rl/p9_replant/2026-08-26_09-09-50_R3_r1plus/model_3680.pt",
    "R1r_3150": "logs/rsl_rl/p9_replant/2026-08-27_10-30-37_R1_resume3110/model_3150.pt",
    "R3r_3760": "logs/rsl_rl/p9_replant/2026-08-27_10-30-37_R3_resume3680/model_3760.pt",
    "K4b_3600": "logs/rsl_rl/p9_replant/2026-08-23_08-56-43_K4b_resume3420/model_3600.pt",
}

for name, rel in CKPTS.items():
    sd = torch.load(os.path.join(RL, rel), map_location="cpu",
                    weights_only=False)["model_state_dict"]
    key = "std" if "std" in sd else next(k for k in sd if k.endswith("std"))
    std = sd[key].flatten()
    joints = " ".join(f"{v:.3f}" for v in std.tolist())
    print(f"{name:10s} sigma_mean={std.mean().item():.4f}  per-joint: {joints}")
