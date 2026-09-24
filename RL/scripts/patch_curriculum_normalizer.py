"""
patch_curriculum_normalizer.py — make a curriculum-OFF checkpoint resumable
with the kneel→stand curriculum ON (S-line, 2026-09-06).

Every K/L/R checkpoint trained with curriculum_enable=False, so the two
bookkeeping obs channels `difficulty` (per-frame index 29) and `grace_norm`
(index 30) were constant 0 and their EmpiricalNormalization stats are
mean=0, std=0. forward() is (x-mean)/(std+0.01): the first kneel-branch
episode would feed difficulty 0.5 → 50 and grace_norm 1.0 → 100 into the
actor AND the critic (obs_groups critic = policy+critic, so the critic's
first 180 dims are the same channels). With count ≈ 1.3e10 the running
stats adapt at ~3e-4 per update — far too slow.

This script sets those channels (all obs_history_steps frames) to a sane
prior for the [0, 1] range and leaves every other channel and `count`
untouched (re-adapting all stats would rescale the joint-pos channels as
kneel poses widen them and perturb the balancing skill we resume for).

Usage (Isaac python — needs the launch-script env vars for numpy):
    python RL/scripts/patch_curriculum_normalizer.py <in.pt> <out.pt>
"""
import sys

import torch

OBS_PER_FRAME = 36
HISTORY_STEPS = 5
CH_DIFFICULTY = 29
CH_GRACE = 30
PRIOR = {
    CH_DIFFICULTY: (0.15, 0.25),   # (mean, std) — kneel-branch d ~ U[0, alpha], 40 % of envs
    CH_GRACE: (0.10, 0.25),        # grace_norm decays d → 0 over the grace window
}


def patch(sd: dict, prefix: str) -> None:
    mean = sd[f"{prefix}._mean"]
    var = sd[f"{prefix}._var"]
    std = sd[f"{prefix}._std"]
    n = mean.shape[-1]
    assert n >= OBS_PER_FRAME * HISTORY_STEPS, f"{prefix}: dim {n} too small"
    before = std.clone()
    touched = []
    for k in range(HISTORY_STEPS):
        for ch, (m, s) in PRIOR.items():
            idx = ch + OBS_PER_FRAME * k
            assert std[..., idx].abs().max() < 1e-6, (
                f"{prefix} ch{idx}: std={std[..., idx].item():.3g} is not zero — "
                "layout assumption broken, refusing to patch")
            mean[..., idx] = m
            std[..., idx] = s
            var[..., idx] = s * s
            touched.append(idx)
    untouched = torch.ones(n, dtype=torch.bool)
    untouched[touched] = False
    assert torch.equal(before[..., untouched], std[..., untouched])
    print(f"  {prefix}: patched channels {touched}; count={int(sd[f'{prefix}.count'])} kept")


def main(in_path: str, out_path: str) -> None:
    ck = torch.load(in_path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    for prefix in ("actor_obs_normalizer", "critic_obs_normalizer"):
        patch(sd, prefix)
    torch.save(ck, out_path)
    print(f"  wrote {out_path} (iter {ck.get('iter')})")

    # Read-back check.
    rb = torch.load(out_path, map_location="cpu", weights_only=False)["model_state_dict"]
    for prefix in ("actor_obs_normalizer", "critic_obs_normalizer"):
        for k in range(HISTORY_STEPS):
            for ch, (m, s) in PRIOR.items():
                idx = ch + OBS_PER_FRAME * k
                assert abs(rb[f"{prefix}._mean"][..., idx].item() - m) < 1e-6
                assert abs(rb[f"{prefix}._std"][..., idx].item() - s) < 1e-6
    print("  read-back OK")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
