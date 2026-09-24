# rsl_rl local patch

Every K / L / R / S training recipe in this repo was run with one change to
`rsl_rl` that is not upstream: the adaptive-LR schedule's ceiling is lowered
from 1e-2 to 1e-3. Without it the recipes do not reproduce (stock 1e-2 rides
diverged three K-line runs late in training).

| | |
|---|---|
| Package | `rsl-rl-lib` **3.1.2** (PyPI wheel `rsl_rl_lib-3.1.2-py3-none-any.whl`) |
| File | `rsl_rl/algorithms/ppo.py` |
| Pristine sha256 | `4373ac1b2f9fdf14d9da57516968fc95d8f605d2967fee01dc61bf0d09423478` |
| Patch | `rsl_rl_lr_ceiling_v2.patch` (one hunk, generated 2026-09-24 with `diff -u` against the wheel) |

## Apply

```sh
SP=$(python -c "import rsl_rl, os; print(os.path.dirname(os.path.dirname(rsl_rl.__file__)))")
patch -p1 -d "$SP" < RL/patches/rsl_rl_lr_ceiling_v2.patch
grep -n "LOCAL PATCH v2" "$SP/rsl_rl/algorithms/ppo.py"   # one hit = applied
```

Re-apply after any `pip install --force-reinstall` or environment rebuild.

## Pairing

Under heavy domain randomisation use `desired_kl` ≈ 0.02 with this ceiling.
A `desired_kl` of 0.008 or lower pins the LR at rsl_rl's 1e-5 floor for most
updates. Never resume a run with `schedule=fixed`; the resume reads cfg
defaults, not the parent's hyperparameters.
