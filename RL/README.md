# RL: training, evaluation and sim2real tools (Isaac Lab)

Isaac Lab extension holding the TipToe stand-up and balance task, the PPO
recipe, and the tools that compared sim to hardware. Everything in the
retrospective's fixes #1–#3, #6–#8 and #15 lives here.

## Install

Pinned versions are in `docs/ENVIRONMENT.md` (Isaac Sim 5.1.0, Isaac Lab
0.54.0, rsl-rl-lib 3.1.2, torch 2.7.0+cu128, Python 3.11). Other Isaac Lab
versions may work; they were not tested.

```bash
# 1. Isaac Lab installed and importable in the active Python (conda env or isaaclab.sh -p)
# 2. this task package
python -m pip install -e RL/source/RL
python RL/scripts/list_envs.py | grep standup_curriculum     # Template-standup_curriculum-v0
# 3. the rsl_rl patch — required; every recipe assumes the 1e-3 LR ceiling
cat RL/patches/README.md
```

On Windows with the Isaac Sim binary install, the Kit runtime needs
environment variables before Python starts. The scripts in `RL/launch/` set
them from `ISAACSIM_PATH` and `CONDA_PREFIX`; on Linux the standard Isaac Lab
conda env is enough.

## Layout

| Path | What |
|---|---|
| `source/RL/RL/tasks/direct/standup_curriculum/` | The task. `rl_env_p6.py` (env), `rl_env_cfg_p6.py` (every knob, sectioned §1–§12 with impact tags), `belt_knee_actuator.py` (the belt-coupled knee: motor-side PD, rotor inertia, off-diagonal coupling, series elasticity), `foot_contact.py` (ball-foot material), `pad_contact.py` (retired pad model, kept for the record), `agents/rsl_rl_ppo_cfg_p6.py`. |
| `scripts/rsl_rl/train.py`, `play.py` | Isaac Lab's standard entry points. |
| `scripts/rsl_rl/play_log.py`, `play_sim2real.py` | Rollouts that log the same npz schema as the robot (`data/README.md`). `play_sim2real.py` is the single-env failure-analysis tool: `--handover {training,deploy}`, `--inject_npz` to start the sim from a recorded hardware state. |
| `scripts/rsl_rl/eval*.py`, `play_with_push.py`, `play_alpha.py` | Headless evaluation: survival, push recovery, stand-up success at fixed curriculum difficulty. |
| `scripts/rsl_rl/probe_*.py`, `verify_*.py`, `sweep_*.py`, `sysid_*.py` | Plant probes (stand-up ramp feasibility, balance manifold, pose dumps, foot contact), DR verification, plant-parameter identification against hardware (`sysid_optuna.py`). |
| `scripts/patch_*.py`, `extend_critic_checkpoint.py`, `reduce_noise_std.py`, `reset_noise_std.py` | Checkpoint surgery (obs-width expansion, curriculum normaliser patch, noise std). |
| `launch/` | Maintained launchers: `train_S7_fullDR.{ps1,sh}` (the last training recipe), `play.{ps1,sh}` (the released checkpoint). |
| `patches/` | The rsl_rl LR-ceiling patch and how to apply it. |
| `../legacy/launch/` | Every historical arm with its design note. |

## Train

```powershell
# Windows
$env:ISAACSIM_PATH = 'C:\path\to\isaac-sim'; conda activate env_isaaclab
powershell -File RL\launch\train_S7_fullDR.ps1 -Gpu 0 -MaxIter 5 -Headless   # smoke: read the [PLANT] lines
powershell -File RL\launch\train_S7_fullDR.ps1 -Gpu 0 -Headless              # full run
```

```bash
# Linux
RL/launch/train_S7_fullDR.sh --gpu 0 --max-iter 5 --headless
RL/launch/train_S7_fullDR.sh --gpu 0 --headless
```

Logs: `RL/logs/train_<run>.{out,err}.log`, TensorBoard under
`RL/logs/rsl_rl/`. The env prints a startup contract (`[PLANT] ...` lines:
belt series band, restitution rows, encoder write slots). Read them before
believing a run.

Three rules that cost the campaign weeks:

1. **Overrides are CLI-only and a resume must re-pass every one of them.**
   `train.py --resume` reads cfg defaults for anything not on the command line.
   The launch scripts are therefore complete recipes, never deltas.
2. **Never resume with `schedule=fixed`**, and pair `desired_kl` ≈ 0.02 with the
   patched 1e-3 LR ceiling (`patches/README.md`).
3. **Curriculum advance must use `curriculum_advance_metric=success_high_rel`.**
   The episode-length criterion is inflated by grace time.

What to expect: the released policy R1r_3150 is iteration 3150 of a lineage
that started from scratch in the K line (`legacy/launch/README.md`); the S7
recipe shipped here has only ever been run as a resume from S4a. A fresh
start with it trains, but its outcome is not on record. Default env count is
in `rsl_rl_ppo_cfg_p6.py` (the campaign used 32 768 envs on one RTX 4090).

## Play the released checkpoint

```powershell
powershell -File RL\launch\play.ps1 -Gpu 0 -NumEnvs 16          # viewer
powershell -File RL\launch\play.ps1 -Gpu 0 -Headless -Video     # mp4
```

```bash
RL/launch/play.sh --gpu 0 --num-envs 16
```

This loads `control/r1r_3150.pt` with the R1 plant recipe (servo gains kp
20/30, hip armature 0.0385, saturation fences) and no pushes. It is the policy
that stood for 6.06 s on 2026-09-03. In sim it balances essentially
indefinitely (hardware-matched survival 0.996 in the 08-27 eval).

## Sim vs hardware

`play_sim2real.py` reproduces a recorded trial: same handover contract, same
trim, optional state injection from the npz. The analysis scripts in
`data/` compare the two. The sim2real checklist in `docs/REPO_MAP.md` lists
the contracts that must be diffed, not assumed.

Known data bug: sim npz written before 2026-08-31 have the knee `target`
channel ×2. Recompute targets from `action` when reading old sim logs.
Hardware npz are unaffected.

## Provenance

`source/RL/` started from the Isaac Lab extension template (BSD-3-Clause,
notices retained). Everything else is MIT (`LICENSE`).
