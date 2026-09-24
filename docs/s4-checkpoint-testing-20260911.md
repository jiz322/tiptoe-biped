# S4 checkpoint testing — sim rollout procedure

**Written 2026-09-11.** How to watch an S-line checkpoint in sim, and the one
rule that governs every command below: **overrides live in the launch script,
not in the checkpoint.** A resume or a play run must re-pass the full recipe or
it silently reverts to `rl_env_cfg_p6` defaults. This has burned the G runs
(CLAUDE.md) and it burned the first S4a play attempt on 09-11.

---

## 0. Preflight — run this before anything else

Every PowerShell session in this workflow starts by de-duplicating `PATH`. The
launch scripts prepend `<conda-env>` and its
`Library\bin` / `Scripts` / `bin` subdirectories to a `PATH` that already
contains them, and repeated launches (or a shell that has sourced several
`launch_*.ps1` files) grow it into duplicates and empty entries. Isaac Sim's
extension loader is sensitive to that; the symptom is a confusing import or
DLL failure at startup rather than a clean error.

Run this in the shell you will launch from, before any `python` invocation:

```powershell
# Remove duplicates while preserving order
$path = ($env:Path -split ';' | Select-Object -Unique) -join ';'
$env:Path = $path

# Remove empty entries and trailing semicolons
$env:Path = ($env:Path -split ';' | Where-Object { $_.Trim() -ne '' }) -join ';'

# Verify new length
$env:Path.Length
```

The final `.Length` is the check. Run it again after the fix and compare: if it
did not shrink, `PATH` was already clean. Do this **before** the runs, not
after a failure has already confused the log.

The environment-variable preamble the launch scripts set is assumed by every
command below:

```powershell
$envRoot = '<conda-env>'
$env:CUDA_VISIBLE_DEVICES = 'GPU-<uuid>'   # RTX 4090
$env:CARB_APP_PATH = '<IsaacLab>\_isaac_sim\kit'
$env:EXP_PATH      = '<IsaacLab>\_isaac_sim\apps'
$env:ISAAC_PATH    = '<IsaacLab>\_isaac_sim'
$env:PYTHONPATH    = ';<IsaacLab>\_isaac_sim\site'
$env:RESOURCE_NAME = 'IsaacSim'
$env:CONDA_PREFIX  = $envRoot
$env:PATH = "$envRoot;$envRoot\Library\mingw-w64\bin;$envRoot\Library\usr\bin;$envRoot\Library\bin;$envRoot\Scripts;$envRoot\bin;" + $env:PATH
```

Then `cd <repo>\RL`. Use `python.exe` directly, never
`conda run` (it buffers stdout, so the live training log goes dark).

---

## 1. Run under test

| | |
|---|---|
| Run dir | `RL/logs/rsl_rl/p9_replant/2026-09-09_22-13-40_S4a_measDR/` |
| Checkpoint | `model_7780.pt` (highest in the dir; confirm with a fresh `ls`) |
| Launch script | `RL/launch_train_S4a_measDR.ps1` (arm A, GPU 0) |
| Sibling arm | `2026-09-09_22-13-44_S4b_measDR_pg05/` — differs in exactly one knob |

Final training numbers, from `RL/logs/train_S4a_measDR.out.log`:

| Channel | Value |
|---|---|
| `curriculum/standup_success_high` | 0.867 |
| `curriculum/standup_success_mid` | 0.880 |
| `curriculum/standup_success_low` | 0.829 |
| `curriculum/standup_success_all` | 0.859 |
| `curriculum/advance_metric_value` | 1.046 (bar 0.7) |
| `curriculum/strat_high_diff_mean` | 0.921 |

The run is healthy and the checkpoint stands up across the whole manifold. A
play run that shows every env falling is a command problem, not a checkpoint
problem — check the recipe re-pass before hypothesising about the policy.

---

## 2. `play_sim2real.py` — deploy-scenario rollout

The script forks `play_log.py` and reproduces the `control/balance_log.py`
scenario: one env, no pickup, curriculum off (stand-branch resets only, d=0),
commands pinned to `vx=0, yaw_rate=0`, gait clock pinned per reset, and the
deploy contract gains (hip_roll 6.0/0.3, hip_pitch 20.0/1.4, knee 30.0/1.5,
LPF alpha 0.6) applied as defaults that any explicit `env.*` CLI arg overrides.

Because it forces `curriculum_enable = False`, this is a **balance-branch** test,
not a stand-up test. Use it to ask "does this checkpoint hold the stand pose on
the deploy plant", not "does it get up".

```powershell
& "$envRoot\python.exe" -u scripts/rsl_rl/play_sim2real.py `
  --task=Template-standup_curriculum-v0 `
  --checkpoint="<repo>\RL\logs\rsl_rl\p9_replant\2026-09-09_22-13-40_S4a_measDR\model_7780.pt" `
  --gait_phase_init 0 `
  --seed=0 `
  env.rew_scale_gait_ref=3.0 `
  env.sat_term_threshold=0.5 env.sat_term_tau_s=0.5 `
  env.swing_clearance_zero_m=0.078 `
  env.servo_kp_hip_pitch=20.0 env.servo_kd_hip_pitch=1.4 `
  env.servo_kp_knee=30.0 env.servo_kd_knee=1.5 `
  env.hip_pitch_armature=0.0385 `
  env.swing_clearance_target_m=0.015 `
  env.rew_scale_no_flight=-15.0 env.rew_scale_vz_penalty=-8.0 `
  env.rew_scale_action_smoothness=-0.24 env.rew_scale_action_snap=-0.06 `
  env.exec_noise_target_bias_max=0.03 env.exec_noise_target_std=0.015 `
  env.noise_joint_pos=0.18 env.init_base_orient_noise_rad=0.12 `
  env.push_enable=true env.push_probability=0.001 `
  env.push_force_min=10.0 env.push_force_max=40.0 env.push_duration_steps=10 `
  env.dr_action_delay_min=1 env.dr_action_delay_max=2 `
  env.dr_imu_delay_min=1 env.dr_imu_delay_max=1 `
  env.dr_joint_delay_min=1 env.dr_joint_delay_max=1 `
  env.dr_floor_friction_min=0.6 env.dr_floor_friction_max=1.1 `
  env.dr_stiffness_mult_min=0.90 env.dr_stiffness_mult_max=1.10 `
  env.dr_armature_mult_min=0.80 env.dr_armature_mult_max=1.20 `
  env.dr_effort_mult_min=0.9 `
  env.dr_com_offset_max=0.01 `
  env.dr_com_shift_x_max=0.02 env.dr_com_shift_y_max=0.01 `
  env.dr_proj_grav_bias_max=0.13 `
  env.dr_mass_mult_min=1.00 env.dr_mass_mult_max=1.14 `
  env.dr_ang_vel_bias_max=0.05 env.dr_gravity_tilt_max_deg=2.0 `
  env.pickup_enable=false `
  env.standup_ramp_mode=path
```

Notes on this specific command:

- **`--gait_phase_init 0`** matches the hardware clock start. The script's own
  default is `pi`, which is the value verified 08-28 to make sim lead with the
  LEFT leg. Drop the flag to get the script default.
- **`--num_envs` defaults to 1** here and line 998 clamps it, so the cfg's
  32768 never leaks in. Pass `--num_envs=4` for a handful.
- The PPO knobs (`entropy_coef`, `desired_kl`, `clip_param`) are deliberately
  omitted: they are agent args, not `env.*`, and play loads them from the
  checkpoint's own config via `cli_args.update_rsl_rl_cfg`.
- `env.dr_proj_grav_bias_max=0.13` is arm A's value. To pin the deploy trim
  instead of sampling the band, add
  `env.dr_proj_grav_bias_pin_x=<value>` (the hw-twin launcher uses `-0.089`
  for the trim-on contract and `0.0` for trim-off).

A launch wrapper already encodes this override family for the balance branch:
`RL/run_s4_hwtwin_GPU2.ps1 -Arm S4a_m089`. It pins `model_6520.pt` and adds a
`dr_proj_grav_bias_pin_x`; bump the path to the latest checkpoint if you want
current weights.

---

## 3. `play_alpha.py` — fixed-difficulty stand-up view

This is the interactive viewer for "does it actually stand up". It pins every
env to one curriculum difficulty and freezes advancement.

```powershell
& "$envRoot\python.exe" -u scripts/rsl_rl/play_alpha.py `
  --task=Template-standup_curriculum-v0 `
  --checkpoint="<repo>\RL\logs\rsl_rl\p9_replant\2026-09-09_22-13-40_S4a_measDR\model_7780.pt" `
  --num_envs=16 --seed=0 `
  --alpha=0.5 --alpha-random `
  env.standup_ramp_mode=path `
  env.standup_grace_seconds=5.0 `
  env.standup_path_spawn_clearance_m=0.006 `
  env.rew_scale_standup_track=4.0 env.rew_scale_standup_contact=2.0 `
  env.swing_clearance_zero_m=0.078 `
  env.servo_kp_hip_pitch=20.0 env.servo_kd_hip_pitch=1.4 `
  env.servo_kp_knee=30.0 env.servo_kd_knee=1.5 `
  env.hip_pitch_armature=0.0385 `
  env.sat_term_threshold=0.5 env.sat_term_tau_s=0.5 `
  env.rew_scale_gait_ref=3.0 `
  env.swing_clearance_target_m=0.015 `
  env.rew_scale_no_flight=-15.0 env.rew_scale_vz_penalty=-8.0 `
  env.rew_scale_action_smoothness=-0.24 env.rew_scale_action_snap=-0.06 `
  env.exec_noise_target_bias_max=0.03 env.exec_noise_target_std=0.015 `
  env.noise_joint_pos=0.18 env.init_base_orient_noise_rad=0.12 `
  env.dr_action_delay_min=1 env.dr_action_delay_max=2 `
  env.dr_imu_delay_min=1 env.dr_imu_delay_max=1 `
  env.dr_joint_delay_min=1 env.dr_joint_delay_max=1 `
  env.dr_floor_friction_min=0.6 env.dr_floor_friction_max=1.1 `
  env.dr_stiffness_mult_min=0.90 env.dr_stiffness_mult_max=1.10 `
  env.dr_armature_mult_min=0.80 env.dr_armature_mult_max=1.20 `
  env.dr_effort_mult_min=0.9 `
  env.dr_com_offset_max=0.01 `
  env.dr_com_shift_x_max=0.02 env.dr_com_shift_y_max=0.01 `
  env.dr_mass_mult_min=1.00 env.dr_mass_mult_max=1.14 `
  env.dr_proj_grav_bias_max=0.13 `
  env.dr_ang_vel_bias_max=0.05 env.dr_gravity_tilt_max_deg=2.0
```

Three things about this command that are easy to get wrong:

- **`--num_envs` default is `None`**, which falls through to
  `env_cfg.scene.num_envs = 32768` (`play_alpha.py:119`). Always set it. This is
  the opposite of `play_sim2real.py`, where the default of 1 is safe.
- **`--alpha=0.5` alone pins every env to one identical pose.** With
  `_force_fixed_difficulty` set, `_reset_idx` gives all envs the same difficulty
  and marks them all as kneel-branch starts (`rl_env_p6.py:2729-2737`). Training
  never samples a fixed d; the kneel branch draws `U[0, alpha]` with a frontier
  share. Add `--alpha-random` to use the distribution the policy actually
  trained on. Use the fixed pin when you deliberately want the worst-case
  single-pose probe.
- **Grace is the action-center travel budget.** It scales with difficulty
  (`difficulty × max_grace_steps`), and the action center slides `s0 → 1.0` at a
  constant rate over that window (`rl_env_p6.py:1415-1418`). `standup_grace_seconds`
  defaults to 4.0; training used 5.0. Leave it at the default and every path
  clock the policy learned is compressed by 20%.

In the interactive loop: `R` resets, `,` / `.` step alpha by 0.05 and re-sample,
`P` pauses.

---

## 4. Startup banner — the fastest falsification test

If a play run does not show the expected environment, the override never
reached it. Read the banner rather than assuming.

The env prints, at construction:

```
  Standup ramp: ... action-center mode=<linear|path|keyframe> ... knots=<n> (spawn on path; track=4.0 σ=... contact=2.0)
  Kneel-start probability: <p>
  handover_prime_prob <p>
```

- `action-center mode=path` with a `knots=<n>` clause is the only acceptable
  value for S3/S4 checkpoints. Anything reading `mode=linear` means
  `standup_ramp_mode=path` was not passed, and the run is testing a different
  trajectory generator than the one that was trained.
- `play_sim2real.py` prints its own block (`SIM2REAL SCENARIO`) listing envs,
  pickup state, curriculum state, and which contract defaults it applied versus
  which were overridden on the CLI. Check the `contract defaults:` line first:
  if it says `all overridden on CLI`, your explicit values won.

---

## 5. Getting a number instead of an impression

A viewer answers "does it look like it stands up". For a rate, use the tool that
mirrors the training metric:

```powershell
& "$envRoot\python.exe" -u scripts/rsl_rl/eval_alpha_success.py `
  --task=Template-standup_curriculum-v0 --headless `
  --num_envs=256 --seed=0 `
  --load_run 2026-09-09_22-13-40_S4a_measDR `
  --checkpoint model_7780.pt `
  --alphas 0.3,0.5,0.7,0.9,1.0 `
  --post_grace_steps 100 `
  <the same env.* recipe overrides as §3>
```

It spawns every env at exactly the requested d, runs the deterministic policy
until each env terminates once, and counts episodes that survived at least
`--post_grace_steps` control steps after grace expiry. That is the same
definition `curriculum/standup_success_*` uses, so its output is directly
comparable to the 0.867 in the training log. Pushes are disabled; re-pass the
recipe because it follows the hydra overrides you give it.

Control loop is 100 Hz (physics 500 Hz, decimation 5), so the 100-step
post-grace bar is 1.0 s of standing. Episode horizon is 30 s.

---

## 6. What a failure looks like, and its most likely cause

Termination is `too_tilted` (tilt beyond `max_tilt_rad` = 1.8) OR `too_low`
(root z below `min_height` = 0.30) OR a hip-roll / hip-pitch / knee split,
gated off during grace (`rl_env_p6.py:2640-2670`). So a robot that never rises
terminates `too_low` the step grace ends.

When every env in a play run fails, check in this order:

1. **Recipe re-passed?** The single most common cause. Overrides are CLI-only
   and are not stored in the checkpoint.
2. **Banner says `mode=path`?** If not, `standup_ramp_mode` never landed.
3. **`--num_envs` set on `play_alpha`?** Unset means 32768 envs, not a few.
4. **`standup_grace_seconds` at 5.0?** The default 4.0 compresses the path clock.
5. **Then** start hypothesising about the policy.

Path geometry is not a suspect at d=0.5: piecewise-linear interpolation of
`standup_path_knots` gives root height 0.228 m and pitch +5.2° there, which is
outside the ground-penetration region that affects the K0→K1 segment
(`s ∈ (0, 0.15]`, see memory `standup-path-k0k1-penetration`).

### Where the recipe and the cfg defaults actually diverge

`rl_env_cfg_p6.py` no longer holds the R-line bands, so the DR mismatch looks
smaller than it is if you only read the cfg. The launch script passes the S4
narrowing as overrides on top of whatever the cfg says:

| Knob | cfg default | S4a trained |
|---|---|---|
| `standup_grace_seconds` | 4.0 | 5.0 |
| `dr_mass_mult_min/max` | 0.94 / (max) | 1.00 / 1.14 |
| `dr_floor_friction_min` | 0.5 | 0.6 |
| `hip_pitch_armature` | 33.2e-3 | 0.0385 |
| `sat_term_threshold` | 0.35 | 0.5 |
| `swing_clearance_zero_m` | 0.075 | 0.078 |
| `noise_joint_pos` | 0.1 | 0.18 |
| `init_base_orient_noise_rad` | 0.08 | 0.12 |
| `rew_scale_no_flight` | −5.0 | −15.0 |
| `rew_scale_vz_penalty` | −2.0 | −8.0 |
| `rew_scale_action_smoothness` | −0.08 | −0.24 |
| `rew_scale_action_snap` | −0.02 | −0.06 |
| `rew_scale_standup_track` | 0.0 | 4.0 |
| `exec_noise_target_bias_max` | 0.0 | 0.03 |
| `exec_noise_target_std` | 0.0 | 0.015 |
| `standup_path_spawn_clearance_m` | 0.003 | 0.006 |
| `kneel_start_probability` | 0.50 | 0.6 |
| `dr_effort_mult_min` | 0.8 | 0.9 |

Any row you leave unset is a knob the policy was not trained under. The
`standup_ramp_mode=path` override is separately load-bearing: `path` is not the
cfg default, and without it the env falls back to the legacy linear ramp, which
cannot stand up from d ≥ 0.7 at all.

---

## 7. Known bad invocation (09-11), kept so it is not repeated

This command was run and every env failed to stand up:

```powershell
& "$envRoot\python.exe" -u scripts/rsl_rl/play_alpha.py `
  --task=Template-standup_curriculum-v0 `
  --checkpoint="...\2026-09-09_22-13-40_S4a_measDR\model_7780.pt" `
  --num_envs=16 --seed=0 --alpha=0.5 `
  env.standup_ramp_mode=path `
  env.rew_scale_standup_track=4.0 env.rew_scale_standup_contact=2.0
```

Diagnosis, from code inspection rather than a re-run: it passed only three of
the recipe's knobs. It left `standup_grace_seconds` at the 4.0 default against
the trained 5.0, and left every plant override unset — `swing_clearance_zero_m`
(0.075 vs 0.078), the servo gains, `hip_pitch_armature` (33.2e-3 vs 0.0385),
the exec-noise and spawn-DR bands, `sat_term_threshold` (0.35 vs 0.5), and the
whole narrowed DR band. That is a checkpoint evaluated on a plant it did not
train on. The run has not been re-executed with the full recipe as of this
writing, so the §3 command is the next test, not a confirmed fix.

---

## 8. Checklist

- [ ] PATH de-duplicated in this shell (§0), `.Length` compared before and after
- [ ] Env-var preamble set, `cd <repo>\RL`
- [ ] Checkpoint path points at the newest `model_*.pt` in the run dir
- [ ] Full recipe re-passed — no relying on cfg defaults
- [ ] `play_alpha`: `--num_envs` set, `--alpha-random` unless a fixed pin is intended
- [ ] Banner read: `action-center mode=path` with the `knots=` clause
- [ ] For a rate: `eval_alpha_success.py` with the same overrides, compared against 0.867
