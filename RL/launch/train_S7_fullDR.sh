#!/usr/bin/env bash
# TipToe: train the S7 "full DR" arm (canonical training template, Linux).
#
# Same recipe as train_S7_fullDR.ps1 (read that file's header, and
# legacy/launch/launch_train_S7_fullDR.ps1 for the full rationale of every band).
# Trains from scratch; see the RESUME block to resume your own run.
#
# Requirements: Isaac Lab installed in the active Python (conda env or
# ./isaaclab.sh -p), this repo installed with `pip install -e RL/source/RL`,
# and the rsl_rl patch applied (RL/patches/README.md).
#
# Usage:  RL/launch/train_S7_fullDR.sh [--gpu 0] [--run-name S7_fullDR] [--max-iter 0] [--headless]
set -euo pipefail
GPU=0; RUN_NAME=S7_fullDR; MAX_ITER=0; HEADLESS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2;;
    --run-name) RUN_NAME="$2"; shift 2;;
    --max-iter) MAX_ITER="$2"; shift 2;;
    --headless) HEADLESS="--headless"; shift;;
    *) echo "unknown arg $1"; exit 2;;
  esac
done
PY="${PYTHON:-python}"
RL="$(cd "$(dirname "$0")/.." && pwd)"
cd "$RL"; mkdir -p logs
export CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1

ARGS=(
  -u scripts/rsl_rl/train.py
  --task Template-standup_curriculum-v0
  --run_name "$RUN_NAME"
  # RESUME block (a resume must re-pass EVERY override below):
  # --resume --load_run <timestamp>_<run_name> --checkpoint model_<n>.pt
  agent.algorithm.entropy_coef=0.005
  agent.algorithm.desired_kl=0.02
  agent.algorithm.clip_param=0.12
  env.rew_scale_gait_ref=3.0
  env.sat_term_threshold=0.5
  env.sat_term_tau_s=0.5
  env.swing_clearance_zero_m=0.078
  env.servo_kp_hip_pitch=20.0
  env.servo_kd_hip_pitch=1.4
  env.servo_kp_knee=30.0
  env.servo_kd_knee=1.5
  env.hip_pitch_armature=0.0385
  env.swing_clearance_target_m=0.015
  env.rew_scale_no_flight=-15.0
  env.rew_scale_vz_penalty=-8.0
  env.rew_scale_action_smoothness=-0.24
  env.rew_scale_action_snap=-0.06
  env.exec_noise_target_bias_max=0.03
  env.exec_noise_target_std=0.015
  env.noise_joint_pos=0.18
  env.init_base_orient_noise_rad=0.12
  env.push_enable=true
  env.push_probability=0.001
  env.push_force_min=10.0
  env.push_force_max=40.0
  env.push_duration_steps=10
  env.dr_action_delay_min=1
  env.dr_action_delay_max=2
  env.dr_imu_delay_min=1
  env.dr_imu_delay_max=1
  env.dr_joint_delay_min=1
  env.dr_joint_delay_max=1
  env.dr_floor_friction_min=0.6
  env.dr_floor_friction_max=1.1
  env.dr_stiffness_mult_min=0.90
  env.dr_stiffness_mult_max=1.10
  env.dr_armature_mult_min=0.80
  env.dr_armature_mult_max=1.20
  env.dr_effort_mult_min=0.9
  env.dr_com_offset_max=0.01
  env.dr_ang_vel_bias_max=0.05
  env.dr_gravity_tilt_max_deg=2.0
  env.dr_com_shift_x_max=0.02
  env.dr_com_shift_y_max=0.01
  env.dr_proj_grav_bias_max=0.00
  env.dr_mass_mult_min=1.00
  env.dr_mass_mult_max=1.14
  env.belt_series_stiffness_min=100.0
  env.belt_series_stiffness_max=1000.0
  env.belt_series_damping_min=0.0
  env.belt_series_damping_max=0.3
  env.foot_restitution=0.63
  env.dr_foot_restitution_min=0.2
  env.dr_foot_restitution_max=0.8
  env.dr_foot_restitution_buckets=64
  env.curriculum_enable=true
  env.curriculum_initial_alpha=0.0
  env.curriculum_max_alpha=1.0
  env.curriculum_alpha_step=0.1
  env.curriculum_advance_metric=success_high_rel
  env.curriculum_success_post_grace_steps=100
  env.curriculum_advance_threshold=0.7
  env.curriculum_frontier_frac=0.5
  env.curriculum_frontier_width=0.3
  env.curriculum_advance_window=2000
  env.curriculum_advance_cooldown=6000
  env.kneel_start_probability=0.6
  env.pickup_enable=false
  env.standup_ramp_mode=path
  env.standup_grace_seconds=5.0
  env.standup_path_spawn_clearance_m=0.006
  env.rew_scale_standup_track=4.0
  env.rew_scale_standup_contact=2.0
)
[[ -n "$HEADLESS" ]] && ARGS+=("$HEADLESS")
[[ "$MAX_ITER" -gt 0 ]] && ARGS+=(--max_iterations "$MAX_ITER")

echo "logs: $RL/logs/train_$RUN_NAME.{out,err}.log"
nohup "$PY" "${ARGS[@]}" > "logs/train_$RUN_NAME.out.log" 2> "logs/train_$RUN_NAME.err.log" &
echo "Launched PID $!"
