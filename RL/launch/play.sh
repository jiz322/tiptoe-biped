#!/usr/bin/env bash
# TipToe: play the released checkpoint (R1r_3150) in the Isaac Lab viewer (Linux).
# See play.ps1 for what the overrides mean.
# Usage:  RL/launch/play.sh [--gpu 0] [--num-envs 16] [--checkpoint <path>] [--headless] [--video]
set -euo pipefail
GPU=0; NUM_ENVS=16; CKPT=""; HEADLESS=""; VIDEO=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2;;
    --num-envs) NUM_ENVS="$2"; shift 2;;
    --checkpoint) CKPT="$2"; shift 2;;
    --headless) HEADLESS="--headless"; shift;;
    --video) VIDEO="1"; shift;;
    *) echo "unknown arg $1"; exit 2;;
  esac
done
PY="${PYTHON:-python}"
RL="$(cd "$(dirname "$0")/.." && pwd)"
cd "$RL"
[[ -z "$CKPT" ]] && CKPT="$RL/../control/r1r_3150.pt"
[[ -f "$CKPT" ]] || { echo "checkpoint missing: $CKPT"; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1
ARGS=(
  -u scripts/rsl_rl/play.py
  --task Template-standup_curriculum-v0
  --num_envs "$NUM_ENVS"
  --checkpoint "$CKPT"
  env.sat_term_threshold=0.5 env.sat_term_tau_s=0.5
  env.servo_kp_hip_pitch=20.0 env.servo_kd_hip_pitch=1.4
  env.servo_kp_knee=30.0 env.servo_kd_knee=1.5
  env.hip_pitch_armature=0.0385
  env.swing_clearance_zero_m=0.078
  env.push_enable=false
  env.curriculum_enable=false
  env.pickup_enable=false
)
[[ -n "$HEADLESS" ]] && ARGS+=("$HEADLESS")
[[ -n "$VIDEO" ]] && ARGS+=(--video --video_length 600 --enable_cameras)
exec "$PY" "${ARGS[@]}"
