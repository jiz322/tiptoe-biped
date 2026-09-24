# Bipedal Point-Foot Balance — v1 Post-Mortem
## checkpoint

<repo>\RL\logs\rsl_rl\bipedal_phase4_add_yaw\2026-04-08_08-05-54\model_13900.pt
特别拽
<repo>\RL\logs\rsl_rl\bipedal_phase4_add_yaw\2026-04-08_13-15-06\model_14995.pt
This actually stepping.
## What the project is

A 6-DOF bipedal robot with point feet trained via PPO (RSL-RL) in Isaac Lab to balance in place.
The robot has two legs, each with hip_roll, hip_pitch, and knee joints. Point feet make static
balance impossible — the policy must discover reactive stepping to stay upright.

## What v1 achieved

- **The robot stays upright.** It survives indefinitely without falling, collecting the alive
  bonus and upright reward consistently across 14,000+ steps of playback.
- **It doesn't drift or spin.** XY velocity and yaw rate stay near zero — the penalty terms
  for those work as intended.
- **Knee and hip pitch joints track reasonably well.** Position errors on joints 3–6 (hip_pitch
  and knee) are small (0.01–0.08 rad), and torques are within physical limits.
- **The policy does produce occasional sign flips on hip roll** (see step 13050, 13200, 13750,
  14050 where columns 1–2 flip from -0.5/+0.5 to +0.5/-0.5), which could be an embryonic
  attempt at weight shifting. But the mechanism is wrong (see below).

## What went wrong

### 1. Hip roll actions permanently saturated — degenerate bang-bang policy

The telemetry tells the whole story:

```
                       hip_roll1  hip_roll2  hip_pitch1  hip_pitch2  knee1   knee2
  target pos :          -0.500    +0.500      +0.041     -0.043    -0.060  +0.057
  actual pos :          -0.015    +0.009      +0.031     -0.020    +0.000  +0.006
  pos error  :          -0.485    +0.491      +0.010     -0.022    -0.060  +0.051
  torque     :          -3.92     +3.95       -0.68      -0.37     -1.86   +1.54
```

Hip roll targets are locked at ±0.500 rad (the action scale limit) in >90% of timesteps.
The joints barely move (actual position stays within ±0.04 rad) because stiffness is only
8.0 N·m/rad. The result is a constant ~4 N·m torque output per hip roll joint — the PD
controller strains against the physical system without producing meaningful motion.

**This is not balancing. It is exploiting the reward structure.**

### 2. Sustained torque exceeds motor continuous rating

The hip roll motor can sustain 1.8 N·m continuously (peak 6.0 N·m burst). The policy
commands ~3.7–4.3 N·m on hip roll every single timestep. This would:
- Overheat and damage the real motor within seconds
- Never transfer to hardware (sim-to-real gap is catastrophic for this behavior)

The sim allowed it because `effort_limit` was set to 6.0 N·m (the burst rating), not
the 1.8 N·m continuous rating.

### 3. Reward structure failed to prevent the exploit

Three compounding failures:

| Reward term | Why it failed |
|---|---|
| **energy** (scale = -0.0005) | Two joints at ~4 N·m = cost of 0.016/step. Against an alive bonus of 2.0/step, this is a rounding error. The scale was ~100× too small. |
| **joint_pos_limits** (scale = -5.0) | Penalizes *actual* joint position, not commanded target. Actual hip roll stays near 0 (the joint can't move much at stiffness 8), so the penalty is ~0. The policy gets free torque with no cost. |
| **action_rate** (scale = -0.05) | Only penalizes *changes* in action. A constant ±1.0 action has zero action rate cost. Bang-bang is actually optimal under this penalty. |
| **No action magnitude penalty** | Nothing penalizes the policy for commanding large actions. Saturating at ±1.0 is free. |

### 4. Uniform action scale across very different joints

All 6 joints share `action_scale = 0.5 rad`, but:
- Hip roll: stiffness 8, effort limit 6 → commanding 0.5 rad offset = requesting 4 N·m
- Hip pitch: stiffness 80, effort limit 17 → commanding 0.5 rad offset = requesting 40 N·m (clamped to 17)
- Knee: stiffness 60, effort limit 23.2 → commanding 0.5 rad offset = requesting 30 N·m (clamped to 23.2)

For hip roll specifically, the action range allows the policy to request torques well above
the continuous limit without any joint displacement, creating a "free torque" exploit.

### 5. Observations from articulation root, not IMU frame

The v1 code computes projected gravity, angular velocity, and linear velocity from the
articulation root (`root_quat_w`, `root_ang_vel_w`, `root_lin_vel_w`). The real robot has
an IMU mounted at a specific body (`frame_imu`). If the IMU is offset or rotated relative
to the articulation root, the policy trained in sim will see different orientation/velocity
signals than what the real sensor reports, creating a sim-to-real gap.

## Summary of the failure mode

The policy discovered that it could collect maximum reward by:
1. Commanding hip rolls to ±0.5 rad permanently (saturated actions)
2. Generating ~4 N·m constant torque that presses the feet into the ground
3. Using the other 4 joints for minor adjustments to stay upright

This "balances" in simulation but is physically invalid — it would burn out the motors
on real hardware. The reward function had no effective mechanism to prevent it because:
- Energy penalty was too small
- Joint position penalty measured the wrong thing (actual vs commanded)
- No penalty existed for action magnitude
- The effort limit was set to burst rating, not continuous rating

## Changes needed for v2

1. **Reduce hip roll effort_limit** from 6.0 to 1.8 N·m (continuous rating)
2. **Add action magnitude penalty** (`rew_scale_action_mag = -0.1`) to penalize saturated commands
3. **Increase energy penalty** from -0.0005 to -0.05 (100×)
4. **Use IMU body frame** for observations (`frame_imu` body instead of articulation root)
5. Consider per-joint action scales in future versions