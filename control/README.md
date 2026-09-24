# Deploy stack (Raspberry Pi 5, SocketCAN)

> **Status 2026-09.** The architecture section below is current. The *procedure*
> below it documents the Phase-6 walking deploy (2026-04) and is historical.
> Today's entry points:
>
> - `balance_log.py`: bench / flight balance loop with keyboard-gated
>   HOLD → OBSERVE → NN handover, logging to npz. `v` = blind handover (the
>   training contract, use this), `b` = sighted (legacy). Default policy is
>   `r1r_3150.pt`, the released checkpoint. `--pg-offset` sets the pitch trim
>   per trial.
> - `main.py`: RC-armed loop (ELRS), same controller.
> - Bring-up: `CAN2SPI_BRINGUP.md` (wiring, overlay, `setup_can0.sh`),
>   `set_host_ids.py` once per motor set, `test_motors.py --no-zero` as the
>   per-session guard, `setzero_crawl.py` to zero at the crawl pose.
> - Desk checks with no hardware: `policy_contract_check.py <ckpt>`
>   (normaliser vs the balance-test constants), `obs_parity_check.py`,
>   `normalizer_gap_check.py`.
> - Transport is SocketCAN since 2026-08-15; `robstride_socketcan_driver.py`
>   is the live driver and `robstride_biped_driver.py` the serial-adapter one it
>   wraps. Scripts named `*_bench.py`, `*_probe.py`, `sysid_*.py`, `friction_*.py`,
>   `kd_*.py` are bench instrumentation from specific investigations (see
>   `docs/replant-runbook.html`).
> - Pi Python: torch is not in the system Python; the campaign used a
>   miniforge environment (`docs/ENVIRONMENT.md`).

## Original Phase-6 deploy notes (2026-04, historical)

Deploy the IsaacLab Phase 6 walking policy on real hardware.

## Architecture

```
                       ┌──────────────────────────────────────────┐
                       │              main.py                     │
                       │  RC loop (state edges, arm/standup)      │
                       └───────┬──────────────────────────────────┘
                               │
                               ▼
              ┌──────────────────────────────┐
              │    BipedController           │
              │  state machine, obs builder, │
              │  action executor             │
              └─┬───────────────┬────────────┘
                │               │
        ┌───────▼─────┐   ┌─────▼─────┐   ┌──────────────┐
        │ MotorBridge │   │ IMUAdapter│   │   Policy     │
        │  6× joint   │   │  proj_g,  │   │ TorchScript  │
        │  decouple   │   │  ang_vel  │   │  obs→action  │
        │  + gear     │   │           │   │              │
        └───────┬─────┘   └────┬──────┘   └──────────────┘
                │              │
        ┌───────▼─────┐   ┌────▼────────┐
        │RobStrideDrv │   │IMUDriver6Ax │
        │  (vendor)   │   │  (vendor)   │
        └─────────────┘   └─────────────┘
```

## Files

| file                        | role                                                      |
|-----------------------------|-----------------------------------------------------------|
| `config.py`                 | All constants (joint mapping, gains, gear ratio, RC, etc) |
| `robstride_biped_driver.py` | MIT-mode CAN driver, per-motor encoding ranges            |
| `motor_bridge.py`           | sim ↔ real joint conversion; gear ratio + hip coupling    |
| `imu_adapter.py`            | IMU → projected_gravity, ang_vel_body, lin_vel, yaw       |
| `biped_rc_mapper.py`        | ELRS channels → vx_cmd, yaw_rate_cmd, arm, standup        |
| `policy.py`                 | rsl_rl JIT loader + checkpoint fallback                   |
| `export_policy.py`          | Run on training machine to export checkpoint → JIT        |
| `controller.py`             | State machine + obs builder + action executor             |
| `main.py`                   | Entry point: hardware init, RC loop, 100 Hz tick          |
| `test_motors.py`            | Calibration: signs, coupling, zeros (run FIRST)           |
| `test_policy.py`            | Dry run policy with real sensors but no motor commands    |

## Bring-up procedure

### 1. Project layout

```
project/
├── sim2real/
│   ├── config.py
│   ├── robstride_biped_driver.py   ← included (updated for 6 motors)
│   ├── motor_bridge.py
│   ├── imu_adapter.py
│   ├── controller.py
│   ├── main.py
│   ├── policy.py
│   ├── biped_rc_mapper.py
│   ├── export_policy.py            ← run on training machine
│   ├── test_motors.py
│   └── test_policy.py
├── imu_driver_6_400hz.py           ← from balancer (IMUDriver6Axis)
├── crsf_reader.py                  ← from balancer
└── policy.pt                       ← exported from rsl_rl
```

### 2. Export the policy (on training machine)

```bash
# From a checkpoint file:
python export_policy.py --checkpoint logs/biped_phase6/models/model_3000.pt

# Or point at the log directory (picks latest checkpoint):
python export_policy.py --log-dir logs/biped_phase6/2025-01-15_12-30-00

# Output: policy.pt — copy to Pi
scp policy.pt pi@biped:~/project/policy.pt
```

The export script auto-detects the rsl_rl actor MLP architecture and
obs normalizer from the checkpoint state dict, wraps them into a single
TorchScript module. If you already have `policy_inference.pt` from
rsl_rl's built-in export, that works directly too — just rename it.

`policy.py` can also load raw checkpoint .pt files as a fallback, but
JIT is preferred (faster inference, no architecture guessing).

### 3. Switch motors to MIT mode

For each of the 6 motors (one at a time on the bench), run the
`switch_to_mit_mode` script you already have, with the right `TARGET_ID`
(101 through 106). Power-cycle each motor after switching. Verify in MIT
mode by running `test_motors.py` step 1.

### 4. Calibrate

```bash
sudo python3 test_motors.py
```

This walks you through:
- **Step 1** — heartbeat check, all 6 motors must respond.
- **Step 2** — push each joint and verify the reading increases in the
  +sim direction. If it decreases, flip the corresponding entry in
  `config.JOINT_SIGN`.
- **Step 3** — measure the knee↔hip-pitch belt coupling. The script fits
  a linear model and prints the implied `KNEE_HIP_COUPLING`. Update
  `config.KNEE_HIP_COUPLING` if it differs.
- **Step 4** — hold the robot in the standing pose, then write zeros.

### 5. Dry-run the policy

```bash
python3 test_policy.py
```

Verify:
- `proj_grav` is roughly `[0, 0, -1]` when the robot stands upright
  (z component ≈ -1).
- `joint_pos_n` is small (current pose - default).
- Loop time stays under 10 ms.
- Actions look reasonable (mostly within ±0.5).

### 6. Live run

```bash
sudo python3 main.py
```

State machine on the transmitter:
1. **CH8 high** → ARM. Motors enable, robot holds standing pose.
2. **CH7 high** → STANDUP. 5 s grace begins; burst PD active; the policy
   slides from kneel pose toward standing. After 5 s, transitions to WALK.
3. **CH2 / CH4** → drive vx and yaw rate.
4. **CH8 low** → DISARM. Motors limp.

## Critical gotchas

### Knee gear ratio

The knee motor (17 Nm) drives the knee joint via a 22:30 reduction. So:
- Joint speed = motor speed × (22/30)
- Joint torque = motor torque × (30/22) → **23.2 Nm joint** (matches sim)

In sim PD-gain space, knees use Kp=16. In motor space, that is
`16 × (22/30)² ≈ 8.6`. `motor_bridge.py` does this scaling automatically.

### Hip-knee mechanical coupling

The knee belt is anchored to the hip body. Rotating the hip pitch shifts
the apparent knee angle even if the knee motor is still:
```
sim_knee_angle = motor_angle × gear_ratio - hip_pitch_angle × coupling
```
Default coupling is 1.0 (full belt coupling). The calibration tool
measures it empirically — trust the measured value over the default.

### Linear velocity in observation

`imu_adapter.lin_vel_body()` returns **zero** by default. The IMU does
not provide absolute linear velocity, and the simulation gave the policy
a perfect oracle. If walking quality suffers because of this, options:

1. Build a leg-odometry estimator from foot kinematics + contact (best).
2. Integrate `acc - gravity` with a low-pass and aggressive decay (drifts).
3. Re-train the policy without `lin_vel_imu` in the observation (cleanest).

### IMU axis convention

Existing balancer uses `pitch_sign = -1`, `gyro_y_sign = -1`,
`gyro_z_sign = -1`. These match a specific IMU mounting orientation. Verify
with `test_policy.py`:
- Tilt the robot **forward** → `proj_grav[0]` should become **positive**.
- Tilt **right** → `proj_grav[1]` should become **positive**.
- Yaw rate **left** (CCW from above) → `ang_vel[2]` should become **positive**.

If signs are wrong, edit `IMU_*_SIGN` in `config.py`.

### Burst torque + thermal limits

During the 5 s standup grace, the bridge sends `BURST_KP / BURST_KD`
gains, allowing 17 Nm motor torques. This matches the motor's 5 s
overload rating. **Do not rapidly re-trigger standup** — give the motors
~30 s to cool between attempts.

### Loop overrun

`main.py` runs at 100 Hz. Each tick reads 6 motors (~6–10 ms of CAN
serial), runs the policy (~1–3 ms on Pi 5 CPU), and updates state. If
loop times exceed 10 ms consistently, options:
- Move the policy to GPU (`--device cuda` if available).
- Use multiple CAN-USB adapters (split legs across two ports).
- Drop to 50 Hz and retrain.

### Safety

`MAX_TILT_RAD = 1.20` (≈ 70°) immediately disarms motors. Loss of RC link
also forces disarm. Always run with the robot in a harness on first runs.

## Exporting the policy from IsaacLab

Use `export_policy.py` (included). See step 2 above. It handles both
rsl_rl's built-in JIT export and manual checkpoint reconstruction.

If you prefer to export manually from an rsl_rl training script:

```python
from rsl_rl.runners import OnPolicyRunner
runner = OnPolicyRunner(env, train_cfg, log_dir)
runner.load(checkpoint_path)
# This writes policy_1.pt (actor only, normalizer baked in)
runner.alg.actor_critic.export(export_dir)
```

Rename `policy_1.pt` → `policy.pt` and copy to the Pi.

### MIT encoding ranges for your motor model

The included `robstride_biped_driver.py` defaults to RS01 encoding
ranges (torque ±6 Nm). If your hip_pitch / knee motors are a larger
model (RS02/RS03), edit `MOTOR_MIT_OVERRIDES` at the top of
`robstride_biped_driver.py`:

```python
MOTOR_MIT_OVERRIDES = {
    101: {'tff_min': -17.0, 'tff_max': 17.0},   # hip_pitch1
    102: {'tff_min': -17.0, 'tff_max': 17.0},   # knee1 motor
    104: {'tff_min': -17.0, 'tff_max': 17.0},   # hip_pitch2
    105: {'tff_min': -17.0, 'tff_max': 17.0},   # knee2 motor
}
```

Check your motor's datasheet under "MIT Control Mode" for the correct
position, velocity, torque, Kp, Kd encoding ranges.
