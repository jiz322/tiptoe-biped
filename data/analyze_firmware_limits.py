"""
analyze_firmware_limits.py — test whether firmware torque/current limits
bind during the fall window of the 2026-08-13 G2i4760 runs.

Firmware ceilings (motor frame, from register dumps 2026-08-14):
  pitch/knee RS01 : limit_torque = 17.0 Nm, limit_cur = 23 A, Kt = 0.562
                    -> current-implied ceiling 23*0.562 = 12.93 Nm
  hip_roll  EL05  : limit_torque = 6.0 Nm,  limit_cur = 11 A, Kt = 0.680
                    -> current-implied ceiling 11*0.680 = 7.48 Nm (6.0 binds)

Question: does ANY commanded torque in the pre-fall window even reach
these ceilings?  motor_tau_cmd in the logs is already motor-frame
(knee scaled by 22/30).
"""
import glob
import os
import sys
import numpy as np

DATA = os.path.dirname(os.path.abspath(__file__))
PATTERN = os.path.join(DATA, "policy_G2i4760_log_20260813_*.npz")

RS01_CUR_CEIL = 23 * 0.562      # 12.93
HR_CEIL = 6.0
NAMES = ['L_roll', 'L_hip', 'L_knee', 'R_roll', 'R_hip', 'R_knee']
PITCH_IDX = [1, 4]
KNEE_IDX = [2, 5]
ROLL_IDX = [0, 3]

files = sorted(glob.glob(PATTERN))
if not files:
    print("no files matched", PATTERN)
    sys.exit(1)

print(f"{len(files)} G2i4760 runs from 2026-08-13")
print()

hdr = (f"{'run':<22s} {'ticks':>5s} {'dur_s':>6s} {'fall_s':>6s} "
       f"{'|hipcmd|max':>11s} {'|kneecmd|max':>12s} {'|rollcmd|max':>12s} "
       f"{'hip>12.9':>8s} {'knee>12.9':>9s} {'roll>=5.7':>9s} "
       f"{'|hipmeas|max':>12s} {'|kneemeas|max':>13s}")
print(hdr)

agg = {'hip_cmd_max': [], 'knee_cmd_max': [], 'roll_cmd_max': [],
       'hip_meas_max': [], 'knee_meas_max': [], 'roll_meas_max': [],
       'fall_t': []}

for f in files:
    d = np.load(f, allow_pickle=True)
    t = d['time']
    n = len(t)
    if n < 10:
        print(f"{os.path.basename(f):<22s}  (only {n} ticks, skipped)")
        continue
    tau_cmd = d['motor_tau_cmd']       # (n,6) motor frame
    tau_meas = d['motor_tau_meas']     # (n,6) motor frame
    pg = d['proj_grav']                # (n,3)

    # fall onset: |proj_grav_x| crossing 0.3 (~17 deg pitch) — past recovery
    tilted = np.abs(pg[:, 0]) > 0.3
    fall_i = int(np.argmax(tilted)) if tilted.any() else n - 1
    fall_t = t[fall_i]

    # analysis window: start -> fall onset (the window where the policy
    # still could have saved it)
    w = slice(0, max(fall_i, 5))
    hc = np.abs(tau_cmd[w][:, PITCH_IDX])
    kc = np.abs(tau_cmd[w][:, KNEE_IDX])
    rc = np.abs(tau_cmd[w][:, ROLL_IDX])
    hm = np.abs(tau_meas[w][:, PITCH_IDX])
    km = np.abs(tau_meas[w][:, KNEE_IDX])
    rm = np.abs(tau_meas[w][:, ROLL_IDX])

    hip_over = (hc > RS01_CUR_CEIL).mean()
    knee_over = (kc > RS01_CUR_CEIL).mean()
    roll_over = (rc >= 0.95 * HR_CEIL).mean()

    agg['hip_cmd_max'].append(hc.max())
    agg['knee_cmd_max'].append(kc.max())
    agg['roll_cmd_max'].append(rc.max())
    agg['hip_meas_max'].append(hm.max())
    agg['knee_meas_max'].append(km.max())
    agg['roll_meas_max'].append(rm.max())
    agg['fall_t'].append(fall_t)

    name = os.path.basename(f).replace('policy_G2i4760_log_20260813_', '').replace('.npz', '')
    print(f"{name:<22s} {n:>5d} {t[-1]:>6.2f} {fall_t:>6.2f} "
          f"{hc.max():>11.2f} {kc.max():>12.2f} {rc.max():>12.2f} "
          f"{hip_over:>8.1%} {knee_over:>9.1%} {roll_over:>9.1%} "
          f"{hm.max():>12.2f} {km.max():>13.2f}")

print()
print("Aggregate over runs:")
for k, v in agg.items():
    v = np.array(v)
    print(f"  {k:<14s} min {v.min():7.3f}  median {np.median(v):7.3f}  max {v.max():7.3f}")

# Also: gains actually used (they're hand-ramped, log carries them per tick)
d = np.load(files[-1], allow_pickle=True)
print()
print("Last run config: kp =", d['kp'][0], " kd =", d['kd'][0],
      " tau_max =", d['tau_max'][0], " lpf_alpha =", d['lpf_alpha'][0])
print("channels:", sorted(d.files))
