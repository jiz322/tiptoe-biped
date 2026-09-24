"""
analyze_velocity_envelope.py — could the DC torque-speed envelope (back-EMF)
or firmware speed limits have clipped torque in the pre-fall window, even
though commanded torque was far below the flat ceilings?

DCMotor availability: tau_avail(v) = sat_effort * (1 - |v|/v_limit)
  hip_pitch: sat 17,  v_limit 27.5 (joint = motor, direct drive)
  knee:      motor sat 17, motor v_limit 27.5; belt-effective motor vel
             = (qd_kn + c*qd_hp)/G  (sim frame), G=22/30, c=0.733
  hip_roll:  sat 6, v_limit 28

Firmware limit_spd: pitch 44, roll 50 (motor rad/s) — check headroom too.

For each run, over the pre-fall window, compute per-tick margin:
  demand = |motor_tau_cmd|, avail = sat*(1-|v_m|/v_lim)
  clip if demand > avail.
"""
import glob, os
import numpy as np

DATA = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(DATA, "policy_G2i4760_log_20260813_*.npz")))

G = 22.0 / 30.0
c = 0.733
worst = {'hip': [], 'knee': [], 'roll': [], 'vmax_hip': [], 'vmax_knee_m': [], 'vmax_roll': []}
clip_ticks = {'hip': 0, 'knee': 0, 'roll': 0}
tot_ticks = 0

for f in files:
    d = np.load(f, allow_pickle=True)
    t = d['time']; n = len(t)
    if n < 10:
        continue
    pg = d['proj_grav']
    tilted = np.abs(pg[:, 0]) > 0.3
    fall_i = int(np.argmax(tilted)) if tilted.any() else n - 1
    w = slice(0, max(fall_i, 5))

    vel = d['sim_joint_vel'][w]          # sim/joint frame
    cmd = np.abs(d['motor_tau_cmd'][w])  # motor frame

    # motor velocities
    v_hip = np.abs(vel[:, [1, 4]])                       # direct drive
    v_knee_m = np.abs(vel[:, [2, 5]] + c * vel[:, [1, 4]]) / G
    v_roll = np.abs(vel[:, [0, 3]])

    avail_hip = 17.0 * np.clip(1 - v_hip / 27.5, 0, None)
    avail_knee_m = 17.0 * np.clip(1 - v_knee_m / 27.5, 0, None)
    avail_roll = 6.0 * np.clip(1 - v_roll / 28.0, 0, None)

    # knee cmd is motor frame already
    m_hip = avail_hip - cmd[:, [1, 4]]
    m_knee = avail_knee_m - cmd[:, [2, 5]]
    m_roll = avail_roll - cmd[:, [0, 3]]

    worst['hip'].append(m_hip.min())
    worst['knee'].append(m_knee.min())
    worst['roll'].append(m_roll.min())
    worst['vmax_hip'].append(v_hip.max())
    worst['vmax_knee_m'].append(v_knee_m.max())
    worst['vmax_roll'].append(v_roll.max())
    clip_ticks['hip'] += int((m_hip < 0).sum())
    clip_ticks['knee'] += int((m_knee < 0).sum())
    clip_ticks['roll'] += int((m_roll < 0).sum())
    tot_ticks += m_hip.shape[0]

for k in ['hip', 'knee', 'roll']:
    v = np.array(worst[k])
    print(f"{k:5s}: worst margin (Nm, avail-demand) min {v.min():6.2f}  median {np.median(v):6.2f} "
          f" | ticks clipped {clip_ticks[k]}/{tot_ticks}")
for k in ['vmax_hip', 'vmax_knee_m', 'vmax_roll']:
    v = np.array(worst[k])
    print(f"{k:12s}: max motor speed rad/s  median {np.median(v):6.2f}  max {v.max():6.2f}")
