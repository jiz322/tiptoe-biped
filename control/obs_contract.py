"""
obs_contract.py — checks a checkpoint's obs normalizer against the constant
channels the balance test feeds it (checklist item 1: verify the contract by
diffing data, not by assumption).

The S-line (S1..S4, 09-06 onward) trains with the kneel->stand curriculum ON,
so obs channels 29 (difficulty), 30 (grace_norm) and 31 (init_h) — constant
in every earlier line (K/L/R) — now vary across episodes. The 36 x 5 layout
itself is unchanged (S4a/S4b actor.0.weight is (512, 180)). A balance-only
hardware test feeds exactly the stand-branch values (d=0, grace=0, standing
init_h), which is what this module scores: for each constant channel of the
newest frame, z = (value - mean) / (std + eps) under the checkpoint's own
EmpiricalNormalization (eps=1e-2 on std, rsl_rl formula). |z| beyond
Z_WARN means the policy was (almost) never trained on that value.

Used by balance_log.py at startup (Pi) and policy_contract_check.py (desk).
"""

import numpy as np

import config as C

NORM_EPS = 1e-2   # rsl_rl EmpiricalNormalization: (x - mean) / (std + eps)
Z_WARN = 3.0

# Bookkeeping tail of one 36-D frame — layout of balance_log.build_obs.
TAIL_LABELS = [
    (27, 'gait_sin'), (28, 'gait_cos'),
    (29, 'difficulty'), (30, 'grace_norm'), (31, 'init_h'),
    (32, 'vx_cmd'), (33, 'yaw_rate_cmd'),
    (34, 'yaw_err_sin'), (35, 'yaw_err_cos'),
]


def balance_tail_values(gait_phase=0.0):
    """The tail values build_obs emits for the balance test (alpha=0, no
    grace, cmd=0, yaw_err=0) at the given gait phase."""
    return {
        27: float(np.sin(gait_phase)), 28: float(np.cos(gait_phase)),
        29: 0.0, 30: 0.0, 31: float(C.SPAWN_HEIGHT),
        32: 0.0, 33: 0.0, 34: 0.0, 35: 1.0,
    }


def tail_report(norm_mean, norm_std, frame_dim=C.OBS_FRAME_DIM):
    """Rows (idx, label, value, mean, std, z) for the newest frame's tail.
    norm_mean/std are the stacked (frame_dim * history,) arrays; frame 0 is
    the newest frame in both sim (_obs_history_buf[:, 0]) and balance_log."""
    vals = balance_tail_values()
    rows = []
    for idx, label in TAIL_LABELS:
        m = float(norm_mean[idx])
        s = float(norm_std[idx])
        v = vals[idx]
        z = (v - m) / (s + NORM_EPS)
        rows.append((idx, label, v, m, s, z))
    return rows


def ood_rows(rows, z_warn=Z_WARN):
    """Constant channels (29..35) whose balance value is OOD for this
    checkpoint. gait sin/cos (27/28) sweep the full circle and are skipped."""
    return [r for r in rows if r[0] >= 29 and abs(r[5]) > z_warn]


def format_report(rows, name=''):
    lines = [f"obs-contract check {name}: balance-test constants vs checkpoint "
             f"normalizer (newest frame, z=(v-mean)/(std+{NORM_EPS:g}))"]
    lines.append(f"  {'ch':>4s} {'label':<13s} {'value':>8s} {'mean':>8s} "
                 f"{'std':>7s} {'z':>7s}")
    for idx, label, v, m, s, z in rows:
        flag = '  <-- OOD' if (idx >= 29 and abs(z) > Z_WARN) else ''
        lines.append(f"  [{idx:2d}] {label:<13s} {v:+8.4f} {m:+8.4f} "
                     f"{s:7.4f} {z:+7.2f}{flag}")
    return "\n".join(lines)


def pg_band(norm_mean, norm_std):
    """(mean, std) of pg_x / pg_y in the newest frame — the band the live
    trimmed IMU reading at upright should sit inside."""
    return ((float(norm_mean[0]), float(norm_std[0])),
            (float(norm_mean[1]), float(norm_std[1])))
