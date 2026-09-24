"""
policy_contract_check.py — desk-side pre-flight for a checkpoint (no hardware).

    python control/policy_contract_check.py control/s4a_6520.pt control/s4b_6520.pt

Per file: loads it exactly as the Pi will (control/policy.py), asserts the
actor input width == config OBS_DIM, prints the training iteration, and
z-scores the balance-test constant channels (obs_contract.py) against the
checkpoint's normalizer. Exit code 1 if any file is OOD or mis-shaped.

Optionally `--ref <ckpt>` prints the same table for a reference policy
(e.g. r1r_3150.pt) so distribution shifts between lines are visible side
by side — this is how the S-line's curriculum channels (29/30/31 going from
std 0 to std 0.2) were confirmed harmless for the balance test.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C                       # noqa: E402
import obs_contract as OC                # noqa: E402
from policy import Policy                # noqa: E402


def check(path):
    pol = Policy(path)
    print(f"  input width {pol.input_dim} (config OBS_DIM {C.OBS_DIM} = "
          f"{C.OBS_FRAME_DIM} x {C.OBS_HISTORY_STEPS}), iter {pol.iter}, "
          f"mode {pol._mode}")
    if pol.norm_mean is None:
        print("  no normalizer stats available — cannot score the tail; "
              "check the checkpoint format")
        return False
    rows = OC.tail_report(pol.norm_mean, pol.norm_std)
    print(OC.format_report(rows, os.path.basename(path)))
    (mx, sx), (my, sy) = OC.pg_band(pol.norm_mean, pol.norm_std)
    print(f"  pg band (newest frame): pg_x {mx:+.3f} +- {sx:.3f}   "
          f"pg_y {my:+.3f} +- {sy:.3f}   "
          f"(config trim IMU_PG_OFFSET = {C.IMU_PG_OFFSET})")
    bad = OC.ood_rows(rows)
    if bad:
        print(f"  FAIL: {len(bad)} balance-test constant(s) OOD for this checkpoint")
        return False
    print("  PASS: all balance-test constants inside the trained band")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoints', nargs='+')
    ap.add_argument('--ref', default=None,
                    help='reference checkpoint printed alongside (not scored)')
    args = ap.parse_args()

    ok = True
    if args.ref:
        print(f"=== reference: {args.ref}")
        check(args.ref)
        print()
    for p in args.checkpoints:
        print(f"=== {p}")
        ok = check(p) and ok
        print()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
