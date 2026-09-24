"""
test_policy.py — dry run for the policy + observation pipeline.

Reads from real IMU + motor encoders, builds the 36-dim observation,
runs the policy, and PRINTS the requested action — but never sends
torque commands to motors. Use this to:

    - sanity-check the policy loads and produces sensible numbers
    - confirm the observation values look reasonable
    - measure end-to-end loop latency

Usage:
    python3 test_policy.py
    python3 test_policy.py --skip-zero   # if already zeroed this session
"""

import argparse
import time
import numpy as np

import config as C
from motor_bridge import MotorBridge
from imu_adapter import IMUAdapter
from policy import Policy
from controller import BipedController


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-zero', action='store_true',
                    help="Skip motor zeroing (use if already zeroed).")
    args = ap.parse_args()

    imu = IMUAdapter()
    imu.open()
    time.sleep(0.5)

    motors = MotorBridge()
    # Motors must be enabled to reply with encoder data, but send_idle
    # uses kp=kd=tff=0 so no torque is produced.
    motors.enable()

    try:
        if not args.skip_zero:
            input("\n>>> Place robot in KNEEL pose, then press ENTER to zero motors <<<\n")
            motors.set_zero_at_current_pose()
            print("[test_policy] Motors zeroed at kneel. Software offsets active.")
            time.sleep(0.5)

        policy = Policy(C.POLICY_PATH, device='cpu')
        ctrl = BipedController(motors, imu, policy)

        print("\nDry run — no motor commands will be sent.")
        print("Ctrl-C to stop.\n")

        t0 = time.monotonic()
        loop_times = []
        for k in range(2000):
            t_start = time.monotonic()
            imu.update()
            t_imu = time.monotonic()
            t0 = time.monotonic()
            motors.send_idle()
            t1 = time.monotonic()
            motors.read_state()
            t2 = time.monotonic()
            print(f"send_idle: {(t1-t0)*1000:.2f}ms, read_state: {(t2-t1)*1000:.2f}ms")
            t_can = time.monotonic()
            obs = ctrl._build_observation()
            t_obs = time.monotonic()
            action = policy.forward(obs)
            t_pol = time.monotonic()

            if k % 50 == 0:
                print(f"\n[t={t_start-t0:5.1f}s]  step {k}")
                print(f"  imu  : {(t_imu-t_start)*1000:5.2f} ms")
                print(f"  can  : {(t_can-t_imu  )*1000:5.2f} ms")
                print(f"  obs  : {(t_obs-t_can  )*1000:5.2f} ms")
                print(f"  pol  : {(t_pol-t_obs  )*1000:5.2f} ms")
                print(f"  total: {(t_pol-t_start)*1000:5.2f} ms")

            time.sleep(max(0.0, C.DT - (time.monotonic() - t_start)))

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n[test_policy] Error: {e}")
    finally:
        motors.send_idle()
        motors.disable()
        imu.close()
        motors.close()
        print("\nDone.")


if __name__ == '__main__':
    main()