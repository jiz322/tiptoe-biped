"""
biped_rc_mapper.py — ELRS channels → bipedal commands.

Channel mapping:
    CH2 (pitch stick, idx 1) → vx_cmd      [-0.15, +0.40] m/s
    CH4 (yaw stick,   idx 3) → yaw_rate    [-0.5, +0.5]   rad/s
    CH7 (Aux3,        idx 6) → standup     low=idle, high=trigger standup
    CH8 (Aux4,        idx 7) → arm         low=disarmed, high=armed

All sticks have a deadband. Sim ranges come from rl_env_cfg.py.
"""

import config as C


CRSF_MIN = 172
CRSF_MID = 992
CRSF_MAX = 1811


def _centered(val, deadband, neg_out, pos_out):
    """Map a centered stick (992 ±) to [neg_out, pos_out] with deadband."""
    centered = val - CRSF_MID
    if abs(centered) < deadband:
        return 0.0
    if centered > 0:
        t = (centered - deadband) / (CRSF_MAX - CRSF_MID - deadband)
        t = max(0.0, min(1.0, t))
        return t * pos_out
    else:
        t = (-centered - deadband) / (CRSF_MID - CRSF_MIN - deadband)
        t = max(0.0, min(1.0, t))
        return -t * abs(neg_out)


class BipedRCMapper:
    def __init__(self,
                 vx_min=C.RC_VX_MIN,
                 vx_max=C.RC_VX_MAX,
                 yaw_max=C.RC_YAW_RATE_MAX,
                 deadband=C.RC_DEADBAND,
                 arm_threshold=C.RC_ARM_THRESHOLD,
                 standup_threshold=C.RC_STANDUP_THRESHOLD):
        self.vx_min = vx_min
        self.vx_max = vx_max
        self.yaw_max = yaw_max
        self.deadband = deadband
        self.arm_threshold = arm_threshold
        self.standup_threshold = standup_threshold

    def map(self, channels):
        return {
            'vx_cmd':       _centered(channels[1], self.deadband,
                                       self.vx_min, self.vx_max),
            'yaw_rate_cmd': _centered(channels[3], self.deadband,
                                       -self.yaw_max, self.yaw_max),
            'standup':      channels[6] > self.standup_threshold,
            'armed':        channels[7] > self.arm_threshold,
        }