#!/usr/bin/env python3
"""
imu_enc_skew_log.py — Replant runbook M1.2: IMU vs encoder staleness.

Logs both host-side ARRIVAL streams while a human oscillates the hip
pitch by hand (robot hanging, motors limp — grab the torso/leg and rock
it about the hip-pitch axis, varying the tempo ~0.5-2 Hz; irregular is
better than metronomic for the cross-correlation).

  IMU:     every gyro frame is timestamped in the reader thread at parse
           time (~0.5 ms after wire arrival). All 3 axes logged; the
           analysis picks the axis most coherent with the encoder.
  Encoder: motor polled as fast as the bus allows (~1.9 kHz) with limp
           commands; every reply timestamped.

Analysis (runs automatically at the end, npz saved for offline redo):
  lag(gyro vs d(enc_pos)/dt) = transport skew, encoder path vs IMU path
      (position is unfiltered in firmware, so this is pure transport).
  lag(gyro vs enc_vel_fb)    = same skew PLUS the firmware velocity EMA
      (spd_filt_gain) — the difference between the two lags is a free
      first estimate of the M2 filter lag.
Positive lag = IMU stream leads (encoder is staler), matching the old
cfg §12b wording "IMU leads encoder".

Usage:
    python3 imu_enc_skew_log.py --motor 101 --secs 30 --out skew_101.npz
"""

import argparse
import time

import numpy as np

import config as C
from robstride_socketcan_driver import make_driver
from imu_driver_6_400hz import IMUDriver6Axis, IMU_PORT, ID_GYRO, SCALE
import struct as _struct


class StampedIMU(IMUDriver6Axis):
    """IMUDriver6Axis that appends (t_mono, gx, gy, gz) per gyro frame."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.gyro_log = []

    def _parse_data(self, payload):
        offset = 0
        while offset + 2 <= len(payload):
            data_id = payload[offset]
            data_len = payload[offset + 1]
            offset += 2
            if offset + data_len > len(payload):
                break
            chunk = payload[offset:offset + data_len]
            offset += data_len
            if data_id == ID_GYRO and data_len >= 12:
                g = _struct.unpack('<iii', chunk[:12])
                self.gyro_log.append((time.monotonic(),
                                      g[0] * SCALE, g[1] * SCALE, g[2] * SCALE))
        super()._parse_data(payload)


def xcorr_lag(t_ref, x_ref, t_sig, x_sig, fs=1000.0, max_lag_s=0.05):
    """Lag of x_sig relative to x_ref (positive = sig arrives later),
    by resampling both to fs and parabolic-refining the xcorr peak."""
    t0 = max(t_ref[0], t_sig[0])
    t1 = min(t_ref[-1], t_sig[-1])
    g = np.arange(t0, t1, 1.0 / fs)
    a = np.interp(g, t_ref, x_ref)
    b = np.interp(g, t_sig, x_sig)
    a = a - a.mean()
    b = b - b.mean()
    n = int(max_lag_s * fs)
    lags = np.arange(-n, n + 1)
    cc = np.array([np.dot(a[max(0, -k):len(a) - max(0, k)],
                          b[max(0, k):len(b) - max(0, -k)])
                   for k in lags])
    i = int(np.argmax(cc))
    if 0 < i < len(cc) - 1:                      # parabolic refinement
        y0, y1, y2 = cc[i - 1], cc[i], cc[i + 1]
        denom = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / denom if denom else 0.0
    else:
        frac = 0.0
    corr = cc[i] / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return (lags[i] + frac) / fs, corr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--motor', type=int, default=101)
    ap.add_argument('--secs', type=float, default=30.0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    imu = StampedIMU(IMU_PORT)
    if not imu.open():
        raise SystemExit("IMU failed to open")
    drv = make_driver(motor_ids=[args.motor])
    et, ep, ev = [], [], []
    try:
        drv.enable_all()
        print(f"[skew] logging {args.secs:.0f}s — rock the hip-pitch by hand "
              f"now, vary the tempo (Ctrl-C stops early)")
        t_start = time.monotonic()
        while time.monotonic() - t_start < args.secs:
            drv.send_command(args.motor, 0.0, 0.0, 0.0, 0.0, 0.0)   # limp poll
            pos, vel, _, _ = drv.get_state(args.motor)
            et.append(time.monotonic())
            ep.append(pos)
            ev.append(vel)
    except KeyboardInterrupt:
        print("\n[skew] stopped early")
    finally:
        try:
            drv.disable_all()
        except Exception:
            pass
        imu.close()

    et = np.asarray(et)
    ep = np.asarray(ep)
    ev = np.asarray(ev)
    g = np.asarray(imu.gyro_log)                 # (N, 4): t, gx, gy, gz
    if len(g) < 400 or len(et) < 2000:
        raise SystemExit(f"too little data (imu {len(g)}, enc {len(et)})")
    print(f"[skew] imu {len(g)} frames ({len(g)/(g[-1,0]-g[0,0]):.0f} Hz), "
          f"enc {len(et)} samples ({len(et)/(et[-1]-et[0]):.0f} Hz)")

    dpdt = np.gradient(ep, et)
    best = None
    for ax, name in ((1, 'gx'), (2, 'gy'), (3, 'gz')):
        gyro = np.deg2rad(g[:, ax])
        for sign in (1, -1):
            lag, corr = xcorr_lag(g[:, 0], sign * gyro, et, dpdt)
            if best is None or corr > best[3]:
                best = (name, sign, lag, corr)
    name, sign, lag_pos, corr = best
    print(f"[skew] best axis {name} (sign {sign:+d}), corr {corr:.3f}")
    # recompute cleanly on the chosen axis
    ax_i = {'gx': 1, 'gy': 2, 'gz': 3}[name]
    gyro = sign * np.deg2rad(g[:, ax_i])
    lag_pos, corr_p = xcorr_lag(g[:, 0], gyro, et, dpdt)
    lag_vel, corr_v = xcorr_lag(g[:, 0], gyro, et, ev)
    print(f"[skew] encoder-path lag vs IMU (dpos/dt, unfiltered): "
          f"{lag_pos*1000:+.2f} ms (corr {corr_p:.3f})")
    print(f"[skew] encoder-path lag vs IMU (vel_fb, filtered):    "
          f"{lag_vel*1000:+.2f} ms (corr {corr_v:.3f})")
    print(f"[skew] implied firmware velocity-filter lag: "
          f"{(lag_vel-lag_pos)*1000:.2f} ms")
    print("[skew] positive = that encoder stream arrives LATER than the IMU")

    if args.out:
        np.savez(args.out, imu=g, enc_t=et, enc_pos=ep, enc_vel=ev)
        print(f"[skew] saved {args.out}")


if __name__ == '__main__':
    main()
