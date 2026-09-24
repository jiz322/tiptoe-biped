#!/usr/bin/env python3
"""
nn_latency_bench.py — how fast can the control loop run if ONLY the policy
network is in the way?

No IMU, no CAN, no motors: this loads the real policy through the same
Policy class the controller uses and times nothing but inference. The
answer it gives is a CEILING. The real loop also pays IMU read, obs build
and CAN send/read (the 9.3 ms/tick in send_batch is what actually pins us
to 100 Hz today) — so treat this number as "the NN is/isn't the wall".

Three things are measured:

1. per-call latency of policy.forward(obs) — what controller.py calls.
   Reported as median / p95 / p99 / max, because a control loop lives or
   dies on the tail, not the mean.
2. the wrapper's share — forward() also does numpy->tensor, clamp and
   tensor->numpy. _infer() alone is the raw network. The gap is overhead
   you could remove without touching the model.
3. a closed-loop rate sweep — run inference on a fixed-period schedule at
   100/200/.../3200 Hz and count deadline misses. This is the direct
   answer to "what Hz does the NN support?"

Thread count matters more than you'd expect: this actor is a tiny MLP
(180-512-256-128-6), so multi-thread BLAS usually LOSES to a single
thread on a Pi-class CPU. --threads sweeps it.

Usage:
    python3 nn_latency_bench.py
    python3 nn_latency_bench.py --policy policy_F3490.pt --iters 20000
    python3 nn_latency_bench.py --threads 1,2,4        # find the best setting
    python3 nn_latency_bench.py --sweep-hz 100,400,1000,2000
    python3 nn_latency_bench.py --rt                  # SCHED_FIFO + CPU pin (Linux, sudo)
    python3 nn_latency_bench.py --csv nn_lat.csv      # per-call latencies
"""

import argparse
import gc
import os
import time

import numpy as np
import torch

import config as C


DEFAULT_POLICY = 'policy_F3490.pt'
OBS_POOL = 256          # pre-generated observations, cycled through


# ════════════════════════════════════════════════════════════
# Setup helpers
# ════════════════════════════════════════════════════════════

def set_realtime(priority=80, cpu=3):
    """Same treatment latency_test/ gives its timing loops. No-op on Windows."""
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
        print(f"[rt] SCHED_FIFO priority {priority}")
    except PermissionError:
        print("[rt] no RT priority (run with sudo for SCHED_FIFO)")
    except (OSError, AttributeError, ValueError) as e:
        print(f"[rt] sched_setscheduler unavailable: {e}")
    try:
        os.sched_setaffinity(0, {cpu})
        print(f"[rt] pinned to CPU {cpu}")
    except (OSError, AttributeError) as e:
        print(f"[rt] could not pin CPU: {e}")


def make_obs_pool(obs_dim, seed=0):
    """A pool of plausible observations. Values are unit-ish because the
    policy's normalizer expects roughly standardized input; the arithmetic
    cost does not depend on the values, but denormals would skew it."""
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(obs_dim).astype(np.float32)
            for _ in range(OBS_POOL)]


# ════════════════════════════════════════════════════════════
# Timing core
# ════════════════════════════════════════════════════════════

def time_calls(fn, pool, iters, warmup):
    """Call fn(pool[i]) `iters` times, timing each call individually.

    GC is off during measurement so a collection cannot masquerade as a
    slow inference. Returns (latencies_ms, wall_seconds).
    """
    for i in range(warmup):
        fn(pool[i % len(pool)])

    lat = np.empty(iters, dtype=np.float64)
    gc.collect()
    gc.disable()
    try:
        t_wall0 = time.perf_counter()
        for i in range(iters):
            obs = pool[i % len(pool)]
            t0 = time.perf_counter()
            fn(obs)
            lat[i] = time.perf_counter() - t0
        wall = time.perf_counter() - t_wall0
    finally:
        gc.enable()
    return lat * 1e3, wall


def report(label, lat_ms, wall, iters):
    med, p95, p99 = np.percentile(lat_ms, [50, 95, 99])
    print(f"\n  {label}")
    print(f"    min {lat_ms.min():6.3f}   median {med:6.3f}   mean {lat_ms.mean():6.3f} ms")
    print(f"    p95 {p95:6.3f}   p99    {p99:6.3f}   max  {lat_ms.max():6.3f} ms")
    print(f"    ceiling: {1e3/med:8.0f} Hz at median, {1e3/p99:7.0f} Hz at p99, "
          f"{iters/wall:7.0f} Hz sustained")
    return {'median': med, 'p99': p99, 'max': lat_ms.max(),
            'sustained_hz': iters / wall}


# ════════════════════════════════════════════════════════════
# Closed-loop rate sweep
# ════════════════════════════════════════════════════════════

def sweep_rate(policy, pool, hz_list, seconds):
    """Run inference on a fixed-period schedule and count deadline misses.

    The wait between ticks is a SPIN, not time.sleep(): sleep granularity
    (~1 ms on Linux, up to ~15 ms on Windows) would dominate above a few
    hundred Hz and we'd be measuring the OS timer instead of the network.
    A real loop at these rates has to spin too, so this is the honest test.

    Deadlines are absolute (next_t += period), so a slow tick must be paid
    back by the next one — the same way a control loop accumulates lateness.
    """
    print("\n  Closed-loop rate sweep (spin-wait scheduler)")
    print(f"    {'target':>8}  {'achieved':>9}  {'duty':>6}  {'misses':>7}  {'worst late':>10}")
    rows = []
    for hz in hz_list:
        period = 1.0 / hz
        n = max(200, int(seconds * hz))
        lat = np.empty(n)
        late = np.zeros(n)
        gc.collect()
        gc.disable()
        try:
            t_start = time.perf_counter()
            next_t = t_start
            for i in range(n):
                next_t += period
                t0 = time.perf_counter()
                policy.forward(pool[i % len(pool)])
                t1 = time.perf_counter()
                lat[i] = t1 - t0
                if t1 > next_t:
                    late[i] = t1 - next_t
                else:
                    while time.perf_counter() < next_t:
                        pass
            wall = time.perf_counter() - t_start
        finally:
            gc.enable()

        misses = int((late > 0).sum())
        duty = lat.mean() / period
        print(f"    {hz:7.0f}H  {n/wall:8.0f}H  {duty*100:5.1f}%  "
              f"{misses:4d}/{n:<4d}  {late.max()*1e3:8.3f} ms")
        rows.append({'hz': hz, 'achieved_hz': n / wall, 'duty': duty,
                     'miss_frac': misses / n, 'worst_late_ms': late.max() * 1e3})
    return rows


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--policy', default=DEFAULT_POLICY, help=f"policy file (default {DEFAULT_POLICY})")
    ap.add_argument('--device', default='cpu', help="cpu | cuda (deployment is cpu)")
    ap.add_argument('--obs-dim', type=int, default=None,
                    help=f"override C.OBS_DIM (default {C.OBS_DIM})")
    ap.add_argument('--iters', type=int, default=10000, help="timed calls per test")
    ap.add_argument('--warmup', type=int, default=500, help="untimed calls first")
    ap.add_argument('--threads', default='1',
                    help="comma list of torch thread counts to sweep, e.g. 1,2,4")
    ap.add_argument('--sweep-hz', default='100,200,400,800,1600,3200',
                    help="target rates for the closed-loop sweep; empty to skip")
    ap.add_argument('--sweep-seconds', type=float, default=1.0,
                    help="duration per rate in the sweep")
    ap.add_argument('--rt', action='store_true', help="SCHED_FIFO + CPU pin (Linux, sudo)")
    ap.add_argument('--csv', default=None, help="write per-call latencies (ms) here")
    args = ap.parse_args()

    if args.obs_dim is not None:
        C.OBS_DIM = args.obs_dim
    if args.rt:
        set_realtime()

    from policy import Policy      # imported after --rt so torch inherits the affinity

    thread_list = [int(t) for t in args.threads.split(',') if t.strip()]
    torch.set_num_threads(thread_list[0])
    policy = Policy(args.policy, device=args.device)
    pool = make_obs_pool(C.OBS_DIM)

    print(f"\n{'='*64}")
    print(f"  NN-only latency — {os.path.basename(args.policy)} "
          f"mode={policy._mode} obs={C.OBS_DIM} act={C.ACT_DIM}")
    print(f"  device={args.device}  iters={args.iters}  "
          f"control budget={C.DT*1e3:.2f} ms ({C.CONTROL_HZ} Hz)")
    print(f"{'='*64}")

    best = None
    for nt in thread_list:
        torch.set_num_threads(nt)
        lat, wall = time_calls(policy.forward, pool, args.iters, args.warmup)
        st = report(f"policy.forward()  threads={nt}", lat, wall, args.iters)
        st['threads'] = nt
        # Ranked on p99, not median: a loop misses its deadline on the tail.
        if best is None or st['p99'] < best['p99']:
            best, best_lat = st, lat

    torch.set_num_threads(best['threads'])
    print(f"\n  Best thread count: {best['threads']}")

    # Raw network, no numpy conversion / clamp: shows what the wrapper costs.
    obs_t = [torch.from_numpy(o).unsqueeze(0) for o in pool]
    with torch.no_grad():
        lat_raw, wall_raw = time_calls(
            lambda o: policy._infer(o), obs_t, args.iters, args.warmup)
    raw = report(f"_infer() only     threads={best['threads']}", lat_raw, wall_raw, args.iters)
    overhead = best['median'] - raw['median']
    print(f"    wrapper overhead: {overhead:.3f} ms median "
          f"({overhead/best['median']*100:.0f}% of forward())")

    hz_list = [float(h) for h in args.sweep_hz.split(',') if h.strip()]
    if hz_list:
        sweep_rate(policy, pool, hz_list, args.sweep_seconds)

    if args.csv:
        np.savetxt(args.csv, best_lat, fmt='%.6f', header='forward_ms', comments='')
        print(f"\n  wrote {args.csv}  ({len(best_lat)} samples)")

    print(f"\n{'='*64}")
    print(f"  NN alone supports ~{1e3/best['p99']:.0f} Hz (p99) / "
          f"{1e3/best['median']:.0f} Hz (median).")
    print(f"  At the current {C.CONTROL_HZ} Hz it eats "
          f"{best['median']/(C.DT*1e3)*100:.1f}% of the {C.DT*1e3:.1f} ms tick "
          f"({best['max']/(C.DT*1e3)*100:.1f}% worst case).")
    print(f"{'='*64}\n")


if __name__ == '__main__':
    main()

