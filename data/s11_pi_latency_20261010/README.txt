S11 Pi latency bench, 2026-10-10.  No flight. A timed forward pass of the baseline
actor and of the S11 history actor on the Pi, plus the history-ring update and a
2000-tick 100 Hz loop test, to decide whether the long-history policy fits the loop.

Policy: control/s11_11880.ts.pt (S11_hist model_11880), TorchScript, on the Pi.
Tool: control/hist_latency_bench.py. Raw output: bench.out (threads 1/2/4);
bench.json (the same, parsed, plus the ring timings and the loop test).

Numbers are milliseconds. Forward pass, 1 thread: baseline p50 0.186 / p99 0.221;
history actor p50 0.609 / p99 0.696. The whole 100 Hz tick with the history actor
(4 threads, 2000 ticks): p50 0.70, p95 0.74, p99 0.80, max 1.70; 0 ticks over 3 ms,
no missed deadline. Ring update (numpy roll) p50 0.033; double write p50 0.012.
