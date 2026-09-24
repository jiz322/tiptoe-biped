# Contributing

This is a one-person project released as open hardware plus a research log.
No support is promised. What is promised: replication reports get read first.

- **Replications** are the most valuable contribution. If you build one, open an
  issue with the label `replication` and include: BOM deviations, the trim
  (`IMU_PG_OFFSET`) that worked, a recording (`balance_log.py` npz), and video.
  Two robots with the same failure signature settle questions one robot cannot.
- **Pull requests** are welcome. A PR that changes the plant model, the
  observation contract or the training recipe should state, before the results,
  what it predicts will change and how that was measured. PRs with a
  pre-registered prediction are reviewed first. See `docs/PREDICTIONS_p12_3.md`
  for the format the project used.
- **Questions** go in GitHub Discussions. The pinned discussion is the open
  problem: what does the `+0.089` pitch trim compensate?
- **Do not** open issues for `legacy/` or `data/_*.py`-style one-off scripts.
  They document overturned hypotheses and are kept as history, not as tools.

Conventions used in the docs and code, so your report reads the same way:
"trial" is a real hardware recording, "rollout" is a sim episode, and every
analysis states the hypothesis it tests and returns supported / contradicted /
inconclusive.
