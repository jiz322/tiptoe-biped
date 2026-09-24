# Pre-RL controllers (2026-03-28, first week)

Classical stand controllers written in the Isaac Sim Script Editor before the
RL task existed: `stand.py`, `bipedal_pd_balance.py`, the two LQR variants and
the `joint_tuner.py` / `alpha_tester_p6d.py` helpers. Historical only. Note
that these set USD DriveAPI stiffness in per-degree units (`JOINT_STIFFNESS =
40` is about 2300 Nm/rad, not the 8 Nm/rad the RL env uses), which is one
reason they are not comparable to the RL work. The live pose viewers are in
`script/`.
