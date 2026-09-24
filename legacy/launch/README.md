# Archived launch scripts

One PowerShell script per training or evaluation arm, 2026-04 to 2026-09,
exactly as run, except that machine paths became `<REPO>`, `<ISAACLAB>`,
`<CONDA_ENV>` and GPU UUIDs became `GPU-<uuid>`. The header of each file is
the arm's design note: what changed, why, the pre-registered kill rule, and
(often) the outcome added afterwards.

Lineage of the released checkpoint `control/r1r_3150.pt` (R1r_3150):

1. `launch_train_K_r400_zero069.ps1`: K line, designed servo gains kp 20/30,
   swing-clearance zero rule.
2. `launch_train_R1_robust.ps1`: P13 robustness bundle (exec-noise on targets,
   pushes in training, delay DR 1–3, wider spawn DR).
3. `launch_train_R1_resume3110.ps1`: resume from iter 3110 with lower
   `desired_kl`, tighter clip and 3× smoothness/snap penalties. `model_3150.pt`
   of this run is R1r_3150.

The S line (S1 … S8) is the post-09-03 work: coherent CoM-shift DR, the seiza
stand-up path mode, measured plant DR, belt-elasticity and restitution DR.
`RL/launch/train_S7_fullDR.ps1` is the maintained form of S7.

Running one of these today: copy its `$argList` into `RL/launch/train_S7_fullDR.ps1`
(or the `.sh`) and drop the `--resume` lines unless you have the parent run.
