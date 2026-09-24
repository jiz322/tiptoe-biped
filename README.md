# TipToe

**A ¥5500 point-foot biped, trained with PPO in Isaac Lab, built by one
person working with an AI coding agent. It stood for 6 seconds. Here is
everything, including what went wrong.**

<!-- TODO(owner): GIF of the 2026-09-03 stand here, and the seiza photo -->
<!-- ![6.06 s free-standing balance, 2026-09-03](docs/img/stand-20260903.gif) -->

Video: *to be added* (Bilibili / YouTube).

TipToe is a 6-DOF biped with no ankles: hip roll, hip pitch and knee per leg,
four RobStride RS01 and two EL05 quasi-direct-drive actuators, a Raspberry Pi 5
running the policy at 100 Hz over CAN, printed structure, and a belt-driven
knee whose motor sits in the hip body (a heron's knee, if you like). Point feet
give no static balance, so the policy has to step reactively, and every
millimetre of centre-of-mass error, degree of IMU bias, or unmodelled
newton-metre in the knee shows up as a fall within a second.

This repository is **open hardware plus an open sim2real research log**. It is
not a platform, and it does not claim to be one: the robot has stood once, for
6.06 s, still upright when the log stopped. What it offers is the complete
record of getting there, from empty repo (2026-03-28) to first stand
(2026-09-03): the plant model, the fifteen fixes without which it would not
stand, the pre-registered predictions that were right and the hypotheses that
were wrong, and an open problem for you to attack.

## Status

| Works | Does not work yet |
|---|---|
| Stands from a ground start and balances in place for 6 s with the released policy and the +0.089 pitch trim (n = 1 at that configuration; 5/5 and 6/6 falls at the two other trim values the same day). | Reliable standing. Before 09-03 the best trial was 0.67 s to a 25° tilt. |
| Full training pipeline in Isaac Lab: belt-knee actuator model, measured plant DR, kneel-to-stand curriculum along a seiza → crouch → squat path. | Stand-up from seiza on hardware (trained in sim, not yet flown). |
| Deploy stack: RobStride MIT-mode driver over SocketCAN (MCP2518FD), 400 Hz IMU, ELRS, rsl_rl loader with the observation normaliser, logging to npz. | Walking. The v1 sim policy walked in April; nothing has walked on hardware. |
| Bench identification: rotor inertia, friction, belt coupling and elasticity, firmware PD replica, CAN timing. | Belt reliability: the right knee skips a tooth under catch-step loads. |

**Open problems** (pinned in Discussions):

1. **What does the +0.089 trim compensate?** The policy only stands when told
   it leans about 7° further back than it does. The measured IMU mounting
   offset is −0.032, so the working knob is 0.12 past the truth. Leading
   candidate: a knee joint-angle bias from belt slip or stretch that the
   motor-side encoder cannot see. Candidate, not fact. The measurements and
   the ruled-out hypotheses are in the retrospective §2 (#14) and
   `docs/imu-offset-and-joint-zero-20260920.md`.
2. **The belt skips.** One-tooth skips at 6.5–12.6 Nm joint-side, three times
   in one session on 09-10. Each shifts the knee zero by 0.21 rad, invisibly.
   Aluminium pulleys, a wider belt or a knee-side encoder are the obvious
   experiments; none has been run.

## Who built it

One person and an AI coding agent. The person designed and built the robot,
ran every hardware session, watched the video frame by frame, and supplied
the physical facts and the questions. The agent (Claude Code, and earlier
Claude through the web app) wrote most of the code, the physics derivations
and the analysis, and diffed the two sides of the sim2real gap when they
disagreed. The retrospective §3 has the fix-by-fix attribution table, the
assessment of who noticed, localised, implemented and verified each of the
fifteen fixes, and a list of the agent's hypotheses the record proved wrong
next to the user's. If you are interested in what agent-driven engineering on
a physical robot looks like over six months, that section and the five
lessons that follow it are the reason this repo is public.

## Build it

- Bill of materials with the prices actually paid: [hardware/BOM.md](hardware/BOM.md).
  Priced lines total about ¥5360 (motors ¥2994, Pi + SD ¥1013, IMU ¥254,
  CAN board ¥40, two 4S packs ¥762, PDB ¥98, print and screws under ¥200).
- Print files: MakerWorld listing (*link at launch*). STEP export of the v2.7
  assembly: GitHub Release v0.1.0 (*link at launch*).
- Assembly notes, wiring, IMU mounting, motor zeroing: [hardware/ASSEMBLY.md](hardware/ASSEMBLY.md).
- CAN bring-up on the Pi: [control/CAN2SPI_BRINGUP.md](control/CAN2SPI_BRINGUP.md).

## Run it

On the Pi, in the `control/` folder (torch needed; the campaign used a
miniforge environment, see `docs/ENVIRONMENT.md`):

```sh
sudo ./setup_can0.sh                        # per boot
sudo python3 test_motors.py --no-zero       # all six motors reply, zeros survive
sudo python3 balance_log.py --policy r1r_3150.pt --pg-offset 0.089
# keys: 1/2/3 select group, +/- kp, ]/[ kd, g observe, v blind handover (start), h hold, q quit
```

The recording lands next to the script as `r1r_3150_log_<timestamp>.npz`, in
the schema described in [data/README.md](data/README.md). Desk-side, with no
hardware, `python control/policy_contract_check.py control/r1r_3150.pt`
checks the checkpoint against the deploy observation contract.

## Train it

Isaac Sim 5.1 + Isaac Lab 0.54 (pins in [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)):

```sh
python -m pip install -e RL/source/RL
# apply the rsl_rl learning-rate-ceiling patch: RL/patches/README.md (required)
RL/launch/play.sh --gpu 0                       # watch the released policy balance
RL/launch/train_S7_fullDR.sh --gpu 0 --headless # the last training recipe
```

Windows equivalents are the `.ps1` files next to them. Details, the three
rules that cost weeks (resumes must re-pass every override; never resume with
a fixed schedule; curriculum advance on relative success), and the sim2real
tools are in [RL/README.md](RL/README.md).

## Read it

The docs are the product. In reading order:

| Doc | What |
|---|---|
| [docs/RETROSPECTIVE-2026-09.md](docs/RETROSPECTIVE-2026-09.md) | The six months, from evidence: timeline, the 15 fixes with symptom / root cause / verification, who did what, current state, lessons on working with an AI agent on a physical robot. |
| [docs/p12-4-session-20260903.md](docs/p12-4-session-20260903.md) | The day it stood: the trim bracket −0.089 / 0 / +0.089. |
| [docs/REPO_MAP.md](docs/REPO_MAP.md) | Where everything is, the robot in numbers, the gotchas, the sim2real deployment checklist. |
| [docs/knee-hip-coupling-review-20260821.md](docs/knee-hip-coupling-review-20260821.md) | The belt-coupled knee: the math, how Mini Cheetah avoided it, the sim2real risk ranking. |
| [reports/belt_coupling_sim2real/report.md](reports/belt_coupling_sim2real/report.md) | Does modelling the belt close the gap? Teacher-forcing replay of 7 hardware logs (08-07). |
| [docs/replant-report-20260817.html](docs/replant-report-20260817.html) | Plant identification bench: friction, rotor inertia, coupling, latency verdicts. The plant-parameter source of truth. |
| [docs/imu-sign-check.md](docs/imu-sign-check.md) | How the mirrored IMU signs were found and certified from a recorded hand manoeuvre. |
| [docs/PREDICTIONS_p12_3.md](docs/PREDICTIONS_p12_3.md) | A pre-registered hardware A/B (sighted vs blind handover) and its scoring. |
| [docs/standup-path-mode-20260908.md](docs/standup-path-mode-20260908.md) | Stand-up redesign: seiza geometry, balance manifold, the path mode. |
| [docs/s4-stepping-gate-failure-analysis-20260911.md](docs/s4-stepping-gate-failure-analysis-20260911.md), [s4-checkpoint-testing](docs/s4-checkpoint-testing-20260911.md), [s5-elastic-dr](docs/s5-elastic-dr-20260914.md), [armature-derivation-note](docs/armature-derivation-note-20260918.md), [imu-offset-and-joint-zero](docs/imu-offset-and-joint-zero-20260920.md) | The post-stand S line: measured DR, belt elasticity DR, armature from rotor facts, the trim is not an IMU bias. |
| [docs/p11-hardware-runbook.md](docs/p11-hardware-runbook.md), [p12-fall-isolation-runbook](docs/p12-fall-isolation-runbook.md), [p12-4-bench-runbook](docs/p12-4-bench-runbook.md), [p9-checkpoint-selection](docs/p9-checkpoint-selection-20260823.md) | Runbooks as written on the day, including the 740-line one that was abandoned and the retracted claims it still carries. |
| [docs/pad-contact-dr-20260912.md](docs/pad-contact-dr-20260912.md) | A contact model that never applied a force (retired 09-17). Kept as a worked example of a knob whose input is never excited. |
| [analysis_motor_filters/](analysis_motor_filters/) | Motor telemetry forensics: the torque feedback is filtered and dead-banded. |
| [docs/v1-postmortem.md](docs/v1-postmortem.md) | April: why the first "balancing" sim policy was a bang-bang hip-roll degenerate. |

## Dataset

Every flight session, one npz per trial at 100 Hz: observations, actions,
targets, joint state, motor torques and temperatures, IMU, loop timing. Nine
sessions, 69 trials, 2026-08-24 to 09-10, including the 6.06 s stand.
Released as zips on the v0.1.0 Release page under CC BY 4.0; the schema, the
per-session contracts (trim, handover, IMU sign state) and a loader snippet
are in [data/README.md](data/README.md).

## Layout

```
hardware/    BOM, assembly notes            control/   deploy stack + released checkpoint
assets/      URDF, meshes, USD (v2.7)       RL/        Isaac Lab task, launchers, patch
docs/        reports and runbooks           data/      analysis scripts, session READMEs
script/      Isaac Sim pose viewers         legacy/    every launch script, old phases, pre-RL
reports/, analysis_motor_filters/           concluded bench studies with figures
```

`big_thigh` in the docs and code is the project's internal codename.

## Licensing

- Code: MIT ([LICENSE](LICENSE)). `RL/source/RL` retains the Isaac Lab
  template's BSD-3-Clause notices.
- Hardware (URDF, meshes, CAD, print files), documents, reports and the
  dataset: CC BY 4.0 ([LICENSE-CC-BY-4.0](LICENSE-CC-BY-4.0)).
- RobStride protocol documents are not redistributed; get them from the vendor.

## Contributing and citation

No support is promised; replication reports are read first. See
[CONTRIBUTING.md](CONTRIBUTING.md). Cite with [CITATION.cff](CITATION.cff).
