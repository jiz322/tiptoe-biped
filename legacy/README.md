# legacy/

History, kept so the retrospective's claims can be checked. Nothing here is
maintained, and most of it encodes hypotheses that were later overturned. Read
`docs/RETROSPECTIVE-2026-09.md` before trusting any file in this folder.

| Folder | What it is | Live replacement |
|---|---|---|
| `launch/` | Every training and eval launcher of the campaign (arms A … Z, K, L, R, S1–S8), with machine paths replaced by placeholders. Each header documents the arm's rationale, pre-registered kill rules and results. | `RL/launch/` |
| `curriculum_history/` | Curriculum phases p1–p5 of the stand-up task (2026-04) plus the pre-belt `rl_env.py` and the `p6_concise` variant. Not importable as a package on purpose. | `RL/source/RL/RL/tasks/direct/standup_curriculum/` (p6) |
| `control_can2usb/` | Deploy-side scripts from the CAN-over-USB era (before 2026-08-15). They construct the serial driver directly and fail on the SocketCAN robot. | `control/robstride_socketcan_driver.py` |
| `script_pre_rl/` | Classical PD / LQR stand controllers and the Isaac Sim Script Editor tools from the first week (2026-03-28). | the RL env |
