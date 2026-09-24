# Curriculum history (p1–p5)

The stand-up task went through six configuration phases. Phase 6
(`rl_env_p6.py` + `rl_env_cfg_p6.py`) is the live one and lives in
`RL/source/RL/RL/tasks/direct/standup_curriculum/`. Phases 1–5 are here as
flat files so the reward and curriculum changes described in the retrospective
(§1, Phase A) can be read, not run:

- `rl_env_cfg_p1.py` … `rl_env_cfg_p5.py`, `rsl_rl_ppo_cfg_p1.py` … `_p5.py`:
  per-phase env and PPO configs. Their `ROBOT_USD_PATH` pointed at an earlier
  URDF export (`bigthigh-urdf`) that is not shipped.
- `rl_env.py`, `rl_env_p3.py` … `rl_env_p5.py`: the env bodies of those phases.
- `rl_env_p6_concise.py`, `rl_env_cfg_p6_concise.py`: a shortened p6 pair used
  for review; not what trained.

These files are not on any import path and are not maintained.
