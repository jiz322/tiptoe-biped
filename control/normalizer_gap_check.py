"""normalizer_gap_check.py — evidence that the DEPLOYED inference skipped the
rsl_rl observation normalizer, and how much that changed the actions.

Root cause (control/policy.py, _load_from_checkpoint):
    the loader searches the state dict for a key containing 'running_mean',
    but rsl_rl 3.x checkpoints store the normalizer as
        actor_obs_normalizer._mean / ._var / ._std / .count
    -> search misses -> self._normalizer = None -> the actor is fed RAW obs.
    Training (ActorCritic.act / act_inference) always normalizes first:
        y = (x - _mean) / (_std + 1e-2)          # rsl_rl EmpiricalNormalization

For each (checkpoint, hardware log) pair this script:
  1. confirms the normalizer buffers exist and are far from identity;
  2. reproduces the LOGGED hardware actions from the logged obs through the
     rebuilt actor WITHOUT normalization (proves hardware ran unnormalized);
  3. computes what the actions WOULD have been WITH training normalization;
  4. reports the divergence + saturation and the most OOD obs channels.

Usage:  conda run -n env_isaaclab python normalizer_gap_check.py
"""
import numpy as np
import torch
import torch.nn as nn

PAIRS = [
    ('2026-06-20_22-02-53_jrk020_grf13.pt', '2026-06-20_22-02-53_jrk020_grf13_log_20260806_083901.npz'),
    ('2026-06-15_07-07-49_jrk032_grf14.pt', '2026-06-15_07-07-49_jrk032_grf14_log_20260806_084106.npz'),
    ('model_4150.pt',                        'model_4150_log_20260806_084306.npz'),
]
K, F = 5, 36
EPS = 1e-2          # rsl_rl EmpiricalNormalization eps — added to STD, not var
NAMES = ['hr1', 'hp1', 'kn1', 'hr2', 'hp2', 'kn2']
LABELS = (
    ["proj_grav_x", "proj_grav_y", "proj_grav_z"]
    + ["ang_vel_x", "ang_vel_y", "ang_vel_z"]
    + ["lin_vel_x", "lin_vel_y", "lin_vel_z"]
    + [f"jpos_{n}" for n in NAMES] + [f"jvel_{n}" for n in NAMES]
    + [f"act_{n}" for n in NAMES]
    + ["gait_sin", "gait_cos"]
    + ["difficulty", "grace", "init_h", "vx_cmd", "yaw_cmd", "yaw_sin", "yaw_cos"]
)


def build_actor(sd):
    """Rebuild the rsl_rl actor MLP (same as control/policy.py _ActorMLP)."""
    wks = sorted(k for k in sd if k.startswith('actor.') and k.endswith('.weight'))
    layers = []
    for i, wk in enumerate(wks):
        w, b = sd[wk], sd[wk.replace('.weight', '.bias')]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data = w
        lin.bias.data = b
        layers.append(lin)
        if i < len(wks) - 1:
            layers.append(nn.ELU())
    return nn.Sequential(*layers).eval()


def run_actor(actor, obs, mean=None, std=None):
    """Replay per-frame obs through the actor with balance_log.py's exact
    history stacking (first frame replicated into all K slots, then
    roll-and-write-slot-0)."""
    n = len(obs)
    hist = np.zeros((K, F), dtype=np.float32)
    hist[:] = obs[0]
    out = np.zeros((n, 6), dtype=np.float32)
    with torch.no_grad():
        for t in range(n):
            if t > 0:
                hist = np.roll(hist, shift=1, axis=0)
                hist[0] = obs[t]
            x = torch.from_numpy(hist.reshape(1, -1))
            if mean is not None:
                x = (x - mean) / (std + EPS)
            out[t] = actor(x).clamp(-1, 1).numpy()
    return out


for pt, npz in PAIRS:
    ck = torch.load(pt, map_location='cpu', weights_only=False)
    sd = ck['model_state_dict']
    mean = sd['actor_obs_normalizer._mean']    # (1, 180)
    std = sd['actor_obs_normalizer._std']
    cnt = int(sd['actor_obs_normalizer.count'])
    actor = build_actor(sd)

    d = np.load(npz, allow_pickle=True)
    obs = d['obs'].astype(np.float32)
    logged_action = d['action'].astype(np.float32)
    n = len(obs)

    print('=' * 78)
    print(f'{pt}   iter={ck.get("iter")}   log={npz} ({n} ticks)')
    m, s = mean.squeeze(0), std.squeeze(0)
    print(f'  [1] normalizer buffers present, count={cnt:.2e} samples; '
          f'std outside [0.9,1.1] on {(((s < 0.9) | (s > 1.1)).sum().item())}/180 dims '
          f'-> NOT an identity transform')

    act_raw = run_actor(actor, obs)
    d_raw = np.abs(act_raw - logged_action).max()
    print(f'  [2] actor(RAW obs) vs logged hw action: max|diff| = {d_raw:.2e}  '
          f'-> hardware provably ran WITHOUT normalization')

    act_norm = run_actor(actor, obs, mean, std)
    diff = np.abs(act_norm - logged_action)
    corr = [float(np.corrcoef(act_norm[:, j], logged_action[:, j])[0, 1]) for j in range(6)]
    sat_hw = (np.abs(logged_action) >= 0.999).mean(axis=0)
    sat_ok = (np.abs(act_norm) >= 0.999).mean(axis=0)
    print(f'  [3] actor(NORMALIZED obs) vs hw action: mean|diff| {diff.mean():.3f}  '
          f'p95 {np.percentile(diff, 95):.3f}  max {diff.max():.3f}   (action units, range ±1)')
    print('      per-joint mean|diff|: '
          + '  '.join(f'{NAMES[j]}={diff[:, j].mean():.2f}' for j in range(6)))
    print('      per-joint corr(correct, hw): '
          + '  '.join(f'{NAMES[j]}={corr[j]:+.2f}' for j in range(6)))
    print(f'      saturation |a|=1 frac  hw:      {np.round(sat_hw, 2)}')
    print(f'      saturation |a|=1 frac  correct: {np.round(sat_ok, 2)}')

    # [4] most OOD channels of the hw obs vs the training running stats
    m36, s36 = m.numpy()[:F], s.numpy()[:F]
    z = (obs - m36) / (s36 + EPS)
    zbar = np.abs(z).mean(axis=0)
    worst = np.argsort(-zbar)[:4]
    print('  [4] most OOD obs channels vs training stats (|z| mean): '
          + ',  '.join(f'{LABELS[i]}={zbar[i]:.1f}' for i in worst))
