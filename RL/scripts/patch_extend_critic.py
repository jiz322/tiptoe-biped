import torch, sys

ckpt = torch.load(sys.argv[1], map_location="cpu")
sd = ckpt["model_state_dict"]

for suffix, fill in [("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)]:
    key = f"critic_obs_normalizer.{suffix}"
    old = sd[key]                          # [1, 245]
    pad = torch.full((1, 1), fill)
    sd[key] = torch.cat([old, pad], dim=1) # [1, 246]
    print(f"{key}: {list(old.shape)} → {list(sd[key].shape)}")

torch.save(ckpt, sys.argv[1])
print("done")