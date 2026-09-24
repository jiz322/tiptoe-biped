# Environment pins

Measured on the machines the campaign ran on, 2026-09-24. Reproducibility
issues are the most common issue type on robot repos, so these are exact.

## Training workstation (Windows 11)

| Component | Version | Note |
|---|---|---|
| Python | 3.11.14 | conda env |
| PyTorch | 2.7.0+cu128 | CUDA 12.8 wheel |
| Isaac Sim | 5.1.0-rc.19 (build 26219) | binary install, `_isaac_sim/VERSION` |
| Isaac Lab | 0.54.0 | source install, upstream commit `cbf51abb5` (2026-01-22) |
| rsl-rl-lib | 3.1.2 **+ local patch** | see `RL/patches/README.md` |
| NVIDIA driver | 591.86 | |
| GPUs | RTX 4090 (training), RTX 4060 (pose viewers) | one training arm per GPU |

Isaac Lab scripts are launched through `RL/launch/*.ps1` (historical arms:
`legacy/launch/`), which export the Kit environment variables (`ISAAC_PATH`,
`CARB_APP_PATH`, `EXP_PATH`) that the Python entry points need. Set
`ISAACSIM_PATH` to the Isaac Sim binary install and activate the conda env first.

## Robot (Raspberry Pi 5)

| Component | Version | Note |
|---|---|---|
| Raspberry Pi OS | **TBD** | robot offline when this file was written |
| Kernel | **TBD** | `mcp251xfd` overlay for the CAN module |
| Python | **TBD** | miniforge3 env; the system Python has no torch |
| torch (CPU) | **TBD** | TorchScript policy inference |
| python-can | **TBD** | SocketCAN `can0`, 1 Mbps classic CAN |

Fill the Pi rows with:

```sh
cat /etc/os-release | grep -E "PRETTY|VERSION_ID"; uname -r
~/miniforge3/bin/python3 -c "import sys,torch,can;print(sys.version.split()[0],torch.__version__,can.__version__)"
```

## Analysis scripts

`data/*.py` and `analysis_motor_filters/*.py` need only numpy, scipy and
matplotlib (any Python ≥ 3.10).
