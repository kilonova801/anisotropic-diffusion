# CIFAR-10 experiment scripts

Maps each CIFAR-10 experiment to its original scripts under `experiments/cifar10/`.
For new work prefer the consolidated entry points in `scripts/` (see `ARCHITECTURE.md`).

Naming: a *basis* is the subspace decomposition (global DCT, or class-conditional PCA);
a *schedule* `g_j(t)` can be global or class-conditional.

## Isotropic baseline

| Role | Script |
|------|--------|
| Train | `g_iso_train_rampup_ema.py` |
| Sample (EDM/Heun) | `g_sample_path_trainednet_edmsampler.py` |
| Sample (standard) | `g_sample_path.py` *(referenced in the original notes but not included in this release)* |

## DCT (global basis, conditional schedule)

| Role | Isotropic | Anisotropic |
|------|-----------|-------------|
| Train | `g_iso_train_rampup_ema_condition_debug.py` | `g_ani_train_rampup_ema_condition.py` |
| Sample (EDM/Heun) | `g_sample_path_trainednet_edmsampler.py` | `g_sample_path_trainednet_ani_edmsampler_v2_condition.py` |
| Sample (standard) | `g_sample_path_condition.py` | `g_sample_path_condition.py` |

## PCA (conditional basis, global schedule)

| Role | Script |
|------|--------|
| Train | `g_ani_train_rampup_ema_basis.py` |
| Sample (EDM/Heun) | `g_sample_path_trainednet_ani_edmsampler_v2_basis.py` |
| Sample (standard) | `g_sample_path_basis.py` |
