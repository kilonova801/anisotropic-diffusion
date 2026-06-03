# ImageNet-64 experiment scripts

Maps each ImageNet-64 experiment to its original scripts under `experiments/imagenet/`.
These use PyTorch DDP for multi-GPU training. For new work prefer the consolidated entry
points in `scripts/` (see `ARCHITECTURE.md`).

## g-iso (conditional schedule, global DCT basis)

| Role | Script |
|------|--------|
| Train | `g_iso_train_rampup_ema_condition_debug32.py` |
| Sample (EDM/Heun) | `g_sample_path_trainednet_edmsampler.py` |
| Sample (standard) | `g_sample_path_condition.py` |

## g-ani (global schedule, conditional PCA basis)

| Role | Script |
|------|--------|
| Train | `g_ani_train_rampup_ema_basis_imagenet_v3_newpca_ddp.py` |
| Sample (EDM/Heun) | `g_sample_path_trainednet_ani_edmsampler_v2_basis_imagenet_v3_newpca.py` |
| Sample (standard) | `g_sample_path_basis_imagenet_v3_newpca.py` |

## g-ani (conditional schedule, global DCT basis)

| Role | Script |
|------|--------|
| Train | `g_ani_train_rampup_ema_condition_imagenet_ddp.py` |
| Sample (EDM/Heun) | `g_sample_path_trainednet_ani_edmsampler_v2_condition.py` |
| Sample (standard) | `g_sample_path_condition.py` |

## g-ani (conditional schedule, conditional PCA basis)

| Role | Script |
|------|--------|
| Train | `g_ani_train_rampup_ema_basis_imagenet_v3_newpca_condition_ddp.py` |
| Sample (EDM/Heun) | `g_sample_path_trainednet_ani_edmsampler_v2_basis_imagenet_v3_newpca_condition.py` |
| Sample (standard) | `g_sample_path_basis_imagenet_v3_newpca_condition.py` |
