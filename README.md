# Variational Trajectory Optimization of Anisotropic Diffusion Schedules

Code for **"Variational Trajectory Optimization of Anisotropic Diffusion Schedules."**
We learn matrix-valued diffusion schedules `M_t(θ) = Σ_j g_j(t;θ) P_j` that allocate noise
across subspaces, jointly training the score network and the schedule with a
trajectory-level objective and a schedule-gradient estimator that needs only x-directional
derivatives of the network. Sampling uses an anisotropic generalization of the second-order
Heun solver. Across CIFAR-10, AFHQv2, FFHQ, and ImageNet-64 this improves over the EDM
baseline across NFE budgets, with the largest gains at low NFE. Built on
[NVLabs/EDM (Karras et al., 2022)](https://github.com/NVlabs/edm).

For a map from paper sections/equations to code, see [`ARCHITECTURE.md`](ARCHITECTURE.md).
For known issues and cleanup notes, see [`CODE_REVIEW.md`](CODE_REVIEW.md).

## Repository structure

```
common_utils.py        Core library: schedules, DCT/PCA bases, operators, solvers, losses
scripts/train/         Training entry points (one per schedule family)
scripts/sample/        Sampling entry points (matrix-Heun / EDM solvers)
scripts/launch/        Convenience shell launchers
experiments/cifar10/   Original per-experiment scripts (CIFAR-10 results)
experiments/imagenet/  Original per-experiment scripts (ImageNet-64 results)
docs/                  Experiment-to-script maps
dnnlib/ torch_utils/ training/ + train.py generate.py fid.py dataset_tool.py
                       Inherited unchanged from NVLabs/EDM (kept at root; see ARCHITECTURE.md)
```

## Environment

```bash
conda create -n anidiff python=3.9
conda activate anidiff
pip install -r requirements.txt
pip install -e .          # exposes common_utils / dnnlib / torch_utils / training as importable modules
```

The pinned `torch==1.12.1+cu116` / `torchvision==0.13.1+cu116` wheels require the PyTorch
CUDA index; if `pip install -r requirements.txt` cannot find them, add
`--extra-index-url https://download.pytorch.org/whl/cu116`.

`pip install -e .` lets the entry points in `scripts/` import the core modules from any
working directory. If you prefer not to install, run every command from the repository
root with `PYTHONPATH=.` prefixed (e.g. `PYTHONPATH=. python scripts/train/g_iso_train.py ...`).

## Dataset preparation

Dataset preparation follows the official
[NVLabs/EDM procedure](https://github.com/NVlabs/edm). Download the original datasets, then
convert them to the required resolution with the inherited `dataset_tool.py`:

```bash
python dataset_tool.py --source=<cifar10_folder> --dest=datasets/cifar10-32x32.zip --resolution=32
python dataset_tool.py --source=<afhqv2_folder>  --dest=datasets/afhqv2-64x64.zip  --resolution=64
python dataset_tool.py --source=<ffhq_folder>    --dest=datasets/ffhq-64x64.zip    --resolution=64
```

The training scripts expect the processed datasets under `datasets/`.

## Training

Each script under `scripts/train/` trains one schedule family (see the variant table in
[`ARCHITECTURE.md`](ARCHITECTURE.md)). Run from the repository root:

```bash
# Isotropic schedule (warm-up / baseline)
python scripts/train/g_iso_train.py --dataset cifar10

# Anisotropic schedule (fixed DCT basis)
python scripts/train/g_ani_train.py --dataset afhqv2

# Anisotropic with learned/PCA basis, and the conditional variants
python scripts/train/g_ani_train_basis.py            --dataset cifar10
python scripts/train/g_ani_train_condition.py        --dataset cifar10
python scripts/train/g_ani_train_condition_basis.py  --dataset imagenet
```

## Sampling

```bash
# Unified sampler (dispatches by --model_tag)
python scripts/sample/g_sample.py --dataset cifar10 --model_tag g-iso --steps 20
python scripts/sample/g_sample.py --dataset afhqv2  --model_tag g-ani --steps 20

# Variant-specific matrix-Heun samplers
python scripts/sample/g_ani_sample_basis_edm.py            --dataset cifar10
python scripts/sample/g_ani_sample_basis_condition_edm.py  --dataset imagenet
```

## Reproducing the paper experiments

`scripts/` holds the consolidated, dataset-agnostic entry points. The exact per-dataset
scripts that produced the reported numbers are under `experiments/cifar10/` and
`experiments/imagenet/`, and `scripts/launch/run_all_train_and_sample.sh` runs training +
sampling + FID end to end. The script-to-experiment mapping is in
[`docs/cifar10-experiments.md`](docs/cifar10-experiments.md) and
[`docs/imagenet-experiments.md`](docs/imagenet-experiments.md).

## Acknowledgments

Builds on [NVLabs/EDM (Karras et al., 2022)](https://github.com/NVlabs/edm). The
`torch_utils/`, `training/`, and `dnnlib/` packages and the `train.py`, `generate.py`,
`fid.py`, `dataset_tool.py` scripts are inherited from EDM.

## License

EDM is released under **CC BY-NC-SA 4.0** (non-commercial, share-alike). As a derivative
work, the inherited components remain subject to that license; add a top-level `LICENSE`
reflecting the terms intended for the new code in this repository before public release.
