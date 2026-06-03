# Architecture

This document maps the code to the paper *"Variational Trajectory Optimization of
Anisotropic Diffusion Schedules"* so a reader can go from a section or equation to the
code that implements it. The method builds on
[NVLabs/EDM](https://github.com/NVlabs/edm); the inherited pieces are listed at the end.

## Directory layout

```
anisotropic-diffusion/
├── common_utils.py          Core library: schedules, bases, operators, solvers, losses
├── data_loader.py           Dataset loading helpers
│
├── scripts/
│   ├── train/               Training entry points (one per schedule family)
│   ├── sample/              Sampling entry points (matrix-Heun / EDM solvers)
│   └── launch/              Convenience shell launchers (run from repo root)
│
├── experiments/
│   ├── cifar10/             Original per-experiment scripts for CIFAR-10 results
│   └── imagenet/            Original per-experiment scripts for ImageNet-64 results
│
├── docs/                    Experiment-to-script maps (see below)
│
│  # inherited unchanged from NVLabs/EDM — kept at the root on purpose (see note)
├── dnnlib/  torch_utils/  training/
└── train.py  generate.py  fid.py  dataset_tool.py
```

The root-level importable modules (`common_utils.py`, `data_loader.py`) and the EDM
packages (`dnnlib/`, `torch_utils/`, `training/`) stay at the root and keep their original
names. EDM's `torch_utils.persistence` pickles fully-qualified module paths into
checkpoints, and the schedule objects in `common_utils` are pickled too, so moving or
renaming them would break loading of existing checkpoints. `pip install -e .` exposes them
under their original names from anywhere, so the relocated entry points in `scripts/`
import them unchanged.

## Method → code

The forward process is `dx_t = (∂_t M_t)^{1/2} dB_t` with a matrix-valued schedule
`M_t(θ) = Σ_j g_j(t;θ) P_j` over orthogonal subspace projectors `P_j` (paper Eq. 2, 29).

| Paper | Component | Where in `common_utils.py` |
|-------|-----------|----------------------------|
| §D.1, §C.1 | Monotone knot-based scalar schedule `g_j(t;θ)` | `GFn`, `GFnInterval`, `GFnSelector`, `Free_GFn` |
| §E.5.1 (Eq. for `Φ_{p,q}`) | 2D-DCT basis and projection | `compute_DCT_basis`, `proj_dct` |
| §E.5.2, §C.2(c) | Class-conditional PCA bases | loaded via `--teacher_basis_path` (`Q`, `class_to_cluster`) |
| §C.1 (Eq. 30) | Subspace-wise matrix functions `f(M_t)` | `mat_mul`, `mat_mul_labelwise` |
| §2 (Eq. 5–6), §5 | EDM preconditioning lifted to the anisotropic net | `ANI_absM_Precond_Wrapper`, `..._basis`, `..._basis_v2` |
| §4.2 (Eq. 18) | Flow parameterization `flow = M_t^{1/2} net` | `ANI_absM_Precond_Flow_Net`, `..._basis`, `..._basis_v3` |
| §3 (Eq. 11/19) | Trajectory-level score-matching loss | `ANILoss_gh_energy`, `ANILoss_gh_energy_condition` |
| §4, Thm 4.1 / Cor 4.2 | Schedule-gradient estimator (flow form) | `flow_matching_energy_debug_basis*` family |
| §4.2 | Variance reduction | `ANILoss_gh_energy_variance_reduce` |
| §5.2 (Eq. 26) | Reverse Euler discretization | `M_euler_ode` |
| §5.3 (Eq. 28) | Matrix-valued Heun solver | `M_EDM_ode`, `M_EDM_ode_mid`, `M_EDM_ode_next` |
| §D.4 | Composition with DPM-Solver-v3 | `M_EDM_ode_v3` |

## Schedule variants → entry scripts

The paper evaluates the families in §C.3. Each canonical entry point in `scripts/train/`
trains one family; the matching sampler is in `scripts/sample/`.

| Paper variant (§C.3) | Train script (`scripts/train/`) | Sampler (`scripts/sample/`) |
|----------------------|--------------------------------|------------------------------|
| (a) Global isotropic `g(t)I` | `g_iso_train.py` | `g_iso_sample_edm.py` |
| (b) Class-conditional isotropic `{g_y}` | `g_iso_train_condition.py` | `g_iso_sample_edm.py` |
| (c) DCT anisotropic `(g_1^{DCT}, g_2^{DCT})` | `g_ani_train.py` | `g_ani_sample_edm.py` |
| (d) PCA anisotropic `(g_1^{PCA}, g_2^{PCA})` | `g_ani_train_basis.py` | `g_ani_sample_basis_edm.py` |
| (e) Class-conditional DCT `{(g_{1,y}, g_{2,y})^{DCT}}` | `g_ani_train_condition.py` | `g_ani_sample_edm.py` |
| (f) Class-conditional PCA `{(g_{1,y}, g_{2,y})^{PCA}}` | `g_ani_train_condition_basis.py` | `g_ani_sample_basis_condition_edm.py` |

Two further training entry points are alternative objectives/paths rather than new
families: `g_iso_train_discretize.py` minimizes the discretization-error objective
(`ANILoss_gh_energy_plus_discretization`), and `g_iso_wrapper_train.py` /
`g_ani_wrapper_train.py` train through the EDM preconditioning wrapper. `g_sample.py` is a
unified sampler that dispatches between the isotropic and anisotropic paths by `--model_tag`.

## Reproducing the paper runs

`scripts/train` and `scripts/sample` are the consolidated, dataset-agnostic entry points
(they branch internally on `--dataset`). The exact per-dataset scripts used for the
reported numbers live under `experiments/cifar10/` and `experiments/imagenet/`; the
script-to-experiment mapping for each is in `docs/cifar10-experiments.md` and
`docs/imagenet-experiments.md`. Prefer the `scripts/` entry points for new work.

## Inherited from NVLabs/EDM

`dnnlib/`, `torch_utils/`, `training/`, and the root scripts `train.py`, `generate.py`,
`fid.py`, `dataset_tool.py` are taken from EDM and used for dataset preparation
(`dataset_tool.py`), the network/preconditioning backbone (`training/networks.py`), and FID
evaluation (`fid.py`).
