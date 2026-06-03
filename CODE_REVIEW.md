# Code Review

A prioritized review of the repository. Items marked **[fixed]** were already applied in
this revision; everything else is a recommendation that needs an author decision because it
depends on experiment-specific intent.

---

## P0 — Anonymity (resolved in this revision)

The repository contained identifying information that would break a double-blind
submission. All of it has been scrubbed:

| Leak | Where | Action |
|------|-------|--------|
| author-name tags in inline comments | `common_utils.py` (10×) | name removed, comment text kept |
| `## note to <name>: ...` | 3 sampling scripts | rewritten to `## note: ...` |
| an absolute cluster path under `--teacher_basis_path` | 7 defaults | replaced with `pretrained/...` |
| a second hardcoded absolute data path | 1 default | replaced with `pretrained/...` |

That single cluster path leaked the lab name, a user/NetID, the cluster (the path layout
is institution-specific), and even the target venue (it was embedded in the directory
name). **Before pushing the anonymous repo, double-check these remain scrubbed in git
history too** — a `git filter-repo`/fresh-history commit is safest, since deleting a line
in a new commit still leaves it in history.

`dnnlib/util.py` uses `os.environ['USERPROFILE']` — this is upstream EDM code reading an
env var at runtime, not a hardcoded identity, and was left untouched.

---

## P1 — Correctness

### `scripts/train/g_ani_train_condition_basis.py` — stray `)` breaks the whole file **[fixed]**
The script ended with an extra closing parenthesis after `train(args)`, so the file failed
to even parse (`SyntaxError: unmatched ')'`) — it could never have run as-is. The stray
`)` has been removed. This was pre-existing, not introduced by the anonymization pass. All
other root and subfolder scripts compile cleanly.

### `scripts/sample/g_sample.py :: load_model` — dead code + redundant load **[fixed]**
Both branches computed a `ckpt_net_path` that was never used, then re-opened the *same*
`ckpt_path` a second time to read `ema`. Since the schedules (`g`, `h`) and the network
(`ema`) live in one checkpoint, this now reads `ema = ckpt["ema"]` directly. Behavior is
identical; the redundant I/O and the misleading unused variable are gone.

### `scripts/sample/g_sample.py :: load_model` — fragile model-type detection **[needs decision]**
```python
if "ani" in model_tag or "gh" in model_tag or "h" in model_tag:
```
The `"h" in model_tag` test is a bare substring match. It happens not to misfire on the
documented tags (`g-iso`, `g-ani`, `g-iso-wrapper`, `g-ani-wrapper`, `g-iso-discretize`),
but it will silently take the anisotropic branch for any future tag containing the letter
`h` (e.g. a dataset or experiment name). Recommend an explicit flag instead of substring
sniffing, e.g. pass `--anisotropic/--isotropic` or detect by checking
`"h" in ckpt` (does the checkpoint actually contain an `h` schedule?) rather than parsing
the tag string.

---

## P2 — Structure & maintainability

### Repository reorganization **[done]**
The repository was reorganized into a paper-aligned layout: consolidated entry points under
`scripts/train` and `scripts/sample`, the original per-dataset scripts under
`experiments/cifar10` and `experiments/imagenet`, launchers under `scripts/launch`, and
experiment maps under `docs/`. A `pyproject.toml` was added so `pip install -e .` exposes
`common_utils`, `dnnlib`, `torch_utils`, and `training` under their original names from any
working directory — these are intentionally kept at the repository root because EDM's
`torch_utils.persistence` and the pickled schedule objects encode module paths into
checkpoints. `ARCHITECTURE.md` maps every component to its paper section. The `scripts/`
entry points are the consolidated versions of the per-dataset scripts (e.g.
`scripts/train/g_iso_train.py` = `experiments/cifar10/g_iso_train_rampup_ema.py` plus
`if dataset == 'imagenet'` branches; ~12-line diff) and cover all paper variants.

Remaining recommendation: treat `experiments/` as archival and avoid editing both a
`scripts/` entry point and its `experiments/` ancestor in tandem — that double-maintenance
is the main residual risk.

### `common_utils.py` is a ~5,300-line monolith
It mixes schedule modules (`GFn`/`GFnInterval`/...), basis construction (DCT/PCA), the
diffusion operators (`mat_mul`), the EDM ODE/precond wrappers, the matrix-Heun solvers
(`M_EDM_ode*`), and the loss/estimator functions. It also still contains many dead
experiment variants — e.g. the `ANILoss_gh_energy_plus_discretization_todo{1..4}*` family,
several `_debug`/`_v2`/`_v3` duplicates. Splitting the live API into `schedules.py`,
`bases.py`, `operators.py`, `solvers.py`, `losses.py` (re-exported from `common_utils.py`
for checkpoint/back-compat) and pruning the dead variants would make the core method far
easier to review. This was **not** done here because it needs the ability to run the code
to confirm nothing live is removed.

### `debug` in a canonical filename
`experiments/cifar10/g_iso_train_rampup_ema_condition_debug.py` and
`experiments/imagenet/..._debug32.py` are referenced as the *real* iso-conditional training
scripts in the experiment maps, yet carry `debug` in the name. Rename for a public release
(e.g. drop `_debug`) so it doesn't read as throwaway code.

---

## P3 — Polish (low risk, optional)

- **Comment typos:** `schore` → `score`, `scoled` → `scaled` (a few occurrences in
  `common_utils.py` comments). Cosmetic only.
- **f-strings with no placeholders:** ~7 `print(f"...")` calls have no `{}` fields; drop
  the `f` prefix. Cosmetic.
- **Magic numbers:** `T=6400.0` and the per-dataset `(num_V_dim, res)` table
  (`cifar10→256/32`, `afhqv2|ffhq→1024/64`) are hardcoded in multiple scripts. Consider a
  small `configs/` or a shared dict in `common_utils.py` so they can't drift between
  training and sampling.
- **`requirements.txt`:** the pinned `torch==1.12.1+cu116` / `torchvision==0.13.1+cu116`
  wheels need the PyTorch CUDA index. The README now documents the
  `--extra-index-url https://download.pytorch.org/whl/cu116` workaround.
- **`pretrained/` path:** the anonymized `--teacher_basis_path` defaults now point to
  `pretrained/clustered_pca_via_classPCA128_K60_r1024_bestseed1.pt`. Document how a user
  obtains/builds that file, or ship a small script to regenerate it.

---

## Summary

| Priority | Item | State |
|----------|------|-------|
| P0 | De-anonymize source (names, lab, cluster path, venue) | **fixed** (verify git history) |
| P1 | `scripts/train/g_ani_train_condition_basis.py` stray `)` (file would not parse) | **fixed** |
| P1 | `load_model` dead code / redundant load | **fixed** |
| P1 | `load_model` fragile `"h" in model_tag` detection | needs decision |
| P2 | Canonical vs. archived script split | documented; consider moving/removing |
| P2 | `common_utils.py` monolith | suggested split |
| P2 | `debug` in canonical filenames | rename suggested |
| P3 | typos / f-strings / magic numbers / deps / data paths | optional |
