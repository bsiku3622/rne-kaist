# rne-kaist

Surrogate modelling of the transient 3-D temperature field a laser scan induces in
a metal plate (laser powder-bed fusion / directed energy deposition).

A solver generates the ground truth; ten models try to reproduce it in a fraction
of the time. They do not agree with each other, and the interesting part is why.

```
simulation/  ──▶  data/  ──▶  model-training/  ──▶  archive/
   solver         datasets      nine networks        one entry
                     │          models/spectral      per run
                     │              the tenth
                     └── config.json: how it was made
```

## Two repositories, and why they stay two

The networks live in a repository of their own, mounted here as a submodule:

| | Owner | What it holds |
|---|---|---|
| **`rne-kaist`** (this) | [@bsiku3622](https://github.com/bsiku3622) | the solver, the datasets, the archive, and the spectral model |
| **`model-training/`** | [@typeulli](https://github.com/typeulli) — [rne-model-training](https://github.com/typeulli/rne-model-training) | nine coordinate networks behind one inference contract |

A submodule is a pointer to a commit over there, not a copy of the code, so the
two halves move independently. Clone with:

```powershell
git clone --recurse-submodules https://github.com/bsiku3622/rne-kaist
# or, in an existing clone:
git submodule update --init
```

**They are kept apart on purpose.** Folding `models/spectral` into the
submodule's `models/` was tried and abandoned: both projects have a `dataset.py`
and the two mean different things — theirs draws batches from the corpus, ours
builds a Fourier basis out of it — and both have an `agent.py`. Merged into one
namespace, the imports resolve to whichever file the interpreter reaches first,
which is a bug that hides until it does not. Two conventions in two directories
cost nothing; two conventions in one directory cost correctness.

What they *do* share is the data, and that needed no negotiation at all. The nine
models train against our runs on `--data-dir` alone, and their
`DEFAULT_FIELD_SHAPE` of `(25, 41, 161)` is
`data/torch_20260710-122446_100_250_0.25` exactly — the same corpus, reached by a
different path.

## Layout

| Path | What it is |
|---|---|
| `simulation/` | The solver. Two independent implementations of the same problem — FEM (`heat_fenics.py`, legacy FEniCS 2019, CPU) and a 7-point finite-difference scheme (`heat_torch.py`, PyTorch, CUDA) — cross-validated against each other to 0.14% RMSE. |
| `data/` | Solver output, one directory per run. Not tracked: a seven-power sweep takes ~20 s on a GPU, and each run carries the `config.json` that reproduces it. |
| `share/` | The grid, the metrics, the figures, the archiver. A library, not an interface. |
| `models/spectral/` | The one model that is not a network over coordinates — it learns the field's spatial **Fourier coefficients** instead, so one forward pass is a whole volume. |
| `model-training/` | **Submodule.** Nine networks: `mlp`, `pimlp`, `gmlp`, `cmlp`, `cgmlp`, `cpimlp`, `pidon`, `gdon`, `gpidon`. |
| `archive/` | One directory per finished run: checkpoint, logs, TensorBoard events, figures, a code snapshot, hard links to the data, and the provenance. Not tracked. |
| `reports/` | Figures that cut *across* runs, so they belong to no single archive entry. |

## Naming

Datasets and the trainings fitted to them are named the same way, so a directory
listing reads without a decoder ring:

```
data/     <solver>_<stamp>_<minP>_<maxP>_<spacing>[-<tag>]
archive/  <model>_<stamp>_<minP>_<maxP>_<spacing>[-<tag>]

data/torch_20260710-132221_100_700_0.125
archive/spectral_20260713-174928_100_700_0.125-derotated-detrended
```

## Running it

```powershell
# 1. make a dataset  (writes data/torch_<stamp>_100_700_0.125/ and its config.json)
python simulation/heat_torch.py --powers 100 200 300 400 500 600 700 `
    --ele_size 0.125 --dt 6.25e-4

# 2a. the networks, from the submodule
cd model-training
python train.py --list
python train.py gpidon --data-dir ../data/torch_20260710-132221_100_700_0.125
python benchmark.py --model gpidon --checkpoint checkpoints/gpidon/best.pt

# 2b. the spectral model, which needs its basis built first
cd ..
python models/spectral/dataset.py --run data/torch_20260710-132221_100_700_0.125
python models/spectral/train.py   --run data/torch_20260710-132221_100_700_0.125 --derotate
```

Every `models/spectral` run opens its own archive entry and writes into it
directly — checkpoint, TensorBoard, figures, code snapshot, provenance — so there
is nothing to clean up afterwards and nothing to collide over. `tensorboard
--logdir archive` sees every run.

## What the two do not share

Stated plainly rather than papered over, because a reader comparing numbers across
the boundary needs to know:

- **Two output conventions.** `model-training/` writes to `runs/` and
  `checkpoints/`; we open one archive entry per run. Both work; neither knows about
  the other.
- **Two data loaders.** `share/grid.py` (millimetres, `[nt, nx, ny, nz]`) and its
  `dataset.py` (SI, `[nt, nz, ny, nx]`) read the same files in different units and
  a different axis order.
- **Two validation splits, and this is the one that bites.** The networks hold out
  a random 10% of *points*, drawn from powers they also train on. The spectral model
  holds out a whole *power* and never sees it. **The two sets of numbers are not
  comparable**, and the gap between the tasks is large — the second is strictly
  harder.

Closing any of these means changing code in a repository this one does not own.

## History

This repository merges two previously separate repositories, with their commit
history preserved:

- `rne-am-simulation` → `simulation/`  ([archived](https://github.com/bsiku3622/rne-am-simulation))
- `rne-am-pi-deeponet` → superseded by `model-training/`  ([archived](https://github.com/bsiku3622/rne-am-pi-deeponet))

Earlier stages (a transfer-learning PINN reproducing Peng et al., JMP 138 (2025)
140–156, and assorted exploratory solvers) live under `../olds/`.
