# rne-kaist

Surrogate modelling of the transient 3-D temperature field a laser scan induces in
a metal plate (laser powder-bed fusion / directed energy deposition).

A solver generates the ground truth; thirteen models try to reproduce it in a
fraction of the time. They do not agree with each other, and the interesting part
is why.

```
simulation/  ──▶  data/  ──▶  models/                    ──▶  archive/
   solver         datasets     deeponet · coord · spectral      one entry
                     │                                          per run
                     │      typeulli-model-training/
                     │         nine coordinate networks
                     └── config.json: how the dataset was made
```

## Two repositories, and why they stay two

| | Owner | What it holds |
|---|---|---|
| **`rne-kaist`** (this) | [@bsiku3622](https://github.com/bsiku3622) | the solver, the datasets, the archive, and three models |
| **`typeulli-model-training/`** | [@typeulli](https://github.com/typeulli) — [rne-model-training](https://github.com/typeulli/rne-model-training) | nine coordinate networks behind one inference contract |

The second is a **submodule**: this repository records a pointer to a commit over
there, not a copy of its code, so the two halves move independently and neither
can break the other. Clone with:

```powershell
git clone --recurse-submodules https://github.com/bsiku3622/rne-kaist
# or, in an existing clone:
git submodule update --init
```

**They are kept apart on purpose.** Folding `models/spectral` into the submodule's
`models/` was tried and abandoned. Both projects have a `dataset.py` and the two
mean different things — theirs draws batches from the corpus, ours builds a Fourier
basis out of it — and both have an `agent.py`. In one namespace the imports resolve
to whichever file the interpreter reaches first, and `from dataset import
DEFAULT_FIELD_SHAPE` starts finding the wrong module. Two conventions in two
directories cost nothing; two conventions in one directory cost correctness.

What the two *do* share is the data, and that needed no negotiation. The nine
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
| `models/` | Ours. `deeponet` (branch/trunk, with the PDE in the loss), `coord` (the plain regression baseline), `spectral` (Fourier coefficients — one forward pass is a whole volume). |
| `typeulli-model-training/` | **Submodule.** `mlp`, `pimlp`, `gmlp`, `cmlp`, `cgmlp`, `cpimlp`, `pidon`, `gdon`, `gpidon`. |
| `archive/` | One directory per finished run: checkpoint, logs, TensorBoard events, figures, a code snapshot, hard links to the data, and the provenance. Not tracked. |
| `reports/` | Figures that cut *across* runs, so they belong to no single archive entry. |

`models/coord` and `models/deeponet` overlap with the submodule's `mlp` and
`pidon`. They are kept anyway, for one reason: **they hold out a whole power and
never see it**, and every run they make lands in an archive entry carrying its own
config, metrics, code snapshot and provenance. The nine over there do neither. The
same architecture measured two ways is worth having; what is not worth having is
the two sets of numbers in one table (see below).

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

# 2. our models. every run opens its own archive entry and writes into it directly --
#    checkpoint, TensorBoard, figures, code snapshot, provenance -- so there is
#    nothing to clean up afterwards and nothing to collide over
$RUN = "data/torch_20260710-132221_100_700_0.125"
python models/deeponet/train.py --run $RUN
python models/coord/train.py    --run $RUN

# the spectral model needs its basis built first
python models/spectral/dataset.py --run $RUN
python models/spectral/train.py   --run $RUN --derotate

tensorboard --logdir archive       # every run at once

# 3. the submodule's nine, which keep their own conventions
cd typeulli-model-training
python train.py --list
python train.py gpidon --data-dir ../data/torch_20260710-122446_100_250_0.25
python benchmark.py --model gpidon --checkpoint checkpoints/gpidon/best.pt
```

## What the two do not share

Stated plainly rather than papered over, because a reader comparing numbers across
the boundary needs to know:

- **Two validation splits, and this is the one that bites.** The nine networks hold
  out a random 10% of *points*, drawn from the same powers they train on. Our three
  hold out a whole *power* and never see it. **The two sets of numbers are not
  comparable**, and the second task is strictly harder.
- **Two output conventions.** `typeulli-model-training/` writes to `runs/` and
  `checkpoints/`; we open one archive entry per run. Both work; neither knows about
  the other.
- **Two data loaders.** `share/grid.py` (millimetres, `[nt, nx, ny, nz]`) and its
  `dataset.py` (SI, `[nt, nz, ny, nx]`) read the same files in different units and
  a different axis order.

Closing any of these means changing code in a repository this one does not own.

## History

This repository merges two previously separate repositories, with their commit
history preserved:

- `rne-am-simulation` → `simulation/`  ([archived](https://github.com/bsiku3622/rne-am-simulation))
- `rne-am-pi-deeponet` → `models/deeponet/`  ([archived](https://github.com/bsiku3622/rne-am-pi-deeponet))

Earlier stages (a transfer-learning PINN reproducing Peng et al., JMP 138 (2025)
140–156, and assorted exploratory solvers) live under `../olds/`.
