# rne-kaist

Surrogate modelling of the transient 3-D temperature field a laser scan induces in
a metal plate (laser powder-bed fusion / directed energy deposition).

A solver generates the ground truth; several models try to reproduce it in a
fraction of the time. The models do not agree with each other, and the interesting
part is why.

```
simulation/  ──▶  data/  ──▶  models/  ──▶  archive/
   solver         datasets      four         one entry
                               attempts      per run
```

## Layout

| Path | What it is |
|---|---|
| `simulation/` | The solver. Two independent implementations of the same problem — FEM (`heat_fenics.py`, legacy FEniCS 2019, CPU) and a 7-point finite-difference scheme (`heat_torch.py`, PyTorch, CUDA) — cross-validated against each other to 0.14% RMSE. See `simulation/README.md`. |
| `data/` | Solver output, one directory per run. Not tracked: a seven-power sweep takes ~20 s on a GPU, and each run carries the `config.json` that reproduces it. |
| `share/` | What every model needs and nothing that binds them together — the grid, the metrics, the figures, the archiver. A library, not an interface: **no model imports another model.** |
| `models/` | One directory per model. Each owns its `train.py` and its `visualize.py`, and shares nothing with its neighbours except `share/`. |
| `archive/` | One directory per finished run: checkpoint, logs, TensorBoard events, figures, a code snapshot, hard links to the data, and the provenance. Not tracked. |
| `reports/` | Figures that cut *across* runs, so they belong to no single archive entry. |

## Naming

Datasets and the trainings fitted to them are named the same way, so a directory
listing reads without a decoder ring:

```
data/     <solver>_<stamp>_<minP>_<maxP>_<spacing>[-<tag>]
archive/  <model>_<stamp>_<minP>_<maxP>_<spacing>[-<tag>]

data/torch_20260713-151246_100_700_0.125
archive/spectral_20260713-152030_100_700_0.125-derotated
```

## The models

All four take the same data and are scored by the same three numbers
(`share/metrics.py`), but they do not solve the same problem, and two of them are
not comparable to the others without care.

| Model | Learns | Params | Validation |
|---|---|---|---|
| `deeponet` | `(P; x,y,z,t) -> T`, branch/trunk, with PDE + BC + IC residuals in the loss | 133k | a random 10% of *points* |
| `deeponet --architecture mlp` | the same, as one MLP over the five raw inputs, at matched parameter count | 133k | a random 10% of *points* |
| `spectral` | `(P, t) -> every spatial Fourier coefficient` | 9.5M | a **held-out power** |
| `coord` | `(P, t, z, y, x) -> T`, pure regression, no physics term | 34k | a **held-out power** |

**The validation splits differ, and the difference matters.** `deeponet` scores itself
on random points drawn from the same powers it trained on; `spectral` and `coord`
hold a whole power out and never see it. Those numbers do not belong in the same
table. `deeponet` is also the only one whose loss contains the PDE — the other two
are fitting data and nothing else.

## Running it

```powershell
# 1. make a dataset  (writes data/torch_<stamp>_100_700_0.125/ and its config.json)
python simulation/heat_torch.py --powers 100 200 300 400 500 600 700 `
    --ele_size 0.125 --dt 6.25e-4

# 2. train.  every run opens its own archive entry and writes into it directly --
#    checkpoint, TensorBoard, figures, code snapshot, provenance -- so there is
#    nothing to clean up afterwards and nothing to collide over
python models/deeponet/train.py --run data/torch_20260713-151246_100_700_0.125
python models/coord/train.py    --run data/torch_20260713-151246_100_700_0.125

# the spectral model needs its basis built first
python models/spectral/dataset.py --run data/torch_20260713-151246_100_700_0.125
python models/spectral/train.py   --run data/torch_20260713-151246_100_700_0.125 --derotate

# 3. every run at once
tensorboard --logdir archive
```

Re-render an old run's figures without retraining:

```powershell
python models/spectral/visualize.py --entry archive/spectral_20260713-152030_100_700_0.125
```

## What an archive entry holds

```
archive/spectral_20260713-152030_100_700_0.125-derotated/
├── checkpoint.pt
├── tensorboard/            event files; --logdir archive sees every run
├── figures/                whatever that model's visualize.py renders
├── code/                   models/<model>/ and share/, as they were
├── data/                   hard links into data/<run>/, plus a manifest
├── config.json             every argument, resolved
├── metrics.json            the scores, so runs compare without loading a checkpoint
├── env.json                python, torch, CUDA, the GPU
└── git.txt                 the commit, and whether the tree was dirty
```

**The data is hard-linked, not copied.** Copying it cost 13 GB per run, and six
entries had quietly accumulated 78 GB of byte-identical duplicates of a dataset the
solver regenerates in 20 s. A hard link is a second name for the same file record: it
costs nothing, it opens exactly like the original, and deleting the original leaves
the archived name intact.

## History

This repository merges two previously separate repositories, with their commit
history preserved:

- `rne-am-simulation` → `simulation/`
- `rne-am-pi-deeponet` → `models/deeponet/`

Earlier stages (a transfer-learning PINN reproducing Peng et al., JMP 138 (2025)
140–156, and assorted exploratory solvers) live under `../archive/`.
