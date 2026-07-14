# rne-kaist

Surrogate modelling of the transient 3-D temperature field a laser scan induces in a metal
plate (laser powder-bed fusion / directed energy deposition).

A solver generates the ground truth; twelve models try to reproduce it in a fraction of the
time. They disagree with each other, and the interesting part is why.

```
simulation/  ──▶  data/  ──▶  models/  ──▶  archive/  ──▶  reports/
   solver         datasets     twelve       one entry      what it
                     │                      per run        all meant
                     └── config.json: how the dataset was made
```

## Running it

```powershell
# a dataset (writes data/torch_<stamp>_100_700_0.125/ and its config.json)
python simulation/heat_torch.py --powers 100 200 300 400 500 600 700 --ele_size 0.125 --dt 6.25e-4

# a model. every run opens its own archive entry and writes into it directly
python train.py --list
python train.py gpidon --run data/torch_20260710-122446_100_250_0.25 --holdout 175
python train.py mlp    --run data/... --optimizer lbfgs --lr 1
python train.py gdon   --help

tensorboard --logdir runs        # what is training now
tensorboard --logdir archive     # everything ever trained
```

`models/spectral` needs its Fourier basis built first:

```powershell
python models/spectral/dataset.py --run data/... [--detrend]
python models/spectral/train.py   --run data/... --derotate
```

## Layout

| Path | What it is |
|---|---|
| `simulation/` | The solver. FEM (`heat_fenics.py`, legacy FEniCS, CPU) and a 7-point finite-difference scheme (`heat_torch.py`, PyTorch, CUDA), cross-validated against each other to 0.14% RMSE. |
| `data/` | Solver output, one directory per run, each carrying the `config.json` that reproduces it. **Not tracked** — a seven-power sweep is ~20 s on a GPU. |
| `share/` | Everything a model does that is not its own architecture. |
| `models/` | One directory per model. Twelve of them. |
| `archive/` | One directory per finished run: checkpoint, TensorBoard events, figures, a code snapshot, hard links to the data, and the provenance. **Not tracked.** |
| `runs/` | Junctions into the archive entries currently training — what TensorBoard watches. Costs nothing; delete freely. **Not tracked.** |
| `reports/` | Figures that answer a question rather than describe a run. **Tracked.** See `reports/README.md`. |
| `typeulli-model-training/` | **Submodule.** [@typeulli](https://github.com/typeulli)'s repository, kept as the channel their code arrives through. Nine of the twelve models were ported out of it. |

## The models

| | Learns | Loss |
|---|---|---|
| `mlp` | `(P, t, z, y, x) -> T` | data |
| `gmlp` | the same, handed the laser's gaussian as a gate | data |
| `pimlp` | the same as `mlp` | data + PDE + BC + IC |
| `pidon` | `(P; x, y, z, t) -> T`, branch over the power, trunk over space-time | data + PDE + BC + IC |
| `gdon` | a DeepONet with the gaussian gate | data |
| `gpidon` | both | data + PDE + BC + IC |
| `cmlp` `cgmlp` `cpimlp` | **controls**: the same networks with the laser power removed | — |
| `coord` | `(P, t, z, y, x) -> T` | data |
| `deeponet` | branch/trunk, with the gaussian trunk feature | data + PDE + BC + IC |
| `spectral` | `(P, t) -> the field's spatial Fourier coefficients` | data |

The **controls** are the floor. The same `(t, z, y, x)` carries a different temperature at
every power, so a network that cannot see `P` can do no better than the mean over the
sweep. Any model that fails to beat them has learned nothing about the laser. (One did.)

`spectral` is the only model that is not a network over coordinates. It predicts the
coefficients of the whole field at once, so a forward pass returns a volume rather than a
point — see `models/spectral/model.py`.

## share/

Not an interface. A library, and the place a thing is defined once instead of twelve times.

| | |
|---|---|
| `corpus.py` | The point cloud, and `split_by_power` — the split every generalisation number here comes from. |
| `grid.py` | The solver's output, back on the grid it was written from. |
| `agent.py` | **The inference contract.** `predict_at([B,5] of (x,y,z,t,P)) -> [B,1] K`, `predict_of([B,2] of (t,P)) -> [B,1,D,H,W] K`. Every tool downstream is written against those two and nothing else, so adding a model costs an `agent.py` and changes no caller. Either end may be the primitive: most models implement `predict_at` and get the volume derived; `spectral` runs the other way. |
| `harness.py` | What a model does around its architecture: read the corpus, hold out a power, build, score, render, archive. |
| `training.py` | The loop. Adam or L-BFGS, the schedule, the validation cadence, the checkpoint. |
| `metrics.py` | RMSE, L∞, **and peak error**. |
| `plotting.py` | The figures, and the colour rules they obey. |
| `archiving.py` | An entry per run, and the hard links that make it cost nothing. |

## Three things this repository does differently, and why

**Validation is a held-out power.** Not a random tenth of the *points* drawn from powers the
model also trains on. Filling in gaps between points it has already seen is not the question
a surrogate exists to answer, and the two tasks are not close: the same network scores 0.83 K
on one and 1.45 K on the other.

**The checkpoint is not chosen on RMSE.** RMSE is a volume average and the melt pool is a
fraction of a percent of the plate, so the two come apart: a model sat at 14 K RMSE while
flattening the peak by 44%, and by the metric it was selected with, it looked like the best
one. Selection is on `rmse + |peak|` (`share/training.select`). The weighting is a judgement,
made in the open — which beats a judgement made by omission.

**A run writes into its own archive entry.** Checkpoint, events, figures, a snapshot of the
code that produced them, hard links to the data, `config.json`, `metrics.json`, `env.json`,
`git.txt`. Two models can train at once, nothing has to be swept up afterwards, and a number
in a report can be walked back to the run, the code and the data that made it. The data is
*linked*, not copied: copying cost 13 GB a run and six entries had accumulated 78 GB of
byte-identical duplicates of a dataset the solver regenerates in 20 s.

## The submodule

`typeulli-model-training/` is [@typeulli](https://github.com/typeulli)'s
[repository](https://github.com/typeulli/rne-model-training), mounted as a submodule. Its
nine models now live under `models/` in this repository's conventions — their `model.py`,
`loss.py`, `dataset.py` and `agent.py` came across **untouched**, because the physics in
them is calibrated in SI and a unit slip inside a PDE residual trains happily and is wrong.
Only the harness around them changed.

The submodule stays as the channel their code arrives through:

```powershell
git submodule update --remote typeulli-model-training
```

Clone this repository with `--recurse-submodules`, or the directory arrives empty.

## History

Two repositories were merged here, with their commit history preserved:

- `rne-am-simulation` → `simulation/` ([archived](https://github.com/bsiku3622/rne-am-simulation))
- `rne-am-pi-deeponet` → `models/deeponet/` ([archived](https://github.com/bsiku3622/rne-am-pi-deeponet))

Earlier stages (a transfer-learning PINN reproducing Peng et al., JMP 138 (2025) 140–156,
and assorted exploratory solvers) live under `../olds/`.
