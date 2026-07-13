# rne-kaist

Surrogate modelling of the transient 3D temperature field induced by a laser
scan on a metal plate (laser powder bed fusion / directed energy deposition).

The repository holds the full pipeline: a thermal solver that generates the
ground-truth data, and a physics-informed neural operator that learns to
predict the same field in a fraction of the time.

```
simulation/  ──  data_<P>W.npy  ──▶  model-training/
   solver           dataset            PI-DeepONet
```

## Layout

| Path | What it is |
|---|---|
| `simulation/` | Transient heat-conduction solver. Two independent implementations — FEM (`heat_fenics.py`, legacy FEniCS 2019, CPU) and a 7-point finite-difference scheme (`heat_torch.py`, PyTorch, CUDA) — cross-validated against each other. See `simulation/README.md`. |
| `simulation/data/` | Solver output, one directory per run (`YYYYMMDD_HHMMSS_<tag>`). Not tracked; regenerate with `simulation/src/heat_torch.py`. |
| `simulation/logs/` | Run logs. The only record of which parameters produced which run. |
| `model-training/` | Physics-informed DeepONet. Branch net takes the laser power `P`, trunk net takes `(x, y, z, t)`; the loss combines data, PDE residual, boundary and initial conditions. See `model-training/README.md`. |
| `model-training/archive/` | One directory per training run: checkpoint, logs, TensorBoard events, figures, and a snapshot of the code that produced them. Delete-protected by ACL (see `model-training/SERVER.md`). |
| `model-training/v2/` | Comparison figures for four architectures (mlp, pidon, gdon, gpidon). |

## Data flow

`simulation/src/heat_torch.py` writes `data_<P>W.npy` into a timestamped run
directory. Each file is an `[N, 6]` array of rows `(x, y, z, t, P, T)`, where
`P` is constant within a file and serves as the branch input.

`model-training/train.py` consumes a directory of such files via `--data-dir`.
The production dataset is `simulation/data/20260710_132221_powersweep_gpu`
(100–700 W, 0.125 mm grid).

```powershell
python model-training/train.py `
  --data-dir ../simulation/data/20260710_132221_powersweep_gpu
```

## History

This repository merges two previously separate repositories, with their commit
history preserved:

- `rne-am-simulation` → `simulation/`
- `rne-am-pi-deeponet` → `model-training/`

Earlier stages of the project (a transfer-learning PINN reproducing Peng et
al., JMP 138 (2025) 140–156, and assorted exploratory solvers) live under
`../archive/`.
