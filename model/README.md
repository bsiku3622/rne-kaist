# pi-deeponet

A physics-informed DeepONet that predicts the transient 3-D temperature field of a
laser scan over a metal substrate. The branch network takes the laser power `P`,
the trunk network takes the space-time query point `(x, y, z, t)`, and their
latent codes are combined by an inner product to give `T(x, y, z, t; P)`.

Training minimises a weighted sum of four residuals:

| Term | Residual | Scale |
|---|---|---|
| Data | `T_hat - T` against the simulation grids | `dT` [K] |
| PDE | `rho*c_p*dT/dt - k*laplacian(T)` | `rho*c_p*dT/t` [W m^-3] |
| BC | bottom (Dirichlet at `T_amb`), top (Gaussian laser flux, convection, radiation), lateral faces (convection, radiation) | `dT` [K] / peak flux [W m^-2] |
| IC | `T_hat - T_amb` at `t = 0` | `dT` [K] |

Each residual is divided by its characteristic magnitude before being summed, so
the weights in `LossWeights` express relative importance rather than unit
conversion. Everything inside the code is SI (metres, seconds, Kelvin); the raw
`.npy` grids store millimetres and are converted on load.

## Layout

| File | Purpose |
|---|---|
| `model.py` | `DeepONet` and `DeepONeuralNet`, plus autograd derivatives of the field |
| `loss.py` | `ThermalProperties`, `ResidualScales`, `LossWeights`, `PINNLoss` |
| `train.py` | Training loop, point sampling, TensorBoard logging, checkpointing |
| `calibrate.py` | Fits the PDE and laser constants from the data; prints a block to paste into `train.py` |
| `visualize.py` | Truth / prediction / error panels on the `top` or `track` slice, or a `scanline` profile through the melt pool peak |

`data/` holds `data_<power>W.npy`, each an `[N, 6]` array of `(x, y, z, t, P, T)`
rows on a structured grid — `81 x 21 x 13 x 31 = 685,503` rows per file at a
0.5 mm spacing over a 40 x 10 x 6 mm block, 0 to 3 s. Power is constant within
a file, so the branch network only has something to learn once several powers are
present. All `.npy` files under `--data-dir` are globbed and concatenated
automatically; the seven shipped powers (100 W to 250 W) come to 4.8M points, so
the whole dataset loads onto the GPU at once. Regenerate them from the `simulation`
repo — they are git-ignored, as are `runs/`, `figures/`, `archive/` and the
checkpoints.

## Setup

Python 3.10+ with PyTorch, NumPy, SciPy, Matplotlib, tqdm and TensorBoard.

```powershell
pip install torch numpy scipy matplotlib tqdm tensorboard
```

## Usage

Fit the constants first, whenever the contents of `data/` change:

```powershell
python calibrate.py --data-dir data
```

It prints an `ABSORPTIVITY` / `BEAM_RADIUS` / `PROPERTIES` block; paste it over
the corresponding block at the top of `train.py`. Only the diffusivity ratio
`alpha = k / (rho * c_p)` and the laser parameters are identifiable — `rho`,
`c_p`, `h` and `epsilon` stay as inputs.

Then train:

```powershell
python train.py --iterations 20000 --lr 1e-3 --logdir runs
```

The best-validation-RMSE checkpoint is written to `checkpoint.pt` and carries the
architecture, properties, scales and weights, so the plotting scripts rebuild the
network without re-reading the dataset. Loss weights are `--w-data`, `--w-pde`,
`--w-bc`, `--w-ic`; `--double` runs in float64.

Then look at the result:

```powershell
python visualize.py --power 200 --plane top
python visualize.py --power 200 --plane track --times 0.5 1.5 2.5
python visualize.py --power 200 --plane scanline --gaussian
```

`--plane top` shows the `z = z_max` surface the laser scans; `--plane track` cuts
along the scan line at `y = y_c`, which is where the melt pool depth lives and
where the model is hardest to fit; `--plane scanline` pulls a single line out of
that surface along the scan track and, with `--gaussian`, overlays a `1/e^2` fit
so the melt pool's peak height and width can be read off directly.

## Monitoring server

Two long-running services are exposed over the VPN interface
(`baeks-server-only-intranet`, `10.8.0.0/24`): TensorBoard for the training
curves at `http://10.8.0.2:6006`, and a file server for browsing the project
folder at `http://10.8.0.2:8080`.

```powershell
$root = "D:\School\KSA\RnE\rne-kaist\model"
Set-Location $root

$tb = Start-Process -FilePath "python" `
  -ArgumentList "-m","tensorboard.main","--logdir","runs","--host","0.0.0.0","--port","6006" `
  -RedirectStandardOutput "$root\tensorboard.log" -RedirectStandardError "$root\tensorboard.err" `
  -WindowStyle Hidden -PassThru

$fs = Start-Process -FilePath "python" `
  -ArgumentList "-u","-m","http.server","8080","--bind","10.8.0.2","--directory","$root" `
  -RedirectStandardOutput "$root\fileserver.log" -RedirectStandardError "$root\fileserver.err" `
  -WindowStyle Hidden -PassThru

"$($tb.Id)" | Out-File -Encoding utf8 "$root\.tensorboard.pid"
"$($fs.Id)" | Out-File -Encoding utf8 "$root\.fileserver.pid"
```

`Start-Process` detaches the services from the shell — backgrounding them with
`&` kills them when the session ends. Shut them down with:

```powershell
Stop-Process -Id (Get-Content "$root\.tensorboard.pid").Trim() -Force
Stop-Process -Id (Get-Content "$root\.fileserver.pid").Trim() -Force
```

TensorBoard binds `0.0.0.0` because its default is localhost-only. The file
server binds `10.8.0.2` on purpose: `0.0.0.0` would expose the whole folder to
the public IP and the home LAN. Do not delete an empty `runs/` — it is the
`--logdir` target and TensorBoard will not start without it.

See [SERVER.md](SERVER.md) for the health checks, the nginx
reverse-proxy config behind `desktop.baeksikoo.com`, the ACL on `archive/`, and
the known constraints of the single-threaded `http.server`.
