"""Fit (P, t, z, y, x) -> T with the same 128x3 MLP, and compare it to the spectral one.

This is the other way to spend the same three hidden layers. ``spectral_nn.py`` hands
the network a 2-dimensional input and asks it for 73,402 numbers at once, because the
Fourier basis is already carrying the space; here the network takes all five
coordinates and returns a single temperature, so it has to carry the space itself.

The trade is stark, and worth seeing before the numbers arrive:

    spectral   2 -> 128 -> 128 -> 128 -> 73402     9.5M params, 217 samples
    coordinate 5 -> 128 -> 128 -> 128 ->     1      34k params, 276M samples

99.6% of the spectral model's weights sit in its output layer; the coordinate model
has almost none, and instead sees every one of the 1.27M grid points in every
snapshot. Full-batch is impossible, so points are drawn at random each step.

Everything else is held fixed so the comparison means something: the same held-out
power, the same optimiser and schedule, the same figures. The network is asked for
dT = T - T_amb internally, scaled to O(1) -- predicting T directly would waste its
range on the 298 K offset that every point shares -- and T is recovered by adding the
ambient back.

Example::

    python coord_nn.py --raw ../simulation/data/20260710_132221_powersweep_gpu \
        --data data-spectral/fft2_powersweep.npz
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from spectral_nn import plot_planes, plot_scanline, plot_signal, score

T_AMB = 298.0


class CoordMLP(nn.Module):
    """(P, t, z, y, x) -> dT, scaled. Same shape of hidden stack as the spectral net."""

    def __init__(self, width: int = 128, depth: int = 3):
        super().__init__()
        layers, d = [], 5
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_grid(raw: Path):
    """Every power's dT on the grid, plus the axes it lives on."""
    files = sorted(
        raw.glob("data_*W.npy"),
        key=lambda p: int(re.search(r"data_(\d+)W", p.name).group(1)),
    )
    powers = np.array([int(re.search(r"data_(\d+)W", f.name).group(1)) for f in files])
    rows = np.load(files[0], mmap_mode="r")
    x, y, z, t = (np.unique(np.asarray(rows[:, c])) for c in range(4))
    shape = (len(t), len(x), len(y), len(z))

    dT = np.empty((len(files), *shape), dtype=np.float32)
    for i, f in enumerate(files):
        r = np.load(f, mmap_mode="r")
        dT[i] = (np.asarray(r[:, 5]) - T_AMB).reshape(shape)
    return dT, powers, x, y, z, t


def predict_grid(model, dev, powers, x, y, z, t, scale, chunk=1 << 21) -> np.ndarray:
    """The model's field on the full grid, one power at a time."""
    nt, nx, ny, nz = len(t), len(x), len(y), len(z)
    axes = [torch.tensor(v, dtype=torch.float32, device=dev) for v in (z, y, x)]
    zn, yn, xn = (v / v.max() for v in axes)
    tn = torch.tensor(t / t.max(), dtype=torch.float32, device=dev)

    out = np.empty((len(powers), nt, nx, ny, nz), dtype=np.float32)
    n = nx * ny * nz
    with torch.no_grad():
        for i, P in enumerate(powers):
            pn = float(P) / float(powers.max())
            for j in range(nt):
                for s in range(0, n, chunk):
                    idx = torch.arange(s, min(s + chunk, n), device=dev)
                    ix = idx // (ny * nz)
                    iy = (idx // nz) % ny
                    iz = idx % nz
                    q = torch.stack(
                        [
                            torch.full_like(zn[iz], pn),
                            torch.full_like(zn[iz], float(tn[j])),
                            zn[iz],
                            yn[iy],
                            xn[ix],
                        ],
                        -1,
                    )
                    out[i, j].reshape(-1)[s : s + idx.numel()] = (
                        (model(q) * scale).cpu().numpy()
                    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True, help="solver run dir")
    ap.add_argument("--data", type=Path, required=True, help="npz from spectral.py, for the signal plot")
    ap.add_argument("--holdout", type=float, default=400.0)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=1 << 16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--figdir", type=Path, default=Path("figures/spectral"))
    ap.add_argument("--times", type=float, nargs="+", default=[0.5, 1.5, 3.0])
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dT, powers, x, y, z, t = load_grid(a.raw)
    nP, nt, nx, ny, nz = dT.shape
    test_p = int(np.argmin(np.abs(powers - a.holdout)))
    train_p = [i for i in range(nP) if i != test_p]
    scale = float(dT[train_p].max())

    # the whole training set lives on the GPU; points are drawn from it at random
    Y = torch.tensor(dT[train_p], device=dev).reshape(len(train_p), -1) / scale
    zn = torch.tensor(z / z.max(), dtype=torch.float32, device=dev)
    yn = torch.tensor(y / y.max(), dtype=torch.float32, device=dev)
    xn = torch.tensor(x / x.max(), dtype=torch.float32, device=dev)
    tn = torch.tensor(t / t.max(), dtype=torch.float32, device=dev)
    pn = torch.tensor(powers[train_p] / powers.max(), dtype=torch.float32, device=dev)

    torch.manual_seed(0)
    model = CoordMLP(a.width, a.depth).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    n_par = sum(q.numel() for q in model.parameters())

    per_snap = nx * ny * nz
    print(
        f"MLP 5 -> {' -> '.join([str(a.width)] * a.depth)} -> 1   "
        f"{n_par / 1e3:.1f}k params\n"
        f"{len(train_p) * nt * per_snap / 1e6:.0f}M training points "
        f"({', '.join(f'{int(p)}W' for p in powers[train_p])}), "
        f"{int(powers[test_p])}W held out\n"
        f"batch {a.batch}, {a.steps} steps -> "
        f"{a.steps * a.batch / (len(train_p) * nt * per_snap):.2f} epochs\n"
    )

    t0 = time.time()
    for step in range(a.steps):
        ip = torch.randint(0, len(train_p), (a.batch,), device=dev)
        it = torch.randint(0, nt, (a.batch,), device=dev)
        ix = torch.randint(0, nx, (a.batch,), device=dev)
        iy = torch.randint(0, ny, (a.batch,), device=dev)
        iz = torch.randint(0, nz, (a.batch,), device=dev)
        q = torch.stack([pn[ip], tn[it], zn[iz], yn[iy], xn[ix]], -1)
        target = Y[ip, (it * nx + ix) * ny * nz + iy * nz + iz]

        opt.zero_grad(set_to_none=True)
        loss = ((model(q) - target) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
        if step % 4000 == 0 or step == a.steps - 1:
            print(f"  step {step:>6}  train {loss.item():.3e}")
    print(f"  {time.time() - t0:.0f} s\n")

    pred = predict_grid(model, dev, powers, x, y, z, t, scale)

    hdr = f"{'P':>6} {'RMSE':>9} {'Linf':>8} {'peak':>9}"
    print(hdr)
    print("-" * len(hdr))
    for i, p in enumerate(powers):
        s = score(pred[i].astype(np.float64), dT[i].astype(np.float64))
        tag = "HELD OUT" if i == test_p else ""
        print(
            f"{int(p):>5}W {s['rmse']:>8.3f}K {s['linf']:>7.1f}K {s['peak']:>8.2f}%   {tag}"
        )
    print()

    d = np.load(a.data)
    meta = {k: d[k] for k in ("mx", "my", "grid", "z", "x", "y")}
    meta["dt"] = float(t[1] - t[0])
    truth = dT[test_p].astype(np.float64)
    mine = pred[test_p].astype(np.float64)
    P = int(powers[test_p])

    # the same modes the spectral model was given, read off this model's own field
    C = np.fft.rfftn(mine, axes=(1, 2), norm="ortho")
    mine_c = C[:, meta["mx"], :, :][:, :, meta["my"], :]

    a.figdir.mkdir(parents=True, exist_ok=True)
    plot_planes(truth, mine, meta, t, P, a.figdir / "nn_coord_field.png", "coord MLP")
    plot_signal(d["coef"][test_p], mine_c, meta, P, a.figdir / "nn_coord_signal.png")
    plot_scanline(
        truth, None, mine, meta, t, a.times, P,
        a.figdir / "nn_coord_scanline.png", "coord MLP",
    )
    for name in ("field", "signal", "scanline"):
        print(f"figure -> {a.figdir / f'nn_coord_{name}.png'}")


if __name__ == "__main__":
    main()
