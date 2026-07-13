"""Fit (P, t) -> spatial Fourier coefficients, and archive the run that did it.

The DeepONet in ``models/deeponet`` learns a function of five variables and is asked
for one temperature at a time. Here the spatial dependence is already carried by the
basis ``dataset.py`` built, so all that is left to learn is how the coefficients move
as the laser power and the clock change -- a map from R^2, which three hidden layers
are enough for.

Two things make the evaluation honest.

**There is a floor.** A network that predicted every stored coefficient exactly would
still be wrong by the truncation error, because the box threw the rest of the spectrum
away. So the run reports both, and the gap between them is the only part the network
is answerable for.

**One power is held out.** Interpolating in P is the whole point of a surrogate, so it
never appears in training and everything quoted for it is inference from its neighbours.

``--norm`` is worth running both ways. Under ``global`` the coefficients keep their
relative sizes, so by Parseval the loss *is* the field's L2 error -- physically the
right thing to minimise, though the low modes carry nearly all of it. Under ``per-coef``
every coefficient is standardised, so the tiny high-frequency ones count as much as the
DC term.

Example::

    python models/spectral/train.py --run data/20260710_132221_powersweep_gpu --derotate
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(Path(__file__).resolve().parents[2]))

from share import archiving, metrics
from share.grid import load_run

from spectral_model import SpectralMLP, derotate_phase, reconstruct
from visualize import render

MODEL = "spectral"
NPZ = "spectral_fft2.npz"


def _fields(model, X, nP, nt, shape, spin, derotate, meta):
    """The model's field for every power, back in the lab frame."""
    dev = next(model.parameters()).device
    with torch.no_grad():
        out = model.denormalise(
            model(torch.tensor(X, dtype=torch.float32, device=dev)).cpu().numpy()
        ).reshape(nP, nt, *shape, 2)
    c = out[..., 0] + 1j * out[..., 1]
    if derotate:
        c = c / spin[None, :, :, None, None]
    return [reconstruct(c[i], meta) for i in range(nP)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True, help="a solver run under data/")
    ap.add_argument("--holdout", type=float, default=400.0, help="power kept out of training")
    ap.add_argument("--norm", choices=("global", "per-coef"), default="global")
    ap.add_argument("--derotate", action="store_true",
                    help="learn the coefficients in the frame that moves with the laser")
    ap.add_argument("--vel", type=float, default=10.0, help="scan speed, mm/s (VEL)")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--times", type=float, nargs="+", default=[0.5, 1.5, 3.0])
    ap.add_argument("--tag", type=str, default=None, help="suffix on the archive entry")
    ap.add_argument("--lock", action="store_true", help="mark the archive entry read-only")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run = load_run(a.run)
    npz_path = run.dir / NPZ
    if not npz_path.is_file():
        raise SystemExit(
            f"no {NPZ} in {run.dir}\n"
            f"build it first:  python models/spectral/dataset.py --run {run.dir}"
        )
    npz = np.load(npz_path)

    coef, powers, times = npz["coef"], npz["powers"], npz["t"]
    nP, nt = coef.shape[:2]
    shape = coef.shape[2:]
    meta = {k: npz[k] for k in ("mx", "my", "grid")}

    spin = derotate_phase(meta["mx"], run, times, a.vel)
    work = coef * spin[None, :, :, None, None] if a.derotate else coef

    X = np.stack(
        np.meshgrid(powers / powers.max(), times / times.max(), indexing="ij"), -1
    ).reshape(-1, 2)
    Y = np.stack([work.real, work.imag], -1).reshape(nP * nt, -1).astype(np.float64)

    test_p = int(np.argmin(np.abs(powers - a.holdout)))
    tr = np.repeat(np.arange(nP) != test_p, nt)

    if a.norm == "global":
        # one scale keeps the coefficients' relative sizes, so by Parseval the loss
        # below is the field's L2 error, up to that constant
        mu, sd = np.zeros(1), np.full(1, Y[tr].std())
    else:
        mu, sd = Y[tr].mean(0), Y[tr].std(0) + 1e-12

    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    Yt = torch.tensor((Y - mu) / sd, dtype=torch.float32, device=dev)
    trt = torch.tensor(tr, device=dev)

    torch.manual_seed(0)
    arch = dict(n_out=Y.shape[1], width=a.width, depth=a.depth)
    model = SpectralMLP(**arch).to(dev)
    model.set_normalisation(mu, sd)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    entry = archiving.open_entry(MODEL, run, a.tag)
    writer = SummaryWriter(log_dir=str(entry.tensorboard))
    n_par = sum(p.numel() for p in model.parameters())
    print(
        f"[archive] {entry.dir.name}\n"
        f"MLP 2 -> {' -> '.join([str(a.width)] * a.depth)} -> {Y.shape[1]}   "
        f"{n_par / 1e6:.1f}M params\n"
        f"{int(tr.sum())} train samples, {nt} held out ({int(powers[test_p])} W), "
        f"norm = {a.norm}, derotate = {a.derotate}\n"
    )

    t0 = time.time()
    for ep in range(a.epochs):
        opt.zero_grad(set_to_none=True)
        loss = ((model(Xt[trt]) - Yt[trt]) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
        if ep % 100 == 0 or ep == a.epochs - 1:
            with torch.no_grad():
                val = ((model(Xt[~trt]) - Yt[~trt]) ** 2).mean().item()
            writer.add_scalar("loss/train", loss.item(), ep)
            writer.add_scalar("loss/holdout", val, ep)
            if ep % 4000 == 0 or ep == a.epochs - 1:
                print(f"  epoch {ep:>6}  train {loss.item():.3e}  held-out {val:.3e}")
    wall = time.time() - t0
    writer.close()
    print(f"  {wall:.0f} s\n")

    model.eval()
    render(model, npz, run, test_p, entry.figures, a.times, a.derotate, a.vel)
    preds = _fields(model, X, nP, nt, shape, spin, a.derotate, meta)

    print("Against the solver.  'floor' is what the kept modes can do at best.\n")
    print(f"{'P':>6} {'':>7} {'RMSE':>9} {'Linf':>8} {'peak':>9}")
    print("-" * 45)
    scores = {}
    for i, p in enumerate(powers):
        truth = run.dT(i)
        row = {
            "floor": metrics.score(reconstruct(coef[i], meta), truth),
            "model": metrics.score(preds[i], truth),
        }
        scores[int(p)] = row
        for name, s in row.items():
            held = "HELD OUT" if i == test_p and name == "model" else ""
            print(
                f"{int(p):>5}W {name:>7} {s['rmse']:>8.3f}K {s['linf']:>7.1f}K "
                f"{s['peak']:>8.2f}%   {held}"
            )

    torch.save(
        {
            "state": model.state_dict(),
            "architecture": arch,
            "mu": mu,
            "sd": sd,
            "test_p": test_p,
            "run_dir": str(run.dir),
            "config": vars(a),
        },
        entry.checkpoint,
    )
    archiving.finalise(
        entry,
        run,
        config={**vars(a), "params": n_par, "device": dev},
        metrics={
            "wall_s": round(wall, 1),
            "holdout_power": int(powers[test_p]),
            "per_power": scores,
        },
        lock=a.lock,
    )


if __name__ == "__main__":
    main()
