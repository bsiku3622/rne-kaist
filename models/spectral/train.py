"""Fit (P, t) -> spatial Fourier coefficients, and archive the run that did it.

The nine networks in ``typeulli-model-training/`` all learn a function of five variables and
are asked for one temperature at a time. Here the spatial dependence is already
carried by the basis ``dataset.py`` built, so all that is left to learn is how the
coefficients move as the laser power and the clock change -- a map from R^2, which
three hidden layers are enough for. One forward pass is a whole volume.

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

from share import archiving, metrics, spectral
from share.checkpoints import BestCheckpoint
from share.grid import load_run

from .model import SpectralMLP
from .visualize import render

MODEL = "spectral"


def split(flat, nP, nt, shape, n_coef, ramp_shape):
    """Undo the packing: the network emits [Re, Im, ramp] in one vector."""
    coef = flat[..., :n_coef].reshape(nP, nt, *shape, 2)
    c = coef[..., 0] + 1j * coef[..., 1]
    ramp = (
        flat[..., n_coef:].reshape(nP, nt, *ramp_shape) if ramp_shape is not None else None
    )
    return c, ramp


def _fields(model, X, nP, nt, shape, spin, derotate, ds, n_coef, ramp_shape):
    """The model's field for every power, back in the lab frame."""
    dev = next(model.parameters()).device
    with torch.no_grad():
        out = model.denormalise(
            model(torch.tensor(X, dtype=torch.float32, device=dev)).cpu().numpy()
        ).reshape(nP * nt, -1)
    c, ramp = split(out, nP, nt, shape, n_coef, ramp_shape)
    if derotate:
        # the coefficients travel with the laser; the ramp is pinned to the domain,
        # so it is the one thing that must NOT be spun back
        c = c / spin[None, :, :, None, None]
    return [ds.reconstruct(c[i], None if ramp is None else ramp[i]) for i in range(nP)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True, help="a solver run under data/")
    ap.add_argument("--holdout", type=float, default=400.0, help="power kept out of training")
    ap.add_argument("--norm", choices=("global", "per-coef"), default="global")
    ap.add_argument("--derotate", action="store_true",
                    help="learn the coefficients in the frame that moves with the laser")
    ap.add_argument("--detrend", action="store_true",
                    help="use the detrended npz: the x-wrap ramp is a separate output")
    ap.add_argument("--energy", type=float, default=0.9999,
                    help="which saved box to load, by its energy target (default 0.9999)")
    ap.add_argument("--vel", type=float, default=10.0, help="scan speed, mm/s (VEL)")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--times", type=float, nargs="+", default=[0.5, 1.5, 3.0])
    ap.add_argument("--tag", type=str, default=None, help="suffix on the archive entry")
    ap.add_argument("--lock", action="store_true", help="mark the archive entry read-only")
    a = ap.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run = load_run(a.run)
    ds = spectral.load(run.dir, a.energy, a.detrend)

    coef, powers, times = ds.coef, ds.powers, ds.t
    nP, nt = coef.shape[:2]
    shape = ds.shape
    ramp = ds.ramp
    ramp_shape = ramp.shape[2:] if ramp is not None else None

    spin = ds.spin(times, a.vel)
    # only the coefficients are spun back. The ramp is the x-wrap cliff, which is
    # pinned to the domain and never moved, so de-rotating it would inject an
    # oscillation rather than remove one -- that is the whole point of splitting them.
    work = coef * spin[None, :, :, None, None] if a.derotate else coef

    X = np.stack(
        np.meshgrid(powers / powers.max(), times / times.max(), indexing="ij"), -1
    ).reshape(-1, 2)
    Y = np.stack([work.real, work.imag], -1).reshape(nP * nt, -1).astype(np.float64)
    n_coef = Y.shape[1]
    if ramp is not None:
        Y = np.concatenate([Y, ramp.reshape(nP * nt, -1).astype(np.float64)], axis=1)

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

    # Keep the checkpoint with the best held-out loss, not the last one: a long run
    # overfits after its best epoch, and the two differ whenever it does. Under
    # --norm global the transform is orthonormal, so by Parseval this held-out
    # coefficient MSE is the held-out field's L2 error up to a fixed constant -- its
    # argmin is the argmin of the real-space RMSE, so this selects the best-
    # generalising volume, not merely the best fit in coefficient space.
    best = BestCheckpoint(entry.checkpoint, mode="min")

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
            best.update(
                val,
                {
                    "state": model.state_dict(),
                    "architecture": arch,
                    "mu": mu,
                    "sd": sd,
                    "test_p": test_p,
                    "run_dir": str(run.dir),
                    "config": vars(a),
                },
                step=ep,
            )
            if ep % 4000 == 0 or ep == a.epochs - 1:
                print(f"  epoch {ep:>6}  train {loss.item():.3e}  held-out {val:.3e}")
    wall = time.time() - t0
    writer.close()
    print(f"  {wall:.0f} s   best held-out {best.best:.3e} at epoch {best.step}\n")

    # score and archive the best checkpoint, not the final weights
    model.load_state_dict(
        torch.load(entry.checkpoint, map_location=dev, weights_only=False)["state"]
    )
    model.eval()
    render(model, ds, run, test_p, entry.figures, a.times, a.derotate, a.vel,
           n_coef=n_coef, ramp_shape=ramp_shape)
    preds = _fields(model, X, nP, nt, shape, spin, a.derotate, ds, n_coef, ramp_shape)

    print("Against the solver.  'floor' is what the kept modes can do at best.\n")
    print(f"{'P':>6} {'':>7} {'RMSE':>9} {'Linf':>8} {'peak':>9}")
    print("-" * 45)
    scores = {}
    for i, p in enumerate(powers):
        truth = run.dT(i)
        row = {
            "floor": metrics.score(ds.floor(i), truth),
            "model": metrics.score(preds[i], truth),
        }
        scores[int(p)] = row
        for name, s in row.items():
            held = "HELD OUT" if i == test_p and name == "model" else ""
            print(
                f"{int(p):>5}W {name:>7} {s['rmse']:>8.3f}K {s['linf']:>7.1f}K "
                f"{s['peak']:>8.2f}%   {held}"
            )

    # the checkpoint is already on disk -- BestCheckpoint wrote the best epoch's
    # weights during training, so there is no final save to make here
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
