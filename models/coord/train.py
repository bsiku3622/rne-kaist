"""Fit (P, t, z, y, x) -> T, and archive the run that did it.

The counterpart to ``models/spectral``: the same hidden stack, but carrying the
space itself instead of leaning on a Fourier basis. That makes it 280x smaller and
gives it 276M training points instead of 217, so full-batch is impossible and points
are drawn at random each step.

Everything else is held fixed so the comparison means something -- the same held-out
power, the same optimiser and schedule, the same figures, the same metrics.

Example::

    python models/coord/train.py --run data/20260710_132221_powersweep_gpu
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

from .model import CoordMLP
from .visualize import predict, render

MODEL = "coord"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True, help="a solver run under data/")
    ap.add_argument("--holdout", type=float, default=400.0)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=1 << 16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--times", type=float, nargs="+", default=[0.5, 1.5, 3.0])
    ap.add_argument("--tag", type=str, default=None, help="suffix on the archive entry")
    ap.add_argument("--lock", action="store_true", help="mark the archive entry read-only")
    a = ap.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run = load_run(a.run)
    nt, nx, ny, nz = run.shape
    nP = len(run.powers)
    test_p = run.index_of(a.holdout)
    train_p = [i for i in range(nP) if i != test_p]

    dT = run.dT_all(np.float32)
    scale = float(dT[train_p].max())

    # the whole training set sits on the GPU; points are drawn from it at random
    Y = torch.tensor(dT[train_p], device=dev).reshape(len(train_p), -1) / scale
    zn = torch.tensor(run.z / run.z.max(), dtype=torch.float32, device=dev)
    yn = torch.tensor(run.y / run.y.max(), dtype=torch.float32, device=dev)
    xn = torch.tensor(run.x / run.x.max(), dtype=torch.float32, device=dev)
    tn = torch.tensor(run.t / run.t.max(), dtype=torch.float32, device=dev)
    pn = torch.tensor(run.powers[train_p] / run.powers.max(), dtype=torch.float32, device=dev)

    torch.manual_seed(0)
    arch = dict(width=a.width, depth=a.depth, scale=scale)
    model = CoordMLP(**arch).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)

    entry = archiving.open_entry(MODEL, run, a.tag)
    writer = SummaryWriter(log_dir=str(entry.tensorboard))
    n_par = sum(q.numel() for q in model.parameters())
    n_pts = len(train_p) * nt * nx * ny * nz
    print(
        f"[archive] {entry.dir.name}\n"
        f"MLP 5 -> {' -> '.join([str(a.width)] * a.depth)} -> 1   {n_par / 1e3:.1f}k params\n"
        f"{n_pts / 1e6:.0f}M training points, {int(run.powers[test_p])} W held out\n"
        f"batch {a.batch}, {a.steps} steps -> {a.steps * a.batch / n_pts:.2f} epochs\n"
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
        if step % 100 == 0 or step == a.steps - 1:
            writer.add_scalar("loss/train", loss.item(), step)
            if step % 4000 == 0 or step == a.steps - 1:
                print(f"  step {step:>6}  train {loss.item():.3e}")
    wall = time.time() - t0
    writer.close()
    print(f"  {wall:.0f} s\n")

    model.eval()
    npz_path = run.dir / "spectral_fft2.npz"
    npz = np.load(npz_path) if npz_path.is_file() else None
    render(model, run, test_p, entry.figures, a.times, npz)

    print(f"{'P':>6} {'RMSE':>9} {'Linf':>8} {'peak':>9}")
    print("-" * 36)
    scores = {}
    for i, p in enumerate(run.powers):
        s = metrics.score(
            predict(model, run, p).astype(np.float64), dT[i].astype(np.float64)
        )
        scores[int(p)] = s
        held = "HELD OUT" if i == test_p else ""
        print(
            f"{int(p):>5}W {s['rmse']:>8.3f}K {s['linf']:>7.1f}K "
            f"{s['peak']:>8.2f}%   {held}"
        )

    torch.save(
        {
            "state": model.state_dict(),
            "architecture": arch,
            "test_p": test_p,
            "run_dir": str(run.dir),
            "config": vars(a),
        },
        entry.checkpoint,
    )
    archiving.finalise(
        entry,
        run,
        config={**vars(a), "params": n_par, "device": dev, "train_points": n_pts},
        metrics={
            "wall_s": round(wall, 1),
            "holdout_power": int(run.powers[test_p]),
            "per_power": scores,
        },
        lock=a.lock,
    )


if __name__ == "__main__":
    main()
