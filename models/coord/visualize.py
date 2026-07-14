"""Render a trained coordinate model against the run it was fitted to.

Same three figures as ``models/spectral``, deliberately, so the two can be laid
side by side. There is no truncation floor here -- this model predicts the field
directly, so nothing was thrown away before it started and the error is all its own.

The ``signal`` figure still applies, and is the most revealing of the three: the
model's *predicted field* is transformed onto the same modes the spectral dataset
kept, and compared against the solver's. A coordinate MLP that is quietly adding
piecewise-linear ripple shows up there as spectral energy sitting *above* the
truth at high ``|mx|`` -- energy that is not in the physics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from share import plotting
from share.grid import load_run

from .model import CoordMLP

CHUNK = 1 << 21


def predict(model, run, power: float) -> np.ndarray:
    """The model's field on the full grid, ``(nt, nx, ny, nz)``."""
    dev = next(model.parameters()).device
    nt, nx, ny, nz = run.shape
    zn = torch.tensor(run.z / run.z.max(), dtype=torch.float32, device=dev)
    yn = torch.tensor(run.y / run.y.max(), dtype=torch.float32, device=dev)
    xn = torch.tensor(run.x / run.x.max(), dtype=torch.float32, device=dev)
    tn = torch.tensor(run.t / run.t.max(), dtype=torch.float32, device=dev)
    pn = float(power) / float(run.powers.max())

    out = np.empty((nt, nx, ny, nz), dtype=np.float32)
    n = nx * ny * nz
    with torch.no_grad():
        for j in range(nt):
            for s in range(0, n, CHUNK):
                idx = torch.arange(s, min(s + CHUNK, n), device=dev)
                iz, iy, ix = idx % nz, (idx // nz) % ny, idx // (ny * nz)
                q = torch.stack(
                    [
                        torch.full_like(zn[iz], pn),
                        torch.full_like(zn[iz], float(tn[j])),
                        zn[iz], yn[iy], xn[ix],
                    ],
                    -1,
                )
                out[j].reshape(-1)[s : s + idx.numel()] = model.dT(q).cpu().numpy()
    return out


def render(model, run, test_p: int, figdir: Path, times, npz=None):
    """Every figure this model is judged by, written into ``figdir``."""
    power = int(run.powers[test_p])
    truth = run.dT(test_p)
    mine = predict(model, run, power).astype(np.float64)

    plotting.planes(truth, mine, run, power, figdir / "field.png", "coord MLP")
    plotting.scanline(truth, mine, run, power, times, figdir / "scanline.png", "coord MLP")

    if npz is not None:
        # read this model's own field on the modes the spectral dataset kept
        mx, my = npz["mx"], npz["my"]
        C = np.fft.rfftn(mine, axes=(1, 2), norm="ortho")
        mine_c = C[:, mx, :, :][:, :, my, :]
        plotting.signal(npz["coef"][test_p], mine_c, mx, power, figdir / "signal.png")
    return truth, mine


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entry", type=Path, required=True, help="an archive/ directory")
    ap.add_argument("--times", type=float, nargs="+", default=[0.5, 1.5, 3.0])
    a = ap.parse_args(argv)

    ckpt = torch.load(a.entry / "checkpoint.pt", map_location="cpu", weights_only=False)
    run = load_run(Path(ckpt["run_dir"]))
    model = CoordMLP(**ckpt["architecture"])
    model.load_state_dict(ckpt["state"])
    model.eval()

    npz_path = run.dir / "spectral_fft2.npz"
    npz = np.load(npz_path) if npz_path.is_file() else None
    render(model, run, ckpt["test_p"], a.entry / "figures", a.times, npz)
    print(f"figures -> {a.entry / 'figures'}")


if __name__ == "__main__":
    main()
