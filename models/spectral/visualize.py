"""Render a trained spectral model against the run it was fitted to.

Importable (``train.py`` calls :func:`render` on the way out, so every run
archives its own figures) and runnable on its own against an archived checkpoint.

Three figures, and the third is the one that earns its keep. ``field`` and
``scanline`` show whether the temperature came out right; ``signal`` shows *which
Fourier modes* the model got, which is the only place a model that has quietly
given up on the high modes is visible. Both are drawn against the *floor* -- what
the kept coefficients reconstruct exactly -- so the gap is the network's alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from share import plotting, spectral
from share.grid import load_run

from .model import SpectralMLP


def render(model, ds, run, test_p: int, figdir: Path, times, derotate: bool, vel: float,
           n_coef: int | None = None, ramp_shape=None):
    """Every figure this model is judged by, written into ``figdir``.

    ``ds`` is the :class:`~share.spectral.SpectralDataset` the model was fitted to; it
    carries the box, the de-rotation phase, and the inverse transform, so this only has
    to run the network and hand the coefficients back to it.
    """
    coef, powers, t = ds.coef, ds.powers, ds.t
    nP, nt = coef.shape[:2]
    power = int(powers[test_p])

    X = np.stack(
        np.meshgrid(powers / powers.max(), t / t.max(), indexing="ij"), -1
    ).reshape(-1, 2)
    with torch.no_grad():
        dev = next(model.parameters()).device
        flat = model.denormalise(
            model(torch.tensor(X, dtype=torch.float32, device=dev)).cpu().numpy()
        ).reshape(nP * nt, -1)
    if n_coef is None:
        n_coef = flat.shape[1]
    c = flat[:, :n_coef].reshape(nP, nt, *ds.shape, 2)
    pred = c[..., 0] + 1j * c[..., 1]
    if derotate:
        pred = pred / ds.spin(t, vel)[None, :, :, None, None]

    pred_ramp = (
        flat[:, n_coef:].reshape(nP, nt, *ramp_shape) if ramp_shape is not None else None
    )

    truth = run.dT(test_p)
    floor = ds.floor(test_p)
    mine = ds.reconstruct(pred[test_p], None if pred_ramp is None else pred_ramp[test_p])

    # where the raw coefficient series starts aliasing -- the reason the spin is
    # divided out analytically rather than left for the network to find. It is not a
    # ceiling on what can be learned; see share/plotting.signal.
    Lx = run.shape[1] * run.spacing
    nyquist = 0.5 / (run.snap_dt * vel / Lx)

    plotting.planes(truth, mine, run, power, figdir / "field.png", "spectral MLP")
    plotting.scanline(truth, mine, run, power, times, figdir / "scanline.png",
                      "spectral MLP", floor=floor)
    plotting.signal(coef[test_p], pred[test_p], ds.mx, power,
                    figdir / "signal.png", nyquist=nyquist)
    return truth, floor, mine


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entry", type=Path, required=True, help="an archive/ directory")
    ap.add_argument("--times", type=float, nargs="+", default=[0.5, 1.5, 3.0])
    a = ap.parse_args(argv)

    ckpt = torch.load(a.entry / "checkpoint.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    run = load_run(Path(ckpt["run_dir"]))
    ds = spectral.load(run.dir, cfg.get("energy", 0.9999), cfg.get("detrend", False))

    model = SpectralMLP(**ckpt["architecture"])
    model.load_state_dict(ckpt["state"])
    model.set_normalisation(ckpt["mu"], ckpt["sd"])
    model.eval()

    # the checkpoint's box may carry a ramp; split it off exactly as training did
    n_coef = 2 * int(np.prod(ds.shape))
    ramp_shape = ds.ramp.shape[2:] if ds.ramp is not None else None
    render(model, ds, run, ckpt["test_p"], a.entry / "figures", a.times,
           cfg["derotate"], cfg["vel"], n_coef=n_coef, ramp_shape=ramp_shape)
    print(f"figures -> {a.entry / 'figures'}")


if __name__ == "__main__":
    main()
