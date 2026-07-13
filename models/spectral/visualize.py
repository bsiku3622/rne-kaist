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

from share import plotting
from share.grid import load_run

from spectral_model import SpectralMLP, derotate_phase, reconstruct


def render(model, npz, run, test_p: int, figdir: Path, times, derotate: bool, vel: float):
    """Every figure this model is judged by, written into ``figdir``."""
    coef, powers, t = npz["coef"], npz["powers"], npz["t"]
    nP, nt = coef.shape[:2]
    meta = {k: npz[k] for k in ("mx", "my", "grid")}
    power = int(powers[test_p])

    X = np.stack(
        np.meshgrid(powers / powers.max(), t / t.max(), indexing="ij"), -1
    ).reshape(-1, 2)
    with torch.no_grad():
        dev = next(model.parameters()).device
        out = model.denormalise(
            model(torch.tensor(X, dtype=torch.float32, device=dev)).cpu().numpy()
        ).reshape(nP, nt, *coef.shape[2:], 2)
    pred = out[..., 0] + 1j * out[..., 1]
    if derotate:
        pred = pred / derotate_phase(meta["mx"], run, t, vel)[None, :, :, None, None]

    truth = run.dT(test_p)
    floor = reconstruct(coef[test_p], meta)
    mine = reconstruct(pred[test_p], meta)

    # where the raw coefficient series starts aliasing -- the reason the spin is
    # divided out analytically rather than left for the network to find. It is not a
    # ceiling on what can be learned; see share/plotting.signal.
    Lx = run.shape[1] * run.spacing
    nyquist = 0.5 / (run.snap_dt * vel / Lx)

    plotting.planes(truth, mine, run, power, figdir / "field.png", "spectral MLP")
    plotting.scanline(truth, mine, run, power, times, figdir / "scanline.png",
                      "spectral MLP", floor=floor)
    plotting.signal(coef[test_p], pred[test_p], meta["mx"], power,
                    figdir / "signal.png", nyquist=nyquist)
    return truth, floor, mine


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entry", type=Path, required=True, help="an archive/ directory")
    ap.add_argument("--times", type=float, nargs="+", default=[0.5, 1.5, 3.0])
    a = ap.parse_args()

    ckpt = torch.load(a.entry / "checkpoint.pt", map_location="cpu", weights_only=False)
    run = load_run(Path(ckpt["run_dir"]))
    npz = np.load(run.dir / "spectral_fft2.npz")

    model = SpectralMLP(**ckpt["architecture"])
    model.load_state_dict(ckpt["state"])
    model.set_normalisation(ckpt["mu"], ckpt["sd"])
    model.eval()

    render(model, npz, run, ckpt["test_p"], a.entry / "figures", a.times,
           ckpt["config"]["derotate"], ckpt["config"]["vel"])
    print(f"figures -> {a.entry / 'figures'}")


if __name__ == "__main__":
    main()
