"""The spatial-Fourier coefficient dataset, and everything needed to use it.

The ``spectral`` model does not learn a temperature at a point; it learns every
stored Fourier coefficient of a whole snapshot, and an inverse transform turns
those back into a field. That coefficient dataset is a *dataset* in its own right --
derived from a solver run, cached to disk, read back at train time -- so it lives
here in the shared data layer beside :mod:`share.grid`, not inside the model.

Only ``spectral`` uses it, but keeping it here draws the line where it belongs:
:class:`SpectralDataset` is data, :class:`~models.spectral.model.SpectralMLP` is a
network, and the reconstruction and de-rotation that map between coefficients and
Kelvin are properties of the *representation*, not of the three dense layers fitted
to it. So they are methods on the dataset.

Three things live here:

* :func:`build` -- transform a run into a truncated ``fft2`` coefficient box and
  write it to disk. The box itself is chosen by :func:`best_box` from the run's
  energy spectrum; :mod:`models.spectral.dataset` is the analysis that justifies
  that choice and calls this to produce the file.
* :class:`SpectralDataset` and :func:`load` -- read a saved box back, and carry the
  operations that use it: :meth:`~SpectralDataset.reconstruct` (coefficients to
  ``dT`` on the grid), :meth:`~SpectralDataset.spin` (the de-rotation phase), and
  :meth:`~SpectralDataset.floor` (what the kept modes reconstruct exactly).
* :func:`npz_name` -- the on-disk name, which encodes the energy target so two
  budgets (0.9999 and 0.99999, say) can sit side by side instead of one silently
  overwriting the other.

The whole module is numpy: no torch, no matplotlib. Building and reconstructing a
field is array work, and keeping torch out of the data layer means a checkpoint can
be scored, or a box rebuilt, without a GPU in the room.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from share.grid import T_AMB, Run, detrend_x, retrend_x


def npz_name(energy_target: float, detrend: bool = False) -> str:
    """The on-disk file name for a given energy budget.

    The target is in the name -- ``spectral_fft2_e0.9999.npz`` -- so a run can hold
    several budgets at once. A single fixed name meant rebuilding at a new target
    clobbered the old box, which is a quiet way to score a model against a dataset
    it was not trained on.
    """
    return f"spectral_fft2_e{energy_target:g}{'_detrended' if detrend else ''}.npz"


# --------------------------------------------------------------------------- #
# the mode budget: fold the spectrum, price a box, pick the cheapest that clears
# --------------------------------------------------------------------------- #


def fold(P: np.ndarray, axis: int) -> np.ndarray:
    """Sum |C|^2 over +m and -m, leaving |m| = 0 .. N//2 along ``axis``.

    N is odd on every axis here, so there is no Nyquist bin to double-count:
    index 0 is m = 0, indices 1..N//2 are m = +1..+N//2, and the reversed tail
    is m = -1..-N//2.
    """
    P = np.moveaxis(P, axis, 0)
    half = P.shape[0] // 2
    out = np.empty((half + 1, *P.shape[1:]), dtype=P.dtype)
    out[0] = P[0]
    out[1:] = P[1 : half + 1] + P[: half : -1]
    return np.moveaxis(out, 0, axis)


def spectrum_shape(variant: str, nx: int, ny: int, nz: int) -> tuple[int, int, int]:
    kz = {"fft3": nz // 2 + 1, "fft2": 1, "dct": nz}[variant]
    return nx // 2 + 1, ny // 2 + 1, kz


def dof(variant: str, nz: int, kx: np.ndarray, ky: np.ndarray, kz: np.ndarray):
    """Stored real numbers for the box |mx| <= kx, |my| <= ky, and kz in z.

    Hermitian symmetry makes half the complex coefficients redundant, so a box of
    n complex coefficients costs n real numbers, not 2n. In ``fft2`` the whole z
    grid is stored; in ``dct`` the z coefficients are real to begin with.
    """
    plane = (2 * kx + 1) * (2 * ky + 1)
    if variant == "fft3":
        return plane * (2 * kz + 1)
    if variant == "fft2":
        return plane * nz
    return plane * (kz + 1)


def best_box(energy: np.ndarray, total: np.ndarray, variant: str, nz: int, target: float):
    """Cheapest box (fewest stored reals) that retains ``target`` for *every* power."""
    retained = energy.cumsum(1).cumsum(2).cumsum(3) / total[:, None, None, None]
    worst = retained.min(axis=0)
    kx, ky, kz = np.meshgrid(*(np.arange(n) for n in worst.shape), indexing="ij")
    cost = dof(variant, nz, kx, ky, kz)
    ok = worst >= target
    if not ok.any():
        return None
    flat = np.where(ok.ravel(), cost.ravel(), np.iinfo(np.int64).max)
    i = int(flat.argmin())
    per_power = retained.reshape(len(total), -1)[:, i]
    return {
        "kx": int(kx.ravel()[i]),
        "ky": int(ky.ravel()[i]),
        "kz": int(kz.ravel()[i]),
        "dof": int(cost.ravel()[i]),
        "retained": float(worst.ravel()[i]),
        "per_power": per_power,
        "pooled": float(per_power @ total / total.sum()),
    }


# --------------------------------------------------------------------------- #
# production: a run -> a truncated fft2 box on disk
# --------------------------------------------------------------------------- #


def build(
    run: Run,
    box: dict,
    out: Path,
    *,
    detrend: bool = False,
    energy_target: float | None = None,
) -> None:
    """Write the truncated ``fft2`` coefficients for ``box``, and check they reconstruct.

    Stored as a real-input FFT over ``(x, y)``: ``y`` keeps only its non-negative
    wavenumbers, since ``C(-kx, -ky) = conj(C(kx, ky))`` makes the rest redundant.
    ``z`` is kept whole. :meth:`SpectralDataset.reconstruct` is the way back.

    Under ``detrend`` the coefficients are those of the *residual* -- the field with
    the x-wrap ramp taken out -- and ``ramp`` holds the ``(nt, ny, nz)`` end-face
    mismatch that was removed, so reconstruction ends with ``retrend_x``. That costs
    ``ny * nz`` extra numbers a snapshot and buys back the high wavenumbers the cliff
    was filling with an artefact.
    """
    nt, nx, ny, nz = run.shape
    kx, ky = box["kx"], box["ky"]
    mx = np.r_[np.arange(kx + 1), np.arange(-kx, 0)]  # 0..kx, -kx..-1
    my = np.arange(ky + 1)

    nP = len(run.powers)
    coef = np.empty((nP, nt, len(mx), len(my), nz), dtype=np.complex64)
    ramp = np.zeros((nP, nt, ny, nz), dtype=np.float32)
    rows = []
    for i, P in enumerate(run.powers):
        dT = run.dT(i)
        work, D = detrend_x(dT) if detrend else (dT, None)
        if D is not None:
            ramp[i] = D

        C = np.fft.rfftn(work, axes=(1, 2), norm="ortho")  # over (x, y), t stays
        coef[i] = C[:, mx, :, :][:, :, my, :]

        full = np.zeros((nt, nx, ny // 2 + 1, nz), dtype=np.complex128)
        full[:, mx[:, None], my[None, :], :] = coef[i]
        rec = np.fft.irfftn(full, s=(nx, ny), axes=(1, 2), norm="ortho")
        if detrend:
            rec = retrend_x(rec, ramp[i].astype(np.float64))
        err = rec - dT

        true_peak = dT.reshape(nt, -1).max(1)
        hot = true_peak > 1.0  # t = 0 is uniformly ambient, so it has no peak
        peak_rel = (rec.reshape(nt, -1).max(1)[hot] - true_peak[hot]) / true_peak[hot]
        rows.append(
            (
                P,
                float(np.sqrt((err**2).mean())),
                float(np.abs(err).max()),
                float(np.abs(rec[:, :, :, 0]).max()),  # Dirichlet says this is 0
                float(peak_rel[np.abs(peak_rel).argmax()] * 100),
            )
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        coef=coef,
        ramp=ramp,
        detrend=bool(detrend),
        mx=mx.astype(np.int32),
        my=my.astype(np.int32),
        kx=kx,
        ky=ky,
        powers=run.powers.astype(np.float32),
        t=run.t.astype(np.float32),
        x=run.x.astype(np.float32),
        y=run.y.astype(np.float32),
        z=run.z.astype(np.float32),
        grid=np.array([nx, ny, nz], dtype=np.int32),
        T_amb=np.float32(T_AMB),
        energy=np.float32(energy_target if energy_target is not None else np.nan),
        norm="ortho",
    )

    print(f"\nsaved {out}  {coef.shape}  {out.stat().st_size / 1e6:.1f} MB")
    print("errors are over every snapshot of that power; peak is the worst one\n")
    print(f"{'P':>5} {'RMSE':>8} {'Linf':>8} {'bottom':>8} {'peak':>9}")
    for P, rmse, linf, bottom, peak in rows:
        print(f"{P:>5} {rmse:>7.3f}K {linf:>7.1f}K {bottom:>7.2f}K {peak:>8.2f}%")


# --------------------------------------------------------------------------- #
# the dataset itself, and the operations that use it
# --------------------------------------------------------------------------- #


@dataclass
class SpectralDataset:
    """A saved coefficient box, and the transforms between it and Kelvin.

    ``coef`` is ``(nP, nt, len(mx), len(my), nz)`` complex; ``ramp`` is the x-wrap
    mismatch of a detrended box, ``(nP, nt, ny, nz)``, or ``None``. Everything else
    is the geometry the inverse transform needs, carried so a checkpoint reconstructs
    the field exactly as the run that wrote it did.
    """

    coef: np.ndarray
    ramp: np.ndarray | None
    mx: np.ndarray
    my: np.ndarray
    grid: np.ndarray  # (nx, ny, nz)
    powers: np.ndarray
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    kx: int
    ky: int
    detrend: bool
    T_amb: float
    energy_target: float | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        """The per-snapshot coefficient shape, ``(len(mx), len(my), nz)``."""
        return self.coef.shape[2:]

    @property
    def nP(self) -> int:
        return self.coef.shape[0]

    @property
    def nt(self) -> int:
        return self.coef.shape[1]

    def spin(self, times: np.ndarray, vel: float) -> np.ndarray:
        """``exp(+2i pi kx v t)``, the de-rotation phase. Shaped ``(len(times), len(mx))``.

        A source moving at ``vel`` makes the field roughly ``f(x - v t)``, whose
        transform is ``g(kx) * exp(-2i pi kx v t)``: every mode spins, the faster the
        mode the faster it spins. This is the phase that undoes it. ``Lx = nx * dx`` is
        the period the DFT actually assumes -- one cell wider than the plate.
        """
        Lx = int(self.grid[0]) * float(self.x[1] - self.x[0])
        return np.exp(2j * np.pi * (self.mx / Lx) * vel * np.asarray(times)[:, None])

    def reconstruct(self, coef: np.ndarray, ramp: np.ndarray | None = None) -> np.ndarray:
        """Coefficients -> ``dT`` on the grid. The inverse of what :func:`build` stored.

        ``coef`` is ``(nt, len(mx), len(my), nz)`` for one power. ``y`` holds only its
        non-negative wavenumbers, since ``C(-kx, -ky) = conj(C(kx, ky))`` makes the
        rest redundant, so the way back is an ``irfftn`` over ``(x, y)`` with ``z``
        left alone. ``ramp`` is the x-wrap mismatch a detrended box took out; putting
        it back is the last step.
        """
        nx, ny, nz = (int(v) for v in self.grid)
        nt = coef.shape[0]
        full = np.zeros((nt, nx, ny // 2 + 1, nz), dtype=np.complex128)
        full[:, self.mx[:, None], self.my[None, :], :] = coef
        dT = np.fft.irfftn(full, s=(nx, ny), axes=(1, 2), norm="ortho")
        return dT if ramp is None else retrend_x(dT, np.asarray(ramp, dtype=np.float64))

    def floor(self, i: int) -> np.ndarray:
        """What the stored box reconstructs for power ``i`` -- the truncation floor.

        No network fitted to these coefficients can beat this, so it is the bound the
        model's own error is measured against.
        """
        return self.reconstruct(self.coef[i], None if self.ramp is None else self.ramp[i])


def load(run_dir: Path, energy_target: float, detrend: bool = False) -> SpectralDataset:
    """Read the box for ``energy_target`` back from ``run_dir``.

    Raises with the command that would build it if the file is not there, so a
    forgotten :mod:`models.spectral.dataset` step fails loudly rather than silently
    loading a different budget.
    """
    path = Path(run_dir) / npz_name(energy_target, detrend)
    if not path.is_file():
        flag = " --detrend" if detrend else ""
        raise SystemExit(
            f"no {path.name} in {run_dir}\n"
            f"build it first:  python models/spectral/dataset.py "
            f"--run {run_dir} --save-target {energy_target:g}{flag}"
        )
    z = np.load(path)
    return SpectralDataset(
        coef=z["coef"],
        ramp=z["ramp"] if detrend else None,
        mx=z["mx"],
        my=z["my"],
        grid=z["grid"],
        powers=z["powers"],
        t=z["t"],
        x=z["x"],
        y=z["y"],
        z=z["z"],
        kx=int(z["kx"]),
        ky=int(z["ky"]),
        detrend=bool(z["detrend"]),
        T_amb=float(z["T_amb"]),
        energy_target=float(z["energy"]) if "energy" in z.files else energy_target,
    )
