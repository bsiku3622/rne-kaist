"""Spatial spectral analysis of the solver output, and the mode budget it implies.

The solver writes ``[x, y, z, t, P, T]`` rows on a uniform grid, so the field can
be reshaped back to ``(nt, nx, ny, nz)`` and transformed directly. The question
this script answers is how many spectral coefficients are actually needed to
represent it -- the mode budget a spectral surrogate (or the ``modes`` hyper-
parameter of an FNO) has to be given.

The field is not periodic, which is what makes the answer non-obvious. Its two
awkward directions are not awkward for the same reason:

* ``z`` runs from a Dirichlet substrate at the bottom (dT = 0 exactly) to the
  laser-heated top face (dT up to 4717 K). A DFT wraps those two faces onto each
  other, so it sees a 4717 K cliff that does not exist. Coefficients then decay
  like 1/m and truncation rings. On top of that, ``z`` carries the one genuinely
  small scale in the problem -- the thermal penetration depth alpha/v = 0.288 mm,
  against a 6 mm domain sampled by 49 points.
* ``x`` and ``y`` carry no scale finer than the 1.5 mm beam, and their wrap
  mismatch is small (41 K across x; y is symmetric about the scan line, so it is
  continuous in value and only kinked in slope).

So three ways to spend the transform are compared, all orthonormal, so that
Parseval holds and the truncation error follows from the spectrum alone:

    fft3   FFT over (x, y, z)              -- the literal reading of "3-D spatial FFT"
    fft2   FFT over (x, y), z left on grid -- Fourier in the homogeneous directions only
    dct    FFT over (x, y), DCT-II over z  -- even-extends z, so no cliff

For each, the optimal truncation box is the one of least stored real degrees of
freedom that still retains a target fraction of the field's energy. Energy and L2
error come free from Parseval; peak error, worst pointwise error and the
Dirichlet violation at the bottom face need an actual reconstruction, so those are
measured only for the chosen boxes.

The npz lands next to the data it came from, as ``data/<run>/spectral_fft2_e<target>.npz``;
the figures land in ``reports/spectral-<run>/``, because they are what justifies
the mode budget and ``data/`` is not tracked.

Example::

    python models/fmlp/dataset.py --run data/20260710_132221_powersweep_gpu
"""

from __future__ import annotations

import argparse
from datetime import datetime
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dctn, idctn

sys.path.append(str(Path(__file__).resolve().parents[2]))

from share.grid import Run, detrend_x, load_run
from share.spectral import best_box, build, fold, npz_name, spectrum_shape

VARIANTS = ("fft3", "fft2", "dct")


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #


def transform(dT: np.ndarray, variant: str) -> np.ndarray:
    """Orthonormal spatial transform of one snapshot, shaped (nx, ny, nz)."""
    if variant == "fft3":
        return np.fft.fftn(dT, axes=(0, 1, 2), norm="ortho")
    if variant == "fft2":
        return np.fft.fftn(dT, axes=(0, 1), norm="ortho")
    if variant == "dct":
        return np.fft.fftn(
            dctn(dT, type=2, axes=(2,), norm="ortho"), axes=(0, 1), norm="ortho"
        )
    raise ValueError(variant)


def inverse(C: np.ndarray, variant: str) -> np.ndarray:
    if variant == "fft3":
        return np.fft.ifftn(C, axes=(0, 1, 2), norm="ortho").real
    if variant == "fft2":
        return np.fft.ifftn(C, axes=(0, 1), norm="ortho").real
    if variant == "dct":
        return idctn(
            np.fft.ifftn(C, axes=(0, 1), norm="ortho").real,
            type=2,
            axes=(2,),
            norm="ortho",
        )
    raise ValueError(variant)


def spectrum(dT: np.ndarray, variant: str) -> np.ndarray:
    """|C|^2 binned by (|mx|, |my|, kz), where kz is |mz| for FFT and j for DCT."""
    P = np.abs(transform(dT, variant)) ** 2
    if variant == "fft2":
        P = P.sum(axis=2, keepdims=True)  # z is kept whole, so it is not a budget axis
    elif variant == "fft3":
        P = fold(P, 2)
    return fold(fold(P, 0), 1)


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #


def field(run: Run, i: int, detrend: bool):
    """The field the transform is applied to, and the ramp taken out of it."""
    dT = run.dT(i)
    if not detrend:
        return dT, None
    return detrend_x(dT)


def accumulate(run: Run, detrend: bool = False) -> dict:
    """One pass over the sweep: folded spectra, kept per power and per snapshot.

    Energy scales like P^2, so 100 W carries under 1% of the sweep's total while
    700 W carries a third of it. A budget fitted to the pooled energy would be
    free to discard the low powers outright -- and they are the *harder* ones to
    represent, because radiation only caps and broadens the peak once the field
    is hot enough. So everything here is kept per power, and the budget is later
    required to hold for the worst of them rather than for their sum.
    """
    nt, nx, ny, nz = run.shape
    nP = len(run.powers)
    energy = {v: np.zeros((nP, *spectrum_shape(v, nx, ny, nz))) for v in VARIANTS}
    total = np.zeros(nP)
    snap = np.zeros((nP, nt, nx // 2 + 1, ny // 2 + 1))  # fft2, per snapshot
    snap_total = np.zeros((nP, nt))
    peak = np.zeros((nP, nt))

    for i, P in enumerate(run.powers):
        dT, _ = field(run, i, detrend)
        peak[i] = dT.reshape(nt, -1).max(axis=1)
        snap_total[i] = (dT**2).reshape(nt, -1).sum(axis=1)
        total[i] = snap_total[i].sum()
        for j in range(nt):
            for v in VARIANTS:
                s = spectrum(dT[j], v)
                energy[v][i] += s
                if v == "fft2":
                    snap[i, j] = s[:, :, 0]
        print(
            f"  {int(P):3d} W  peak dT {peak[i].max():7.1f} K   "
            f"{100 * total[i] / total.sum():5.1f}% of the energy so far"
        )

    return {
        "energy": energy,
        "total": total,
        "snap": snap,
        "snap_total": snap_total,
        "peak": peak,
    }


def hardest_snapshot(acc: dict, kx: int, ky: int) -> tuple[int, int]:
    """The snapshot that keeps least of its own energy inside the (kx, ky) box."""
    cum = acc["snap"].cumsum(2).cumsum(3)[:, :, kx, ky]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(acc["snap_total"] > 0, cum / acc["snap_total"], 1.0)
    return np.unravel_index(frac.argmin(), frac.shape)


def mask_of(box: dict, variant: str, shape: tuple[int, int, int]) -> np.ndarray:
    """Keep-mask over the untransformed coefficient layout."""
    nx, ny, nz = shape
    mx = np.abs(np.fft.fftfreq(nx, 1 / nx).astype(int))
    my = np.abs(np.fft.fftfreq(ny, 1 / ny).astype(int))
    keep = (mx <= box["kx"])[:, None, None] & (my <= box["ky"])[None, :, None]
    if variant == "fft3":
        mz = np.abs(np.fft.fftfreq(nz, 1 / nz).astype(int))
        keep = keep & (mz <= box["kz"])[None, None, :]
    elif variant == "dct":
        keep = keep & (np.arange(nz) <= box["kz"])[None, None, :]
    return keep


def reconstruct_error(dT: np.ndarray, variant: str, box: dict) -> dict:
    """Errors a truncated box makes on one snapshot, in Kelvin."""
    C = transform(dT, variant)
    rec = inverse(C * mask_of(box, variant, dT.shape), variant)
    err = rec - dT
    return {
        "rmse": float(np.sqrt((err**2).mean())),
        "linf": float(np.abs(err).max()),
        "peak": float(rec.max() - dT.max()),
        "peak_rel": float((rec.max() - dT.max()) / dT.max() * 100),
        "bottom": float(np.abs(rec[:, :, 0]).max()),  # Dirichlet says this is 0
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def plot_marginals(run: Run, acc: dict, out: Path) -> None:
    """Energy per axis against wavenumber, which is where the anisotropy shows.

    Each power is normalised by its own energy before averaging, so the 700 W run
    does not simply draw the picture on its own.
    """
    nt, nx, ny, nz = run.shape
    d = float(run.x[1] - run.x[0])
    w = 1 / acc["total"] / len(acc["total"])
    E = (acc["energy"]["fft3"] * w[:, None, None, None]).sum(0)
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")

    for name, axis, n in (("kx", 0, nx), ("ky", 1, ny), ("kz", 2, nz)):
        m = np.arange(E.shape[axis])
        e = E.sum(axis=tuple(a for a in range(3) if a != axis))
        ax.loglog(m[1:] / (n * d), e[1:], label=f"{name}  (L = {n * d:.3f} mm)")

    Edct = (acc["energy"]["dct"] * w[:, None, None, None]).sum(0)
    j = np.arange(Edct.shape[2])
    e = Edct.sum(axis=(0, 1))
    ax.loglog(j[1:] / (2 * nz * d), e[1:], "--", label="kz, DCT-II")

    for k, lab in ((1 / 1.5, "1/beam"), (1 / 0.288, "1/(alpha/v)"), (1 / (2 * d), "Nyquist")):
        ax.axvline(k, color="0.7", lw=0.8, ls=":")
        ax.text(k, ax.get_ylim()[1], f" {lab}", rotation=90, va="top", fontsize=7, color="0.4")

    ax.set_xlabel("wavenumber  [cycles/mm]")
    ax.set_ylabel("fraction of total energy")
    ax.set_title("Marginal power spectrum, averaged over the sweep")
    ax.legend()
    fig.savefig(out / "marginal_spectrum.png", dpi=150)
    plt.close(fig)


def plot_kspace(run: Run, box: dict, cases: list, out: Path) -> None:
    """|C(mx, my)|^2 over the scan plane, with the retained box drawn on it.

    The trail is long in x and narrow in y, but it is the *moving* source that
    sets the spectrum: the melt pool's leading edge is far steeper than its
    lateral profile, so x carries the higher wavenumbers of the two.
    """
    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 4.2), layout="constrained")
    for ax, (dT, label) in zip(np.atleast_1d(axes), cases):
        C = np.fft.fftn(dT, axes=(0, 1), norm="ortho")
        E = fold(fold(np.abs(C) ** 2, 0), 1).sum(axis=2)
        E = E / E.sum()
        im = ax.imshow(
            np.log10(np.maximum(E[:81, :31], 1e-16)).T,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=-12,
            vmax=0,
        )
        ax.add_patch(
            plt.Rectangle(
                (-0.5, -0.5),
                box["kx"] + 1,
                box["ky"] + 1,
                fill=False,
                ec="white",
                lw=1.6,
                ls="--",
            )
        )
        ax.text(
            2,
            box["ky"] + 2,
            f"kept: |mx| <= {box['kx']}, |my| <= {box['ky']}",
            color="white",
            fontsize=8,
            va="bottom",
        )
        ax.annotate(
            "the x-wrap ridge:\nan artefact, and it is\nwhat sets |mx| <= 53",
            xy=(70, 0.5),
            xytext=(46, 17),
            color="white",
            fontsize=7.5,
            ha="center",
            arrowprops=dict(arrowstyle="->", color="white", lw=0.9),
        )
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("|mx|")
        ax.set_ylabel("|my|")
    fig.suptitle(
        "Energy over the scan plane   (mx: 1 mode = 0.025 cycles/mm,  "
        "my: 1 mode = 0.099 cycles/mm)",
        fontsize=10,
    )
    fig.colorbar(im, ax=np.atleast_1d(axes)[-1], label="log10 energy fraction")
    fig.savefig(out / "kspace.png", dpi=150)
    plt.close(fig)


def plot_reconstruction(run: Run, box: dict, cases: list, out: Path) -> None:
    """The laser-scanned face, before and after the truncation, and the difference."""
    ext = [0, float(run.x[-1]), 0, float(run.y[-1])]
    fig, axes = plt.subplots(
        len(cases), 2, figsize=(11, 2.4 * len(cases)), layout="constrained", squeeze=False
    )
    for row, (dT, label) in zip(axes, cases):
        rec = inverse(transform(dT, "fft2") * mask_of(box, "fft2", dT.shape), "fft2")
        truth, pred = dT[:, :, -1], rec[:, :, -1]  # z = z_max, the face the laser scans
        err = pred - truth
        bound = float(np.abs(err).max()) or 1.0

        style = dict(origin="lower", extent=ext, aspect="equal")
        a = row[0].imshow(pred.T, **style, vmin=0, vmax=float(truth.max()), cmap="inferno")
        b = row[1].imshow(err.T, **style, vmin=-bound, vmax=bound, cmap="RdBu_r")
        fig.colorbar(a, ax=row[0], label="dT [K]", fraction=0.03, pad=0.02)
        fig.colorbar(b, ax=row[1], label="error [K]", fraction=0.03, pad=0.02)

        row[0].set_title(f"{label}   reconstructed from the kept modes", fontsize=9)
        row[1].set_title(
            f"minus the solver   (peak {100 * (pred.max() - truth.max()) / truth.max():+.2f}%)",
            fontsize=9,
        )
        for ax in row:
            ax.set_xlabel("x [mm]", fontsize=8)
            ax.set_ylabel("y [mm]", fontsize=8)
            ax.tick_params(labelsize=8)
    fig.suptitle(
        "Laser-scanned face: what the truncation keeps, and what it costs", fontsize=11
    )
    fig.savefig(out / "reconstruction.png", dpi=150)
    plt.close(fig)


def plot_profiles(run: Run, box: dict, dT: np.ndarray, label: str, out: Path) -> None:
    """Two cuts through the beam: along the scan line, and down into the plate.

    The right-hand panel is the whole argument for leaving z alone. Both curves
    there spend the *same* 13 real coefficients out of 49 on the depth profile;
    the FFT spends them fighting a 4717 K cliff that only exists because it wrapped
    the hot top face onto the cold substrate, and rings by hundreds of kelvin --
    including below the bottom face, where the Dirichlet condition says the answer
    is exactly zero.
    """
    nx, ny, nz = dT.shape
    ix, iy, _ = np.unravel_index(dT.argmax(), dT.shape)
    rec = inverse(transform(dT, "fft2") * mask_of(box, "fft2", dT.shape), "fft2")

    keep = 6  # |mz| <= 6 -> 13 real numbers, the same as DCT's j <= 12
    Cz = np.fft.fft(dT, axis=2, norm="ortho")
    mz = np.abs(np.fft.fftfreq(nz, 1 / nz).astype(int))
    z_fft = np.fft.ifft(Cz * (mz <= keep), axis=2, norm="ortho").real
    Dz = dctn(dT, type=2, axes=(2,), norm="ortho")
    z_dct = idctn(Dz * (np.arange(nz) <= 2 * keep), type=2, axes=(2,), norm="ortho")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")

    ax1.plot(run.x, dT[:, iy, -1], color="0.25", lw=2.4, label="solver")
    ax1.plot(
        run.x, rec[:, iy, -1], color="#0072b2", lw=1.6, ls="--",
        label=f"fft2, |mx| <= {box['kx']}, |my| <= {box['ky']}",
    )
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("dT [K]")
    ax1.set_title(f"along the scan line  (y = {run.y[iy]:.1f} mm, top face)", fontsize=10)
    ax1.legend(fontsize=8)

    ax2.plot(dT[ix, iy], run.z, color="0.25", lw=2.4, label="solver")
    ax2.plot(z_fft[ix, iy], run.z, color="#d55e00", lw=1.6, label="z by FFT,  |mz| <= 6")
    ax2.plot(z_dct[ix, iy], run.z, color="#009e73", lw=1.6, label="z by DCT-II,  j <= 12")
    ax2.axvline(0, color="0.75", lw=0.8)
    ax2.axhline(0, color="0.75", lw=0.8)
    ax2.set_xlabel("dT [K]")
    ax2.set_ylabel("z [mm]   (0 = substrate, 6 = laser)")
    ax2.set_title("down through the beam, on the same 13 of 49 coefficients", fontsize=10)
    ax2.legend(fontsize=8)

    for ax in (ax1, ax2):
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(label, fontsize=11)
    fig.savefig(out / "profiles.png", dpi=150)
    plt.close(fig)


def plot_pareto(run: Run, acc: dict, out: Path) -> None:
    """Stored degrees of freedom against the L2 error they leave behind.

    The error plotted is the worst power's, not the sweep's, for the reason given
    in ``accumulate``.
    """
    nt, nx, ny, nz = run.shape
    full = nx * ny * nz
    N = full * nt  # points behind one power's RMSE
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")

    targets = 1 - np.logspace(-1, -6, 40)
    for v in VARIANTS:
        pts = []
        for tgt in targets:
            b = best_box(acc["energy"][v], acc["total"], v, nz, tgt)
            if b is None:
                continue
            lost = (acc["total"] * (1 - b["per_power"])).max()
            pts.append((b["dof"] / full, np.sqrt(lost / N)))
        if pts:
            c, e = np.array(pts).T
            ax.loglog(c, e, "o-", ms=3, label=v)

    ax.set_xlabel("stored coefficients / grid points")
    ax.set_ylabel("truncation RMSE of the worst power  [K]")
    ax.set_title("Cost of a given accuracy, three ways to transform")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.savefig(out / "pareto.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #


REPO = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run", type=Path, required=True,
        help="a solver run under data/, e.g. data/20260710_132221_powersweep_gpu",
    )
    ap.add_argument(
        "--figdir", type=Path, default=None,
        help="default: reports/<stamp>_spectral-mode-budget[-detrended]/ -- these are what "
             "justifies the mode budget, so they are kept where they can be read, not under "
             "the ignored data/. See reports/README.md.",
    )
    ap.add_argument(
        "--targets", type=float, nargs="+", default=[0.99, 0.999, 0.9999, 0.99999]
    )
    ap.add_argument(
        "--no-save", action="store_true", help="analyse only; do not write the npz"
    )
    ap.add_argument(
        "--detrend",
        action="store_true",
        help="take the x-wrap ramp out before transforming, and store it alongside",
    )
    ap.add_argument(
        "--save-target",
        type=float,
        default=0.9999,
        help=(
            "energy the saved box must retain, for every power (default 0.9999; "
            "the tighter 0.99999 lowers the truncation floor further but roughly "
            "doubles the stored coefficients -- which are ~99%% of a spectral "
            "model's weights -- for little end-to-end gain, so it is not the default)"
        ),
    )
    a = ap.parse_args(argv)

    run = load_run(a.run)
    a.save = None if a.no_save else run.dir / npz_name(a.save_target, a.detrend)
    if a.figdir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        suffix = "-detrended" if a.detrend else ""
        a.figdir = REPO / "reports" / f"{stamp}_spectral-mode-budget{suffix}"
    nt, nx, ny, nz = run.shape
    d = float(run.x[1] - run.x[0])
    print(
        f"grid {nx}x{ny}x{nz} = {nx * ny * nz} nodes, {nt} snapshots, "
        f"{len(run.powers)} powers, spacing {d} mm"
    )
    print(f"implied FFT periods: {nx * d:.3f} x {ny * d:.3f} x {nz * d:.3f} mm\n")

    acc = accumulate(run, a.detrend)

    full = nx * ny * nz
    cache: dict[tuple[int, int], np.ndarray] = {}
    hdr = (
        f"{'target':>8} {'variant':>6} {'kx':>4} {'ky':>4} {'kz':>4} "
        f"{'dof':>9} {'x fewer':>8} | {'hardest snapshot':>16} "
        f"{'RMSE':>8} {'peak':>9} {'Linf':>8} {'bottom':>8}"
    )
    print("\nBudget is set by the *worst* power, and errors are quoted on the")
    print("snapshot that keeps least of its own energy inside the box.\n")
    print(hdr)
    print("-" * len(hdr))
    for tgt in a.targets:
        for v in VARIANTS:
            b = best_box(acc["energy"][v], acc["total"], v, nz, tgt)
            if b is None:
                print(f"{tgt:>8.5f} {v:>6}   unreachable")
                continue
            ip, it = hardest_snapshot(acc, b["kx"], b["ky"])
            if (ip, it) not in cache:
                cache[(ip, it)] = field(run, ip, a.detrend)[0][it]
            e = reconstruct_error(cache[(ip, it)], v, b)
            where = f"{run.powers[ip]}W t={run.t[it]:.1f}s"
            print(
                f"{tgt:>8.5f} {v:>6} {b['kx']:>4} {b['ky']:>4} {b['kz']:>4} "
                f"{b['dof']:>9} {full / b['dof']:>7.1f}x | {where:>16} "
                f"{e['rmse']:>7.2f}K {e['peak_rel']:>8.2f}% {e['linf']:>7.1f}K "
                f"{e['bottom']:>7.1f}K"
            )
        print()

    box = best_box(acc["energy"]["fft2"], acc["total"], "fft2", nz, a.save_target)
    print(
        f"fft2 box for {a.save_target:.5f} of every power's energy: "
        f"kx = {box['kx']}, ky = {box['ky']}, z kept whole\n"
    )

    # the two ends of the sweep: the coldest run keeps the sharpest peak, because
    # radiation has not yet grown big enough to cap and broaden it
    lo, hi = 0, len(run.powers) - 1
    cases = [
        (run.dT(lo)[-1], f"{run.powers[lo]} W, t = {run.t[-1]:.1f} s"),
        (run.dT(hi)[-1], f"{run.powers[hi]} W, t = {run.t[-1]:.1f} s"),
    ]

    a.figdir.mkdir(parents=True, exist_ok=True)
    plot_marginals(run, acc, a.figdir)
    plot_pareto(run, acc, a.figdir)
    plot_kspace(run, box, cases, a.figdir)
    plot_reconstruction(run, box, cases, a.figdir)
    plot_profiles(run, box, *cases[1], a.figdir)
    print(f"figures -> {a.figdir}")

    if a.save:
        build(run, box, a.save, detrend=a.detrend, energy_target=a.save_target)


if __name__ == "__main__":
    main()
