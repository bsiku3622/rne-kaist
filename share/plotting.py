"""The figures every model is judged by, and the colour rules they obey.

Three views, because each hides what the others show.

:func:`planes` is the colour map -- it shows *where* on the plate the error is,
but a 20 K error on a 3000 K scale is invisible in colour, so it cannot tell you
how big.

:func:`scanline` cuts a line through the beam and plots the profile against the
signed error beside it. This is where a flattened peak or a shifted pool shows up,
because the error is drawn on its own axis with its own scale.

:func:`signal` leaves the field entirely and plots the Fourier coefficients. The
first two say whether the answer came out right; this one says *which modes* the
model got right, which is the only way to see a model that is fitting the
energetic low modes and giving up on the rest.

Colour follows what the field is, not what looks nice. Temperature is a magnitude,
so it gets a single-hue sequential ramp (``inferno``); signed error is a polarity
about zero, so it gets a diverging pair with a neutral midpoint (``RdBu_r``, blue
under, red over). Lines use Okabe-Ito, which passes CVD separation, the lightness
band, and 3:1 contrast against the surface.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIELD_CMAP = "inferno"  # sequential: temperature is a magnitude
ERROR_CMAP = "RdBu_r"  # diverging about zero: blue under-predicts, red over
SPECTRUM_CMAP = "viridis"  # sequential, and distinct from the temperature ramp

TRUTH = "#2a78d6"  # what visualize.py has always used for the simulation
PREDICTION = "#eb6834"
INK = "#52514e"
REFERENCE = "#009e73"  # Okabe-Ito green, for the floor / baseline curve


def _gaussian_fit(coord: np.ndarray, rise: np.ndarray, window: float) -> dict | None:
    """Least-squares ``A*exp(-2(x-c)^2/w^2)`` around a profile's peak, or None.

    Fits the temperature *rise* (``dT``, which decays to zero away from the beam,
    so no ambient offset is needed) within ``window`` of the peak. The trailing
    thermal wake is not Gaussian, so fitting the whole line would let the tail set
    the width; the window measures the peak itself. Truth and prediction get the
    same treatment, so the width they report is comparable. Ported from
    ``typeulli-model-training/scanline.py``; ``w`` is the 1/e^2 radius.
    """
    from scipy.optimize import curve_fit

    def model(x: np.ndarray, amplitude: float, centre: float, width: float) -> np.ndarray:
        return amplitude * np.exp(-2.0 * (x - centre) ** 2 / width**2)

    peak = int(np.argmax(rise))
    if float(rise[peak]) <= 1.0:  # t = 0 and cold cuts have no peak to fit
        return None
    selected = np.abs(coord - coord[peak]) <= window
    if int(selected.sum()) < 4:  # three parameters need at least four points
        return None
    try:
        (amplitude, centre, width), _ = curve_fit(
            model, coord[selected], rise[selected],
            p0=(float(rise[peak]), float(coord[peak]), window / 2.5), maxfev=10000,
        )
    except (RuntimeError, ValueError):
        return None
    return {"amplitude": float(amplitude), "centre": float(centre), "width": abs(float(width))}


def planes(truth, pred, run, power, out: Path, label: str = "model", snap: int = -1) -> None:
    """The scanned face and the depth section, at one snapshot: truth, model, error."""
    x, y, z = run.x, run.y, run.z
    iy = len(y) // 2  # the scan line runs down the middle of the plate

    views = (
        (f"top   z = {z[-1]:.0f} mm", truth[snap, :, :, -1], pred[snap, :, :, -1],
         [0, float(x[-1]), 0, float(y[-1])], "y [mm]"),
        (f"track  y = {y[iy]:.0f} mm", truth[snap, :, iy, :], pred[snap, :, iy, :],
         [0, float(x[-1]), 0, float(z[-1])], "z [mm]"),
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 5.2), layout="constrained")
    for row, (name, tr, pr, ext, ylab) in zip(axes, views):
        err = pr - tr
        bound = float(np.abs(err).max()) or 1.0
        style = dict(origin="lower", extent=ext, aspect="auto")

        a = row[0].imshow(tr.T, **style, vmin=0, vmax=float(tr.max()), cmap=FIELD_CMAP)
        row[1].imshow(pr.T, **style, vmin=0, vmax=float(tr.max()), cmap=FIELD_CMAP)
        b = row[2].imshow(err.T, **style, vmin=-bound, vmax=bound, cmap=ERROR_CMAP)
        fig.colorbar(a, ax=row[1], label="dT [K]", fraction=0.04, pad=0.02)
        fig.colorbar(b, ax=row[2], label="error [K]", fraction=0.04, pad=0.02)

        row[0].set_title(f"{name}   solver", fontsize=9)
        row[1].set_title(f"{label} prediction", fontsize=9)
        row[2].set_title(f"{label} - solver   (max |err| {bound:.1f} K)", fontsize=9)
        for ax in row:
            ax.set_xlabel("x [mm]", fontsize=8)
            ax.set_ylabel(ylab, fontsize=8)
            ax.tick_params(labelsize=8)

    fig.suptitle(f"{power} W  --  t = {run.t[snap]:.1f} s", fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def scanline(
    truth, pred, run, power, want, out: Path, label: str = "model", floor=None,
    gaussian: float | None = None,
) -> None:
    """Two cuts through the beam, each with its signed error beside it.

    The left pair runs along the scan line, so the pool's height and the steepness
    of its leading edge read off directly; the right pair cuts across the track at
    the beam centre, which is where its width lives.

    ``floor`` is an optional second field -- for a model that predicts truncated
    Fourier coefficients it is what those coefficients reconstruct *exactly*, so the
    gap between the two error curves is the only part the network is answerable
    for. A model that predicts the field directly has no such floor and passes None.

    ``gaussian`` is an optional fit half-window in mm. When given, a 1/e^2 Gaussian
    is fitted to each profile's peak (truth and prediction alike) and overlaid
    dotted, so a flattened or widened pool is measured rather than eyeballed; the
    fitted radii go in the panel title. Off by default -- the fit costs a
    ``scipy.optimize`` import and only means anything on the cuts that hit the beam.
    """
    x, y = run.x, run.y
    iy = len(y) // 2
    rows = [int(np.argmin(np.abs(run.t - w))) for w in want]

    fig, axes = plt.subplots(
        len(rows), 4, figsize=(19, 3.1 * len(rows)), squeeze=False, layout="constrained"
    )
    for r, j in enumerate(rows):
        ix = int(np.argmax(truth[j, :, iy, -1])) if truth[j].max() > 1 else len(x) // 2
        cuts = (
            (f"along the scan line   y = {y[iy]:.1f} mm", x, "x [mm]",
             truth[j, :, iy, -1], pred[j, :, iy, -1],
             None if floor is None else floor[j, :, iy, -1]),
            (f"across the track   x = {x[ix]:.2f} mm", y, "y [mm]",
             truth[j, ix, :, -1], pred[j, ix, :, -1],
             None if floor is None else floor[j, ix, :, -1]),
        )
        for c, (name, axis, xlab, tr, pr, fl) in enumerate(cuts):
            prof, errax = axes[r][2 * c], axes[r][2 * c + 1]
            rmse = float(np.sqrt(((pr - tr) ** 2).mean()))

            prof.plot(axis, tr, color=TRUTH, lw=1.8, label="simulation")
            prof.plot(axis, pr, color=PREDICTION, lw=1.8, ls="--", label=label)
            prof.set_ylabel("dT [K]")

            title = f"{name}\nt = {run.t[j]:.1f} s   RMSE {rmse:.2f} K"
            if gaussian is not None:
                dense = np.linspace(float(axis[0]), float(axis[-1]), 400)
                widths = []
                for series, colour in ((tr, TRUTH), (pr, PREDICTION)):
                    fit = _gaussian_fit(axis, series, gaussian)
                    if fit is None:
                        widths.append(None)
                        continue
                    inside = np.abs(dense - fit["centre"]) <= gaussian
                    prof.plot(dense[inside],
                              fit["amplitude"] * np.exp(-2.0 * (dense[inside] - fit["centre"]) ** 2 / fit["width"] ** 2),
                              color=colour, lw=1.0, ls=":",
                              label="gaussian fit" if (r == 0 and c == 0 and series is tr) else None)
                    widths.append(fit["width"])
                if widths[0] is not None and widths[1] is not None:
                    title += f"   w {widths[0]:.2f} -> {widths[1]:.2f} mm"
            prof.set_title(title, fontsize=9)
            if r == 0 and c == 0:
                prof.legend(frameon=False, fontsize=8)

            errax.axhline(0.0, color=INK, lw=0.8)
            errax.plot(axis, pr - tr, color=INK, lw=1.5, label=f"{label} - simulation")
            title = f"max |error|  {np.abs(pr - tr).max():.1f} K"
            if fl is not None:
                errax.plot(axis, fl - tr, color=REFERENCE, lw=1.2, ls="--",
                           label="floor: truncation alone")
                title += f"   (floor {np.abs(fl - tr).max():.1f} K)"
            errax.set_ylabel("error [K]")
            errax.set_title(title, fontsize=9)
            if r == 0 and c == 0:
                errax.legend(frameon=False, fontsize=8)

            for ax in (prof, errax):
                ax.set_xlabel(xlab, fontsize=8)
                ax.grid(alpha=0.2)
                ax.set_axisbelow(True)

    fig.suptitle(f"{power} W  --  never trained on", fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def signal(true_c, pred_c, mx, power, out: Path, nyquist: float | None = None) -> None:
    """The Fourier coefficients themselves: which modes the model got, and which it lost.

    ``nyquist`` marks where the *raw* coefficient time series becomes aliased: a
    moving source spins every coefficient at ``kx * v``, and past this mode that spin
    is faster than the snapshot interval can sample. It is why the spin has to be
    divided out analytically rather than left for the network to discover.

    It is **not** a ceiling on what can be learned, and the line should not be read as
    one. De-rotation evaluates the phase at the exact sample times, so it does not have
    to infer anything from the samples and the aliasing costs it nothing: a run with
    this mark at |mx| = 5 learns every mode out to 38 with under 14% error. Where the
    high modes *do* fail, the cause is that they belong to the stationary x-wrap
    artefact rather than to the travelling pool, and de-rotating those injects a spin
    instead of removing one.
    """
    j = -1
    ct, cp = true_c[j], pred_c[j]
    m = np.abs(mx)
    nb = int(m.max()) + 1
    k = np.arange(nb)

    et = np.bincount(m, (np.abs(ct) ** 2).sum(axis=(1, 2)), nb)
    ep = np.bincount(m, (np.abs(cp) ** 2).sum(axis=(1, 2)), nb)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 4.4), layout="constrained")

    ax0.semilogy(k, et, color=INK, lw=2.6, label="solver")
    ax0.semilogy(k, ep, color=PREDICTION, lw=1.6, ls="--", label="model")
    ax0.set_xlabel("|mx|")
    ax0.set_ylabel("energy in the mode")
    ax0.set_title("Where the energy sits, per mode", fontsize=10)
    ax0.legend(fontsize=8)

    rep = np.repeat(m, ct.shape[1] * ct.shape[2])
    t_all = np.concatenate([ct.real.ravel(), ct.imag.ravel()])
    p_all = np.concatenate([cp.real.ravel(), cp.imag.ravel()])
    s = ax1.scatter(t_all, p_all, c=np.concatenate([rep, rep]), s=2, alpha=0.3,
                    cmap=SPECTRUM_CMAP, linewidths=0)
    lim = float(np.abs(t_all).max()) * 1.05
    ax1.plot([-lim, lim], [-lim, lim], color="0.4", lw=1, ls=":", zorder=0)
    ax1.set_xscale("symlog", linthresh=1.0)
    ax1.set_yscale("symlog", linthresh=1.0)
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_xlabel("solver coefficient  (Re and Im)")
    ax1.set_ylabel("model coefficient")
    ax1.set_title("Every coefficient, predicted vs true", fontsize=10)
    fig.colorbar(s, ax=ax1, label="|mx|")

    num = np.bincount(m, (np.abs(cp - ct) ** 2).sum(axis=(1, 2)), nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.sqrt(np.where(et > 0, num / et, np.nan))
    ax2.semilogy(k, rel, color=TRUTH, lw=1.8)
    ax2.set_ylim(top=4.0)
    ax2.axhline(1.0, color=PREDICTION, lw=1.2, ls="--")
    ax2.text(1, 1.25, "error as big as the mode itself", color=PREDICTION, fontsize=7.5)
    if nyquist is not None:
        ax2.axvline(nyquist, color="0.4", lw=1.2, ls=":")
        ax2.text(nyquist + 1, 0.02,
                 f"raw series aliased above |mx| = {nyquist:.0f}\n"
                 "(not a ceiling: de-rotation is analytic)",
                 color="0.35", fontsize=7.5)
    ax2.set_xlabel("|mx|")
    ax2.set_ylabel("relative L2 error of the mode")
    ax2.set_title("Which modes the model got wrong", fontsize=10)

    for ax in (ax0, ax1, ax2):
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(f"{power} W  --  coefficients at the last snapshot", fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)
