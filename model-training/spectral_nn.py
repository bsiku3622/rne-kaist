"""Fit (P, t) -> spatial Fourier coefficients with a plain MLP.

The spectral dataset from ``spectral.py`` turns the surrogate problem inside out.
The DeepONet in ``train.py`` has to learn a function of five variables and is asked
for one temperature at a time; here the spatial dependence is already carried by
the basis, so all that is left to learn is how 36,701 coefficients move as the
laser power and the clock change. That is a map from R^2, and an MLP is enough.

Two things make the evaluation honest.

**There is a floor.** Even a network that predicted every stored coefficient
exactly would still be wrong by the truncation error, because the box threw the
rest of the spectrum away. So the model is scored against the field the *true*
coefficients reconstruct, not just against the solver -- the gap to that floor is
the only part the network is responsible for.

**One power is held out.** Interpolating in P is the whole point of a surrogate,
so 400 W never appears in training and everything reported for it is extrapolation
from its neighbours.

The ``--norm`` flag is the experiment worth running twice. Under ``global`` the
coefficients keep their relative sizes, so by Parseval the loss *is* the field's
L2 error -- physically the right thing to minimise, but the low modes carry almost
all of it and the network can ignore the rest. Under ``per-coef`` every coefficient
is standardised to unit variance, so the tiny high-frequency ones count as much as
the DC term. Which one wins is not obvious in advance.

Example::

    python spectral_nn.py --data data-spectral/fft2_powersweep.npz \
        --raw ../simulation/data/20260710_132221_powersweep_gpu
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_out: int, width: int = 128, depth: int = 3):
        super().__init__()
        layers, d = [], 2  # (P, t)
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers += [nn.Linear(d, n_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def reconstruct(coef: np.ndarray, meta: dict) -> np.ndarray:
    """Coefficients -> dT on the grid. Inverse of what spectral.py stored."""
    nx, ny, nz = meta["grid"]
    mx, my = meta["mx"], meta["my"]
    nt = coef.shape[0]
    full = np.zeros((nt, nx, ny // 2 + 1, nz), dtype=np.complex128)
    full[:, mx[:, None], my[None, :], :] = coef
    return np.fft.irfftn(full, s=(nx, ny), axes=(1, 2), norm="ortho")


def score(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Field errors in Kelvin, against whatever field is passed as the truth."""
    nt = truth.shape[0]
    err = pred - truth
    peak_t = truth.reshape(nt, -1).max(1)
    hot = peak_t > 1.0  # t = 0 is uniformly ambient and has no peak
    rel = (pred.reshape(nt, -1).max(1)[hot] - peak_t[hot]) / peak_t[hot]
    return {
        "rmse": float(np.sqrt((err**2).mean())),
        "linf": float(np.abs(err).max()),
        "peak": float(rel[np.abs(rel).argmax()] * 100),
    }


def plot_planes(truth, mlp, meta, times, p, out: Path, label: str = "MLP") -> None:
    """The two planes ``visualize.py`` uses, for the power the network never saw.

    ``top`` is the face the laser scans, over (x, y); ``track`` cuts along the scan
    line and shows (x, z), which is where the melt pool depth lives. Truth and
    prediction share a colour scale so they can be compared by eye; the error panel
    is symmetric about zero, so blue is under-prediction and red is over.
    """
    x, y, z = meta["x"], meta["y"], meta["z"]
    iy = len(y) // 2  # the scan line runs down y = y_c
    j = -1  # last snapshot

    planes = (
        ("top   z = %.0f mm" % z[-1], truth[j, :, :, -1], mlp[j, :, :, -1],
         [0, float(x[-1]), 0, float(y[-1])], "y [mm]"),
        ("track  y = %.0f mm" % y[iy], truth[j, :, iy, :], mlp[j, :, iy, :],
         [0, float(x[-1]), 0, float(z[-1])], "z [mm]"),
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 5.2), layout="constrained")
    for row, (name, tr, pr, ext, ylab) in zip(axes, planes):
        err = pr - tr
        hi = float(tr.max())
        bound = float(np.abs(err).max()) or 1.0
        style = dict(origin="lower", extent=ext, aspect="auto")

        a = row[0].imshow(tr.T, **style, vmin=0, vmax=hi, cmap="inferno")
        row[1].imshow(pr.T, **style, vmin=0, vmax=hi, cmap="inferno")
        b = row[2].imshow(err.T, **style, vmin=-bound, vmax=bound, cmap="RdBu_r")
        fig.colorbar(a, ax=row[1], label="dT [K]", fraction=0.04, pad=0.02)
        fig.colorbar(b, ax=row[2], label="error [K]", fraction=0.04, pad=0.02)

        row[0].set_title(f"{name}   solver", fontsize=9)
        row[1].set_title(f"{label} prediction", fontsize=9)
        row[2].set_title(
            f"{label} - solver   (max |err| {bound:.1f} K)", fontsize=9
        )
        for ax in row:
            ax.set_xlabel("x [mm]", fontsize=8)
            ax.set_ylabel(ylab, fontsize=8)
            ax.tick_params(labelsize=8)

    fig.suptitle(
        f"{p} W -- never trained on -- at t = {times[j]:.1f} s", fontsize=11
    )
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_scanline(
    truth, floor, mlp, meta, times, want, p, out: Path, label: str = "spectral MLP"
) -> None:
    """Two cuts through the beam on the scanned face, with the signed error beside each.

    The left pair runs along the scan line -- the ``y = y_c`` cut ``visualize.py``
    calls ``scanline`` -- so the melt pool's height and the steepness of its leading
    edge can be read off directly. The right pair cuts across the track at the beam
    centre, which is where the pool's width lives.

    Each error panel also carries the floor: the error the truncation alone commits,
    before the network has done anything. The gap between the two curves is the only
    part the network is answerable for. A model that predicts the field directly has
    no truncation to answer for, so it passes ``floor=None`` and that curve is dropped.
    """
    x, y = meta["x"], meta["y"]
    iy = len(y) // 2  # the scan line, y = y_c
    rows = [int(np.argmin(np.abs(times - t))) for t in want]

    fig, axes = plt.subplots(
        len(rows), 4, figsize=(19, 3.1 * len(rows)), squeeze=False, layout="constrained"
    )
    print(f"{p} W, held out -- cuts on the scanned face\n")

    for r, j in enumerate(rows):
        ix = int(np.argmax(truth[j, :, iy, -1])) if truth[j].max() > 1 else len(x) // 2
        cuts = (
            ("along the scan line   y = %.1f mm" % y[iy], x, "x [mm]",
             truth[j, :, iy, -1], None if floor is None else floor[j, :, iy, -1],
             mlp[j, :, iy, -1]),
            ("across the track   x = %.2f mm" % x[ix], y, "y [mm]",
             truth[j, ix, :, -1], None if floor is None else floor[j, ix, :, -1],
             mlp[j, ix, :, -1]),
        )
        for c, (name, axis, xlab, tr, fl, ml) in enumerate(cuts):
            prof, errax = axes[r][2 * c], axes[r][2 * c + 1]
            rmse = float(np.sqrt(((ml - tr) ** 2).mean()))

            prof.plot(axis, tr, color="#2a78d6", lw=1.8, label="simulation")
            prof.plot(axis, ml, color="#eb6834", lw=1.8, ls="--", label=label)
            prof.set_ylabel("dT [K]")
            prof.set_title(f"{name}\nt = {times[j]:.1f} s   RMSE {rmse:.2f} K", fontsize=9)
            if r == 0 and c == 0:
                prof.legend(frameon=False, fontsize=8)

            errax.axhline(0.0, color="#52514e", lw=0.8)
            errax.plot(axis, ml - tr, color="#52514e", lw=1.5, label=f"{label} - simulation")
            title = f"max |error|  {np.abs(ml - tr).max():.1f} K"
            if fl is not None:
                errax.plot(
                    axis, fl - tr, color="#009e73", lw=1.2, ls="--",
                    label="floor: truncation alone",
                )
                title += f"   (floor {np.abs(fl - tr).max():.1f} K)"
            errax.set_ylabel("error [K]")
            errax.set_title(title, fontsize=9)
            if r == 0 and c == 0:
                errax.legend(frameon=False, fontsize=8)

            for ax in (prof, errax):
                ax.set_xlabel(xlab, fontsize=8)
                ax.grid(alpha=0.2)
                ax.set_axisbelow(True)

        print(
            f"  t = {times[j]:4.1f}s   x-cut RMSE {np.sqrt(((mlp[j, :, iy, -1] - truth[j, :, iy, -1]) ** 2).mean()):7.3f} K"
            f"   peak {truth[j, :, iy, -1].max():7.1f} -> {mlp[j, :, iy, -1].max():7.1f} K"
            f"  ({100 * (mlp[j, :, iy, -1].max() - truth[j, :, iy, -1].max()) / truth[j, :, iy, -1].max():+.2f}%)"
        )

    fig.suptitle(f"{p} W -- never trained on", fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print()


def plot_signal(true_c, pred_c, meta, p, out: Path) -> None:
    """The coefficients themselves -- what the network is actually asked to output.

    The field plots show whether the answer came out right; these show *which modes*
    it got right. A model that is only fitting the energetic low modes and giving up
    on the rest looks fine in a colour map and shows up here immediately: the two
    spectra separate, and the scatter fans out as |mx| grows.
    """
    j = -1  # last snapshot
    ct, cp = true_c[j], pred_c[j]
    mx = np.abs(meta["mx"])

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 4.4), layout="constrained")

    # A. energy per |mx|, truth against prediction
    nb = mx.max() + 1
    et = np.bincount(mx, (np.abs(ct) ** 2).sum(axis=(1, 2)), nb)
    ep = np.bincount(mx, (np.abs(cp) ** 2).sum(axis=(1, 2)), nb)
    k = np.arange(nb)
    ax0.semilogy(k, et, color="0.25", lw=2.6, label="solver")
    ax0.semilogy(k, ep, color="#d55e00", lw=1.6, ls="--", label="MLP")
    ax0.set_xlabel("|mx|")
    ax0.set_ylabel("energy in the mode")
    ax0.set_title("Where the energy sits, per mode", fontsize=10)
    ax0.legend(fontsize=8)

    # B. every stored coefficient, predicted against true
    rep = np.repeat(mx, ct.shape[1] * ct.shape[2])
    t_all = np.concatenate([ct.real.ravel(), ct.imag.ravel()])
    p_all = np.concatenate([cp.real.ravel(), cp.imag.ravel()])
    c_all = np.concatenate([rep, rep])
    s = ax1.scatter(t_all, p_all, c=c_all, s=2, alpha=0.3, cmap="viridis", linewidths=0)
    lim = float(np.abs(t_all).max()) * 1.05
    ax1.plot([-lim, lim], [-lim, lim], color="0.4", lw=1, ls=":", zorder=0)
    ax1.set_xscale("symlog", linthresh=1.0)
    ax1.set_yscale("symlog", linthresh=1.0)
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_xlabel("solver coefficient  (Re and Im)")
    ax1.set_ylabel("MLP coefficient")
    ax1.set_title("Every coefficient, predicted vs true", fontsize=10)
    fig.colorbar(s, ax=ax1, label="|mx|")

    # C. how much of each mode's amplitude the network actually captured
    num = np.bincount(mx, (np.abs(cp - ct) ** 2).sum(axis=(1, 2)), nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.sqrt(np.where(et > 0, num / et, np.nan))
    ax2.semilogy(k, rel, color="#0072b2", lw=1.8)
    ax2.set_ylim(top=4.0)
    ax2.axhline(1.0, color="#d55e00", lw=1.2, ls="--")
    ax2.text(1, 1.25, "error as big as the mode itself", color="#d55e00", fontsize=7.5)

    # the snapshot interval cannot resolve a mode spinning faster than its Nyquist
    # rate; above this |mx| the coefficient's time series is aliased, and no amount
    # of training recovers what the sampling threw away
    Lx = float(meta["grid"][0]) * float(meta["x"][1] - meta["x"][0])
    m_alias = 0.5 / (float(meta["dt"]) * 10.0 / Lx)
    ax2.axvline(m_alias, color="0.4", lw=1.2, ls=":")
    ax2.text(
        m_alias + 1, 0.02, f"temporal Nyquist\n|mx| = {m_alias:.0f}",
        color="0.35", fontsize=7.5,
    )
    ax2.set_xlabel("|mx|")
    ax2.set_ylabel("relative L2 error of the mode")
    ax2.set_title("Which modes the network got wrong", fontsize=10)

    for ax in (ax0, ax1, ax2):
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(f"{p} W -- never trained on -- coefficients at the last snapshot", fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True, help="npz from spectral.py")
    ap.add_argument("--raw", type=Path, required=True, help="solver run dir, for the truth")
    ap.add_argument("--holdout", type=float, default=400.0, help="power kept out of training")
    ap.add_argument("--norm", choices=("global", "per-coef"), default="global")
    ap.add_argument(
        "--derotate",
        action="store_true",
        help="learn the coefficients in the frame that moves with the laser",
    )
    ap.add_argument("--vel", type=float, default=10.0, help="scan speed, mm/s (VEL)")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--figdir", type=Path, default=Path("figures/spectral"))
    ap.add_argument(
        "--times", type=float, nargs="+", default=[0.5, 1.5, 3.0],
        help="snapshots to cut through, one row each",
    )
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(a.data)
    meta = {k: d[k] for k in ("mx", "my", "grid", "z", "x", "y")}
    coef, powers, times = d["coef"], d["powers"], d["t"]
    meta["dt"] = float(times[1] - times[0])  # snapshot interval, for the Nyquist mark
    nP, nt = coef.shape[:2]
    shape = coef.shape[2:]

    # A source moving at v makes the field roughly f(x - v t), whose transform is
    # g(kx) * exp(-2i pi kx v t): every coefficient spins in the complex plane, and
    # the faster the mode, the faster it spins. At mx = 21 that spin already passes
    # the Nyquist rate of the 0.1 s snapshot interval, so above it the sampled time
    # series is aliased and no smooth model can fit it. Undoing the spin analytically
    # -- a change to the frame that travels with the laser -- leaves an amplitude that
    # barely moves in time. It is exact, and it is undone again before scoring.
    Lx = float(meta["grid"][0]) * float(meta["x"][1] - meta["x"][0])
    spin = np.exp(
        2j * np.pi * (meta["mx"] / Lx) * a.vel * times[:, None]
    )  # (nt, len(mx))
    work = coef * spin[None, :, :, None, None] if a.derotate else coef

    # (P, t) -> [Re, Im] of every stored coefficient
    X = np.stack(
        np.meshgrid(powers / powers.max(), times / times.max(), indexing="ij"), -1
    ).reshape(-1, 2)
    Y = np.stack([work.real, work.imag], -1).reshape(nP * nt, -1).astype(np.float64)

    test_p = int(np.argmin(np.abs(powers - a.holdout)))
    is_test = np.zeros(nP, dtype=bool)
    is_test[test_p] = True
    tr = np.repeat(~is_test, nt)
    te = np.repeat(is_test, nt)

    if a.norm == "global":
        # a single scale keeps the coefficients' relative sizes, so by Parseval
        # the loss below is the field's L2 error, up to that constant
        mu, sd = 0.0, Y[tr].std()
    else:
        mu, sd = Y[tr].mean(0), Y[tr].std(0) + 1e-12
    Yn = (Y - mu) / sd

    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    Yt = torch.tensor(Yn, dtype=torch.float32, device=dev)
    trt = torch.tensor(tr, device=dev)

    torch.manual_seed(0)
    model = MLP(Y.shape[1], a.width, a.depth).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    n_par = sum(p.numel() for p in model.parameters())

    print(
        f"MLP 2 -> {' -> '.join([str(a.width)] * a.depth)} -> {Y.shape[1]}   "
        f"{n_par / 1e6:.1f}M params\n"
        f"{int(tr.sum())} train samples ({', '.join(f'{int(p)}W' for p in powers[~is_test])}), "
        f"{int(te.sum())} held out ({int(powers[test_p])}W), norm = {a.norm}\n"
    )

    hist = []
    t0 = time.time()
    for ep in range(a.epochs):
        opt.zero_grad(set_to_none=True)
        loss = ((model(Xt[trt]) - Yt[trt]) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
        if ep % 500 == 0 or ep == a.epochs - 1:
            with torch.no_grad():
                vl = ((model(Xt[~trt]) - Yt[~trt]) ** 2).mean().item()
            hist.append((ep, loss.item(), vl))
            if ep % 4000 == 0 or ep == a.epochs - 1:
                print(f"  epoch {ep:>6}  train {loss.item():.3e}  held-out {vl:.3e}")
    print(f"  {time.time() - t0:.0f} s\n")

    with torch.no_grad():
        P = (model(Xt).cpu().numpy().astype(np.float64) * sd + mu).reshape(
            nP, nt, *shape, 2
        )
    pred = P[..., 0] + 1j * P[..., 1]
    if a.derotate:
        pred = pred / spin[None, :, :, None, None]  # back to the lab frame

    hdr = f"{'P':>6} {'':>10} {'RMSE':>9} {'Linf':>8} {'peak':>9}"
    print("Against the solver.  'floor' is what the kept modes can do at best;")
    print("'MLP' is what the network actually got.\n")
    print(hdr)
    print("-" * len(hdr))
    for i, p in enumerate(powers):
        rows = np.load(a.raw / f"data_{int(p)}W.npy", mmap_mode="r")
        truth = (np.asarray(rows[:, 5]) - 298.0).reshape(nt, *meta["grid"])
        floor = reconstruct(coef[i], meta)
        mlp = reconstruct(pred[i], meta)
        tag = "HELD OUT" if i == test_p else ""
        for name, f in (("floor", floor), ("MLP", mlp)):
            s = score(f, truth)
            print(
                f"{int(p):>5}W {name:>10} {s['rmse']:>8.3f}K {s['linf']:>7.1f}K "
                f"{s['peak']:>8.2f}%   {tag if name == 'MLP' else ''}"
            )
        if i == test_p:
            held = (truth, floor, mlp, int(p))
    print()

    # where in time the error actually sits, on the power it never saw
    truth, floor, mlp, p = held
    peak_t = truth.reshape(nt, -1).max(1)
    rmse_t = np.sqrt(((mlp - truth) ** 2).reshape(nt, -1).mean(1))
    floor_t = np.sqrt(((floor - truth) ** 2).reshape(nt, -1).mean(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_t = np.where(
            peak_t > 1, (mlp.reshape(nt, -1).max(1) - peak_t) / np.maximum(peak_t, 1), 0.0
        )
    print(f"{p} W, held out -- where the error is, in time\n")
    print(f"{'t [s]':>6} {'peak [K]':>10} {'floor RMSE':>11} {'MLP RMSE':>10} {'MLP peak':>10}")
    for j in range(0, nt, 3):
        print(
            f"{times[j]:>6.1f} {peak_t[j]:>10.1f} {floor_t[j]:>10.3f}K "
            f"{rmse_t[j]:>9.3f}K {100 * rel_t[j]:>9.2f}%"
        )
    print()

    a.figdir.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax2, ax1) = plt.subplots(1, 3, figsize=(16, 4.2), layout="constrained")

    ax2.semilogy(times, np.maximum(floor_t, 1e-3), color="#009e73", lw=1.8, label="floor")
    ax2.semilogy(times, np.maximum(rmse_t, 1e-3), color="#d55e00", lw=1.8, label="MLP")
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("field RMSE [K]")
    ax2.set_title(f"{p} W: when is the MLP wrong?", fontsize=10)
    ax2.legend(fontsize=8)

    ep, tl, vl = np.array(hist).T
    ax0.semilogy(ep, tl, color="#0072b2", lw=1.6, label="train (6 powers)")
    ax0.semilogy(ep, vl, color="#d55e00", lw=1.6, label=f"held out ({p} W)")
    ax0.set_xlabel("epoch")
    ax0.set_ylabel(f"MSE on {a.norm}-normalised coefficients")
    ax0.set_title("Learning curve", fontsize=10)
    ax0.legend(fontsize=8)

    iy = meta["grid"][1] // 2
    ax1.plot(meta["x"], truth[-1, :, iy, -1], color="0.25", lw=2.6, label="solver")
    ax1.plot(
        meta["x"], floor[-1, :, iy, -1], color="#009e73", lw=1.6,
        label="floor: the kept modes, exactly",
    )
    ax1.plot(
        meta["x"], mlp[-1, :, iy, -1], color="#d55e00", lw=1.6, ls="--",
        label="MLP prediction",
    )
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("dT [K]")
    ax1.set_title(
        f"{p} W (never trained on), t = 3.0 s, along the scan line", fontsize=10
    )
    ax1.legend(fontsize=8)
    for ax in (ax0, ax1, ax2):
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
    tag = f"{a.norm}{'_derotated' if a.derotate else ''}"
    fig.savefig(a.figdir / f"nn_{tag}.png", dpi=150)
    plt.close(fig)

    plot_planes(truth, mlp, meta, times, p, a.figdir / f"nn_{tag}_field.png")
    plot_signal(coef[test_p], pred[test_p], meta, p, a.figdir / f"nn_{tag}_signal.png")
    plot_scanline(
        truth, floor, mlp, meta, times, a.times, p, a.figdir / f"nn_{tag}_scanline.png"
    )
    for name in ("", "_field", "_signal", "_scanline"):
        print(f"figure -> {a.figdir / f'nn_{tag}{name}.png'}")


if __name__ == "__main__":
    main()
