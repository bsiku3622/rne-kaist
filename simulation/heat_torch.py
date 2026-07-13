"""GPU finite-difference twin of heat_fenics.py.

Same PDE, same boundary conditions, same material model -- but on the uniform
BoxMesh grid the CG1 stencil collapses to a 7-point difference, so the whole
sweep runs as one batched tensor on the GPU.

    rho*Cp*dT/dt = k * laplacian(T)                     in the volume
    k * grad(T).n = q_laser - q_conv - q_rad            on the top face
    k * grad(T).n =        - q_conv - q_rad             on the four sides
    T = ambient                                          on the bottom face

Robin faces are imposed with ghost nodes, which keeps the interior stencil
second-order right up to the boundary:

    ghost = neighbour_inside + 2*dx*Q/k

Heun / predictor-corrector stepping gives second-order accuracy in time, but it
still has an explicit diffusion stability limit, so the caller is expected to
respect that (assert below).  Powers are stacked along the batch dimension, so
the seven runs advance together.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- geometry
DOMAIN = (40.0, 10.0, 6.0)  # mm
X0, Y0 = 5.0, 5.0  # toolpath origin, mm
R_BEAM = 1.5  # mm
VEL = 10.0  # mm/s
T_AMB = 298.0  # K
RBOLTZ = 5.6704e-14  # W/(mm^2 K^4)


def make_grid(ele, device, dtype):
    nx, ny, nz = (int(round(L / ele)) + 1 for L in DOMAIN)
    x = torch.linspace(0.0, DOMAIN[0], nx, device=device, dtype=dtype)
    y = torch.linspace(0.0, DOMAIN[1], ny, device=device, dtype=dtype)
    z = torch.linspace(0.0, DOMAIN[2], nz, device=device, dtype=dtype)
    return x, y, z


def surface_flux(T_face, k, h, eta, emiss, q_laser=None):
    """Q = k*grad(T).n_out, i.e. net flux *into* the body."""
    Q = -h * (T_face - T_AMB) - RBOLTZ * emiss * (T_face**4 - T_AMB**4)
    if q_laser is not None:
        Q = Q + q_laser
    return Q


def rhs_temperature(T, t, xg, yg, amp, alpha, k, h, eta, emiss, dx, dy, dz):
    """Return dT/dt for the finite-difference heat model at time t."""
    q_laser = amp * torch.exp(
        -2.0 * ((xg - X0 - VEL * t) ** 2 + (yg - Y0) ** 2) / R_BEAM**2
    )

    Tp = torch.empty(
        (T.shape[0], T.shape[1] + 2, T.shape[2] + 2, T.shape[3] + 2),
        device=T.device,
        dtype=T.dtype,
    )
    Tp[:, 1:-1, 1:-1, 1:-1] = T

    Qx0 = surface_flux(T[:, 0, :, :], k, h, eta, emiss)
    Qx1 = surface_flux(T[:, -1, :, :], k, h, eta, emiss)
    Tp[:, 0, 1:-1, 1:-1] = T[:, 1, :, :] + 2.0 * dx * Qx0 / k
    Tp[:, -1, 1:-1, 1:-1] = T[:, -2, :, :] + 2.0 * dx * Qx1 / k

    Qy0 = surface_flux(T[:, :, 0, :], k, h, eta, emiss)
    Qy1 = surface_flux(T[:, :, -1, :], k, h, eta, emiss)
    Tp[:, 1:-1, 0, 1:-1] = T[:, :, 1, :] + 2.0 * dy * Qy0 / k
    Tp[:, 1:-1, -1, 1:-1] = T[:, :, -2, :] + 2.0 * dy * Qy1 / k

    Qz1 = surface_flux(T[:, :, :, -1], k, h, eta, emiss, q_laser=q_laser)
    Tp[:, 1:-1, 1:-1, -1] = T[:, :, :, -2] + 2.0 * dz * Qz1 / k
    Tp[:, 1:-1, 1:-1, 0] = T[:, :, :, 1]  # unused: bottom is Dirichlet

    c = Tp[:, 1:-1, 1:-1, 1:-1]
    lap = (
        (Tp[:, 2:, 1:-1, 1:-1] + Tp[:, :-2, 1:-1, 1:-1] - 2.0 * c) / dx**2
        + (Tp[:, 1:-1, 2:, 1:-1] + Tp[:, 1:-1, :-2, 1:-1] - 2.0 * c) / dy**2
        + (Tp[:, 1:-1, 1:-1, 2:] + Tp[:, 1:-1, 1:-1, :-2] - 2.0 * c) / dz**2
    )

    dTdt = alpha * lap
    dTdt[:, :, :, 0] = 0.0
    return dTdt


def solve(
    powers,
    ele,
    dt,
    total_t,
    snap_dt,
    rho,
    Cp,
    k,
    h,
    eta,
    emiss,
    device="cuda",
    dtype=torch.float64,
    progress=True,
):
    alpha = k / (rho * Cp)
    dt_max = ele**2 / (6.0 * alpha)
    assert (
        dt < dt_max
    ), f"explicit diffusion stepping unstable: dt={dt:.3e} >= {dt_max:.3e}"

    x, y, z = make_grid(ele, device, dtype)
    nx, ny, nz = len(x), len(y), len(z)
    B = len(powers)
    P = torch.tensor(powers, device=device, dtype=dtype).view(B, 1, 1)

    T = torch.full((B, nx, ny, nz), T_AMB, device=device, dtype=dtype)

    # laser footprint depends on (x, y, t); precompute the spatial parts
    xg = x.view(nx, 1)
    yg = y.view(1, ny)
    amp = 2.0 * P * eta / (np.pi * R_BEAM**2)  # (B,1,1)

    nsteps = int(round(total_t / dt))
    snap_every = max(1, int(round(snap_dt / dt)))
    dx = dy = dz = ele

    snaps_T, snaps_t = [T.clone()], [0.0]
    t0 = time.time()

    for n in range(nsteps):
        t = n * dt
        k1 = rhs_temperature(T, t, xg, yg, amp, alpha, k, h, eta, emiss, dx, dy, dz)

        T_pred = T + dt * k1
        T_pred[:, :, :, 0] = T_AMB

        k2 = rhs_temperature(
            T_pred, t + dt, xg, yg, amp, alpha, k, h, eta, emiss, dx, dy, dz
        )

        T = T + 0.5 * dt * (k1 + k2)
        T[:, :, :, 0] = T_AMB  # Dirichlet bottom

        if (n + 1) % snap_every == 0:
            snaps_T.append(T.clone())
            snaps_t.append((n + 1) * dt)
            if progress:
                el = time.time() - t0
                done = (n + 1) / nsteps
                print(
                    f"\r{done*100:5.1f}%  step {n+1}/{nsteps}  "
                    f"t={(n+1)*dt:.2f}s  Tmax={T.max().item():7.1f}K  "
                    f"[{el:.0f}s elapsed, {el/done - el:.0f}s left]",
                    end="",
                    flush=True,
                )
    if progress:
        print()

    return x, y, z, torch.tensor(snaps_t, dtype=dtype), torch.stack(snaps_T, 1)


def to_rows(x, y, z, ts, Tsnap, power):
    """[x, y, z, t, P, T] rows for one power, matching the FEM npy layout."""
    nx, ny, nz = len(x), len(y), len(z)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    coords = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], 1)  # (N,3)
    N = coords.shape[0]
    out = []
    for j, t in enumerate(ts):
        blk = torch.empty((N, 6), dtype=coords.dtype)
        blk[:, 0:3] = coords.cpu()
        blk[:, 3] = t
        blk[:, 4] = power
        blk[:, 5] = Tsnap[j].reshape(-1).cpu()
        out.append(blk)
    return torch.cat(out, 0).numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--powers", type=float, nargs="+", default=[100, 125, 150, 175, 200, 225, 250]
    )
    ap.add_argument("--ele_size", type=float, default=0.25)
    ap.add_argument("--dt", type=float, default=2.5e-3)
    ap.add_argument("--total_t", type=float, default=3.0)
    ap.add_argument("--snap_dt", type=float, default=0.1)
    ap.add_argument("--rho", type=float, default=4.43e-3)
    ap.add_argument("--Cp", type=float, default=0.526)
    ap.add_argument("--k", type=float, default=0.0067)
    ap.add_argument("--h", type=float, default=2e-5)
    ap.add_argument("--eta", type=float, default=0.35)
    ap.add_argument("--emiss", type=float, default=0.40)
    ap.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="where to write; default data/torch_<stamp>_<minP>_<maxP>_<ele>[-<tag>]",
    )
    ap.add_argument("--tag", type=str, default=None, help="suffix on the run directory")
    ap.add_argument("--device", type=str, default="cuda")
    a = ap.parse_args()

    dev = a.device if torch.cuda.is_available() else "cpu"
    print(
        f"device={dev}  ele={a.ele_size}  dt={a.dt:.2e}  "
        f"powers={[int(p) for p in a.powers]}"
    )

    started = time.time()
    x, y, z, ts, Tsnap = solve(
        a.powers,
        a.ele_size,
        a.dt,
        a.total_t,
        a.snap_dt,
        a.rho,
        a.Cp,
        a.k,
        a.h,
        a.eta,
        a.emiss,
        device=dev,
    )
    print(
        f"grid {len(x)}x{len(y)}x{len(z)} = {len(x)*len(y)*len(z)} nodes, "
        f"{len(ts)} snapshots"
    )

    # The run directory is named the way archive/ entries are, so a dataset and the
    # trainings fitted to it read the same: <who>_<stamp>_<minP>_<maxP>_<spacing>.
    if a.outdir:
        outdir = Path(a.outdir)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        lo, hi = int(min(a.powers)), int(max(a.powers))
        name = f"torch_{stamp}_{lo}_{hi}_{a.ele_size:g}"
        outdir = REPO / "data" / (f"{name}-{a.tag}" if a.tag else name)
    outdir.mkdir(parents=True, exist_ok=True)

    for b, P in enumerate(a.powers):
        rows = to_rows(x, y, z, ts, Tsnap[b], P)
        out = outdir / f"data_{int(P)}W.npy"
        np.save(out, rows)
        print(
            f"saved {out.name}  shape={rows.shape}  "
            f"T=[{rows[:,5].min():.1f}, {rows[:,5].max():.1f}] K"
        )

    # Every number needed to reproduce this run, next to the bytes it produced.
    # Without it the only record of which parameters made which dataset was a log
    # file somewhere else, which is no record at all.
    (outdir / "config.json").write_text(
        json.dumps(
            {
                "solver": "heat_torch.py",
                "scheme": "7-point finite difference, Heun in time",
                "wall_s": round(time.time() - started, 1),
                "device": dev,
                "powers_W": [float(p) for p in a.powers],
                "grid": {"nx": len(x), "ny": len(y), "nz": len(z), "nt": len(ts)},
                "ele_size_mm": a.ele_size,
                "dt_s": a.dt,
                "total_t_s": a.total_t,
                "snap_dt_s": a.snap_dt,
                "domain_mm": list(DOMAIN),
                "beam": {
                    "radius_mm": R_BEAM,
                    "scan_speed_mm_s": VEL,
                    "start_x_mm": X0,
                    "y_mm": Y0,
                },
                "material": {
                    "rho_g_mm3": a.rho,
                    "Cp_J_gK": a.Cp,
                    "k_W_mmK": a.k,
                    "h_W_mm2K": a.h,
                    "eta": a.eta,
                    "emissivity": a.emiss,
                },
                "T_ambient_K": T_AMB,
            },
            indent=2,
        )
    )
    print(f"\nwrote {outdir}/config.json")
    print(f"     {outdir}")
