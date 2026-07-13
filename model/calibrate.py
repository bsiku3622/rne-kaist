"""Recover the PDE and boundary-condition constants from the simulation grids.

``train.py`` needs material properties and laser parameters that the ``.npy``
files do not carry. Rather than guessing them, this script fits them to the
data and prints a block that can be pasted into ``train.py``.

What is identifiable, and what is not:

* **Thermal diffusivity** ``alpha = k / (rho * c_p)`` is recovered by regressing
  ``dT/dt`` on ``laplacian(T)`` over the interior. Only the ratio is
  identifiable -- ``rho``, ``c_p`` and ``k`` individually are not, so ``rho``
  and ``c_p`` must be supplied and ``k`` follows.
* **Laser parameters** (absorptivity, beam radius, start position, scan speed,
  track centre) are recovered from the top-surface energy balance, fitted
  jointly across every power. The conductive flux there is 10^7-10^8 W/m^2, so
  the one-sided finite difference is accurate and the fit is well conditioned.
  Note that ``A`` is only identifiable given ``k``: the balance constrains
  ``q / k``.
* **Convection coefficient and emissivity are NOT identifiable.** On the
  lateral faces the normal derivative is at the noise floor of the exported
  0.5 mm grid -- neighbouring nodes differ by ~1.5 K while the second-order
  one-sided stencil extracts a ~0.26 K signal, a 0.07% cancellation. They
  contribute ~2.5% of the top-surface balance, so they are left as inputs.

Run with ``python calibrate.py`` (needs SciPy for the non-linear fit).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

MM = 1.0e-3
STEFAN_BOLTZMANN = 5.670374419e-8
AMBIENT_TEMPERATURE = 298.0

# Not identifiable from the data (see module docstring); supplied by the user.
DENSITY = 7990.0  # rho [kg m^-3]
SPECIFIC_HEAT = 500.0  # c_p [J kg^-1 K^-1]
CONVECTION_COEFF = 20.0  # h [W m^-2 K^-1]
EMISSIVITY = 0.35  # epsilon [-]


@dataclass
class Grid:
    """A simulation file reshaped onto its structured ``(x, y, z, t)`` axes."""

    name: str
    x: np.ndarray  # [nx] in metres
    y: np.ndarray  # [ny] in metres
    z: np.ndarray  # [nz] in metres
    t: np.ndarray  # [nt] in seconds
    temperature: np.ndarray  # [nx, ny, nz, nt] in Kelvin
    power: float  # W

    @property
    def spacing(self) -> tuple[float, float, float, float]:
        return (
            float(self.x[1] - self.x[0]),
            float(self.y[1] - self.y[0]),
            float(self.z[1] - self.z[0]),
            float(self.t[1] - self.t[0]),
        )


def load_grid(path: Path) -> Grid:
    """Reshape an ``[N, 6]`` file of ``(x, y, z, t, P, T)`` rows onto its grid."""
    raw = np.load(path).astype(np.float64)
    if raw.ndim != 2 or raw.shape[1] != 6:
        raise ValueError(f"{path}: expected [N, 6] of (x, y, z, t, P, T), got {raw.shape}")

    axes = [np.unique(raw[:, i]) for i in range(4)]
    index = tuple(np.searchsorted(axis, raw[:, i]) for i, axis in enumerate(axes))
    field = np.full([len(axis) for axis in axes], np.nan)
    field[index] = raw[:, 5]
    if np.isnan(field).any():
        raise ValueError(f"{path}: rows do not fill a complete structured grid")

    powers = np.unique(raw[:, 4])
    if powers.size != 1:
        raise ValueError(f"{path}: expected a single laser power, got {powers.tolist()}")

    x, y, z, t = axes
    return Grid(
        name=path.name, x=x * MM, y=y * MM, z=z * MM, t=t, temperature=field, power=float(powers[0])
    )


def _interior_terms(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """``(dT/dt, laplacian(T))`` on the interior, both raveled."""
    field = grid.temperature
    dx, dy, dz, dt = grid.spacing
    core = field[1:-1, 1:-1, 1:-1, 1:-1]

    time_derivative = (field[1:-1, 1:-1, 1:-1, 2:] - field[1:-1, 1:-1, 1:-1, :-2]) / (2 * dt)
    laplacian = (
        (field[2:, 1:-1, 1:-1, 1:-1] - 2 * core + field[:-2, 1:-1, 1:-1, 1:-1]) / dx**2
        + (field[1:-1, 2:, 1:-1, 1:-1] - 2 * core + field[1:-1, :-2, 1:-1, 1:-1]) / dy**2
        + (field[1:-1, 1:-1, 2:, 1:-1] - 2 * core + field[1:-1, 1:-1, :-2, 1:-1]) / dz**2
    )
    return time_derivative.ravel(), laplacian.ravel()


def fit_diffusivity(grids: Sequence[Grid], laplacian_floor: float = 1.0e3) -> tuple[float, float, int]:
    """Pooled least-squares ``alpha`` in ``dT/dt = alpha * laplacian(T)``.

    Points where the Laplacian is below ``laplacian_floor`` are dropped: both
    sides vanish there and they only add noise. Returns ``(alpha, corr, count)``.
    """
    time_blocks, laplacian_blocks = [], []
    for grid in grids:
        time_derivative, laplacian = _interior_terms(grid)
        selected = np.abs(laplacian) > laplacian_floor
        time_blocks.append(time_derivative[selected])
        laplacian_blocks.append(laplacian[selected])

    dTdt = np.concatenate(time_blocks)
    lap = np.concatenate(laplacian_blocks)
    alpha = float((lap * dTdt).sum() / (lap * lap).sum())
    correlation = float(np.corrcoef(lap, dTdt)[0, 1])
    return alpha, correlation, int(lap.size)


def top_surface_gradient(grid: Grid) -> np.ndarray:
    """Outward ``dT/dz`` on ``z = z_max``, second-order one-sided, ``[nx, ny, nt]``."""
    _, _, dz, _ = grid.spacing
    surface = grid.temperature[:, :, -1, :]
    first = grid.temperature[:, :, -2, :]
    second = grid.temperature[:, :, -3, :]
    return (3 * surface - 4 * first + second) / (2 * dz)


@dataclass
class LaserFit:
    absorptivity: float
    beam_radius: float
    start_x: float
    scan_speed: float
    track_y: float
    peak_flux_per_watt: float
    residual_rms: float
    reference_peak_flux: float

    @property
    def relative_residual(self) -> float:
        return self.residual_rms / self.reference_peak_flux


def fit_top_bc(
    grids: Sequence[Grid], conductivity: float, convection: float, emissivity: float
) -> LaserFit:
    """Jointly fit a moving Gaussian source across every power.

    Enforces ``k*dT/dn = q - h(T-T_amb) - sigma*eps*(T^4-T_amb^4)`` on ``z = z_max``
    with one shared ``(A, r_b, x0, v, y_c)`` and each grid's own ``P``.
    """
    from scipy.optimize import least_squares

    x_blocks, y_blocks, t_blocks, power_blocks = [], [], [], []
    conductive_blocks, loss_blocks = [], []
    ambient = AMBIENT_TEMPERATURE

    for grid in grids:
        surface = grid.temperature[:, :, -1, :]
        gradient = top_surface_gradient(grid)
        shape = surface.shape

        x_blocks.append(np.broadcast_to(grid.x[:, None, None], shape).ravel())
        y_blocks.append(np.broadcast_to(grid.y[None, :, None], shape).ravel())
        t_blocks.append(np.broadcast_to(grid.t[None, None, :], shape).ravel())
        power_blocks.append(np.full(surface.size, grid.power))

        flat = surface.ravel()
        conductive_blocks.append(conductivity * gradient.ravel())
        loss_blocks.append(
            convection * (flat - ambient)
            + STEFAN_BOLTZMANN * emissivity * (flat**4 - ambient**4)
        )

    x = np.concatenate(x_blocks)
    y = np.concatenate(y_blocks)
    t = np.concatenate(t_blocks)
    power = np.concatenate(power_blocks)
    conductive = np.concatenate(conductive_blocks)
    surface_loss = np.concatenate(loss_blocks)

    def residual(parameters: np.ndarray) -> np.ndarray:
        absorptivity, radius, start_x, speed, track_y = parameters
        centre_x = start_x + speed * t
        squared_distance = (x - centre_x) ** 2 + (y - track_y) ** 2
        flux = (2 * absorptivity * power / (np.pi * radius**2)) * np.exp(
            -2 * squared_distance / radius**2
        )
        return (conductive - (flux - surface_loss)) / 1.0e6  # scale to O(1)

    guess = [0.4, 2.0 * MM, 5.0 * MM, 10.0 * MM, 0.5 * float(grids[0].y[-1])]
    lower = [0.01, 0.2 * MM, 0.0, 1.0 * MM, float(grids[0].y[0])]
    upper = [1.0, 6.0 * MM, 20.0 * MM, 50.0 * MM, float(grids[0].y[-1])]
    solution = least_squares(residual, guess, bounds=(lower, upper), xtol=1e-15, ftol=1e-15)

    absorptivity, radius, start_x, speed, track_y = solution.x
    peak_per_watt = 2 * absorptivity / (np.pi * radius**2)
    return LaserFit(
        absorptivity=float(absorptivity),
        beam_radius=float(radius),
        start_x=float(start_x),
        scan_speed=float(speed),
        track_y=float(track_y),
        peak_flux_per_watt=float(peak_per_watt),
        residual_rms=float(np.sqrt(((solution.fun * 1.0e6) ** 2).mean())),
        reference_peak_flux=float(peak_per_watt * power.max()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    paths = sorted(args.data_dir.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"no .npy files under {args.data_dir}")

    grids = [load_grid(path) for path in paths]
    for grid in grids:
        print(f"[grid] {grid.name}: {grid.temperature.shape} at P = {grid.power} W")

    alpha, correlation, count = fit_diffusivity(grids)
    conductivity = alpha * DENSITY * SPECIFIC_HEAT
    print(f"[pde ] alpha = {alpha:.6e} m^2/s from {count} interior points (corr = {correlation:.4f})")
    print(f"[pde ] k = alpha * rho * c_p = {conductivity:.4f} W/m/K  (rho, c_p supplied)")

    fit = fit_top_bc(grids, conductivity, CONVECTION_COEFF, EMISSIVITY)
    print(f"[bc  ] peak flux at P_max = {fit.reference_peak_flux:.4e} W/m^2")
    print(f"[bc  ] residual RMS = {fit.residual_rms:.4e} W/m^2 ({fit.relative_residual:.2%} of peak)")

    print("\n--- paste into train.py ---")
    print(f"ABSORPTIVITY = {fit.absorptivity:.4f}")
    print(f"BEAM_RADIUS = {fit.beam_radius / MM:.4f} * MM")
    print(f"LASER_START_X = {fit.start_x / MM:.4f} * MM")
    print(f"LASER_Y = {fit.track_y / MM:.4f} * MM")
    print(f"SCAN_SPEED = {fit.scan_speed / MM:.4f} * MM")
    print("PROPERTIES = ThermalProperties(")
    print(f"    density={DENSITY},")
    print(f"    specific_heat={SPECIFIC_HEAT},")
    print(f"    conductivity={conductivity:.4f},")
    print(f"    convection_coeff={CONVECTION_COEFF},")
    print(f"    emissivity={EMISSIVITY},")
    print(f"    ambient_temperature={AMBIENT_TEMPERATURE},")
    print(")")


if __name__ == "__main__":
    main()
