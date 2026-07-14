"""The PINN objective for CPiMLP, and the physical constants it is built from.

A copy of ``models/pimlp/loss.py`` with the laser power removed from everything
the network touches. The residuals themselves are untouched -- the heat equation
does not care what shape the network inside it has, nor how many inputs it takes
-- so the objective CPiMLP is fitted against is term-for-term the one PiMLP is
fitted against. Two things follow from dropping ``P``:

``PointSet`` no longer carries ``laser_power``
    Nothing consumes it. PiMLP's point sets carry it because every residual calls
    ``model(laser_power, coords)``; here every residual calls ``model(coords)``.

``q_laser`` is still a function of power
    It has to be: the top-surface source term ``2*A*P / (pi*r_b^2)`` *is* the
    power, and a heat equation with no source in it is not the problem this
    project is about. So the power survives in the physics even though it has left
    the network, and :mod:`models.cpimlp.dataset` pins it to a single
    representative value rather than sampling it per batch. That choice, and the
    tension it leaves behind, is set out in the docstring of
    :mod:`models.cpimlp.train`.

The residuals need material properties and laser parameters that the ``.npy``
files do not carry. They live here, next to the terms that consume them, and are
recovered from the data by ``python calibrate.py`` -- re-run it and paste the
block back whenever the contents of the data directory change.

Every quantity is SI (metres, seconds, Kelvin, watts).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .model import ControlPhysicsMLP, FieldDerivatives

STEFAN_BOLTZMANN = 5.670374419e-8  # W m^-2 K^-4
MM = 1.0e-3


@dataclass
class ThermalProperties:
    """Material and environment constants of the heat-conduction problem.

    SI units throughout: lengths in metres, time in seconds, temperature in
    Kelvin. Absolute temperature is required because of the radiation term.
    """

    density: float  # rho [kg m^-3]
    specific_heat: float  # c_p [J kg^-1 K^-1]
    conductivity: float  # k [W m^-1 K^-1]
    convection_coeff: float  # h [W m^-2 K^-1]
    emissivity: float  # epsilon [-]
    ambient_temperature: float  # T_amb [K]
    stefan_boltzmann: float = STEFAN_BOLTZMANN


# ---------------------------------------------------------------------------
# Fitted jointly across all seven powers by `python calibrate.py`. The
# top-surface energy balance closes to 0.86% of peak flux.
# ---------------------------------------------------------------------------
ABSORPTIVITY = 0.4593  # A [-]
BEAM_RADIUS = 1.6971 * MM  # r_b [m]
LASER_START_X = 4.8683 * MM
LASER_Y = 4.9929 * MM
SCAN_SPEED = 10.0000 * MM  # [m s^-1]

# `conductivity` follows from the fitted diffusivity alpha = 2.3935e-6 m^2/s
# (corr 0.979 against the interior Laplacian) and the assumed rho and c_p; only
# the ratio k/(rho*c_p) is identifiable from the data. `convection_coeff` and
# `emissivity` are NOT identifiable -- on the lateral faces the normal
# derivative sits at the noise floor of the 0.5 mm export grid -- so they stay
# as inputs. They account for ~2.5% of the top-surface balance.
PROPERTIES = ThermalProperties(
    density=7990.0,  # rho [kg m^-3]      (assumed)
    specific_heat=500.0,  # c_p [J kg^-1 K^-1] (assumed)
    conductivity=9.5619,  # k [W m^-1 K^-1]    (= alpha * rho * c_p)
    convection_coeff=20.0,  # h [W m^-2 K^-1]    (assumed)
    emissivity=0.35,  # epsilon [-]        (assumed)
    ambient_temperature=298.0,  # T_amb [K]  (matches t=0 and z=0 in the data)
)


def peak_laser_flux(power: Tensor | float) -> Tensor | float:
    """Centreline intensity of the Gaussian beam, ``2*A*P / (pi*r_b^2)`` in W m^-2."""
    return 2.0 * ABSORPTIVITY * power / (math.pi * BEAM_RADIUS**2)


def laser_flux(coords: Tensor, power: Tensor | float, exponent_scale: float = 1.0) -> Tensor:
    """Gaussian moving heat source on the top surface, ``[batch, 1]`` in W m^-2.

    ``q(x, y, t) = (2*A*P / (pi*r_b^2)) * exp(-2*k*((x - x_l(t))^2 + (y - y_l)^2) / r_b^2)``
    with the beam centre ``x_l(t) = LASER_START_X + SCAN_SPEED * t`` and ``k`` the
    ``exponent_scale``. ``k`` only reshapes the decay -- the amplitude at the
    centre is unchanged, since ``exp(0) = 1`` regardless of ``k`` -- so ``k > 1``
    narrows the deposited spot and ``k < 1`` broadens it while the total peak
    flux from :func:`peak_laser_flux` stays fixed.

    ``power`` is a plain ``float`` here where PiMLP passes a ``[batch, 1]`` tensor:
    the batch of powers has collapsed to the one value the physics is pinned to.
    """
    x, y, t = coords[:, 0:1], coords[:, 1:2], coords[:, 3:4]
    centre_x = LASER_START_X + SCAN_SPEED * t
    squared_distance = (x - centre_x) ** 2 + (y - LASER_Y) ** 2
    return peak_laser_flux(power) * torch.exp(
        -2.0 * exponent_scale * squared_distance / BEAM_RADIUS**2
    )


@dataclass
class ResidualScales:
    """Characteristic magnitudes used to non-dimensionalise each residual.

    Without this the loss terms cannot be compared: the Dirichlet residuals are
    in Kelvin, the PDE residual in W m^-3 and the Neumann residuals in W m^-2,
    so summing the three boundary terms under one weight would let the flux
    terms dominate by many orders of magnitude. Dividing each residual by its
    characteristic value makes every component O(1), after which the weights in
    :class:`LossWeights` express genuine relative importance.
    """

    temperature: float = 1.0  # [K]   - data, bottom BC, IC
    pde: float = 1.0  # [W m^-3]      - conduction residual
    flux: float = 1.0  # [W m^-2]     - top and surrounding BC

    def __post_init__(self) -> None:
        for name in ("temperature", "pde", "flux"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} scale must be positive")

    @classmethod
    def characteristic(
        cls,
        properties: ThermalProperties,
        temperature_rise: float,
        time_scale: float,
        peak_flux: float,
    ) -> "ResidualScales":
        """Derive scales from the problem: ``dT``, ``rho*c_p*dT/t`` and the peak laser flux."""
        return cls(
            temperature=temperature_rise,
            pde=properties.density * properties.specific_heat * temperature_rise / time_scale,
            flux=peak_flux,
        )


@dataclass
class LossWeights:
    """Coefficients of L = w_D * L_D + w_PDE * L_PDE + w_BC * L_BC + w_IC * L_IC."""

    data: float = 1.0
    pde: float = 1.0
    bc: float = 1.0
    ic: float = 1.0


@dataclass
class PointSet:
    """A batch of collocation points fed to one loss term.

    ``coords`` is the space-time point ``(x, y, z, t)``. The remaining fields are
    only needed by some terms: ``temperature`` by the data loss, ``normal`` by the
    Neumann boundary losses, and ``q_laser`` by the top-surface loss.

    PiMLP's :class:`~models.pimlp.loss.PointSet` also carries ``laser_power``,
    because every one of its residuals passes it to the network. None of these do,
    so it is not here; the power reaches the objective only through the already
    evaluated ``q_laser``.
    """

    coords: Tensor
    temperature: Tensor | None = None
    normal: Tensor | None = None
    q_laser: Tensor | None = None


def _mean_squared(residual: Tensor, scale: float = 1.0) -> Tensor:
    return (residual / scale).pow(2).mean()


def _with_grad(coords: Tensor) -> Tensor:
    if coords.requires_grad:
        return coords
    if coords.is_leaf:
        return coords.requires_grad_(True)
    raise ValueError("non-leaf coords must be created with requires_grad=True")


def normal_derivative(derivatives: FieldDerivatives, normal: Tensor) -> Tensor:
    """``dT/dn = grad(T) . n`` for outward unit normals ``normal`` of shape ``[batch, 3]``."""
    if normal.dim() != 2 or normal.size(-1) != 3:
        raise ValueError(f"normal must have shape [batch, 3], got {tuple(normal.shape)}")
    return (derivatives.spatial_gradient * normal).sum(dim=-1, keepdim=True)


def surface_heat_flux(temperature: Tensor, properties: ThermalProperties) -> Tensor:
    """Convective plus radiative loss ``h(T - T_amb) + sigma*eps*(T^4 - T_amb^4)``."""
    ambient = properties.ambient_temperature
    convection = properties.convection_coeff * (temperature - ambient)
    radiation = (
        properties.stefan_boltzmann
        * properties.emissivity
        * (temperature.pow(4) - ambient**4)
    )
    return convection + radiation


def data_loss(predicted: Tensor, measured: Tensor, scale: float = 1.0) -> Tensor:
    """``L_D = T_hat - T`` against labelled temperature samples."""
    return _mean_squared(predicted - measured, scale)


def pde_loss(
    derivatives: FieldDerivatives, properties: ThermalProperties, scale: float = 1.0
) -> Tensor:
    """``L_cond = rho*c_p*dT/dt - k*(d2T/dx2 + d2T/dy2 + d2T/dz2)``."""
    residual = (
        properties.density * properties.specific_heat * derivatives.T_t
        - properties.conductivity * derivatives.laplacian
    )
    return _mean_squared(residual, scale)


def bottom_bc_loss(
    predicted: Tensor, properties: ThermalProperties, scale: float = 1.0
) -> Tensor:
    """``L_bottom = T_hat - T_amb``: the base plate is held at ambient temperature."""
    return _mean_squared(predicted - properties.ambient_temperature, scale)


def top_bc_loss(
    derivatives: FieldDerivatives,
    normal: Tensor,
    q_laser: Tensor,
    properties: ThermalProperties,
    scale: float = 1.0,
) -> Tensor:
    """``L_top = k*dT/dn - [q_laser - h(T_hat - T_amb) - sigma*eps*(T_hat^4 - T_amb^4)]``."""
    flux = surface_heat_flux(derivatives.T, properties)
    residual = (
        properties.conductivity * normal_derivative(derivatives, normal) - q_laser + flux
    )
    return _mean_squared(residual, scale)


def surrounding_bc_loss(
    derivatives: FieldDerivatives,
    normal: Tensor,
    properties: ThermalProperties,
    scale: float = 1.0,
) -> Tensor:
    """``L_surr = k*dT/dn - [-h(T_hat - T_amb) - sigma*eps*(T_hat^4 - T_amb^4)]``."""
    flux = surface_heat_flux(derivatives.T, properties)
    residual = properties.conductivity * normal_derivative(derivatives, normal) + flux
    return _mean_squared(residual, scale)


def initial_condition_loss(
    predicted: Tensor, properties: ThermalProperties, scale: float = 1.0
) -> Tensor:
    """``L_IC = T_hat - T_amb`` at ``t = 0``."""
    return _mean_squared(predicted - properties.ambient_temperature, scale)


class PINNLoss(nn.Module):
    """Weighted sum of the data, PDE, boundary and initial-condition residuals.

    Every point set is optional; an omitted one contributes zero, so the same
    module handles a purely physics-driven run and a data-assimilating one.
    The boundary term is the sum of the bottom, top and surrounding residuals,
    each non-dimensionalised by :class:`ResidualScales` first.
    """

    def __init__(
        self,
        properties: ThermalProperties,
        weights: LossWeights | None = None,
        scales: ResidualScales | None = None,
    ) -> None:
        super().__init__()
        self.properties = properties
        self.weights = weights or LossWeights()
        self.scales = scales or ResidualScales()

    def forward(
        self,
        model: ControlPhysicsMLP,
        *,
        data: PointSet | None = None,
        collocation: PointSet | None = None,
        bottom: PointSet | None = None,
        top: PointSet | None = None,
        surrounding: PointSet | None = None,
        initial: PointSet | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return ``(total_loss, components)``.

        ``components`` holds the unweighted ``data``, ``pde``, ``bottom``,
        ``top``, ``surrounding``, ``ic`` terms plus the aggregated ``bc``.
        """
        reference = next(model.parameters())
        zero = torch.zeros((), device=reference.device, dtype=reference.dtype)
        properties = self.properties
        scales = self.scales

        components: dict[str, Tensor] = {
            "data": zero,
            "pde": zero,
            "bottom": zero,
            "top": zero,
            "surrounding": zero,
            "ic": zero,
        }

        if data is not None:
            if data.temperature is None:
                raise ValueError("data point set requires `temperature` targets")
            predicted = model(data.coords)
            components["data"] = data_loss(predicted, data.temperature, scales.temperature)

        if collocation is not None:
            derivatives = model.derivatives(_with_grad(collocation.coords))
            components["pde"] = pde_loss(derivatives, properties, scales.pde)

        if bottom is not None:
            predicted = model(bottom.coords)
            components["bottom"] = bottom_bc_loss(predicted, properties, scales.temperature)

        if top is not None:
            if top.normal is None or top.q_laser is None:
                raise ValueError("top point set requires `normal` and `q_laser`")
            derivatives = model.derivatives(_with_grad(top.coords), second_order=False)
            components["top"] = top_bc_loss(
                derivatives, top.normal, top.q_laser, properties, scales.flux
            )

        if surrounding is not None:
            if surrounding.normal is None:
                raise ValueError("surrounding point set requires `normal`")
            derivatives = model.derivatives(
                _with_grad(surrounding.coords), second_order=False
            )
            components["surrounding"] = surrounding_bc_loss(
                derivatives, surrounding.normal, properties, scales.flux
            )

        if initial is not None:
            predicted = model(initial.coords)
            components["ic"] = initial_condition_loss(
                predicted, properties, scales.temperature
            )

        components["bc"] = (
            components["bottom"] + components["top"] + components["surrounding"]
        )

        weights = self.weights
        total = (
            weights.data * components["data"]
            + weights.pde * components["pde"]
            + weights.bc * components["bc"]
            + weights.ic * components["ic"]
        )
        return total, components
