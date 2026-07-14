from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .model import DeepONeuralNet, FieldDerivatives

STEFAN_BOLTZMANN = 5.670374419e-8  # W m^-2 K^-4


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

    ``laser_power`` is the branch input and ``coords`` the trunk input
    ``(x, y, z, t)``. The remaining fields are only needed by some terms:
    ``temperature`` by the data loss, ``normal`` by the Neumann boundary
    losses, and ``q_laser`` by the top-surface loss.
    """

    laser_power: Tensor
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
        model: DeepONeuralNet,
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
            predicted = model(data.laser_power, data.coords)
            components["data"] = data_loss(predicted, data.temperature, scales.temperature)

        if collocation is not None:
            derivatives = model.derivatives(
                collocation.laser_power, _with_grad(collocation.coords)
            )
            components["pde"] = pde_loss(derivatives, properties, scales.pde)

        if bottom is not None:
            predicted = model(bottom.laser_power, bottom.coords)
            components["bottom"] = bottom_bc_loss(predicted, properties, scales.temperature)

        if top is not None:
            if top.normal is None or top.q_laser is None:
                raise ValueError("top point set requires `normal` and `q_laser`")
            derivatives = model.derivatives(
                top.laser_power, _with_grad(top.coords), second_order=False
            )
            components["top"] = top_bc_loss(
                derivatives, top.normal, top.q_laser, properties, scales.flux
            )

        if surrounding is not None:
            if surrounding.normal is None:
                raise ValueError("surrounding point set requires `normal`")
            derivatives = model.derivatives(
                surrounding.laser_power, _with_grad(surrounding.coords), second_order=False
            )
            components["surrounding"] = surrounding_bc_loss(
                derivatives, surrounding.normal, properties, scales.flux
            )

        if initial is not None:
            predicted = model(initial.laser_power, initial.coords)
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
