from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn


def _as_layer_sizes(layer_sizes: Iterable[int] | None) -> list[int]:
    if layer_sizes is None:
        return []
    return [int(size) for size in layer_sizes]


def _normalisation_buffer(
    values: Iterable[float] | None, dim: int, default: float
) -> Tensor:
    """Build a ``[1, dim]`` buffer, broadcasting a scalar and validating the length."""
    if values is None:
        return torch.full((1, dim), float(default))
    tensor = torch.as_tensor(list(values), dtype=torch.get_default_dtype()).reshape(1, -1)
    if tensor.numel() == 1:
        tensor = tensor.expand(1, dim).contiguous()
    if tensor.numel() != dim:
        raise ValueError(f"expected {dim} normalisation values, got {tensor.numel()}")
    return tensor


def _build_mlp(
    input_dim: int,
    hidden_layers: Sequence[int],
    output_dim: int,
    activation: type[nn.Module] = nn.ReLU,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim

    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(activation())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim

    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    """Deep Operator Network.

    The branch network encodes the input function, and the trunk network
    encodes the query location. Their outputs are combined by an inner
    product to predict the operator value.
    """

    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        hidden_layers: Sequence[int] | None = None,
        latent_dim: int = 128,
        output_dim: int = 1,
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_layers = _as_layer_sizes(hidden_layers)

        if branch_input_dim <= 0:
            raise ValueError("branch_input_dim must be positive")
        if trunk_input_dim <= 0:
            raise ValueError("trunk_input_dim must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")

        self.branch_input_dim = branch_input_dim
        self.trunk_input_dim = trunk_input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        branch_output_dim = latent_dim * output_dim
        trunk_output_dim = latent_dim * output_dim

        self.branch_net = MLP(
            branch_input_dim,
            hidden_layers,
            branch_output_dim,
            activation=activation,
            dropout=dropout,
        )
        self.trunk_net = MLP(
            trunk_input_dim,
            hidden_layers,
            trunk_output_dim,
            activation=activation,
            dropout=dropout,
        )
        self.bias = nn.Parameter(torch.zeros(output_dim)) if use_bias else None

    def forward(self, branch_input: Tensor, trunk_input: Tensor) -> Tensor:
        if branch_input.dim() != 2:
            raise ValueError("branch_input must have shape [batch, branch_input_dim]")
        if trunk_input.dim() != 2:
            raise ValueError("trunk_input must have shape [batch, trunk_input_dim]")
        if branch_input.size(-1) != self.branch_input_dim:
            raise ValueError(
                f"Expected branch_input last dimension {self.branch_input_dim}, got {branch_input.size(-1)}"
            )
        if trunk_input.size(-1) != self.trunk_input_dim:
            raise ValueError(
                f"Expected trunk_input last dimension {self.trunk_input_dim}, got {trunk_input.size(-1)}"
            )

        branch_features = self.branch_net(branch_input)
        trunk_features = self.trunk_net(trunk_input)

        branch_features = branch_features.view(-1, self.output_dim, self.latent_dim)
        trunk_features = trunk_features.view(-1, self.output_dim, self.latent_dim)

        branch_batch = branch_features.size(0)
        trunk_batch = trunk_features.size(0)

        if branch_batch != trunk_batch:
            if branch_batch == 1:
                branch_features = branch_features.expand(trunk_batch, -1, -1)
            elif trunk_batch == 1:
                trunk_features = trunk_features.expand(branch_batch, -1, -1)
            else:
                raise ValueError(
                    "branch_input and trunk_input must have the same batch size, or one of them must have batch size 1"
                )

        output = (branch_features * trunk_features).sum(dim=-1)
        if self.bias is not None:
            output = output + self.bias
        return output


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: Sequence[int] | None,
        output_dim: int,
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network = _build_mlp(
            input_dim,
            _as_layer_sizes(hidden_layers),
            output_dim,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


@dataclass
class FieldDerivatives:
    """Predicted temperature field and its autograd derivatives.

    Every field is shaped ``[batch, 1]``. The second-order fields are ``None``
    when :meth:`DeepONeuralNet.derivatives` is called with ``second_order=False``.
    """

    T: Tensor
    T_x: Tensor
    T_y: Tensor
    T_z: Tensor
    T_t: Tensor
    T_xx: Tensor | None = None
    T_yy: Tensor | None = None
    T_zz: Tensor | None = None

    @property
    def spatial_gradient(self) -> Tensor:
        """``[batch, 3]`` gradient ``(dT/dx, dT/dy, dT/dz)``."""
        return torch.cat((self.T_x, self.T_y, self.T_z), dim=-1)

    @property
    def laplacian(self) -> Tensor:
        """``[batch, 1]`` value of ``d2T/dx2 + d2T/dy2 + d2T/dz2``."""
        if self.T_xx is None or self.T_yy is None or self.T_zz is None:
            raise ValueError(
                "second-order derivatives are unavailable; call derivatives(..., second_order=True)"
            )
        return self.T_xx + self.T_yy + self.T_zz


class _DifferentiableTemperatureField(nn.Module):
    """Shared autograd machinery for models predicting ``T(coords; laser_power)``.

    Subclasses implement ``forward(laser_power, coords) -> T`` over the
    physical ``(x, y, z, t)`` coordinates; this base class differentiates that
    prediction for the PDE residual.
    """

    physical_dim = 4  # (x, y, z, t)

    def derivatives(
        self,
        laser_power: Tensor,
        coords: Tensor,
        second_order: bool = True,
    ) -> FieldDerivatives:
        """Evaluate the network and differentiate it w.r.t. ``coords``.

        ``coords`` must carry ``requires_grad=True``. The graph is retained
        (``create_graph=True``) so the residuals built from these derivatives
        remain differentiable w.r.t. the network parameters.
        """
        if not coords.requires_grad:
            raise ValueError(
                "coords must have requires_grad=True to differentiate through the trunk input"
            )

        temperature = self(laser_power, coords)
        first = torch.autograd.grad(
            temperature,
            coords,
            grad_outputs=torch.ones_like(temperature),
            create_graph=True,
        )[0]
        T_x, T_y, T_z, T_t = (first[:, i : i + 1] for i in range(self.physical_dim))

        if not second_order:
            return FieldDerivatives(T=temperature, T_x=T_x, T_y=T_y, T_z=T_z, T_t=T_t)

        second = []
        for axis, gradient in enumerate((T_x, T_y, T_z)):
            row = torch.autograd.grad(
                gradient,
                coords,
                grad_outputs=torch.ones_like(gradient),
                create_graph=True,
            )[0]
            second.append(row[:, axis : axis + 1])

        return FieldDerivatives(
            T=temperature,
            T_x=T_x,
            T_y=T_y,
            T_z=T_z,
            T_t=T_t,
            T_xx=second[0],
            T_yy=second[1],
            T_zz=second[2],
        )


class DeepONeuralNet(_DifferentiableTemperatureField):
    """Physics-informed DeepONet for a transient 3-D temperature field.

    The branch network encodes the laser process parameter(s) ``P`` and the
    trunk network encodes the space-time query point ``(x, y, z, t)`` plus one
    engineered feature, the known Gaussian laser-proximity term also used to
    build the top-surface flux BC in ``train.py``. A tanh MLP is biased toward
    smooth, low-frequency functions (spectral bias), so it systematically
    widens and flattens the laser's narrow moving spike no matter how the data
    is sampled; handing the network this feature directly -- computed from
    ``coords`` inside :meth:`forward` so autograd still differentiates through
    it -- gives it a sharp, correctly-scaled building block instead of asking
    it to synthesise one from scratch.

    The activation defaults to ``tanh`` because the PDE residual needs a
    non-vanishing second derivative; piecewise-linear activations such as
    ``ReLU`` make the Laplacian identically zero.

    Input and output normalisation is applied *inside* :meth:`forward`, so
    ``coords`` are passed and differentiated in physical units while the
    subnetworks still see well-scaled O(1) values. Autograd carries the chain
    rule through the affine maps, which keeps the residuals in ``loss.py``
    dimensionally correct without any manual rescaling.
    """

    trunk_input_dim = _DifferentiableTemperatureField.physical_dim + 1  # + laser-proximity feature

    # Declared so type checkers see `Tensor` rather than `Tensor | Module`, which
    # is what `nn.Module.__getattr__` is annotated to return for buffers.
    coord_mean: Tensor
    coord_scale: Tensor
    branch_mean: Tensor
    branch_scale: Tensor
    temperature_offset: Tensor
    temperature_scale: Tensor
    laser_beam_radius: Tensor
    laser_start_x: Tensor
    laser_y: Tensor
    laser_scan_speed: Tensor

    def __init__(
        self,
        branch_input_dim: int = 1,
        hidden_layers: Sequence[int] | None = (64, 64, 64, 64),
        latent_dim: int = 128,
        activation: type[nn.Module] = nn.Tanh,
        dropout: float = 0.0,
        use_bias: bool = True,
        coord_mean: Sequence[float] | None = None,
        coord_scale: Sequence[float] | None = None,
        branch_mean: Sequence[float] | None = None,
        branch_scale: Sequence[float] | None = None,
        temperature_offset: float = 0.0,
        temperature_scale: float = 1.0,
        laser_beam_radius: float = 1.0,
        laser_start_x: float = 0.0,
        laser_y: float = 0.0,
        laser_scan_speed: float = 0.0,
    ) -> None:
        super().__init__()
        self.branch_input_dim = branch_input_dim
        self.operator = DeepONet(
            branch_input_dim=branch_input_dim,
            trunk_input_dim=self.trunk_input_dim,
            hidden_layers=hidden_layers,
            latent_dim=latent_dim,
            output_dim=1,
            activation=activation,
            dropout=dropout,
            use_bias=use_bias,
        )

        self.register_buffer(
            "coord_mean", _normalisation_buffer(coord_mean, self.physical_dim, 0.0)
        )
        self.register_buffer(
            "coord_scale", _normalisation_buffer(coord_scale, self.physical_dim, 1.0)
        )
        self.register_buffer(
            "branch_mean", _normalisation_buffer(branch_mean, branch_input_dim, 0.0)
        )
        self.register_buffer(
            "branch_scale", _normalisation_buffer(branch_scale, branch_input_dim, 1.0)
        )
        if temperature_scale == 0.0:
            raise ValueError("temperature_scale must be non-zero")
        self.register_buffer("temperature_offset", torch.tensor(float(temperature_offset)))
        self.register_buffer("temperature_scale", torch.tensor(float(temperature_scale)))
        self.register_buffer("laser_beam_radius", torch.tensor(float(laser_beam_radius)))
        self.register_buffer("laser_start_x", torch.tensor(float(laser_start_x)))
        self.register_buffer("laser_y", torch.tensor(float(laser_y)))
        self.register_buffer("laser_scan_speed", torch.tensor(float(laser_scan_speed)))

    def _laser_feature(self, coords: Tensor) -> Tensor:
        """``exp(-2*((x - x_l(t))^2 + (y - y_l)^2) / r_b^2)``, the same Gaussian
        ``train.laser_flux`` scales by the peak flux for the top BC."""
        x, y, t = coords[:, 0:1], coords[:, 1:2], coords[:, 3:4]
        centre_x = self.laser_start_x + self.laser_scan_speed * t
        squared_distance = (x - centre_x) ** 2 + (y - self.laser_y) ** 2
        return torch.exp(-2.0 * squared_distance / self.laser_beam_radius**2)

    def forward(self, laser_power: Tensor, coords: Tensor) -> Tensor:
        """Predict temperature at ``coords`` for the process parameters ``laser_power``.

        ``laser_power`` is ``[batch, branch_input_dim]`` (a batch size of 1 is
        broadcast over the query points) and ``coords`` is ``[batch, 4]``
        holding ``(x, y, z, t)`` in physical units. Returns ``[batch, 1]`` in
        the same temperature unit as ``temperature_offset``.
        """
        if coords.dim() != 2 or coords.size(-1) != self.physical_dim:
            raise ValueError(
                f"coords must have shape [batch, {self.physical_dim}] for (x, y, z, t), "
                f"got {tuple(coords.shape)}"
            )

        normalised_coords = (coords - self.coord_mean) / self.coord_scale
        trunk_input = torch.cat([normalised_coords, self._laser_feature(coords)], dim=-1)
        normalised_power = (laser_power - self.branch_mean) / self.branch_scale
        latent = self.operator(normalised_power, trunk_input)
        return self.temperature_offset + self.temperature_scale * latent


class PlainNeuralNet(_DifferentiableTemperatureField):
    """A single MLP over the raw ``(x, y, z, t, P)`` input -- no branch/trunk
    split, no inner product, no engineered features.

    A companion to :class:`DeepONeuralNet` at roughly the same parameter
    count, for testing whether the operator's branch/trunk factorisation (and
    the gaussian trunk feature) actually earns its keep here, or whether one
    MLP over all five raw inputs does comparably.
    """

    coord_mean: Tensor
    coord_scale: Tensor
    branch_mean: Tensor
    branch_scale: Tensor
    temperature_offset: Tensor
    temperature_scale: Tensor

    def __init__(
        self,
        branch_input_dim: int = 1,
        hidden_layers: Sequence[int] | None = (209, 209, 209, 209),
        activation: type[nn.Module] = nn.Tanh,
        dropout: float = 0.0,
        coord_mean: Sequence[float] | None = None,
        coord_scale: Sequence[float] | None = None,
        branch_mean: Sequence[float] | None = None,
        branch_scale: Sequence[float] | None = None,
        temperature_offset: float = 0.0,
        temperature_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.branch_input_dim = branch_input_dim
        self.network = MLP(
            self.physical_dim + branch_input_dim,
            hidden_layers,
            output_dim=1,
            activation=activation,
            dropout=dropout,
        )

        self.register_buffer(
            "coord_mean", _normalisation_buffer(coord_mean, self.physical_dim, 0.0)
        )
        self.register_buffer(
            "coord_scale", _normalisation_buffer(coord_scale, self.physical_dim, 1.0)
        )
        self.register_buffer(
            "branch_mean", _normalisation_buffer(branch_mean, branch_input_dim, 0.0)
        )
        self.register_buffer(
            "branch_scale", _normalisation_buffer(branch_scale, branch_input_dim, 1.0)
        )
        if temperature_scale == 0.0:
            raise ValueError("temperature_scale must be non-zero")
        self.register_buffer("temperature_offset", torch.tensor(float(temperature_offset)))
        self.register_buffer("temperature_scale", torch.tensor(float(temperature_scale)))

    def forward(self, laser_power: Tensor, coords: Tensor) -> Tensor:
        """Predict temperature at ``coords`` for the process parameters ``laser_power``.

        Same signature and broadcasting rules as :meth:`DeepONeuralNet.forward`.
        """
        if coords.dim() != 2 or coords.size(-1) != self.physical_dim:
            raise ValueError(
                f"coords must have shape [batch, {self.physical_dim}] for (x, y, z, t), "
                f"got {tuple(coords.shape)}"
            )

        normalised_coords = (coords - self.coord_mean) / self.coord_scale
        normalised_power = (laser_power - self.branch_mean) / self.branch_scale
        if normalised_power.size(0) != normalised_coords.size(0):
            if normalised_power.size(0) != 1:
                raise ValueError(
                    "laser_power and coords must have the same batch size, "
                    "or laser_power must have batch size 1"
                )
            normalised_power = normalised_power.expand(normalised_coords.size(0), -1)

        network_input = torch.cat([normalised_coords, normalised_power], dim=-1)
        output = self.network(network_input)
        return self.temperature_offset + self.temperature_scale * output