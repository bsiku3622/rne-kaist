"""GDoN -- GPiDoN with the physics taken out.

Same architecture as :mod:`models.gpidon`: a DeepONet whose inner product is
multiplied by a Gaussian riding on the moving beam before the bias is added.
What is missing is the ``Pi``. There is no PDE residual, so nothing here needs a
second derivative and :meth:`GPiDoN.derivatives` has no counterpart; the network
is fitted to labelled temperatures alone.

That makes GDoN the controlled comparison for GPiDoN: same map, same gate, same
normalisation, only the objective differs.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import Tensor, nn

from .laser import beam_gaussian


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


class DeepONet(nn.Module):
    """Deep Operator Network.

    The branch network encodes the input function, and the trunk network
    encodes the query location. Their outputs are combined by an inner
    product to predict the operator value, which an optional ``gate`` may
    modulate before the bias is added.
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

    def forward(
        self, branch_input: Tensor, trunk_input: Tensor, gate: Tensor | None = None
    ) -> Tensor:
        """``gate`` is ``[batch, 1]`` and scales the inner product before the bias."""
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
        if gate is not None:
            if gate.dim() != 2 or gate.size(-1) != 1:
                raise ValueError(f"gate must have shape [batch, 1], got {tuple(gate.shape)}")
            output = output * gate
        if self.bias is not None:
            output = output + self.bias
        return output


class GDoN(nn.Module):
    """Gaussian-gated Deep Operator Network, fitted to data alone.

    The branch network encodes the laser process parameter(s) ``P`` and the trunk
    network encodes the space-time query point ``(x, y, z, t)``. Their latent
    codes are combined by an inner product, multiplied by the gate and offset by
    the operator bias, to predict the temperature ``T_hat(x, y, z, t; P)``::

        T_hat = T_amb + dT * ( <branch(P), trunk(x,y,z,t)> * G + b )
        G     = g(x, y, z, t) + p

    ``g`` is :func:`~models.gdon.laser.beam_gaussian`, a unit-peak Gaussian riding
    on the moving beam, and ``p`` is a single learnable scalar -- the floor the gate
    relaxes to away from the beam. This is the same ``G = g + p`` that
    :class:`~models.gmlp.model.GatedMLP` gates its dense stack with, and it is here
    for the same reason.

    ``p`` is what keeps the gate from being a straitjacket. With ``G = g`` alone the
    prediction is pinned to ``temperature_offset`` plus the bias everywhere ``g``
    has died off, which is everywhere beyond a beam radius or two; the diffused
    field and the trail the beam drags behind it are then not merely hard to fit but
    *unrepresentable*. With ``G = g + p`` the operator still gets the beam's shape
    for free near the source, while far from it the gate tends to ``p`` and the
    inner product is free to describe whatever is there. ``p`` therefore reads
    directly as how far the fit had to back away from the Gaussian prior: at
    ``p = 0`` this is the pure gate, and as ``p`` grows the gate flattens.

    The operator bias ``b`` is added *after* the gate, so it is not gated along with
    the inner product -- which is exactly where ``GatedMLP`` adds its own bias.

    The activation defaults to ``tanh`` to match :class:`~models.gpidon.model.GPiDoN`,
    so the two differ only in their objective. No residual is ever taken, so a
    piecewise-linear activation would work equally well here.

    Input and output normalisation is applied *inside* :meth:`forward`, so
    ``coords`` are passed in physical units while the subnetworks see well-scaled
    O(1) values -- and the gate, which is built from those same physical
    coordinates, stays in metres.
    """

    trunk_input_dim = 4

    # Declared so type checkers see `Tensor` rather than `Tensor | Module`, which
    # is what `nn.Module.__getattr__` is annotated to return for buffers.
    coord_mean: Tensor
    coord_scale: Tensor
    branch_mean: Tensor
    branch_scale: Tensor
    temperature_offset: Tensor
    temperature_scale: Tensor
    gaussian_exponent_scale: Tensor

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
        gaussian_exponent_scale: float = 1.0,
        gate_offset: float = 0.5,
    ) -> None:
        """See the class docstring for the trunk/branch normalisation. ``gaussian_exponent_scale``
        is the ``k`` of the Gaussian gate applied in :meth:`forward`: ``k > 1`` tightens the
        envelope around the beam, ``k < 1`` widens it. It is registered as a buffer so the
        sharpness this checkpoint was trained with travels with it.

        ``gate_offset`` is only the *initial* value of the learnable ``p`` in
        ``G = g + p``; training moves it, and the trained value is what rides in the
        state dict. It is a constructor argument rather than a buffer for that
        reason -- the number worth carrying is the one learned, not the one guessed.
        """
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
        # The `p` of `G = g + p`: the floor the gate relaxes to away from the beam.
        self.gate_offset = nn.Parameter(torch.tensor(float(gate_offset)))

        self.register_buffer(
            "coord_mean", _normalisation_buffer(coord_mean, self.trunk_input_dim, 0.0)
        )
        self.register_buffer(
            "coord_scale", _normalisation_buffer(coord_scale, self.trunk_input_dim, 1.0)
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
        self.register_buffer(
            "gaussian_exponent_scale", torch.tensor(float(gaussian_exponent_scale))
        )

    def forward(self, laser_power: Tensor, coords: Tensor) -> Tensor:
        """Predict temperature at ``coords`` for the process parameters ``laser_power``.

        ``laser_power`` is ``[batch, branch_input_dim]`` (a batch size of 1 is
        broadcast over the query points) and ``coords`` is ``[batch, 4]``
        holding ``(x, y, z, t)`` in physical units. Returns ``[batch, 1]`` in
        the same temperature unit as ``temperature_offset``.
        """
        if coords.dim() != 2 or coords.size(-1) != self.trunk_input_dim:
            raise ValueError(
                f"coords must have shape [batch, {self.trunk_input_dim}] for (x, y, z, t), "
                f"got {tuple(coords.shape)}"
            )

        normalised_coords = (coords - self.coord_mean) / self.coord_scale
        normalised_power = (laser_power - self.branch_mean) / self.branch_scale
        gate = beam_gaussian(coords, self.gaussian_exponent_scale) + self.gate_offset
        latent = self.operator(normalised_power, normalised_coords, gate=gate)
        return self.temperature_offset + self.temperature_scale * latent
