"""The MLP baseline with the process parameter taken away.

:class:`~models.mlp.model.SimpleMLP` maps ``(P, t, z, y, x) -> T``. This maps

    ``(t, z, y, x) -> T``

and is otherwise the same network: the same dense stack, the same ``SiLU``, the
same normalisation buffers, the same supervised objective. The laser power is
the only thing removed.

It is the control for ``mlp``. The corpus holds seven powers, and the same
``(t, z, y, x)`` carries a different temperature in each of them, so a network
that cannot see ``P`` cannot resolve which one it is being asked about; the best
it can do is the power-averaged field. Whatever ``mlp`` achieves over this is
what the process parameter, as an input, is worth -- and the gap is the yardstick
the ``c``-prefixed models exist to provide.

The input order is ``(t, z, y, x)``: the axes from slowest to fastest varying,
which is the order the field is laid out in as ``[nt, nz, ny, nx]``, and exactly
``mlp``'s order with its leading ``P`` column dropped. The agent still answers
the shared ``(x, y, z, t, P)`` contract of ``predict_at`` and discards the ``P``
column on the way in.

Normalisation is applied inside :meth:`forward` so that callers pass and read
physical units while the hidden layers see O(1) values.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import Tensor, nn

INPUT_DIM = 4  # (t, z, y, x)
INPUT_NAMES = ("t", "z", "y", "x")


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


class ControlMLP(nn.Module):
    """``T_hat(t, z, y, x)`` as a dense stack, blind to the laser power.

    ``input_mean`` and ``input_scale`` are ``[4]`` in the input column order;
    ``temperature_offset`` and ``temperature_scale`` map the network's O(1)
    output back to Kelvin. All four are buffers, so they ride along in the state
    dict and a checkpoint reproduces the model exactly.
    """

    input_mean: Tensor
    input_scale: Tensor
    temperature_offset: Tensor
    temperature_scale: Tensor

    def __init__(
        self,
        hidden_layers: Sequence[int] = (256, 256, 256, 256),
        activation: type[nn.Module] = nn.SiLU,
        dropout: float = 0.0,
        input_mean: Sequence[float] | None = None,
        input_scale: Sequence[float] | None = None,
        temperature_offset: float = 0.0,
        temperature_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature_scale == 0.0:
            raise ValueError("temperature_scale must be non-zero")

        layers: list[nn.Module] = []
        current = INPUT_DIM
        for width in hidden_layers:
            layers.append(nn.Linear(current, width))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current = width
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

        self.register_buffer("input_mean", _normalisation_buffer(input_mean, INPUT_DIM, 0.0))
        self.register_buffer("input_scale", _normalisation_buffer(input_scale, INPUT_DIM, 1.0))
        self.register_buffer("temperature_offset", torch.tensor(float(temperature_offset)))
        self.register_buffer("temperature_scale", torch.tensor(float(temperature_scale)))

    def forward(self, inputs: Tensor) -> Tensor:
        """``[batch, 4]`` of ``(t, z, y, x)`` in SI units to ``[batch, 1]`` in Kelvin."""
        if inputs.dim() != 2 or inputs.size(-1) != INPUT_DIM:
            raise ValueError(
                f"inputs must have shape [batch, {INPUT_DIM}] for {INPUT_NAMES}, "
                f"got {tuple(inputs.shape)}"
            )

        normalised = (inputs - self.input_mean) / self.input_scale
        return self.temperature_offset + self.temperature_scale * self.network(normalised)
