"""The MLP baseline with a Gaussian riding on the moving beam multiplied into it.

:mod:`models.gdon` gates the inner product of a DeepONet; this gates the output
of the plain dense stack of :mod:`models.mlp`, and changes nothing else:

    ``T_hat = T_amb + dT * (net(P, t, z, y, x) * G(x, y, z, t) + b)``
    ``G = g(x, y, z, t) + p``

``g`` is :func:`~models.gmlp.laser.beam_gaussian`, a unit-peak Gaussian centred on
the beam, and ``p`` is a single learnable scalar -- the floor the gate relaxes to
away from the beam.

``p`` is what keeps the gate from being a straitjacket. With ``G = g`` alone the
prediction is pinned to ``temperature_offset`` plus the bias everywhere ``g``
has died off, which is everywhere beyond a beam radius or two; the diffused field
and the trail the beam drags behind it are then not merely hard to fit but
*unrepresentable*. With ``G = g + p`` the network still gets the beam's shape for
free near the source, while far from it the gate tends to ``p`` and the dense
stack is free to describe whatever is there. ``p`` therefore reads directly as how
far the model had to back away from the Gaussian prior: at ``p = 0`` this is the
pure gate, and as ``p`` grows the gate flattens and the model approaches
:class:`~models.mlp.model.SimpleMLP`.

The head ``Linear`` carries no bias of its own -- it would be gated along with
everything else -- and ``b`` is a separate parameter added *after* the gate, which
is exactly where :class:`~models.gdon.model.GDoN` adds its operator bias.

Everything else is :class:`~models.mlp.model.SimpleMLP`: the same
``(P, t, z, y, x)`` input order, the same normalisation buffers, the same
supervised objective and no derivatives anywhere. The gate is the only
difference, which is what makes ``mlp`` the control for this model.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import Tensor, nn

from .laser import beam_gaussian

INPUT_DIM = 5  # (P, t, z, y, x)
INPUT_NAMES = ("P", "t", "z", "y", "x")

# Columns of a `(P, t, z, y, x)` row holding `(x, y, z, t)`, which is the order
# the beam geometry in `laser.py` is written against.
COORD_COLUMNS = (4, 3, 2, 1)


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


class GatedMLP(nn.Module):
    """``T_hat(P, t, z, y, x)`` as a dense stack, gated by a Gaussian on the beam.

    ``input_mean`` and ``input_scale`` are ``[5]`` in the input column order;
    ``temperature_offset`` and ``temperature_scale`` map the network's O(1) output
    back to Kelvin. All of them are buffers, as is ``gaussian_exponent_scale``, so
    they ride along in the state dict and a checkpoint reproduces the model
    exactly.
    """

    input_mean: Tensor
    input_scale: Tensor
    temperature_offset: Tensor
    temperature_scale: Tensor
    gaussian_exponent_scale: Tensor

    def __init__(
        self,
        hidden_layers: Sequence[int] = (256, 256, 256, 256),
        activation: type[nn.Module] = nn.SiLU,
        dropout: float = 0.0,
        use_bias: bool = True,
        input_mean: Sequence[float] | None = None,
        input_scale: Sequence[float] | None = None,
        temperature_offset: float = 0.0,
        temperature_scale: float = 1.0,
        gaussian_exponent_scale: float = 1.0,
        gate_offset: float = 0.5,
    ) -> None:
        """``gaussian_exponent_scale`` is the ``k`` of the Gaussian gate applied in
        :meth:`forward`: ``k > 1`` tightens the envelope around the beam, ``k < 1``
        widens it. It is registered as a buffer so the sharpness this checkpoint was
        trained with travels with it.

        ``gate_offset`` is only the *initial* value of the learnable ``p`` in
        ``G = g + p``; training moves it, and the trained value is what rides in the
        state dict. It is a constructor argument rather than a buffer for that
        reason -- the number worth carrying is the one learned, not the one guessed.
        """
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
        # No bias on the head: it would be multiplied by the gate along with the
        # rest of the output, and the gate-independent offset is `self.bias`.
        layers.append(nn.Linear(current, 1, bias=False))
        self.network = nn.Sequential(*layers)
        self.bias = nn.Parameter(torch.zeros(1)) if use_bias else None
        # The `p` of `G = g + p`: the floor the gate relaxes to away from the beam.
        self.gate_offset = nn.Parameter(torch.tensor(float(gate_offset)))

        self.register_buffer("input_mean", _normalisation_buffer(input_mean, INPUT_DIM, 0.0))
        self.register_buffer("input_scale", _normalisation_buffer(input_scale, INPUT_DIM, 1.0))
        self.register_buffer("temperature_offset", torch.tensor(float(temperature_offset)))
        self.register_buffer("temperature_scale", torch.tensor(float(temperature_scale)))
        self.register_buffer(
            "gaussian_exponent_scale", torch.tensor(float(gaussian_exponent_scale))
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """``[batch, 5]`` of ``(P, t, z, y, x)`` in SI units to ``[batch, 1]`` in Kelvin."""
        if inputs.dim() != 2 or inputs.size(-1) != INPUT_DIM:
            raise ValueError(
                f"inputs must have shape [batch, {INPUT_DIM}] for {INPUT_NAMES}, "
                f"got {tuple(inputs.shape)}"
            )

        normalised = (inputs - self.input_mean) / self.input_scale
        # The gaussian is built from the physical columns, so it stays in metres and
        # seconds while the dense stack sees O(1) values.
        gate = (
            beam_gaussian(inputs[:, COORD_COLUMNS], self.gaussian_exponent_scale)
            + self.gate_offset
        )

        latent = self.network(normalised) * gate
        if self.bias is not None:
            latent = latent + self.bias
        return self.temperature_offset + self.temperature_scale * latent
