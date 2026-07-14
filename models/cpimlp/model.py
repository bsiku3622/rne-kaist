"""The PiMLP network with the process parameter taken away.

:class:`~models.pimlp.model.PhysicsMLP` maps ``(x, y, z, t; P) -> T``. This maps

    ``(x, y, z, t) -> T``

and is otherwise the same network: the same dense stack, the same ``SiLU``, the
same normalisation, the same :meth:`derivatives` the PINN residuals are built
from. The laser power is the only thing removed.

That makes :meth:`forward` simpler than PiMLP's rather than more complicated. Its
sibling takes ``(laser_power, coords)`` as two tensors because the residuals are
written against ``(x, y, z, t)`` and autograd needs a tensor of exactly those four
columns to differentiate with respect to -- the power had to be kept out of the
way. Here there is no power to keep out of the way, so ``coords`` *is* the input,
and the reordering into the ``(t, z, y, x)`` the stack wants happens inside. The
chain rule is carried through that reordering and through the normalisation alike,
which keeps the residuals in ``loss.py`` dimensionally correct without any manual
rescaling.

The activation stays ``SiLU``, as in ``pimlp``: it is smooth, so the second
derivative the PDE residual needs exists. A piecewise-linear activation such as
``ReLU`` would not do -- it makes the Laplacian identically zero.

``physics_power`` is a buffer for the same reason ``gaussian_exponent_scale`` is
one in ``pimlp``: :meth:`forward` does not consume it, but the laser flux the
checkpoint was *fitted against* is a property of the checkpoint, and pinning it to
a single value is the substantive choice this model makes (see the module
docstring of :mod:`models.cpimlp.train`). It travels with the weights instead of
living as an easy-to-lose CLI flag.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class FieldDerivatives:
    """Predicted temperature field and its autograd derivatives.

    Every field is shaped ``[batch, 1]``. The second-order fields are ``None``
    when :meth:`ControlPhysicsMLP.derivatives` is called with
    ``second_order=False``.
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


class ControlPhysicsMLP(nn.Module):
    """``T_hat(x, y, z, t)`` as a dense stack trained against the PDE, blind to ``P``.

    ``input_mean`` and ``input_scale`` are ``[4]`` in the ``(t, z, y, x)`` column
    order the stack is fed in -- not the ``(x, y, z, t)`` order :meth:`forward`
    takes, which is the order the residuals are written against.
    ``temperature_offset`` and ``temperature_scale`` map the network's O(1) output
    back to Kelvin. All of them are buffers, so they ride along in the state dict
    and a checkpoint reproduces the model exactly.
    """

    coord_dim = 4  # (x, y, z, t)

    # Declared so type checkers see `Tensor` rather than `Tensor | Module`, which
    # is what `nn.Module.__getattr__` is annotated to return for buffers.
    input_mean: Tensor
    input_scale: Tensor
    temperature_offset: Tensor
    temperature_scale: Tensor
    gaussian_exponent_scale: Tensor
    physics_power: Tensor

    def __init__(
        self,
        hidden_layers: Sequence[int] = (256, 256, 256, 256),
        activation: type[nn.Module] = nn.SiLU,
        dropout: float = 0.0,
        input_mean: Sequence[float] | None = None,
        input_scale: Sequence[float] | None = None,
        temperature_offset: float = 0.0,
        temperature_scale: float = 1.0,
        gaussian_exponent_scale: float = 1.0,
        physics_power: float = 0.0,
    ) -> None:
        """See the class docstring for the normalisation. Neither
        ``gaussian_exponent_scale`` nor ``physics_power`` is consumed by
        :meth:`forward` -- they have no effect on the network itself -- they are
        carried here only so the laser source this checkpoint was *trained against*
        (see :func:`models.cpimlp.loss.laser_flux`) travels with the checkpoint
        instead of living as a pair of separate, easy-to-lose CLI flags.
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
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

        self.register_buffer("input_mean", _normalisation_buffer(input_mean, INPUT_DIM, 0.0))
        self.register_buffer("input_scale", _normalisation_buffer(input_scale, INPUT_DIM, 1.0))
        self.register_buffer("temperature_offset", torch.tensor(float(temperature_offset)))
        self.register_buffer("temperature_scale", torch.tensor(float(temperature_scale)))
        self.register_buffer(
            "gaussian_exponent_scale", torch.tensor(float(gaussian_exponent_scale))
        )
        self.register_buffer("physics_power", torch.tensor(float(physics_power)))

    def forward(self, coords: Tensor) -> Tensor:
        """Predict temperature at ``coords``.

        ``coords`` is ``[batch, 4]`` holding ``(x, y, z, t)`` in physical units.
        Returns ``[batch, 1]`` in the same temperature unit as
        ``temperature_offset``.
        """
        if coords.dim() != 2 or coords.size(-1) != self.coord_dim:
            raise ValueError(
                f"coords must have shape [batch, {self.coord_dim}] for (x, y, z, t), "
                f"got {tuple(coords.shape)}"
            )

        x, y, z, t = (coords[:, i : i + 1] for i in range(self.coord_dim))
        inputs = torch.cat((t, z, y, x), dim=-1)

        normalised = (inputs - self.input_mean) / self.input_scale
        return self.temperature_offset + self.temperature_scale * self.network(normalised)

    def derivatives(
        self,
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
                "coords must have requires_grad=True to differentiate through the input"
            )

        temperature = self(coords)
        first = torch.autograd.grad(
            temperature,
            coords,
            grad_outputs=torch.ones_like(temperature),
            create_graph=True,
        )[0]
        T_x, T_y, T_z, T_t = (first[:, i : i + 1] for i in range(self.coord_dim))

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
