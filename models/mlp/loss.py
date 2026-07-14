"""The supervised objective of the MLP baseline: mean squared error, and nothing else.

The residual is divided by a characteristic temperature rise before squaring, so
the reported loss is dimensionless and comparable across datasets whose ``T``
ranges differ. That is the only thing separating this from a bare ``MSELoss``.
"""

from __future__ import annotations

from torch import Tensor, nn


class ScaledMSELoss(nn.Module):
    """``mean(((T_hat - T) / scale)^2)`` with ``scale`` a characteristic ``dT`` in Kelvin."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        if scale <= 0.0:
            raise ValueError(f"scale must be positive, got {scale}")
        self.scale = scale

    def forward(self, predicted: Tensor, measured: Tensor) -> Tensor:
        return ((predicted - measured) / self.scale).pow(2).mean()
