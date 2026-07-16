"""The network, and nothing else.

:class:`FourierMLP` is three dense layers that map ``(P, t)`` -- two numbers -- to
every stored Fourier coefficient of a snapshot at once. That is the whole model; what
makes it work is not the architecture but *what it is asked to predict*, and that --
the coefficient dataset, its reconstruction back to Kelvin, and the de-rotation that
makes the target learnable -- lives in :mod:`share.spectral`, because it is a property
of the representation, not of these layers.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class FourierMLP(nn.Module):
    """(P, t) -> every stored Fourier coefficient, as interleaved (Re, Im).

    The output layer is 99.6% of the weights, which sounds alarming and is not: the
    targets span a 28-dimensional subspace (measured, at 99.99% of their variance),
    so a 128-wide last hidden layer has four times the room it needs. The bottleneck
    was never capacity.
    """

    def __init__(self, n_out: int, width: int = 128, depth: int = 3):
        super().__init__()
        layers, d = [], 2  # (P, t)
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers += [nn.Linear(d, n_out)]
        self.net = nn.Sequential(*layers)
        self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("sd", torch.ones(1))

    def set_normalisation(self, mu, sd) -> None:
        self.mu = torch.as_tensor(mu, dtype=torch.float32)
        self.sd = torch.as_tensor(sd, dtype=torch.float32)

    def forward(self, x):
        return self.net(x)

    def denormalise(self, y: np.ndarray) -> np.ndarray:
        return y.astype(np.float64) * self.sd.cpu().numpy() + self.mu.cpu().numpy()
