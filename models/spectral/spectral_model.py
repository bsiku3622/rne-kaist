"""The model, and the two transforms that decide whether it works at all.

Nothing here is a network trick. :class:`SpectralMLP` is three hidden layers; what
makes the difference is *what it is asked to predict*.

**Reconstruction.** The Fourier basis already carries the space, so the network
takes ``(P, t)`` -- two numbers -- and returns every stored coefficient at once. One
forward pass is a whole temperature field; :func:`reconstruct` inverts it back onto
the grid.

**De-rotation.** A source moving at ``v`` makes the field roughly ``f(x - v t)``,
whose transform is ``g(kx) * exp(-2i pi kx v t)``: every coefficient spins in the
complex plane, and the faster the mode, the faster it spins. Over a 3 s run the
energetic modes turn 15-30 times, and an MLP cannot fit a target that oscillates
like that -- left alone, it put the peak 17% low. Undoing the spin analytically,
which is a change to the frame that travels with the laser, leaves an amplitude that
barely moves in time, and the same network lands at 1.3%. It is an exact, invertible
multiplication, so nothing is lost.

The phase is evaluated at the exact snapshot times, so **the snapshot interval does
not have to resolve the spin.** A dataset whose raw coefficients alias above
``|mx| = 5`` still learns every mode out to 38. That is worth stating because the
opposite is easy to assume: aliasing stops you *inferring* a frequency from samples,
and here the frequency is not inferred, it is known.

What does defeat de-rotation is a mode that is not travelling. Above ``|mx| ~ 30`` the
spectrum is dominated by the x-wrap discontinuity -- the 41 K step the DFT sees
between the far end of the plate and the near one -- which is pinned to the domain,
not to the laser. Multiplying that by a spin *adds* an oscillation instead of
removing one, and the fix for it is to detrend x, not to sample time more finely.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SpectralMLP(nn.Module):
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


def derotate_phase(mx: np.ndarray, run, times: np.ndarray, vel: float) -> np.ndarray:
    """``exp(+2i pi kx v t)``, the spin to divide out. Shaped ``(nt, len(mx))``."""
    Lx = run.shape[1] * run.spacing  # the period the DFT actually assumes
    return np.exp(2j * np.pi * (mx / Lx) * vel * times[:, None])


def reconstruct(coef: np.ndarray, meta: dict) -> np.ndarray:
    """Coefficients -> dT on the grid. The inverse of what dataset.py stored.

    ``y`` holds only its non-negative wavenumbers, since ``C(-kx, -ky) = conj(C(kx,
    ky))`` makes the rest redundant, so the way back is an ``irfftn`` over ``(x, y)``
    with ``z`` left alone.
    """
    nx, ny, nz = meta["grid"]
    mx, my = meta["mx"], meta["my"]
    nt = coef.shape[0]
    full = np.zeros((nt, nx, ny // 2 + 1, nz), dtype=np.complex128)
    full[:, mx[:, None], my[None, :], :] = coef
    return np.fft.irfftn(full, s=(nx, ny), axes=(1, 2), norm="ortho")
