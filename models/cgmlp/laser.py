"""Geometry of the moving laser beam, used to gate the network output.

A copy of ``models/gmlp/laser.py``. Nothing in it mentions the laser power --
the beam's *position* is what the gate is built from, and that is a function of
time alone -- so the gate crosses over to the control model unchanged. Kept as a
copy rather than an import so an experiment on one model cannot shift the other
underneath it.

``BEAM_RADIUS``, ``LASER_START_X``, ``LASER_Y`` and ``SCAN_SPEED`` are recovered
from the data by ``python calibrate.py``.

Every quantity is SI (metres, seconds).
"""

from __future__ import annotations

import torch
from torch import Tensor

MM = 1.0e-3

# ---------------------------------------------------------------------------
# Fitted jointly across all seven powers by `python calibrate.py`.
# ---------------------------------------------------------------------------
BEAM_RADIUS = 1.6971 * MM  # r_b [m]
LASER_START_X = 4.8683 * MM
LASER_Y = 4.9929 * MM
SCAN_SPEED = 10.0000 * MM  # [m s^-1]

# Not fitted: the beam scans the top face of the exported domain, z = z_max.
LASER_Z = 6.0000 * MM


def beam_centre_x(t: Tensor | float) -> Tensor | float:
    """``x_l(t) = LASER_START_X + SCAN_SPEED * t``, the beam centre at time ``t``."""
    return LASER_START_X + SCAN_SPEED * t


def beam_gaussian(coords: Tensor, exponent_scale: Tensor | float = 1.0) -> Tensor:
    """``[batch, 1]`` unit-peak Gaussian around the beam, in space and time.

    ``g(x, y, z, t) = exp(-2*k*((x - x_l(t))^2 + (y - y_l)^2 + (z - z_l)^2) / r_b^2)``
    with ``k`` the ``exponent_scale``. It is 1 at the beam centre on the top face
    and falls off in every direction away from it. ``k > 1`` tightens that
    envelope, ``k < 1`` widens it; the peak is 1 either way because ``exp(0) = 1``
    regardless of ``k``.
    """
    x, y, z, t = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3], coords[:, 3:4]
    squared_distance = (
        (x - beam_centre_x(t)) ** 2 + (y - LASER_Y) ** 2 + (z - LASER_Z) ** 2
    )
    return torch.exp(-2.0 * exponent_scale * squared_distance / BEAM_RADIUS**2)
