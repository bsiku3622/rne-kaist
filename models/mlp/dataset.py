"""Reorder the shared corpus into the ``(P, t, z, y, x)`` columns the MLP expects.

:class:`~dataset.SimulationDataset` hands out ``coords[N,4]`` as
``(x, y, z, t)`` alongside ``power[N,1]``. The network wants them in one tensor,
process parameter first and spatial axes reversed, so the transposition lives
here rather than being repeated at every call site.
"""

from __future__ import annotations

import torch
from torch import Tensor

from share.corpus import SimulationDataset

# Column of `(x, y, z, t, P)` that each `(P, t, z, y, x)` column is drawn from.
FROM_CONTRACT = (4, 3, 2, 1, 0)


def to_inputs(coords: Tensor, power: Tensor) -> Tensor:
    """``(coords[N,4] as (x,y,z,t), power[N,1])`` to ``[N, 5]`` of ``(P, t, z, y, x)``."""
    x, y, z, t = (coords[:, i : i + 1] for i in range(4))
    return torch.cat((power, t, z, y, x), dim=-1)


def from_contract(inputs: Tensor) -> Tensor:
    """``[N, 5]`` of ``(x, y, z, t, P)`` to ``[N, 5]`` of ``(P, t, z, y, x)``."""
    return inputs[:, FROM_CONTRACT]


class MLPDataset:
    """Random ``(inputs[B,5], temperature[B,1])`` batches drawn from an in-memory split."""

    def __init__(self, labelled: SimulationDataset, generator: torch.Generator) -> None:
        self.labelled = labelled
        self.generator = generator

    def batch(self, count: int) -> tuple[Tensor, Tensor]:
        coords, power, temperature = self.labelled.sample(count, self.generator)
        return to_inputs(coords, power), temperature

    def all(self) -> tuple[Tensor, Tensor]:
        """Every row of the split, in ``(inputs, temperature)`` form -- for evaluation."""
        return to_inputs(self.labelled.coords, self.labelled.power), self.labelled.temperature

    def normalisation(self) -> tuple[list[float], list[float]]:
        """``(mean, scale)`` per input column, taken from the corpus bounds.

        The box centre and half-width, not the sample mean and standard
        deviation: the grid is uniform, so they agree up to a constant, and the
        bounds are already known from the domain.
        """
        domain = self.labelled.domain
        centre, half_width = domain.center.tolist(), domain.half_width.tolist()
        max_power = self.labelled.max_power

        # (P, t, z, y, x); P is scaled from zero because a laser power of zero is
        # physically meaningful (no heating) while its centre is an artefact of
        # whichever powers happen to be in the sweep.
        mean = [0.0, centre[3], centre[2], centre[1], centre[0]]
        scale = [max_power, half_width[3], half_width[2], half_width[1], half_width[0]]
        return mean, [value if value > 0.0 else 1.0 for value in scale]
