"""Reorder the shared corpus into the ``(t, z, y, x)`` columns CGMLP expects.

A copy of :mod:`models.cmlp.dataset`: the gate lives inside the network and is
built from the coordinate columns of the input, so it changes nothing about how
batches are drawn. :class:`~dataset.SimulationDataset` hands out ``coords[N,4]``
as ``(x, y, z, t)`` alongside ``power[N,1]``; the network wants the spatial axes
reversed and the power not at all.
"""

from __future__ import annotations

import torch
from torch import Tensor

from share.corpus import SimulationDataset

# Column of `(x, y, z, t, P)` that each `(t, z, y, x)` column is drawn from; `P`
# has no column to be drawn into, which is the whole point of this model.
FROM_CONTRACT = (3, 2, 1, 0)


def to_inputs(coords: Tensor) -> Tensor:
    """``coords[N,4]`` as ``(x, y, z, t)`` to ``[N, 4]`` of ``(t, z, y, x)``."""
    x, y, z, t = (coords[:, i : i + 1] for i in range(4))
    return torch.cat((t, z, y, x), dim=-1)


def from_contract(inputs: Tensor) -> Tensor:
    """``[N, 5]`` of ``(x, y, z, t, P)`` to ``[N, 4]`` of ``(t, z, y, x)``; ``P`` is dropped."""
    return inputs[:, FROM_CONTRACT]


class CGMLPDataset:
    """Random ``(inputs[B,4], temperature[B,1])`` batches drawn from an in-memory split."""

    def __init__(self, labelled: SimulationDataset, generator: torch.Generator) -> None:
        self.labelled = labelled
        self.generator = generator

    def batch(self, count: int) -> tuple[Tensor, Tensor]:
        coords, _power, temperature = self.labelled.sample(count, self.generator)
        return to_inputs(coords), temperature

    def all(self) -> tuple[Tensor, Tensor]:
        """Every row of the split, in ``(inputs, temperature)`` form -- for evaluation."""
        return to_inputs(self.labelled.coords), self.labelled.temperature

    def normalisation(self) -> tuple[list[float], list[float]]:
        """``(mean, scale)`` per input column, taken from the corpus bounds.

        The box centre and half-width, not the sample mean and standard
        deviation: the grid is uniform, so they agree up to a constant, and the
        bounds are already known from the domain.
        """
        domain = self.labelled.domain
        centre, half_width = domain.center.tolist(), domain.half_width.tolist()

        # (t, z, y, x). `gmlp` normalises a fifth column, P, from zero; there is
        # no such column here.
        mean = [centre[3], centre[2], centre[1], centre[0]]
        scale = [half_width[3], half_width[2], half_width[1], half_width[0]]
        return mean, [value if value > 0.0 else 1.0 for value in scale]
