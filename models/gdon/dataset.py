"""Random labelled batches for GDoN, in the ``(power, coords)`` form its model takes.

There are no collocation, boundary or initial-condition batches here: without a
PDE residual the only points GDoN ever sees are the simulation's own rows. That
is the entire difference from :mod:`models.gpidon.dataset`.
"""

from __future__ import annotations

import torch
from torch import Tensor

from share.corpus import SimulationDataset


class GDoNDataset:
    """Random ``(power[B,1], coords[B,4], temperature[B,1])`` batches from a split."""

    def __init__(self, labelled: SimulationDataset, generator: torch.Generator) -> None:
        self.labelled = labelled
        self.generator = generator

    def batch(self, count: int) -> tuple[Tensor, Tensor, Tensor]:
        coords, power, temperature = self.labelled.sample(count, self.generator)
        return power, coords, temperature
