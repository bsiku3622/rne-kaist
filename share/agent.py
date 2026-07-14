"""The inference contract every model exposes through its ``agent.py``.

A trained network is only useful to the plotting and analysis scripts through
two questions, so those are the only two an agent has to answer:

``predict_at(inputs)``
    ``inputs`` is ``[B, 5]`` of ``(x, y, z, t, P)`` and the result is ``[B, 1]``
    of temperature in Kelvin. Arbitrary points, no structure assumed -- this is
    what a slice, a line cut or a scattered probe needs.

``predict_of(inputs)``
    ``inputs`` is ``[B, 2]`` of ``(t, P)`` and the result is ``[B, 1, D, H, W]``
    of temperature in Kelvin: the whole three-dimensional field, one volume per
    row. The trailing axes follow the ``Conv3d`` convention ``D = z``, ``H = y``,
    ``W = x``, so ``field[b, 0, k, j, i]`` is ``T(x[i], y[j], z[k])`` at that
    row's ``(t, P)``.

Only :meth:`BaseAgent.predict_at` is abstract. ``predict_of`` is derived from it
here by evaluating the reconstruction grid, so a new model gets the volumetric
view for free and the two can never disagree.

A model may run the other way. ``models/spectral`` returns the coefficients of the
whole field in one pass, so the volume is *its* primitive and a single point costs
exactly as much as all 1.27 million of them; it overrides ``predict_of`` with the
direct route and derives ``predict_at`` by sampling the reconstruction. Either end
may be the primitive. What the contract insists on is that the two agree, and they
do, because one is always computed from the other.

Spatial and temporal bounds come from the checkpoint, not from the dataset, so
plotting a saved model never touches the 13 GB corpus.
"""

from __future__ import annotations

import abc
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from share.corpus import DEFAULT_FIELD_SHAPE

ArrayLike = Tensor | np.ndarray | Sequence


class BaseAgent(abc.ABC):
    """Wraps a trained network as a temperature field ``T(x, y, z, t; P)``.

    ``bounds`` is ``[4, 2]`` of ``(lower, upper)`` for ``(x, y, z, t)`` in SI
    units and defines the box :meth:`predict_of` reconstructs on. ``shape`` is
    ``(D, H, W) = (nz, ny, nx)``, defaulting to the resolution of the simulation
    the model was fitted to.
    """

    def __init__(
        self,
        bounds: Tensor,
        shape: tuple[int, int, int] = DEFAULT_FIELD_SHAPE,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        chunk: int = 65536,
    ) -> None:
        bounds = torch.as_tensor(bounds, dtype=dtype, device=device)
        if bounds.shape != (4, 2):
            raise ValueError(f"bounds must be [4, 2], got {tuple(bounds.shape)}")
        if len(shape) != 3 or any(int(size) < 1 for size in shape):
            raise ValueError(f"shape must be three positive ints, got {shape}")

        self.bounds = bounds
        self.shape = (int(shape[0]), int(shape[1]), int(shape[2]))
        self.device = torch.device(device)
        self.dtype = dtype
        self.chunk = chunk
        self._grid: Tensor | None = None

    # -- the grid ``predict_of`` reconstructs on -----------------------------

    @property
    def axes(self) -> tuple[Tensor, Tensor, Tensor]:
        """``(z, y, x)`` node positions in metres, matching ``(D, H, W)``."""
        depth, height, width = self.shape
        return tuple(  # type: ignore[return-value]
            torch.linspace(
                float(self.bounds[axis, 0]),
                float(self.bounds[axis, 1]),
                count,
                dtype=self.dtype,
                device=self.device,
            )
            for axis, count in ((2, depth), (1, height), (0, width))
        )

    @property
    def grid(self) -> Tensor:
        """``[D*H*W, 3]`` of ``(x, y, z)``, in the row order of a ``[D, H, W]`` volume."""
        if self._grid is None:
            z, y, x = self.axes
            grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing="ij")
            self._grid = torch.stack(
                (grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)), dim=-1
            )
        return self._grid

    # -- inference -----------------------------------------------------------

    @abc.abstractmethod
    def predict_at(self, inputs: ArrayLike) -> Tensor:
        """``[B, 5]`` of ``(x, y, z, t, P)`` to ``[B, 1]`` of Kelvin."""

    @torch.no_grad()
    def predict_of(self, inputs: ArrayLike) -> Tensor:
        """``[B, 2]`` of ``(t, P)`` to ``[B, 1, D, H, W]`` of Kelvin."""
        inputs = self._as_tensor(inputs, columns=2, name="predict_of")
        depth, height, width = self.shape
        grid = self.grid
        nodes = grid.size(0)

        volumes = []
        for row in inputs:
            time, power = row[0], row[1]
            query = torch.empty((nodes, 5), dtype=self.dtype, device=self.device)
            query[:, :3] = grid
            query[:, 3] = time
            query[:, 4] = power
            volumes.append(self.predict_at(query).reshape(depth, height, width))

        return torch.stack(volumes).unsqueeze(1)

    # -- helpers -------------------------------------------------------------

    def _as_tensor(self, inputs: ArrayLike, columns: int, name: str) -> Tensor:
        tensor = torch.as_tensor(inputs, dtype=self.dtype, device=self.device)
        if tensor.dim() != 2 or tensor.size(1) != columns:
            raise ValueError(
                f"{name} expects [B, {columns}], got {tuple(tensor.shape)}"
            )
        return tensor
