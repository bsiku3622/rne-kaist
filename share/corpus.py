"""Shared access to the solver output: the point cloud every coordinate model reads.

Lifted from ``typeulli-model-training/dataset.py`` and kept close to it on purpose --
the models copied alongside it were written against this exact API, in SI units and
this exact axis order, and a unit slip inside a PDE residual is the kind of bug that
trains happily and is wrong. Three things changed, and nothing else:

* it reads ``data/<run>/`` rather than a sibling checkout;
* :meth:`SimulationDataset.split_by_power` holds out a *power*, not a random tenth
  of the points -- the split the surrogate is actually meant to survive;
* the ``train``/``valid`` directory convention is gone with it.

The original docstring follows.

Shared access to the simulation output.

Every model in ``models/`` reads the same ``.npy`` corpus through this module, so
the on-disk layout, the unit conversion and the train/validation split are
defined exactly once.

The corpus is split on disk by laser power, not by row:

``rne-am-simulation/data/train``
    the powers a model may fit -- 100, 125, 150, 175, 200, 225, 250 W.
``rne-am-simulation/data/valid``
    powers held out entirely, to measure generalisation *across ``P``* rather
    than across points of a power the model has already seen.

:data:`DEFAULT_DATA_DIR` points at ``train`` alone, so the held-out powers
cannot be swept up by the recursive glob and silently trained on.

Each file is a structured grid of ``(x, y, z, t, P, T)`` rows -- ``161 x 41 x
25 x 8 = 1320200`` for the shipped sweep. ``P`` is constant within a file, so a
branch network only has something to learn once several files at different
powers are present; all of them are globbed and concatenated automatically.

Two views of the same data are offered:

* :class:`SimulationDataset` -- the flat point cloud, held entirely in memory,
  for random-batch training. Coordinates are in SI units (metres, seconds,
  Kelvin); the raw files store millimetres and are converted on load.
* :func:`load_grid` -- one file reshaped back onto its structured axes, for the
  calibration fit and the plotting scripts.

Grid fields are stored as ``[nt, nz, ny, nx]``, i.e. ``field[n, k, j, i]`` is
``T(x[i], y[j], z[k], t[n])``. The trailing three axes are the ``(D, H, W)`` of
a PyTorch ``Conv3d`` tensor, which is the layout every agent's ``predict_of``
returns; the raw files are in the opposite ``(x, y, z, t)`` order and are
transposed once here rather than at every call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

MM = 1.0e-3

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO / "data"

# A run directory holds every power together; which one is held out is a training
# argument, not a directory layout. See SimulationDataset.split_by_power.
DEFAULT_DATA_DIR = DATA_ROOT

# The resolution of the shipped simulation, as ``(D, H, W) = (nz, ny, nx)``.
# Agents default to reconstructing the field on exactly this grid.
DEFAULT_FIELD_SHAPE = (25, 41, 161)


def find_data_files(data_dir: Path) -> list[Path]:
    """Every ``.npy`` under ``data_dir``, recursively, in a stable order."""
    paths = sorted(data_dir.rglob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"no .npy files under {data_dir}")
    return paths


def _check_raw(path: Path, raw: np.ndarray) -> None:
    if raw.ndim != 2 or raw.shape[1] != 6:
        raise ValueError(
            f"{path}: expected [N, 6] of (x, y, z, t, P, T), got {raw.shape}"
        )


# ---------------------------------------------------------------------------
# Structured view
# ---------------------------------------------------------------------------


@dataclass
class Grid:
    """A simulation file reshaped onto its structured ``(x, y, z, t)`` axes."""

    name: str
    x: np.ndarray  # [nx] in metres
    y: np.ndarray  # [ny] in metres
    z: np.ndarray  # [nz] in metres
    t: np.ndarray  # [nt] in seconds
    temperature: np.ndarray  # [nt, nz, ny, nx] in Kelvin
    power: float  # W

    @property
    def spacing(self) -> tuple[float, float, float, float]:
        """``(dx, dy, dz, dt)``, valid because every axis is uniformly sampled."""
        return (
            float(self.x[1] - self.x[0]),
            float(self.y[1] - self.y[0]),
            float(self.z[1] - self.z[0]),
            float(self.t[1] - self.t[0]),
        )

    @property
    def field_shape(self) -> tuple[int, int, int]:
        """``(D, H, W) = (nz, ny, nx)``."""
        return (self.z.size, self.y.size, self.x.size)


def load_grid(path: Path) -> Grid:
    """Reshape an ``[N, 6]`` file of ``(x, y, z, t, P, T)`` rows onto its grid."""
    raw = np.load(path).astype(np.float64)
    _check_raw(path, raw)

    axes = [np.unique(raw[:, i]) for i in range(4)]
    index = tuple(np.searchsorted(axis, raw[:, i]) for i, axis in enumerate(axes))
    field = np.full([len(axis) for axis in axes], np.nan)
    field[index] = raw[:, 5]
    if np.isnan(field).any():
        raise ValueError(f"{path}: rows do not fill a complete structured grid")

    powers = np.unique(raw[:, 4])
    if powers.size != 1:
        raise ValueError(f"{path}: expected a single laser power, got {powers.tolist()}")

    x, y, z, t = axes
    return Grid(
        name=path.name,
        x=x * MM,
        y=y * MM,
        z=z * MM,
        t=t,
        temperature=np.transpose(field, (3, 2, 1, 0)),  # (x,y,z,t) -> (t,z,y,x)
        power=float(powers[0]),
    )


def file_power(path: Path) -> float:
    """The laser power of a file, read from its first row without loading it.

    ``P`` is constant within a file, so one row settles it -- and probing a
    multi-gigabyte file just to read a single scalar column would otherwise
    load the whole thing.
    """
    return float(np.load(path, mmap_mode="r")[0, 4])


def find_grid(
    data_dir: Path, power: float = 0.0, tolerance: float = 1e-9
) -> Grid:
    """Load the simulation file whose laser power matches ``power``."""
    available = []
    for path in find_data_files(data_dir):
        candidate = file_power(path)
        if abs(candidate - power) < tolerance:
            return load_grid(path)
        available.append(candidate)
    raise ValueError(f"no file at P = {power} W; available: {sorted(available)}")


# ---------------------------------------------------------------------------
# Flat view
# ---------------------------------------------------------------------------


@dataclass
class Domain:
    """Axis-aligned space-time bounds, ``[4, 2]`` as ``(lower, upper)`` per axis."""

    bounds: Tensor

    @property
    def lower(self) -> Tensor:
        return self.bounds[:, 0]

    @property
    def upper(self) -> Tensor:
        return self.bounds[:, 1]

    @property
    def center(self) -> Tensor:
        return 0.5 * (self.lower + self.upper)

    @property
    def half_width(self) -> Tensor:
        return 0.5 * (self.upper - self.lower)

    def uniform(self, count: int, generator: torch.Generator) -> Tensor:
        """``[count, 4]`` points drawn uniformly from the box."""
        unit = torch.rand(
            (count, self.bounds.size(0)),
            generator=generator,
            device=self.bounds.device,
            dtype=self.bounds.dtype,
        )
        return self.lower + unit * (self.upper - self.lower)

    def face(
        self, axis: int, upper: bool, count: int, generator: torch.Generator
    ) -> Tensor:
        """Uniform points on one axis-aligned face, with that axis pinned to a bound."""
        coords = self.uniform(count, generator)
        coords[:, axis] = self.upper[axis] if upper else self.lower[axis]
        return coords


class SimulationDataset(Dataset):
    """The whole corpus as a flat point cloud of ``(x, y, z, t) -> T`` at a given ``P``.

    Held entirely in memory (and on the training device) because every loss term
    draws random batches from it; there is no sequential epoch to stream.
    """

    def __init__(self, coords: Tensor, power: Tensor, temperature: Tensor) -> None:
        if coords.size(0) != power.size(0) or coords.size(0) != temperature.size(0):
            raise ValueError("coords, power and temperature must agree in length")
        if coords.dim() != 2 or coords.size(1) != 4:
            raise ValueError(f"coords must be [N, 4], got {tuple(coords.shape)}")

        self.coords = coords
        self.power = power
        self.temperature = temperature

    # -- construction --------------------------------------------------------

    @classmethod
    def from_files(
        cls,
        paths: list[Path],
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        clip_below: float | None = None,
        verbose: bool = True,
    ) -> "SimulationDataset":
        """Load and concatenate ``paths``.

        ``clip_below`` raises every temperature to that floor. Sub-ambient rows
        (~5% of the corpus) are a solver artefact -- with heating only, ``T`` can
        never fall below ``T_amb`` -- but they are left untouched unless asked
        for, rather than silently editing the source data.
        """
        if not paths:
            raise FileNotFoundError("no .npy files given")

        coord_blocks, power_blocks, temperature_blocks = [], [], []
        for path in paths:
            raw = np.load(path).astype(np.float64)
            _check_raw(path, raw)

            coords = raw[:, :4].copy()
            coords[:, :3] *= MM
            power = raw[:, 4:5].copy()
            temperature = raw[:, 5:6].copy()

            if clip_below is not None:
                below = temperature < clip_below
                if below.any():
                    if verbose:
                        print(
                            f"[data] {path.name}: {below.sum()} of {below.size} rows "
                            f"below {clip_below} K (min {temperature.min():.2f} K) -- clipped"
                        )
                    temperature = np.maximum(temperature, clip_below)

            if verbose:
                print(
                    f"[data] {path.name}: {raw.shape[0]} rows, "
                    f"P={np.unique(power).tolist()} W"
                )
            coord_blocks.append(coords)
            power_blocks.append(power)
            temperature_blocks.append(temperature)

        def stack(blocks: list[np.ndarray]) -> Tensor:
            return torch.as_tensor(np.concatenate(blocks), dtype=dtype, device=device)

        return cls(stack(coord_blocks), stack(power_blocks), stack(temperature_blocks))

    @classmethod
    def from_dir(
        cls, data_dir: Path, **kwargs
    ) -> "SimulationDataset":
        """Load every ``.npy`` under ``data_dir``, recursively."""
        return cls.from_files(find_data_files(data_dir), **kwargs)

    # -- Dataset protocol ----------------------------------------------------

    def __len__(self) -> int:
        return self.coords.size(0)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.coords[index], self.power[index], self.temperature[index]

    # -- statistics ----------------------------------------------------------

    @property
    def domain(self) -> Domain:
        """The bounding box of the sampled ``(x, y, z, t)``."""
        return Domain(
            bounds=torch.stack(
                (self.coords.min(dim=0).values, self.coords.max(dim=0).values), dim=1
            )
        )

    @property
    def powers(self) -> Tensor:
        """The distinct laser powers present, ascending."""
        return torch.unique(self.power)

    @property
    def max_power(self) -> float:
        return float(self.powers.max())

    def temperature_rise(self, reference: float) -> float:
        """``T_max - reference``, the characteristic scale of the output."""
        return float(self.temperature.max()) - reference

    # -- slicing -------------------------------------------------------------

    def subset(self, index: Tensor) -> "SimulationDataset":
        return SimulationDataset(
            self.coords[index], self.power[index], self.temperature[index]
        )

    def split_by_power(
        self, holdout: float, tolerance: float = 1e-9
    ) -> tuple["SimulationDataset", "SimulationDataset"]:
        """``(train, validation)`` split by laser power: the held-out one is never seen.

        A random split by row asks whether the model can fill in points of a power it
        has already been shown -- a far easier question than the one a surrogate exists
        to answer, which is what happens at a power nobody simulated. Every number this
        repository quotes for generalisation comes from here.
        """
        held = (self.power.squeeze(-1) - holdout).abs() < tolerance
        if not held.any():
            available = [float(p) for p in self.powers]
            raise ValueError(f"no rows at {holdout} W; the corpus holds {available}")
        index = torch.arange(len(self), device=self.coords.device)
        return self.subset(index[~held]), self.subset(index[held])

    def split(
        self, val_fraction: float, generator: torch.Generator
    ) -> tuple["SimulationDataset", "SimulationDataset"]:
        """Random ``(train, validation)`` split by row.

        Kept for the models that were written against it, but it is not the split this
        repository reports: see :meth:`split_by_power`.
        """
        if not 0.0 <= val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

        permutation = torch.randperm(
            len(self), generator=generator, device=self.coords.device
        )
        cut = int(val_fraction * len(self))
        return self.subset(permutation[cut:]), self.subset(permutation[:cut])

    def sample(
        self, count: int, generator: torch.Generator
    ) -> tuple[Tensor, Tensor, Tensor]:
        """A random ``(coords, power, temperature)`` batch drawn with replacement."""
        index = torch.randint(
            0, len(self), (count,), generator=generator, device=self.coords.device
        )
        return self.coords[index], self.power[index], self.temperature[index]

    def sample_power(self, count: int, generator: torch.Generator) -> Tensor:
        """``[count, 1]`` branch inputs drawn from the powers present in the corpus."""
        values = self.powers
        index = torch.randint(
            0, values.numel(), (count,), generator=generator, device=values.device
        )
        return values[index].reshape(count, 1)
