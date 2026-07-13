"""Load a solver run and put its rows back on the grid they were written from.

``heat_torch.py`` flattens a uniform grid into ``[N, 6]`` rows of
``(x, y, z, t, P, T)``, one file per power. Nothing about the file says so, but
the rows are written with ``t`` outermost and then ``meshgrid(x, y, z,
indexing="ij")``, so they reshape back to ``(nt, nx, ny, nz)`` exactly. That is
worth checking rather than trusting, and :func:`load_run` does check it: if the
coordinate columns do not come back constant along the axes they should be
constant along, it refuses to hand you the field.

Every model in ``models/`` reads its data through here, so there is one
definition of "the grid" and one place for the reshape to be wrong.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

T_AMB = 298.0  # K, the ambient the solver holds the substrate at


@dataclass
class Run:
    """One power sweep on disk: the axes it lives on, and the fields themselves."""

    dir: Path
    files: list[Path]
    powers: np.ndarray  # (nP,) W
    x: np.ndarray  # (nx,) mm
    y: np.ndarray
    z: np.ndarray
    t: np.ndarray  # (nt,) s
    config: dict = field(default_factory=dict)  # config.json, if the run has one

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return len(self.t), len(self.x), len(self.y), len(self.z)

    @property
    def spacing(self) -> float:
        return float(self.x[1] - self.x[0])

    @property
    def snap_dt(self) -> float:
        return float(self.t[1] - self.t[0])

    def dT(self, i: int) -> np.ndarray:
        """``T - T_amb`` for power ``i``, shaped ``(nt, nx, ny, nz)``.

        Read as float64 -- the files are float64 and the FFTs downstream want the
        headroom -- which is ~316 MB per power on the production grid.
        """
        rows = np.load(self.files[i], mmap_mode="r")
        return (np.asarray(rows[:, 5], dtype=np.float64) - T_AMB).reshape(self.shape)

    def dT_all(self, dtype=np.float32) -> np.ndarray:
        """Every power at once, ``(nP, nt, nx, ny, nz)``. 1.1 GB on the fine grid."""
        out = np.empty((len(self.powers), *self.shape), dtype=dtype)
        for i in range(len(self.powers)):
            out[i] = self.dT(i)
        return out

    def index_of(self, power: float) -> int:
        i = int(np.argmin(np.abs(self.powers - power)))
        if abs(self.powers[i] - power) > 1e-9:
            raise ValueError(f"no {power} W in {self.dir}; have {list(self.powers)}")
        return i


def _power_of(path: Path) -> int:
    match = re.search(r"data_(\d+)W\.npy$", path.name)
    if match is None:
        raise ValueError(f"cannot read a power off {path.name}")
    return int(match.group(1))


def load_run(run_dir: Path) -> Run:
    """Read the axes and verify the rows really do fill the grid they claim to."""
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("data_*W.npy"), key=_power_of)
    if not files:
        raise SystemExit(f"no data_*W.npy under {run_dir}")

    rows = np.load(files[0], mmap_mode="r")
    x, y, z, t = (np.unique(np.asarray(rows[:, c])) for c in range(4))
    nt, nx, ny, nz = len(t), len(x), len(y), len(z)
    if nt * nx * ny * nz != rows.shape[0]:
        raise SystemExit(
            f"{files[0].name}: {rows.shape[0]} rows do not fill a "
            f"{nt}x{nx}x{ny}x{nz} grid"
        )

    # t outermost, then meshgrid(x, y, z, indexing="ij"). Check it, do not trust it.
    for col, axis in enumerate((x, y, z)):
        got = np.asarray(rows[:, col]).reshape(nt, nx, ny, nz)
        want = axis.reshape([-1 if a == col + 1 else 1 for a in range(4)])
        if not np.array_equal(got, np.broadcast_to(want, got.shape)):
            raise SystemExit(
                f"{files[0].name}: column {col} varies along an axis it should not; "
                "the row order is not what this loader assumes"
            )

    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}

    return Run(
        dir=run_dir,
        files=files,
        powers=np.array([_power_of(f) for f in files], dtype=float),
        x=x,
        y=y,
        z=z,
        t=t,
        config=config,
    )
