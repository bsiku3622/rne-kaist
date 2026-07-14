"""Saving a checkpoint, and keeping the best one.

Lifted from ``typeulli-model-training/utils.py``. The run-naming and directory helpers
are not: an archive entry decides where a run writes (``share/archiving.py``), so there
is no global ``runs/`` or ``checkpoints/`` for them to resolve against any more.

What is kept is the part worth keeping. :func:`save_checkpoint` writes beside the target
and moves it into place, so a run killed mid-``torch.save`` cannot leave a truncated file
where the best checkpoint used to be -- and this session killed several.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn


def resolve_device(name: str | None = None) -> torch.device:
    """``name``, or CUDA when it is available and ``name`` is None."""
    if name is not None:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (p for p in parameters if p.requires_grad)
    return sum(p.numel() for p in parameters)


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically.

    A long run interrupted mid-``torch.save`` would otherwise leave a truncated
    file where the best checkpoint used to be, so the new one is written beside
    it and moved into place only once it is complete.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path, map_location: torch.device | str = "cpu"
) -> dict[str, Any]:
    """Read a checkpoint written by :class:`BestCheckpoint`."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    for key in ("model", "architecture"):
        if key not in checkpoint:
            raise KeyError(f"{path}: checkpoint is missing the `{key}` key")
    return checkpoint


class BestCheckpoint:
    """Keeps the single checkpoint with the best value of a monitored metric.

    ``mode='min'`` for losses and errors, ``mode='max'`` for scores. The payload
    is supplied by the caller so each model can store whatever ``agent.py`` needs
    to rebuild itself -- but ``model`` (the state dict) and ``architecture`` (the
    keyword arguments of its constructor) are mandatory, since the loaders
    depend on them.
    """

    def __init__(self, path: Path, mode: str = "min") -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.path = Path(path)
        self.mode = mode
        self.best = math.inf if mode == "min" else -math.inf
        self.step: int | None = None

    def is_better(self, value: float) -> bool:
        return value < self.best if self.mode == "min" else value > self.best

    def update(self, value: float, payload: dict[str, Any], step: int | None = None) -> bool:
        """Save ``payload`` if ``value`` improves on the best seen. Returns whether it did."""
        if not self.is_better(value):
            return False

        self.best = value
        self.step = step
        save_checkpoint(self.path, {**payload, "metric": value, "step": step})
        return True
