"""Which models exist, found by looking rather than by being told.

A model is a directory under ``models/`` with a ``train.py`` that exposes ``main(argv)``.
Adding one is enough to make ``python train.py <name>`` work; there is no list here to
forget to update, which is the failure mode a registry exists to prevent.

Taken from ``typeulli-model-training/models/__init__.py``, which had the same idea.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

MODELS = Path(__file__).resolve().parent.parent / "models"


def available() -> list[str]:
    """Every directory under ``models/`` that looks like a model."""
    return sorted(
        p.name
        for p in MODELS.iterdir()
        if p.is_dir() and not p.name.startswith(("_", ".")) and (p / "train.py").exists()
    )


def _import(name: str, module: str) -> ModuleType:
    known = available()
    if name not in known:
        raise ValueError(f"unknown model {name!r}; available: {', '.join(known)}")
    return importlib.import_module(f"models.{name}.{module}")


def train_module(name: str) -> ModuleType:
    """The model's ``train`` module, which must expose ``main(argv)``."""
    return _import(name, "train")


def build_agent(name: str, *args, **kwargs):
    """Construct the model's agent; arguments go straight to its ``build_agent``."""
    return _import(name, "agent").build_agent(*args, **kwargs)
