"""The control for mlp: the same network with the laser power taken away.

Ported from ``typeulli-model-training/models/cmlp``. Its ``model.py``, ``loss.py``,
``dataset.py`` and ``agent.py`` came across untouched: the physics in them is calibrated
in SI, and a unit slip inside a residual trains happily and is wrong. Only the harness
around them changed -- the split is a held-out power, the checkpoint is chosen on more
than RMSE, and the run writes into an archive entry rather than a global ``runs/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from share import harness

from .agent import CMLPAgent
from .dataset import CMLPDataset
from .loss import ScaledMSELoss
from .model import ControlMLP

NAME = "cmlp"
AMBIENT = 298.0


def add_args(ap) -> None:
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 64, 64],
                    help="hidden layer widths")


def main(argv: list[str] | None = None) -> None:
    s = harness.prepare(NAME, __doc__, add_args, argv)
    a = s.args

    sampler = CMLPDataset(s.train, s.generator)
    input_mean, input_scale = sampler.normalisation()
    architecture = dict(
        hidden_layers=tuple(a.hidden),
        input_mean=input_mean,
        input_scale=input_scale,
        temperature_offset=AMBIENT,
        temperature_scale=s.rise,
    )
    model = s.to(ControlMLP(**architecture))
    criterion = ScaledMSELoss(scale=s.rise)

    def step():
        inputs, target = sampler.batch(a.batch_size)
        loss = criterion(model(inputs), target)
        return loss, {"data": loss.detach()}

    harness.go(NAME, s, model, architecture, CMLPAgent, step)


if __name__ == "__main__":
    main()
