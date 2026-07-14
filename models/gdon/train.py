"""A DeepONet with the gaussian gate, fitted to data alone.

Ported from ``typeulli-model-training/models/gdon``. Its ``model.py``, ``loss.py``,
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

from .agent import GDoNAgent
from .dataset import GDoNDataset
from .loss import ScaledMSELoss
from .model import GDoN

NAME = "gdon"
AMBIENT = 298.0


def add_args(ap) -> None:
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 64, 64],
                    help="branch and trunk hidden widths")
    ap.add_argument("--latent", type=int, default=64,
                    help="the inner-product dimension the branch and trunk meet in")
    ap.add_argument("--gaussian-exponent-scale", type=float, default=1.0)
    ap.add_argument("--gate-offset", type=float, default=0.5)


def main(argv: list[str] | None = None) -> None:
    s = harness.prepare(NAME, __doc__, add_args, argv)
    a = s.args

    sampler = GDoNDataset(s.train, s.generator)
    architecture = dict(
        branch_input_dim=1,
        hidden_layers=tuple(a.hidden),
        latent_dim=a.latent,
        coord_mean=domain.center.tolist(),
        coord_scale=domain.half_width.tolist(),
        branch_mean=[0.0],
        branch_scale=[max_power],
        temperature_offset=AMBIENT,
        temperature_scale=s.rise,
        gaussian_exponent_scale=a.gaussian_exponent_scale,
        gate_offset=a.gate_offset,
    )
    model = s.to(GDoN(**architecture))
    criterion = ScaledMSELoss(scale=s.rise)

    def step():
        inputs, target = sampler.batch(a.batch_size)
        loss = criterion(model(inputs), target)
        return loss, {"data": loss.detach()}

    harness.go(NAME, s, model, architecture, GDoNAgent, step)


if __name__ == "__main__":
    main()
