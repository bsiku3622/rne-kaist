"""A physics-informed DeepONet: a branch over P, a trunk over (x, y, z, t).

Ported from ``typeulli-model-training/models/pidon``. Its ``model.py``, ``loss.py``,
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

from .agent import PiDoNAgent
from .dataset import PiDoNDataset
from .loss import (
    PROPERTIES,
    LossWeights,
    PINNLoss,
    ResidualScales,
    peak_laser_flux,
)
from .model import PiDoN

NAME = "pidon"
AMBIENT = 298.0


def add_args(ap) -> None:
    ap.add_argument("--batch-data", type=int, default=4096)
    ap.add_argument("--batch-physics", type=int, default=2048)
    ap.add_argument("--batch-boundary", type=int, default=1024)
    ap.add_argument("--w-data", type=float, default=1.0)
    ap.add_argument("--w-pde", type=float, default=1.0)
    ap.add_argument("--w-bc", type=float, default=1.0)
    ap.add_argument("--w-ic", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 64, 64],
                    help="branch and trunk hidden widths")
    ap.add_argument("--latent", type=int, default=64,
                    help="the inner-product dimension the branch and trunk meet in")
    ap.add_argument("--gaussian-exponent-scale", type=float, default=1.0)


def main(argv: list[str] | None = None) -> None:
    s = harness.prepare(NAME, __doc__, add_args, argv)
    a = s.args
    domain, max_power = s.corpus.domain, s.corpus.max_power

    sampler = PiDoNDataset(s.train, s.generator)
    architecture = dict(
        branch_input_dim=1,
        hidden_layers=tuple(a.hidden),
        latent_dim=a.latent,
        coord_mean=domain.center.tolist(),
        coord_scale=domain.half_width.tolist(),
        branch_mean=[0.0],
        branch_scale=[max_power],
        temperature_offset=PROPERTIES.ambient_temperature,
        temperature_scale=s.rise,
        gaussian_exponent_scale=a.gaussian_exponent_scale,
    )
    model = s.to(PiDoN(**architecture))

    scales = ResidualScales.characteristic(
        properties=PROPERTIES,
        temperature_rise=s.rise,
        time_scale=float(domain.upper[3]),
        peak_flux=float(peak_laser_flux(max_power)),
    )
    weights = LossWeights(data=a.w_data, pde=a.w_pde, bc=a.w_bc, ic=a.w_ic)
    criterion = PINNLoss(PROPERTIES, weights=weights, scales=scales)

    def step():
        total, components = criterion(
            model, **sampler.batches(a.batch_data, a.batch_physics, a.batch_boundary)
        )
        return total, {k: v.detach() for k, v in components.items()}

    harness.go(NAME, s, model, architecture, PiDoNAgent, step,
               extra={"scales": repr(scales), "weights": repr(weights)})


if __name__ == "__main__":
    main()
