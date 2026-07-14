"""mlp with the PDE, the boundaries and the initial condition added to the loss.

Ported from ``typeulli-model-training/models/pimlp``. Its ``model.py``, ``loss.py``,
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

from .agent import PiMLPAgent
from .dataset import PiMLPDataset, normalisation
from .loss import (
    PROPERTIES,
    LossWeights,
    PINNLoss,
    ResidualScales,
    peak_laser_flux,
)
from .model import PhysicsMLP

NAME = "pimlp"
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
                    help="hidden layer widths")
    ap.add_argument("--gaussian-exponent-scale", type=float, default=1.0)


def main(argv: list[str] | None = None) -> None:
    s = harness.prepare(NAME, __doc__, add_args, argv)
    a = s.args
    domain, max_power = s.corpus.domain, s.corpus.max_power

    sampler = PiMLPDataset(s.train, s.generator)
    input_mean, input_scale = normalisation(domain, max_power)
    architecture = dict(
        hidden_layers=tuple(a.hidden),
        input_mean=input_mean,
        input_scale=input_scale,
        temperature_offset=PROPERTIES.ambient_temperature,
        temperature_scale=s.rise,
        gaussian_exponent_scale=a.gaussian_exponent_scale,
    )
    model = s.to(PhysicsMLP(**architecture))

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

    harness.go(NAME, s, model, architecture, PiMLPAgent, step,
               extra={"scales": repr(scales), "weights": repr(weights)})


if __name__ == "__main__":
    main()
