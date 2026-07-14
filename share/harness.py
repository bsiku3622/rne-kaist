"""Everything a model does around its own architecture, done once.

The nine coordinate models fall into exactly two shapes.

**data** -- ``loss(model(inputs), target)``. One batch of labelled points, a scaled MSE,
nothing else.

**pinn** -- ``loss(model, data=…, physics=…, boundary=…)``. Three batches, and the loss
comes back as a total plus a dict of components (pde, top, bottom, ic, …) because it is
the sum of residuals evaluated in different places.

That is the whole difference. Everything else -- reading the corpus, holding out a power,
building the network from the data's own statistics, the optimiser, the schedule, the
validation cadence, the checkpoint, the archive entry, the figures -- was copied nine
times upstream and lives here once instead.

A model contributes three things and no more: ``build(domain, rise, max_power, args)``
returning ``(network, architecture)``, an ``add_args(parser)``, and the classes its
sampler and agent come from. Its ``train.py`` is the twenty lines that say so.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))

from share import archiving, metrics, plotting, training
from share.corpus import MM, SimulationDataset
from share.grid import load_run

AMBIENT = 298.0


def base_parser(name: str, doc: str) -> argparse.ArgumentParser:
    """The arguments every model takes, whatever it is inside."""
    ap = argparse.ArgumentParser(prog=f"train.py {name}", description=doc)
    ap.add_argument("--run", type=Path, required=True, help="a solver run under data/")
    ap.add_argument(
        "--holdout", type=float, default=175.0,
        help="the power kept out of training entirely, and scored on. Not a random "
             "sample of points from powers the model has already seen.",
    )
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--optimizer", choices=("adam", "lbfgs"), default="adam",
        help="lbfgs runs one inner update per step with a strong-Wolfe line search, so "
             "a fresh minibatch each step does not poison its curvature estimate",
    )
    ap.add_argument("--log-every", type=int, default=250, help="validation cadence")
    ap.add_argument("--scalar-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--double", action="store_true", help="run in float64")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--times", type=float, nargs="+", default=None,
                    help="snapshots to cut through in the figures; default: three spread over the run")
    ap.add_argument("--tag", type=str, default=None, help="suffix on the archive entry")
    ap.add_argument("--lock", action="store_true")
    return ap


class Predictor:
    """The model's field on the grid, through its own agent -- the contract, live.

    Validation needs the whole volume at every snapshot of the held-out power, and every
    model already knows how to produce one: that is what ``agent.py`` is for. Wrapping the
    *live* network rather than a checkpoint means the agent sees each update as it happens,
    so the figures, the metrics and the saved weights can never drift apart.
    """

    def __init__(self, agent_cls, model, run, device):
        nt, nx, ny, nz = run.shape
        bounds = torch.tensor(
            [
                [0.0, float(run.x[-1]) * MM],
                [0.0, float(run.y[-1]) * MM],
                [0.0, float(run.z[-1]) * MM],
                [0.0, float(run.t[-1])],
            ]
        )
        self.agent = agent_cls(model, bounds, shape=(nz, ny, nx), device=device)
        self.query = [[float(t), 0.0] for t in run.t]

    def __call__(self, power: float) -> np.ndarray:
        """dT on the grid, ``(nt, nx, ny, nz)`` -- our axis order, not the contract's."""
        q = [[t, power] for t, _ in self.query]
        volume = self.agent.predict_of(q)[:, 0]  # [nt, D, H, W] = (nt, nz, ny, nx)
        return volume.permute(0, 3, 2, 1).cpu().numpy() - AMBIENT


class Setup:
    """What every model needs before it can build anything, gathered once."""

    def __init__(self, a):
        self.args = a
        self.device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.float64 if a.double else torch.float32
        torch.manual_seed(a.seed)
        self.generator = torch.Generator(device=self.device).manual_seed(a.seed)

        self.run = load_run(a.run)
        self.corpus = SimulationDataset.from_dir(
            self.run.dir, device=self.device, dtype=self.dtype
        )
        self.train, self.held = self.corpus.split_by_power(a.holdout)
        self.rise = self.corpus.temperature_rise(AMBIENT)
        self.truth = self.run.dT(self.run.index_of(a.holdout))

        print(
            f"[data]  {self.run.dir.name}: {len(self.corpus):,} points, "
            f"{[int(p) for p in self.corpus.powers]} W\n"
            f"[split] {len(self.train):,} train, {len(self.held):,} held out at "
            f"{int(a.holdout)} W -- a whole power, never seen"
        )

    def to(self, model):
        return model.to(device=self.device, dtype=self.dtype)


def prepare(name: str, doc: str, add_args, argv) -> Setup:
    """Parse the arguments a model takes and load what it will be fitted to."""
    ap = base_parser(name, doc)
    add_args(ap)
    return Setup(ap.parse_args(argv))


def finish(name, a, run, model, architecture, predict, truth, entry, result, extra=None):
    """Score every power, render the figures, close the archive entry."""
    scores = {}
    for i, p in enumerate(run.powers):
        scores[int(p)] = metrics.score(predict(float(p)), run.dT(i))

    print(f"\n{'P':>6} {'RMSE':>9} {'Linf':>8} {'peak':>9}")
    print("-" * 36)
    for p, s in scores.items():
        held = "HELD OUT" if abs(p - a.holdout) < 1e-9 else ""
        print(f"{p:>5}W {s['rmse']:>8.3f}K {s['linf']:>7.1f}K {s['peak']:>8.2f}%   {held}")

    power = int(a.holdout)
    mine = predict(a.holdout)
    want = a.times or [float(run.t[len(run.t) // 4]), float(run.t[len(run.t) // 2]), float(run.t[-1])]
    plotting.planes(truth, mine, run, power, entry.figures / "field.png", name)
    plotting.scanline(truth, mine, run, power, want, entry.figures / "scanline.png", name)

    archiving.finalise(
        entry, run,
        config={**vars(a), "architecture": architecture, **(extra or {})},
        metrics={
            "wall_s": round(result.wall_s, 1),
            "select": result.best,
            "select_step": result.best_step,
            "holdout_power": power,
            "per_power": scores,
        },
        lock=a.lock,
    )


def go(name, s: Setup, model, architecture, agent_cls, step, extra=None) -> None:
    """Open the entry, run the loop, score every power, render, close.

    The last thing a model's ``train.py`` does, once it has built the three things only it
    knows how to build: the network, its optimisation step, and the agent that turns it
    back into a temperature field.
    """
    a, run = s.args, s.run
    predict = Predictor(agent_cls, model, run, s.device)
    entry = archiving.open_entry(name, run, a.tag)
    payload = {
        "architecture": architecture,
        "bounds": predict.agent.bounds.cpu(),
        "shape": predict.agent.shape,
        "holdout": a.holdout,
    }
    result = training.run(
        entry, model, step, lambda: predict(a.holdout), s.truth, a, payload
    )
    finish(name, a, run, model, architecture, predict, s.truth, entry, result, extra)
