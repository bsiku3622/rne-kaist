"""The training loop every model shares, and the two things it fixes.

The nine models this repository absorbed each carried their own copy of the loop --
2,300 lines of it, differing in one line. Data-only models compute
``criterion(model(inputs), target)``; physics-informed ones compute
``criterion(model, **sampler.batches(...))`` and get a dict of components back. Everything
around that -- the schedule, the gradient clip, the logging, the validation cadence, the
best-checkpoint bookkeeping -- was copied nine times, which is nine places for a fix to
land in eight of them.

So a model supplies a ``step()`` returning ``(total, components)``, and the rest lives
here once. That also means the optimiser is chosen in one place, which is what makes
``--optimizer lbfgs`` a flag rather than nine rewrites.

**Selection is on more than RMSE.** The upstream loop kept the checkpoint with the lowest
validation RMSE. RMSE is a volume average, the melt pool is a fraction of a percent of the
plate, and the two come apart: a model can sit at 14 K RMSE while flattening the peak by
44%, and that model looked best by the metric it was chosen with. So the score here is
composite -- see :func:`select` -- and the peak has a say in it.

**Validation is a held-out power.** Not a random tenth of the points drawn from powers the
model already trains on. The harder question is the one a surrogate exists to answer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from share import archiving, metrics
from share.checkpoints import BestCheckpoint, count_parameters

StepFn = Callable[[], tuple[torch.Tensor, dict[str, torch.Tensor]]]
FieldFn = Callable[[], np.ndarray]  # the model's dT on the grid, (nt, nx, ny, nz)


@dataclass
class Result:
    best: float
    best_step: int
    wall_s: float
    scores: dict = field(default_factory=dict)


def select(score: dict) -> float:
    """The number the best checkpoint is chosen by. Lower is better.

    RMSE alone is what the melt pool hides behind: it is an average over a plate that is
    almost entirely cold, so a model can flatten the peak by 40% and barely move it. The
    peak error is the thing the physics is actually about, and it is a percentage, so the
    two need a common footing before they can be added.

    A percent of peak error is worth 1 K of RMSE here. That is a judgement, not a
    derivation -- but it is a judgement made in the open, and it beats the alternative,
    which is a judgement made by omission.
    """
    return score["rmse"] + abs(score["peak"])


HISTORY = 20  # curvature pairs L-BFGS keeps; see optimiser_for


def optimiser_for(model, args):
    """Adam, or L-BFGS, and the schedule that goes with each.

    L-BFGS is a *deterministic* method wearing a stochastic one's interface. It approximates
    the inverse Hessian from the last ``HISTORY`` pairs ``(s_k, y_k)``, where
    ``y_k = grad f(x_{k+1}) - grad f(x_k)`` -- a difference that only measures curvature if
    ``f`` is the same function at both ends. Hand it a fresh minibatch every step and ``y_k``
    measures the sampling noise instead, and the strong-Wolfe line search, which evaluates
    the objective several times per step, is comparing values of different functions.

    So :func:`run` holds the batch still and this is where the rest of that follows from:
    the line search may take as many inner steps as it likes (``--lbfgs-inner``) because
    they are all on one objective, and no schedule sits on top of ``lr`` because the line
    search sets its own step length.
    """
    if args.optimizer == "lbfgs":
        opt = torch.optim.LBFGS(
            model.parameters(),
            lr=args.lr,
            max_iter=args.lbfgs_inner,
            history_size=HISTORY,
            line_search_fn="strong_wolfe",
        )
        return opt, None
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iterations)


def run(
    entry: archiving.Entry,
    model: torch.nn.Module,
    step: StepFn,
    field: FieldFn,
    truth: np.ndarray,
    args,
    payload: dict,
    generator: torch.Generator,
) -> Result:
    """Train, validate on the held-out power, and leave the best checkpoint in ``entry``.

    ``step()`` runs one optimisation step's forward pass and returns its loss and the
    components to log. ``field()`` returns the model's dT on the full grid of the held-out
    power, which ``truth`` is measured against with :mod:`share.metrics`.

    ``generator`` is the one every sampler draws from, and it is how L-BFGS gets the fixed
    objective it needs without a single model having to know that it exists: rewind the
    generator and ``step()`` redraws the batch it drew last time, to the point.
    """
    opt, sched = optimiser_for(model, args)
    best = BestCheckpoint(entry.checkpoint, mode="min")
    writer = SummaryWriter(log_dir=str(entry.tensorboard))
    archiving.live_link(entry)  # the event file exists now, so TensorBoard can be shown it
    lbfgs = args.optimizer == "lbfgs"

    print(
        f"[archive] {entry.dir.name}\n"
        f"[setup] {count_parameters(model):,} params, {args.optimizer}, lr {args.lr:g}, "
        f"{args.iterations} iterations"
        + (f", {args.lbfgs_inner} inner, batch held {args.lbfgs_cycle or args.iterations} steps"
           if lbfgs else "")
        + "\n"
    )

    # The state that draws the batch L-BFGS is currently minimising over. Rewinding to it
    # before every evaluation is what makes `step()` a function rather than a sample of one.
    batch = generator.get_state() if lbfgs else None
    seen: dict = {}

    def closure():
        generator.set_state(batch)
        opt.zero_grad(set_to_none=True)
        total, components = step()
        total.backward()
        seen["total"], seen["components"] = total.detach(), components
        return total

    started = time.time()
    progress = tqdm(
        range(1, args.iterations + 1),
        desc="train", unit="it", disable=args.no_progress, dynamic_ncols=True,
    )
    for iteration in progress:
        model.train()

        if lbfgs:
            opt.step(closure)
            total, components = seen["total"], seen["components"]

            if args.lbfgs_cycle and iteration % args.lbfgs_cycle == 0:
                # A new batch is a new objective, so every curvature pair in the history
                # now describes a function that is no longer being minimised. Keeping them
                # would let the old landscape steer steps on the new one, so the optimiser
                # is rebuilt: same weights, empty history. This is the cost of not being
                # full-batch, paid explicitly rather than absorbed as noise.
                batch = generator.get_state()
                opt, _ = optimiser_for(model, args)
        else:
            opt.zero_grad(set_to_none=True)
            total, components = step()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            sched.step()

        if iteration % args.scalar_every == 0 or iteration == 1:
            writer.add_scalar("loss/total", total.detach().item(), iteration)
            for name, value in components.items():
                writer.add_scalar(f"loss/{name}", float(value), iteration)
            if sched is not None:
                writer.add_scalar("lr", sched.get_last_lr()[0], iteration)
            progress.set_postfix(loss=f"{total.detach().item():.3e}", best=f"{best.best:.2f}")

        if iteration % args.log_every == 0 or iteration == args.iterations:
            model.eval()
            score = metrics.score(field(), truth)
            for name, value in score.items():
                writer.add_scalar(f"val/{name}", value, iteration)
            writer.add_scalar("val/select", select(score), iteration)

            progress.write(
                f"[{iteration:6d}] loss={total.detach().item():.4e} | "
                f"rmse={score['rmse']:7.3f}K  linf={score['linf']:7.1f}K  "
                f"peak={score['peak']:+7.2f}%  select={select(score):.3f}"
            )
            if best.update(select(score), {**payload, "model": model.state_dict(), **score}, iteration):
                progress.write(f"{'':>9}  ^ best so far")

    progress.close()
    writer.close()
    wall = time.time() - started
    print(f"\n[done] best select {best.best:.3f} at step {best.step}  ({wall:.0f} s)")
    return Result(best=best.best, best_step=best.step or 0, wall_s=wall)
