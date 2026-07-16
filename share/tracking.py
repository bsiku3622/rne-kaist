"""Where a run's numbers go while it trains -- TensorBoard, or Weights & Biases.

The loop logs the same handful of scalars every model reports -- ``loss/total``, its
components, the learning rate, the held-out scores -- and it has no reason to care which
of these is listening. So it logs to a :class:`Tracker`, and the choice of backend is one
flag (``--logger``) resolved here, the same way ``--optimizer`` is resolved in one place
in :mod:`share.training`.

**TensorBoard** writes event files into the archive entry, and a hard link in ``runs/``
lets it be watched live while the run is still going (see :func:`share.archiving.live_link`).

**Weights & Biases** streams to the project instead. It keeps its own ``wandb/`` directory
*under* the entry and takes the entry's name, so the online run and the on-disk archive line
up one-to-one. Either way the entry is the record; the tracker is only the monitor.

Both third-party imports are deferred into the tracker that needs them. A machine that only
ever runs TensorBoard need not have ``wandb`` installed, and one that only ever streams to
W&B is not made to carry ``tensorboard`` -- the flag decides, at the moment of choosing, and
nothing is imported for the road not taken.
"""

from __future__ import annotations

from typing import Protocol

from share import archiving


class Tracker(Protocol):
    """The three things the training loop asks of whatever it is logging to."""

    def add_scalar(self, tag: str, value: float, step: int) -> None: ...

    def add_hparams(self, hparams: dict, metrics: dict) -> None: ...

    def close(self) -> None: ...


class TensorBoardTracker:
    """A ``SummaryWriter``, plus the live link that lets TensorBoard watch the run.

    The writer is created against the entry's ``tensorboard/`` directory, which is where the
    archive keeps the event files; :func:`share.archiving.live_link` is called the moment the
    writer exists, because an event file has to be on disk before ``runs/`` can hard-link to
    it.
    """

    def __init__(self, entry: archiving.Entry):
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=str(entry.tensorboard))
        archiving.live_link(entry)  # the event file exists now; TensorBoard can be shown it

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self._writer.add_scalar(tag, value, step)

    def add_hparams(self, hparams: dict, metrics: dict) -> None:
        self._writer.add_hparams(hparams, metrics)

    def close(self) -> None:
        self._writer.close()


def _config_value(value):
    """Make a CLI argument safe to hand W&B: keep the scalars, stringify the rest.

    ``vars(args)`` carries ``Path`` objects (``--run``) and the odd namespace value that
    W&B's config would not know how to store. Lists survive -- ``--times`` is one -- and
    everything unfamiliar becomes its string, which is all the config panel wants of it.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_config_value(v) for v in value]
    return str(value)


class WandbTracker:
    """A Weights & Biases run, named and stored so it lines up with the archive entry.

    Multiple ``add_scalar`` calls at the same ``step`` are how the loop logs -- ``loss/total``,
    each component and ``lr`` all land on one iteration -- and W&B collects keys logged at the
    same step into a single history point, flushing it only when the step moves on. So the
    obvious call is also the correct one; nothing has to be buffered here.
    """

    def __init__(self, entry: archiving.Entry, args, *, config: dict):
        import wandb

        self._run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=entry.dir.name,          # the archive entry's name, so the two are the same run
            dir=str(entry.dir),           # wandb/ lands inside the entry, not the repo root
            mode=args.wandb_mode,
            config={k: _config_value(v) for k, v in config.items()},
        )

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self._run.log({tag: value}, step=step)

    def add_hparams(self, hparams: dict, metrics: dict) -> None:
        self._run.config.update({k: _config_value(v) for k, v in hparams.items()},
                                allow_val_change=True)
        self._run.summary.update(metrics)

    def close(self) -> None:
        self._run.finish()


def add_tracker_args(ap) -> None:
    """The logging flags every ``train.py`` takes, added in one place.

    The shared harness pulls these in through :func:`share.harness.base_parser`; the models
    that keep their own parser (fmlp, rmlp, deeponet) call this directly, so ``--logger`` means
    the same thing wherever a run is started from.
    """
    ap.add_argument(
        "--logger", choices=("tensorboard", "wandb"), default="tensorboard",
        help="where the run's scalars go. tensorboard writes into the archive entry; "
             "wandb streams to a project. See share/tracking",
    )
    ap.add_argument("--wandb-project", type=str, default="rne-kaist",
                    help="W&B project the run is logged under (default: rne-kaist)")
    ap.add_argument("--wandb-entity", type=str, default=None,
                    help="W&B team or user; default: whoever is logged in")
    ap.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online",
                    help="online streams live; offline logs to disk to sync later; "
                         "disabled is a no-op run")


def make_tracker(entry: archiving.Entry, args, *, config: dict) -> Tracker:
    """The tracker the run logs to, chosen by ``--logger``. TensorBoard unless told otherwise.

    ``config`` is the run's resolved arguments (and whatever else is worth recording); it is
    ignored by TensorBoard, which the loop logs ``add_hparams`` to at the end instead, and it
    becomes the W&B run's config panel.
    """
    if getattr(args, "logger", "tensorboard") == "wandb":
        return WandbTracker(entry, args, config=config)
    return TensorBoardTracker(entry)
