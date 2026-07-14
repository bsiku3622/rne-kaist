"""Train one of the models under ``models/``.

The first argument names the model; everything after it goes, untouched, to that model's
own ``train.main(argv)``. Each therefore owns its hyperparameters instead of sharing a
lowest-common-denominator parser here -- including ``--help``, which is why the name is
split off before argparse ever sees the command line.

Every run opens an archive entry and writes into it directly: checkpoint, TensorBoard
events, figures, a snapshot of the code, hard links to the data it was fitted to, and the
provenance. There is nothing to clean up afterwards, nothing to collide over, and two
models can train at once.

Examples::

    python train.py --list
    python train.py gpidon --run data/torch_20260710-122446_100_250_0.25 --holdout 175
    python train.py mlp --run data/... --optimizer lbfgs --lr 1
    python train.py gdon --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from share.registry import available, train_module


def main() -> None:
    argv = sys.argv[1:]

    if argv and not argv[0].startswith("-"):
        try:
            module = train_module(argv[0])
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            raise SystemExit(2)
        module.main(argv[1:])
        return

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("model", nargs="?", help=f"one of: {', '.join(available())}")
    ap.add_argument("--list", action="store_true", help="print the models and exit")
    args = ap.parse_args(argv)

    if args.list:
        for name in available():
            print(name)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
