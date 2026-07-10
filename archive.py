"""Archive a finished training run as a complete, read-only historical record.

Copies the checkpoint, the TensorBoard run, the data it was trained on, any
figures already rendered from it, and the exact code that produced it into
``archive/<run>_<N>powers/``, then locks every file read-only and clears the
root (every subdirectory under ``runs/`` -- not just ``<run>`` -- plus
``checkpoint.pt``, ``train.log``, ``train.err``, ``.train.pid``,
``figures/*.png``). ``runs/`` itself is kept (TensorBoard needs the directory
to exist) and ``data/`` is left alone -- it is the live working dataset, not
part of any one run.

The code files (``calibrate.py``, ``loss.py``, ``model.py``, ``train.py``,
``visualize.py``) land directly in the entry, alongside ``checkpoint.pt`` and
``train.log`` -- not nested in their own subfolder.

Run after training finishes and, if wanted, after ``visualize.py`` has
written its figures.

Example::

    python archive.py --run 20260710-124432
"""

from __future__ import annotations

import argparse
import shutil
import stat
from pathlib import Path

CODE_FILES = ("calibrate.py", "loss.py", "model.py", "train.py", "visualize.py")
ROOT_CLEANUP = ("checkpoint.pt", "train.log", "train.err", ".train.pid")


def lock(path: Path) -> None:
    """Recursively mark every file under ``path`` (or ``path`` itself) read-only."""
    if path.is_dir():
        for file in path.rglob("*"):
            if file.is_file():
                file.chmod(stat.S_IREAD)
    else:
        path.chmod(stat.S_IREAD)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run name under --logdir, e.g. 20260710-124432")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint.pt"))
    parser.add_argument("--train-log", type=Path, default=Path("train.log"))
    parser.add_argument("--logdir", type=Path, default=Path("runs"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--archive-dir", type=Path, default=Path("archive"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = args.logdir / args.run
    if not run_dir.is_dir():
        raise FileNotFoundError(f"no TensorBoard run at {run_dir}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"no checkpoint at {args.checkpoint}")

    data_files = sorted(args.data_dir.glob("*.npy"))
    if not data_files:
        raise FileNotFoundError(f"no .npy files under {args.data_dir}")

    entry = args.archive_dir / f"{args.run}_{len(data_files)}powers"
    if entry.exists():
        raise FileExistsError(f"{entry} already exists")

    (entry / "data").mkdir(parents=True)
    (entry / "figures").mkdir()

    shutil.copy2(args.checkpoint, entry / args.checkpoint.name)
    shutil.copy2(args.train_log, entry / args.train_log.name)
    shutil.copytree(run_dir, entry / "tensorboard_run", dirs_exist_ok=True)
    for path in data_files:
        shutil.copy2(path, entry / "data" / path.name)
    for path in sorted(args.figures_dir.glob("*.png")):
        shutil.copy2(path, entry / "figures" / path.name)
    for name in CODE_FILES:
        source = Path(name)
        if source.is_file():
            shutil.copy2(source, entry / name)
        else:
            print(f"[warn] {name} not found, skipped")

    lock(entry)
    print(f"[archive] {entry}")

    for child in args.logdir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
    for name in ROOT_CLEANUP:
        path = Path(name)
        if path.is_file():
            path.unlink()
    for path in args.figures_dir.glob("*.png"):
        path.unlink()
    print("[clean] root cleared (data/ left in place)")


if __name__ == "__main__":
    main()
