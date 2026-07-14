"""Give a training run a directory of its own, and make it self-explanatory later.

The old scheme had one global slot -- ``checkpoint.pt``, ``runs/``, ``figures/`` --
that every run wrote into and that had to be swept clean afterwards, so only one
model could be mid-flight at a time and archiving was a separate, forgettable step.
Here a run opens its archive entry *before* it starts and writes into it directly.
There is nothing to sweep, nothing to collide over, and TensorBoard can point at
``archive/`` and see every run at once.

An entry is named ``<model>_<stamp>_<minP>_<maxP>_<spacing>[-<tag>]`` and holds:

    checkpoint.pt      whatever the model chose to save
    train.log          stdout of the run
    tensorboard/       the event files
    figures/           whatever the model's visualize.py rendered
    code/              a snapshot of models/<model>/ and share/
    data/              *hard links* into the data run it was fitted to
    config.json        every CLI argument, resolved
    metrics.json       the final scores, so runs compare without loading a checkpoint
    env.json           python, torch, CUDA, the GPU
    git.txt            the commit the code came from, and whether the tree was dirty

**The data is hard-linked, not copied.** Copying was costing 13 GB per run -- six
entries had accumulated 78 GB of byte-identical duplicates of a dataset the solver
regenerates in about 20 s. A hard link is a second name for the same file record,
so it costs nothing, it opens exactly like the original, and deleting the original
leaves the archived name intact.

The one thing to know about hard links on NTFS: a file's attributes live in the
record the links share, so marking the archived name read-only marks the source
read-only too. That is why :func:`finalise` locks everything *except* ``data/``.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
ARCHIVE = REPO / "archive"
SHARE = REPO / "share"
RUNS = REPO / "runs"  # what TensorBoard watches; see live_link()


@dataclass
class Entry:
    """An open archive directory. The run writes into it while it is still going."""

    dir: Path
    model: str

    @property
    def checkpoint(self) -> Path:
        return self.dir / "checkpoint.pt"

    @property
    def tensorboard(self) -> Path:
        return self.dir / "tensorboard"

    @property
    def figures(self) -> Path:
        return self.dir / "figures"


def _name(model: str, run, tag: str | None, stamp: str) -> str:
    lo, hi = int(run.powers.min()), int(run.powers.max())
    base = f"{model}_{stamp}_{lo}_{hi}_{run.spacing:g}"
    return f"{base}-{tag}" if tag else base


def live_link(entry: Entry) -> None:
    """Point ``runs/<entry>`` at the entry's event files, for TensorBoard to watch.

    The archive is a *record*; TensorBoard is a *monitor*, and pointing it straight
    at ``archive/`` conflates the two -- every run ever made shows up in the sidebar,
    and the one actually training is lost among them.

    So the events stay inside the entry, where they belong, and ``runs/`` holds a
    junction per run for TensorBoard to read. :func:`drop_live_link` takes it away
    again when the run finishes, so what is listed is what is *running*. Point
    ``--logdir archive`` when the whole history is what you want to compare.
    """
    RUNS.mkdir(exist_ok=True)
    link = RUNS / entry.dir.name
    if link.exists():
        return
    # a junction, not a symlink: NTFS symlinks need administrator or developer mode,
    # junctions do not, and for a directory they behave the same
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(entry.tensorboard)],
        capture_output=True,
        check=False,
    )


def drop_live_link(entry: Entry) -> None:
    """Take ``runs/<entry>`` away again, now that the run is no longer live.

    ``rmdir`` without ``/s`` unlinks the junction and leaves what it pointed at alone.
    Deleting a junction with a tool that *follows* it -- ``rm -rf``, ``shutil.rmtree`` --
    would walk into the archive entry and delete the events themselves, which is the one
    thing this must not do.
    """
    link = RUNS / entry.dir.name
    if not link.exists():
        return
    subprocess.run(["cmd", "/c", "rmdir", str(link)], capture_output=True, check=False)


def open_entry(model: str, run, tag: str | None = None, root: Path = ARCHIVE) -> Entry:
    """Create the entry and the directories the run is about to write into."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    entry = Entry(dir=root / _name(model, run, tag, stamp), model=model)
    if entry.dir.exists():
        raise FileExistsError(entry.dir)
    entry.figures.mkdir(parents=True)
    entry.tensorboard.mkdir()
    live_link(entry)
    return entry


def _git() -> str:
    def run(*args) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "?"

    sha, dirty = run("rev-parse", "HEAD"), run("status", "--porcelain")
    lines = [f"commit {sha}", f"dirty  {'yes' if dirty else 'no'}"]
    if dirty:
        lines += ["", "--- uncommitted at the time of the run ---", dirty]
    return "\n".join(lines) + "\n"


def _env() -> dict:
    import torch

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _link_data(run, into: Path) -> None:
    """Hard-link the run's files in, and record enough to notice if they ever change."""
    into.mkdir(exist_ok=True)
    for src in run.files:
        dst = into / src.name
        if not dst.exists():
            os.link(src, dst)  # not a copy, and not a symlink

    source = run.dir.relative_to(REPO) if run.dir.is_relative_to(REPO) else run.dir
    (into / "manifest.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "note": "hard links into the source run, not copies; "
                        "deleting the source leaves these intact",
                "powers": [int(p) for p in run.powers],
                "grid": dict(zip(("nt", "nx", "ny", "nz"), run.shape)),
                "spacing_mm": run.spacing,
                "snap_dt_s": run.snap_dt,
                "solver_config": run.config,
                "files": [{"name": f.name, "bytes": f.stat().st_size} for f in run.files],
            },
            indent=2,
        )
    )


def _snapshot_code(model: str, into: Path) -> None:
    """The code that produced this, so the entry still reads after a refactor."""
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(REPO / "models" / model, into / "model", ignore=ignore, dirs_exist_ok=True)
    shutil.copytree(SHARE, into / "share", ignore=ignore, dirs_exist_ok=True)


def _lock(path: Path) -> None:
    for file in path.rglob("*"):
        if file.is_file():
            file.chmod(stat.S_IREAD)


def finalise(entry: Entry, run, config: dict, metrics: dict, lock: bool = False) -> Entry:
    """Close the entry: link the data, snapshot the code, write down the provenance.

    ``lock`` is off by default while the layout is still settling -- a locked entry
    is a nuisance to move. Turn it on once the structure is fixed.
    """
    _link_data(run, entry.dir / "data")
    _snapshot_code(entry.model, entry.dir / "code")

    (entry.dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    (entry.dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (entry.dir / "env.json").write_text(json.dumps(_env(), indent=2))
    (entry.dir / "git.txt").write_text(_git())

    drop_live_link(entry)  # it is a record now, not a monitor

    if lock:
        # everything but data/: NTFS shares a hard link's attributes with its source,
        # so locking the linked names would lock the live dataset along with them
        for child in entry.dir.iterdir():
            if child.name == "data":
                continue
            _lock(child) if child.is_dir() else child.chmod(stat.S_IREAD)

    print(f"[archive] {entry.dir.relative_to(REPO)}")
    return entry
