"""A flat gallery of every model's latest figures, collected out of ``archive/``.

Each run renders its figures into ``archive/<entry>/figures/`` and leaves them
there -- the right home for a *record*, and the wrong one for a *look*. Comparing
the nine models means opening nine archive directories and holding two windows in
your head at once. This mirrors those figures into a single top-level ``figures/``,
keyed by the run they came from, so a whole sweep reads at a glance.

The mirror is derived and never authoritative. ``archive/`` is the record; this is
a view built from it, and it can be deleted and rebuilt at any time with
``python -m share.figures``. Two consequences follow, and both are deliberate:

* Figures are **copied, not hard-linked.** ``archive/`` links its data and its
  TensorBoard events to save space, but a figure is a few hundred kB and a link
  would tie the gallery to the archive's lock state -- a locked entry marks its
  figures read-only (see :func:`share.archiving.finalise`), and a hard link would
  carry that into ``figures/``, so the copy could not be swept or overwritten. A
  plain copy stays writable and free of the archive entirely.
* Names carry the run's identity: ``<model>[-<tag>]_<view>_<stamp>.png``, the
  ``<stamp>`` read straight out of the archive entry's own name. Because the stamp
  is in the name the figures **accumulate** across a sweep instead of overwriting,
  and the latest of any variant is simply the one with the newest stamp. An
  ``adam`` run and an ``lbfgs`` run of the same model sit side by side rather than
  clobbering each other, which is the comparison a sweep exists to make.

``share.harness.finish`` calls :func:`publish` at the end of every run, so the
gallery stays current on its own. The command line rebuilds it from what is
already on disk -- the latest of each ``model+tag`` by default, so it works
retroactively over runs made before this existed; ``--all`` mirrors every run.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARCHIVE = REPO / "archive"
FIGURES = REPO / "figures"

# An archive entry is <model>_<stamp>_<minP>_<maxP>_<spacing>[-<tag>]; the stamp is
# the one YYYYmmdd-HHMMSS field in it. See share.archiving._name for the other half.
_STAMP = re.compile(r"\d{8}-\d{6}")


def _identity(entry_name: str) -> tuple[str, str | None, str]:
    """``(model, tag, stamp)`` pulled back out of an archive entry's name.

    The model is the first field and the stamp is the timestamped one; the tag --
    which may itself contain hyphens (``lbfgs-wide``) -- is whatever trails the
    spacing after its ``-``. A hand-named entry with no recognisable stamp yields
    an empty stamp, which :func:`latest_per_variant` treats as not-a-run.
    """
    fields = entry_name.split("_")
    model = fields[0]
    stamp = next((f for f in fields if _STAMP.fullmatch(f)), "")
    tail = fields[-1]  # <spacing>[-<tag>]
    tag = tail.split("-", 1)[1] if "-" in tail else None
    return model, tag, stamp


def _variant(entry_name: str) -> str:
    """``<model>[-<tag>]`` -- the prefix a run's figures share in the gallery."""
    model, tag, _stamp = _identity(entry_name)
    return f"{model}-{tag}" if tag else model


def publish(entry_dir: Path, gallery: Path = FIGURES) -> list[Path]:
    """Copy one archive entry's figures into the flat gallery, stamped by its run.

    Returns the gallery paths written. An entry that has rendered nothing yet (no
    ``figures/``) contributes nothing and is not an error, so this is safe to call
    the moment an entry is opened as well as once it is full.
    """
    source = Path(entry_dir) / "figures"
    if not source.is_dir():
        return []

    model, tag, stamp = _identity(Path(entry_dir).name)
    prefix = f"{model}-{tag}" if tag else model
    stamp = stamp or "unknown"

    gallery.mkdir(parents=True, exist_ok=True)
    written = []
    for figure in sorted(source.glob("*.png")):
        destination = gallery / f"{prefix}_{figure.stem}_{stamp}.png"
        shutil.copyfile(figure, destination)  # bytes only: the copy stays writable
        written.append(destination)
    return written


def _entries(root: Path) -> list[Path]:
    """Every archive subdirectory whose name parses as a run."""
    if not root.is_dir():
        return []
    return sorted(
        entry
        for entry in root.iterdir()
        if entry.is_dir() and _identity(entry.name)[2]
    )


def latest_per_variant(root: Path = ARCHIVE) -> list[Path]:
    """The newest entry for each ``model+tag``, by the stamp in its name.

    Keyed by the variant rather than the model alone so an ``adam`` run and an
    ``lbfgs`` run of the same model both survive -- collapsing to the model would
    keep only whichever finished last and lose the comparison between them.
    """
    best: dict[str, tuple[str, Path]] = {}
    for entry in _entries(root):
        variant, stamp = _variant(entry.name), _identity(entry.name)[2]
        if variant not in best or stamp > best[variant][0]:
            best[variant] = (stamp, entry)
    return [entry for _stamp, entry in sorted(best.values())]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m share.figures", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--archive", type=Path, default=ARCHIVE,
                        help="where the runs are (default: archive/)")
    parser.add_argument("--into", type=Path, default=FIGURES,
                        help="where the gallery goes (default: figures/)")
    parser.add_argument("--all", action="store_true",
                        help="mirror every run, not just the latest of each model+tag")
    args = parser.parse_args(argv)

    entries = _entries(args.archive) if args.all else latest_per_variant(args.archive)
    if not entries:
        print(f"[figures] no runs found under {args.archive}")
        return

    total = 0
    for entry in entries:
        for figure in publish(entry, args.into):
            print(f"[figures] {figure.name}")
            total += 1

    into = args.into.relative_to(REPO) if args.into.is_relative_to(REPO) else args.into
    print(f"[figures] {total} figure(s) from {len(entries)} run(s) in {into}/")


if __name__ == "__main__":
    main()
