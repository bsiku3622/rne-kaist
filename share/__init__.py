"""Assets every model needs, and nothing that binds them to each other.

This is a library, not an interface. No model imports another model, and nothing
here knows what a model looks like -- it is the grid, the metrics, the figures and
the archive, factored out so there is one definition of each instead of four.

Scripts under ``models/`` reach it by putting the repository root on the path:

    import sys; from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from share import archiving, grid, metrics, plotting
"""
