"""One definition of "how wrong is this field", used by every model.

Three numbers, because they disagree and the disagreement is the point.

``rmse`` is an L2 average over the whole plate, so it is dominated by the large,
slowly varying background and is nearly blind to the melt pool -- which occupies
a fraction of a percent of the volume and is the only part anyone cares about.

``linf`` is the worst single point. It catches ringing and it catches a smeared
peak, but it does not say which.

``peak`` is the one with physical meaning: how far off the melt pool's maximum
temperature is, as a percentage of the true maximum, taken from whichever
snapshot is worst. The solver's own grid already costs about 0.35% here (see
simulation/README.md), so a model that lands inside that is as good as the data
it was fitted to, and a model reporting a fine RMSE with a 15% peak error is
telling you it flattened the pool.
"""

from __future__ import annotations

import numpy as np


def score(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Field errors in Kelvin. Both arrays are ``(nt, ...)`` of ``dT``."""
    nt = truth.shape[0]
    err = pred - truth
    peak = truth.reshape(nt, -1).max(1)
    hot = peak > 1.0  # t = 0 is uniformly ambient, so it has no peak to miss
    rel = (pred.reshape(nt, -1).max(1)[hot] - peak[hot]) / peak[hot]
    return {
        "rmse": float(np.sqrt((err**2).mean())),
        "linf": float(np.abs(err).max()),
        "peak": float(rel[np.abs(rel).argmax()] * 100) if rel.size else 0.0,
    }


def score_per_power(pred: np.ndarray, truth: np.ndarray, powers) -> dict:
    """``score`` for each power, keyed by the power in watts."""
    return {int(p): score(pred[i], truth[i]) for i, p in enumerate(powers)}


def per_snapshot_rmse(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """RMSE of each snapshot, for looking at *when* a model goes wrong."""
    nt = truth.shape[0]
    return np.sqrt(((pred - truth) ** 2).reshape(nt, -1).mean(1))
