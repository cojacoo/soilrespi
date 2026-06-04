"""Sliding-window best linear regression."""

import numpy as np
from typing import Tuple


def find_best_window(
    x: np.ndarray,
    y: np.ndarray,
    window_frac: float = 0.8,
    undercut: float = 0.3,
) -> Tuple[int, int, float]:
    """Find the most linear segment of a time series via sliding window.

    Searches all (size, position) combinations where size ranges from
    ``window_frac × n`` down to ``window_frac × (1 − undercut) × n``.
    Returns the window with the highest R². When two windows have equal R²,
    the larger one is preferred (honour the requested fraction where possible).

    Parameters
    ----------
    x           : elapsed time [s], shape (n,)
    y           : concentration values, shape (n,)
    window_frac : target fraction of total series length (0 < frac ≤ 1)
    undercut    : maximum fractional reduction below ``window_frac`` (0 ≤ undercut < 1).
                  ``undercut=0.3`` allows the window to shrink by up to 30 %.

    Returns
    -------
    i_start, i_end : slice indices into x / y for the best window
    best_r2        : R² of the best-fit linear regression within that window
    """
    n = len(x)
    max_size = max(int(n * window_frac), 5)
    min_size = max(int(n * window_frac * (1.0 - undercut)), 5)

    best_r2    = -np.inf
    best_start = 0
    best_size  = max_size

    # Outer loop descends so equal-R² ties keep the larger window
    for size in range(max_size, min_size - 1, -1):
        for start in range(n - size + 1):
            x_w = x[start : start + size]
            y_w = y[start : start + size]
            slope, intercept = np.polyfit(x_w, y_w, 1)
            fitted = intercept + slope * x_w
            ss_res = ((y_w - fitted) ** 2).sum()
            ss_tot = ((y_w - y_w.mean()) ** 2).sum()
            r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            if r2 > best_r2:
                best_r2    = r2
                best_start = start
                best_size  = size

    return best_start, best_start + best_size, float(best_r2)
