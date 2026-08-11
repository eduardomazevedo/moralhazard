"""Utility functions for finding globally profitable one-dimensional deviations."""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from moralhazard.core import _compute_expected_utility


def _maximize_agent_utility(
    v: np.ndarray,
    a_left: float,
    a_right: float,
    problem: Any,
    n_intervals: Optional[int] = 5,
) -> Tuple[Optional[float], float]:
    """Find the action that maximizes agent utility on a bounded interval.

    A vectorized grid first identifies *all* candidate local maxima.  Each
    candidate is then polished in its neighboring grid cell.  This is more
    reliable than launching a local optimizer from a few interval midpoints:
    that approach could converge to the intended-action peak while entirely
    missing a narrower profitable deviation.

    The result is still a numerical separation oracle, but grid bracketing
    makes its resolution explicit and deterministic.
    """
    if not (np.isfinite(a_left) and np.isfinite(a_right) and a_left <= a_right):
        raise ValueError("action-search bounds must be finite and ordered")
    if a_left == a_right:
        utility = float(np.asarray(_compute_expected_utility(v, a_left, problem)).item())
        return float(a_left), utility

    # At least 101 points (unit spacing on the paper's [0, 100] examples).
    # n_intervals remains useful to callers requesting a finer initial scan.
    n_grid = max(101, 20 * int(n_intervals or 1) + 1)
    grid = np.linspace(a_left, a_right, n_grid)
    utilities = np.asarray(_compute_expected_utility(v, grid, problem), dtype=float)
    finite = np.isfinite(utilities)
    if not np.any(finite):
        return None, -np.inf

    candidates = [0, n_grid - 1]
    candidates.extend(
        (
            np.flatnonzero(
                finite[1:-1]
                & (utilities[1:-1] >= utilities[:-2])
                & (utilities[1:-1] >= utilities[2:])
            )
            + 1
        ).tolist()
    )

    best_index = int(np.nanargmax(np.where(finite, utilities, -np.inf)))
    best_action = float(grid[best_index])
    best_utility = float(utilities[best_index])

    def negative_utility(action: float) -> float:
        utility = _compute_expected_utility(v, float(action), problem)
        return -float(np.asarray(utility).item())

    for index in candidates:
        # Endpoints are already evaluated exactly.  Interior maxima are
        # bracketed by adjacent grid points before bounded polishing.
        if index == 0 or index == n_grid - 1:
            action = float(grid[index])
            utility = float(utilities[index])
        else:
            result = minimize_scalar(
                negative_utility,
                bounds=(float(grid[index - 1]), float(grid[index + 1])),
                method="bounded",
                options={"xatol": 1e-9, "maxiter": 100},
            )
            if not result.success or not np.isfinite(result.fun):
                continue
            action = float(result.x)
            utility = float(-result.fun)
        if utility > best_utility:
            best_action = action
            best_utility = utility

    return best_action, best_utility
