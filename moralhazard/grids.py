from __future__ import annotations

import numpy as np


def _make_grid(y_min: float, y_max: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build an odd-length outcome grid and Simpson weights.

    Grid policy (generalized):
      - y_grid = linspace(y_min, y_max, n)
      - n must be odd and >= 3
      - weights = Simpson weights scaled by step/3,
        where step = (y_max - y_min) / (n - 1)

    Returns
    -------
    y_grid : np.ndarray (n,), dtype float64
    w      : np.ndarray (n,), dtype float64
    """
    if not isinstance(n, int):
        raise TypeError(f"n must be an int; got {type(n).__name__}")
    if n < 3:
        raise ValueError(f"n must be >= 3; got {n}")
    if n % 2 == 0:
        raise ValueError(f"n must be odd for Simpson's rule; got even n={n}")
    y_min = float(y_min)
    y_max = float(y_max)
    if not (y_max > y_min):
        raise ValueError(f"Require y_max > y_min; got y_min={y_min}, y_max={y_max}")

    y_grid = np.linspace(y_min, y_max, n, dtype=np.float64)
    step = (y_max - y_min) / float(n - 1)

    # Simpson weights for an odd-length, evenly spaced grid
    w = np.zeros_like(y_grid, dtype=np.float64)
    w[0] = 1.0
    w[-1] = 1.0
    w[1:-1:2] = 4.0
    w[2:-2:2] = 2.0
    w *= (step / 3.0)
    return y_grid, w
