from __future__ import annotations

import numpy as np


def _make_grid(distribution_type: str, computational_params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Build an outcome grid and weights.

    Grid policy:
      - For continuous: y_grid = linspace(y_min, y_max, n)
      - For discrete: y_grid = arange(y_min, y_max + step_size, step_size)
      - For continuous: n must be odd and >= 3 (for Simpson's rule)
      - For discrete: no odd number requirement
      - weights = Simpson weights scaled by step/3 for continuous, or ones for discrete

    Parameters
    ----------
    distribution_type : str
        Either 'continuous' or 'discrete'
    computational_params : dict
        For continuous: must contain 'y_min', 'y_max', 'n'
        For discrete: must contain 'y_min', 'y_max', 'step_size'

    Returns
    -------
    y_grid : np.ndarray (n,), dtype float64
    w      : np.ndarray (n,), dtype float64
    """
    if distribution_type not in ['continuous', 'discrete']:
        raise ValueError(f"distribution_type must be 'continuous' or 'discrete'; got '{distribution_type}'")

    # Validate common parameters
    for key in ("y_min", "y_max"):
        if key not in computational_params:
            raise KeyError(f"computational_params['{key}'] is required")
    
    y_min = float(computational_params["y_min"])
    y_max = float(computational_params["y_max"])
    
    if not (y_max > y_min):
        raise ValueError(f"Require y_max > y_min; got y_min={y_min}, y_max={y_max}")

    if distribution_type == "continuous":
        # Validate continuous-specific parameters
        if "n" not in computational_params:
            raise KeyError("computational_params['n'] is required for continuous distribution")
        n = int(computational_params["n"])
        if n < 3:
            raise ValueError(f"n must be >= 3; got {n}")
        if n % 2 == 0:
            raise ValueError(f"n must be odd for continuous distribution; got {n}")

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
    
    else:  # distribution_type == 'discrete'
        # Validate discrete-specific parameters
        if "step_size" not in computational_params:
            raise KeyError("computational_params['step_size'] is required for discrete distribution")
        step_size = float(computational_params["step_size"])
        
        # Check if step_size perfectly divides y_max - y_min
        total_range = y_max - y_min
        if abs(total_range % step_size) > 1e-10:  # Allow for floating point precision
            raise ValueError(f"step_size must perfectly divide y_max - y_min; got step_size={step_size}, range={total_range}")
        
        # Create discrete grid from y_min to y_max with step_size
        y_grid = np.arange(y_min, y_max + step_size, step_size, dtype=np.float64)
        
        # For discrete distributions, weights are just ones (probability mass function)
        w = np.ones_like(y_grid, dtype=np.float64)
        
        return y_grid, w
