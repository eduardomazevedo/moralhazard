from __future__ import annotations

from typing import Callable, Optional, Dict, Any, TYPE_CHECKING
import numpy as np
from scipy.optimize import minimize_scalar

if TYPE_CHECKING:
    from .problem import MoralHazardProblem


def _maximize_1d_robust(
    objective: Callable[[float | np.ndarray], float | np.ndarray],
    lower_bound: float,
    upper_bound: float,
    n_grid_points: int,
    *,
    xatol: float = 1e-8,
) -> tuple[float, float]:
    """
    Robust 1D maximization using grid search followed by local optimization.
    
    First performs a grid search to find the best candidate, then optimizes
    in the two intervals adjacent to that candidate (left and right).
    
    Parameters
    ----------
    objective : Callable[[float | np.ndarray], float | np.ndarray]
        Function to maximize. Should accept both scalar and 1D array inputs
        and return corresponding scalar or array outputs. Vectorized evaluation
        is used for the grid search for performance.
    lower_bound : float
        Lower bound for the search
    upper_bound : float
        Upper bound for the search
    n_grid_points : int
        Number of grid points for initial search
    xatol : float, optional
        Absolute tolerance for optimization (default: 1e-8)
    
    Returns
    -------
    tuple[float, float]
        A tuple of (best_x, best_value)
    """
    # First do a grid search to find the best candidate (vectorized)
    x_grid = np.linspace(lower_bound, upper_bound, n_grid_points)
    values = objective(x_grid)  # Vectorized evaluation
    values = np.asarray(values)  # Ensure it's an array
    best_idx = np.argmax(values)
    x_candidate = float(x_grid[best_idx])
    candidate_value = float(values[best_idx])
    
    # Define negative objective function for minimization (scalar only, for optimizer)
    def neg_objective(x: float) -> float:
        result = objective(x)  # Scalar evaluation
        return -float(np.asarray(result).item())
    
    # Determine intervals: left (previous grid point to candidate) and right (candidate to next grid point)
    if best_idx > 0:
        x_left_bound = float(x_grid[best_idx - 1])
    else:
        x_left_bound = lower_bound
    
    if best_idx < len(x_grid) - 1:
        x_right_bound = float(x_grid[best_idx + 1])
    else:
        x_right_bound = upper_bound
    
    candidates = [(x_candidate, candidate_value)]
    
    # Optimize in the left interval (previous grid point to candidate)
    if x_left_bound < x_candidate:
        try:
            left_result = minimize_scalar(
                neg_objective,
                bounds=(x_left_bound, x_candidate),
                method='bounded',
                options={'xatol': xatol}
            )
            if left_result.success:
                left_x = left_result.x
                left_value = -left_result.fun
                candidates.append((left_x, left_value))
        except (ValueError, RuntimeError):
            pass
    
    # Optimize in the right interval (candidate to next grid point)
    if x_candidate < x_right_bound:
        try:
            right_result = minimize_scalar(
                neg_objective,
                bounds=(x_candidate, x_right_bound),
                method='bounded',
                options={'xatol': xatol}
            )
            if right_result.success:
                right_x = right_result.x
                right_value = -right_result.fun
                candidates.append((right_x, right_value))
        except (ValueError, RuntimeError):
            pass
    
    # Find the best candidate from all options
    best_x, best_value = max(candidates, key=lambda x: x[1])
    
    return best_x, best_value


def _solve_principal_problem(
    *,
    revenue_function: Callable[[float], float],
    expected_wage_fun: Callable[[float], float],
    a_min: float,
    a_max: float,
    a_init: Optional[float] = None,
    minimize_scalar_options: Optional[Dict[str, Any]] = None,
):
    """
    Real worker for the principal's problem:
      maximize_a  revenue_function(a) - expected_wage_fun(a)

    Returns:
      dict with:
        - optimal_action: float
        - objective_value: float (value at optimum)
        - outer_solver_state: dict with metadata from minimize_scalar
    """
    try:
        from scipy.optimize import minimize_scalar
    except Exception as e:
        raise ImportError(
            "scipy is required for _solve_principal_problem (minimize_scalar)."
        ) from e

    bounded_ok = np.isfinite(a_min) and np.isfinite(a_max)

    def neg_obj(a: float) -> float:
        rev = revenue_function(a)
        ew  = expected_wage_fun(a)
        return -(rev - ew)

    method = "bounded" if bounded_ok else "brent"
    options = dict(minimize_scalar_options or {})

    # Note: a_init isn't used by 'bounded'; we keep it for API symmetry/future use.
    res = minimize_scalar(
        neg_obj,
        bounds=(a_min, a_max) if bounded_ok else None,
        method=method,
        options=options
    )

    outer_state = {
        "method": method,
        "success": bool(getattr(res, "success", True)),
        "fun_negated": getattr(res, "fun", np.nan),
        "nfev": int(getattr(res, "nfev", -1)),
        "nit": int(getattr(res, "nit", -1)) if hasattr(res, "nit") else None,
        "message": getattr(res, "message", None),
    }

    opt_a = res.x
    opt_val = -res.fun

    return {
        "optimal_action": opt_a,
        "profit": opt_val,
        "outer_solver_state": outer_state,
    }
