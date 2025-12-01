from __future__ import annotations

from typing import Callable, Optional, Dict, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .problem import MoralHazardProblem


def _make_expected_wage_fun(
    *,
    problem: "MoralHazardProblem",
    Ubar: float,
    a_ic_lb: float,
    a_ic_ub: float,
    n_a_iterations: int = 1,
    n_a_grid_points: int = 10,
    clip_ratio: float = 1e6,
    a_always_check_global_ic: np.ndarray = np.array([0.0]),
) -> Callable[[float], float]:
    """
    Factory returning F(a) = E[w(v*(a))].

    Returns a callable that takes an action a and returns the expected wage.
    """
    def F(a: float) -> float:
        results = problem.solve_cost_minimization_problem(
            intended_action=a,
            reservation_utility=Ubar,
            a_ic_lb=a_ic_lb,
            a_ic_ub=a_ic_ub,
            n_a_grid_points=n_a_grid_points,
            n_a_iterations=n_a_iterations,
            clip_ratio=clip_ratio,
            a_always_check_global_ic=a_always_check_global_ic,
        )
        return results.expected_wage

    return F


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
