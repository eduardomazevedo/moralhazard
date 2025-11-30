from __future__ import annotations

from typing import Callable, Optional, Dict, Any
import numpy as np

from .solver import _minimize_cost_a_hat, _minimize_cost_iterative


def _make_expected_wage_fun(
    *,
    y_grid: np.ndarray,
    w: np.ndarray,
    f: Callable[[np.ndarray, float | np.ndarray], np.ndarray],
    score: Callable[[np.ndarray, float | np.ndarray], np.ndarray],
    C: Callable[[float | np.ndarray], float | np.ndarray],
    Cprime: Callable[[float | np.ndarray], float | np.ndarray],
    g: Callable[[np.ndarray], np.ndarray],
    k: Callable[[np.ndarray], np.ndarray],
    Ubar: float,
    solver: str = "a_hat",
    a_hat: np.ndarray | None = None,
    n_a_iterations: int = 1,
    warm_start: bool = True,
    clip_ratio: float = 1e6,
    a_ic_lb: float = -np.inf,
    a_ic_ub: float = np.inf,
    a_ic_initial: float = 0.0,
) -> Callable[[float], float]:
    """
    Factory returning F(a) = E[w(v*(a))] with an optional warm start across calls.

    Attributes on returned callable (kept live across calls):
      - .last_theta: np.ndarray | None
      - .call_count: int
      - .reset(): None
    """
    last_theta_ref: np.ndarray | None = None
    call_count = 0


    
    def F(a: float) -> float:
        nonlocal last_theta_ref, call_count
        theta_init = last_theta_ref if warm_start else None
        
        if solver == "a_hat":
            if a_hat is None:
                raise ValueError("a_hat is required when solver='a_hat'")
            results, theta_opt = _minimize_cost_a_hat(
                a,
                Ubar,
                a_hat,
                y_grid=y_grid,
                w=w,
                f=f,
                score=score,
                C=C,
                Cprime=Cprime,
                g=g,
                k=k,
                theta_init=theta_init,
                clip_ratio=clip_ratio,
            )
        else:  # solver == "iterative"
            results, theta_opt = _minimize_cost_iterative(
                a0=a,
                Ubar=Ubar,
                n_a_iterations=int(n_a_iterations),
                y_grid=y_grid,
                w=w,
                f=f,
                score=score,
                C=C,
                Cprime=Cprime,
                g=g,
                k=k,
                theta_init=theta_init,
                clip_ratio=clip_ratio,
                a_ic_lb=a_ic_lb,
                a_ic_ub=a_ic_ub,
                a_ic_initial=a_ic_initial,
            )
        
        if warm_start:
            last_theta_ref = theta_opt
        call_count += 1
        # keep attributes in sync
        F.last_theta = last_theta_ref  # type: ignore[attr-defined]
        F.call_count = call_count      # type: ignore[attr-defined]
        return results.expected_wage

    def _reset():
        nonlocal last_theta_ref, call_count
        last_theta_ref = None
        call_count = 0
        F.last_theta = last_theta_ref  # type: ignore[attr-defined]
        F.call_count = call_count      # type: ignore[attr-defined]

    # initialize lightweight attributes
    F.last_theta = last_theta_ref      # type: ignore[attr-defined]
    F.call_count = call_count          # type: ignore[attr-defined]
    F.reset = _reset                   # type: ignore[attr-defined]
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
