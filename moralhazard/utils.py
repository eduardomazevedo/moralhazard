from __future__ import annotations

from typing import Callable
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
    a_min: float | None = None,
    a_max: float | None = None,
    n_a_iterations: int = 1,
    warm_start: bool = True,
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

    # Set default a_min if not provided for iterative solver
    a_min_final = 0.0 if a_min is None else a_min
    
    def F(a: float) -> float:
        nonlocal last_theta_ref, call_count
        theta_init = last_theta_ref if warm_start else None
        
        if solver == "a_hat":
            if a_hat is None:
                raise ValueError("a_hat is required when solver='a_hat'")
            results, theta_opt = _minimize_cost_a_hat(
                float(a),
                float(Ubar),
                np.asarray(a_hat, dtype=np.float64),
                y_grid=y_grid,
                w=w,
                f=f,
                score=score,
                C=C,
                Cprime=Cprime,
                g=g,
                k=k,
                theta_init=theta_init,
            )
        else:  # solver == "iterative"
            if a_max is None:
                raise ValueError("a_max is required when solver='iterative'")
            results, theta_opt = _minimize_cost_iterative(
                a0=float(a),
                Ubar=float(Ubar),
                a_min=float(a_min_final),
                a_max=float(a_max),
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
            )
        
        if warm_start:
            last_theta_ref = theta_opt
        call_count += 1
        # keep attributes in sync
        F.last_theta = last_theta_ref  # type: ignore[attr-defined]
        F.call_count = call_count      # type: ignore[attr-defined]
        return float(results.expected_wage)

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






