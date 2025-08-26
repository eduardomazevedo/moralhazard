from __future__ import annotations

from typing import Callable
import numpy as np

from .solver import _minimize_cost_a_hat


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
    a_hat: np.ndarray,
    warm_start: bool,
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
        results, _cache, theta_opt = _minimize_cost_a_hat(
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


def _compute_expected_utility(
    v: np.ndarray,
    a: float | np.ndarray,
    y_grid: np.ndarray,
    w: np.ndarray,
    f: Callable[[np.ndarray, float | np.ndarray], np.ndarray],
    C: Callable[[float | np.ndarray], float | np.ndarray],
) -> float | np.ndarray:
    """
    Compute U(a) = ∫ v(y) f(y|a) dy - C(a), evaluated on the Simpson grid.

    Inputs:
      - v : must have shape equal to y_grid.shape
      - a : scalar or 1D array
      - y_grid : outcome grid
      - w : Simpson weights
      - f : density function
      - C : cost function

    Returns:
      - scalar if a is scalar; 1D array otherwise
    """
    # Check input types but don't convert
    if not isinstance(v, np.ndarray):
        raise TypeError(f"v must be a numpy array; got {type(v)}")
    if v.shape != y_grid.shape:
        raise ValueError(f"v must have shape {y_grid.shape}; got {v.shape}")
    
    if not isinstance(a, (float, int, np.ndarray)):
        raise TypeError(f"a must be scalar or numpy array; got {type(a)}")

    # Let NumPy broadcasting handle both scalar and array inputs
    if isinstance(a, np.ndarray) and a.ndim != 1:
        raise ValueError(f"a must be 1D array; got shape {a.shape}")
    
    # f(y_grid[:, None], a) works for both scalar and array a due to broadcasting
    f_a = f(y_grid[:, None], a)
    integrals = w @ (v[:, None] * f_a)  # shape (m,) for array a, scalar for scalar a
    costs = C(a)  # shape (m,) for array a, scalar for scalar a
    return integrals - costs
