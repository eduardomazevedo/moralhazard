from __future__ import annotations

from typing import Dict, Any, Callable, TYPE_CHECKING
import numpy as np
from scipy.optimize import minimize_scalar

if TYPE_CHECKING:
    from .problem import MoralHazardProblem


def _make_cache(
    a0: float,
    a_hat: np.ndarray,
    *,
    problem: "MoralHazardProblem",
    clip_ratio: float = 1e6,
) -> Dict[str, Any]:
    """
    Build the *per-solve* precomputations needed by the inner/outer problems.

    This cache now contains **only** derived arrays/scalars that are expensive or
    convenient to reuse. It does **not** include primitives (functions) or
    problem parameters/inputs like y_grid, w, a0, Ubar, or a_hat.

    Returns keys:
      - f0, s0                 : (n,)
      - D, R                   : (n, m) and (n, m)
      - wf0, wf0s0             : (n,)
      - WD_T                   : (m, n) = (w[:,None] * D).T
      - C0, Cprime0, C_hat     : floats / (m,)
    """
    y_grid = problem.y_grid
    w = problem.w
    f = problem.f
    score = problem.score
    C = problem.C
    Cprime = problem.Cprime

    # Baseline density and score at a0 on the fixed grid
    f0 = f(y_grid, a0)           # (n,)
    s0 = score(y_grid, a0)       # (n,)

    # Density matrix at fixed comparison actions a_hat: (n, m)
    D = f(y_grid[:, None], a_hat[None, :])  # (n, m)

    # Cached weights/products
    wf0 = w * f0
    wf0s0 = wf0 * s0

    # Ratio for the global IC constraints: R = 1 - D / f0 (broadcast along columns)
    # Add numerical safeguards to prevent extreme values that could cause optimization issues
    f0_safe = np.maximum(f0, 1e-12)  # Ensure f0 is not too small
    ratio = D / f0_safe[:, None]  # (n, m)
    
    # Clip the ratio to prevent extreme values that could destabilize the dual optimization
    ratio_clipped = np.clip(ratio, -clip_ratio, clip_ratio)
    R = 1.0 - ratio_clipped  # (n, m)

    # Precompute C-related terms
    C0 = C(a0)
    Cprime0 = Cprime(a0)
    C_hat = C(a_hat)  # (m,)

    # Precompute weighted D for Uhat integrals: (w[:, None] * D).T @ v
    WD_T = (w[:, None] * D).T  # (m, n)

    return {
        "f0": f0,
        "s0": s0,
        "D": D,
        "R": R,
        "wf0": wf0,
        "wf0s0": wf0s0,
        "WD_T": WD_T,
        "C0": C0,
        "Cprime0": Cprime0,
        "C_hat": C_hat,
    }


def _canonical_contract(
    lam: float,
    mu: float,
    mu_hat: np.ndarray,
    s0: np.ndarray,
    R: np.ndarray,
    problem: "MoralHazardProblem",
) -> np.ndarray:
    """
    Canonical contract map v = g(λ + μ s0 + R μ̂).

    Parameters
    ----------
    lam: float
        IR multiplier
    mu: float
        FOC multiplier
    mu_hat: np.ndarray
        IC multipliers
    s0 : np.ndarray (n,)
    R  : np.ndarray (n, m)
    problem: MoralHazardProblem
        Problem instance containing primitives

    Returns
    -------
    v : np.ndarray (n,)
    """
    g = problem.g
    z = lam + mu * s0 + R @ mu_hat
    v = g(z)
    return v


def _constraints(
    v: np.ndarray,
    *,    
    cache: Dict[str, Any],
    problem: "MoralHazardProblem",
    Ubar: float,
) -> Dict[str, Any]:
    """
    Evaluate all constraints and E[wage] given a contract v on the internal grid.

    Uses only precomputed arrays from `cache` plus primitive inputs from problem and `Ubar`.

    Returns:
      - U0, IR, FOC : floats
      - Uhat, IC    : np.ndarray (m,)
      - Ewage       : float
    """
    k = problem.k_func
    wf0 = cache["wf0"]
    wf0s0 = cache["wf0s0"]
    WD_T = cache["WD_T"]
    C0 = cache["C0"]
    Cprime0 = cache["Cprime0"]
    C_hat = cache["C_hat"]

    # U0 = ∫ v f0 - C(a0)
    U0 = wf0 @ v - C0

    # FOC = ∫ v s0 f0 - C'(a0)
    FOC = wf0s0 @ v - Cprime0

    # Uhat (m,) and IC = Uhat - U0
    Uhat = WD_T @ v - C_hat  # (m,)
    IC = Uhat - U0

    # IR
    IR = Ubar - U0

    # Expected wage: ∫ k(v) f0
    Ewage = wf0 @ k(v)

    return {"U0": U0, "IR": IR, "FOC": FOC, "Uhat": Uhat, "IC": IC, "Ewage": Ewage}


def _compute_expected_utility(
    v: np.ndarray,
    a: float | np.ndarray,
    *,
    problem: "MoralHazardProblem",
) -> float | np.ndarray:
    """
    Compute U(a) = ∫ v(y) f(y|a) dy - C(a), evaluated on the Simpson grid.

    Inputs:
      - v : must have shape equal to problem.y_grid.shape
      - a : scalar or 1D array
      - problem: MoralHazardProblem instance

    Returns:
      - scalar if a is scalar; 1D array otherwise
    """
    y_grid = problem.y_grid
    w = problem.w
    f = problem.f
    C = problem.C

    # Let NumPy broadcasting handle both scalar and array inputs
    if isinstance(a, np.ndarray) and a.ndim != 1:
        raise ValueError(f"a must be 1D array; got shape {a.shape}")
    
    # f(y_grid[:, None], a) works for both scalar and array a due to broadcasting
    f_a = f(y_grid[:, None], a)
    integrals = w @ (v[:, None] * f_a)  # shape (m,) for array a, scalar for scalar a
    costs = C(a)  # shape (m,) for array a, scalar for scalar a
    result = integrals - costs
    
    # Return result as-is (numpy scalar for scalar input, array for array input)
    return result


def _agent_best_action(
    v: np.ndarray,
    a_lb,
    a_ub,
    n_a_grid_points,
    *,
    problem: "MoralHazardProblem",
) -> tuple[float, float]:
    """
    Find the action that maximizes expected utility within the specified bounds.
    
    First performs a grid search to find the best candidate, then optimizes
    in the two intervals adjacent to that candidate (left and right).
    
    Returns:
        tuple[float, float]: A tuple of (best_action, utility_attained)
    """
    # First do a grid search to find the best candidate
    a_grid = np.linspace(a_lb, a_ub, n_a_grid_points)
    expected_utilities = _compute_expected_utility(v, a_grid, problem=problem)
    best_idx = np.argmax(expected_utilities)
    a_candidate = float(a_grid[best_idx])
    candidate_utility = float(np.asarray(expected_utilities[best_idx]).item())
    
    # Define negative utility function for minimization
    def neg_utility(a: float) -> float:
        utility = _compute_expected_utility(v, a, problem=problem)
        # Extract scalar value properly to avoid deprecation warning
        return -float(np.asarray(utility).item())
    
    # Determine intervals: left (previous grid point to candidate) and right (candidate to next grid point)
    if best_idx > 0:
        a_left_bound = float(a_grid[best_idx - 1])
    else:
        a_left_bound = a_lb
    
    if best_idx < len(a_grid) - 1:
        a_right_bound = float(a_grid[best_idx + 1])
    else:
        a_right_bound = a_ub
    
    candidates = [(a_candidate, candidate_utility)]
    
    # Optimize in the left interval (previous grid point to candidate)
    if a_left_bound < a_candidate:
        try:
            left_result = minimize_scalar(
                neg_utility,
                bounds=(a_left_bound, a_candidate),
                method='bounded',
                options={'xatol': 1e-8}
            )
            if left_result.success:
                left_action = left_result.x
                left_utility = -left_result.fun
                candidates.append((left_action, left_utility))
        except (ValueError, RuntimeError):
            pass
    
    # Optimize in the right interval (candidate to next grid point)
    if a_candidate < a_right_bound:
        try:
            right_result = minimize_scalar(
                neg_utility,
                bounds=(a_candidate, a_right_bound),
                method='bounded',
                options={'xatol': 1e-8}
            )
            if right_result.success:
                right_action = right_result.x
                right_utility = -right_result.fun
                candidates.append((right_action, right_utility))
        except (ValueError, RuntimeError):
            pass
    
    # Find the best candidate from all options
    best_action, best_utility = max(candidates, key=lambda x: x[1])
    
    return best_action, best_utility