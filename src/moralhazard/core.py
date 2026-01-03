"""Core computational functions for the algorithm 1 dual formulation.

Provides cache construction, canonical contract mapping, constraint evaluation,
and expected utility computation used by the solver.
"""
from __future__ import annotations

from typing import Dict, Any, Callable, TYPE_CHECKING, Tuple
import numpy as np

if TYPE_CHECKING:
    from .problem import MoralHazardProblem


def _make_cache(
    a0: float,
    a_hat: np.ndarray,
    *,
    problem: "MoralHazardProblem",
    clip_ratio: float = 1e6,
) -> Dict[str, Any]:
    """Build per-solve precomputations needed by the solver.

    This cache contains only derived arrays/scalars that are
    convenient to reuse. It does not include primitives (functions) or
    problem parameters/inputs like y_grid, w, a0, Ubar, or a_hat.

    Args:
        a0: The intended action.
        a_hat: Array of comparison actions for global IC constraints, shape (m,).
        problem: The MoralHazardProblem instance containing primitives.
        clip_ratio: Maximum absolute value for ratio clipping to prevent
            numerical instability. Defaults to 1e6.

    Returns:
        Dictionary with precomputed arrays (m = len(a_hat), n = len(y_grid)):
            - f0: Density at a0, shape (n,).
            - s0: Score at a0, shape (n,).
            - D: Density matrix at a_hat, shape (m, n).
            - R: Ratio matrix 1 - D/f0, shape (m, n).
            - wf0: Weighted density w * f0, shape (n,).
            - wf0s0: Weighted density times score w * f0 * s0, shape (n,).
            - weighted_D: Weighted density matrix w * D, shape (m, n).
            - C0: Cost at a0, float.
            - Cprime0: Cost derivative at a0, float.
            - C_hat: Costs at a_hat, shape (m,).
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

    # Density matrix at fixed comparison actions a_hat: (m, n) with n as last dimension for efficiency
    D = f(y_grid[None, :], a_hat[:, None])  # (m, n)

    # Cached weights/products
    wf0 = w * f0
    wf0s0 = wf0 * s0

    # Ratio for the global IC constraints: R = 1 - D / f0 (broadcast along columns)
    # Add numerical safeguards to prevent extreme values that could cause optimization issues
    # D is (m, n), f0 is (n,), so we broadcast f0 along rows
    f0_safe = np.maximum(f0, 1e-12)  # Ensure f0 is not too small
    ratio = D / f0_safe[None, :]  # (m, n)
    
    # Clip the ratio to prevent extreme values that could destabilize the dual optimization
    ratio_clipped = np.clip(ratio, -clip_ratio, clip_ratio)
    R = 1.0 - ratio_clipped  # (m, n)

    # Precompute C-related terms
    C0 = C(a0)
    Cprime0 = Cprime(a0)
    C_hat = C(a_hat)  # (m,)

    # Precompute weighted D for Uhat integrals: weighted_D @ v where weighted_D is (m, n)
    weighted_D = w[None, :] * D  # (m, n)

    return {
        "f0": f0,
        "s0": s0,
        "D": D,
        "R": R,
        "wf0": wf0,
        "wf0s0": wf0s0,
        "weighted_D": weighted_D,
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
    """Compute the canonical contract from dual multipliers.

    Implements the contract map v = g(λ + μ s0 + μ̂ᵀ R) where g is the
    link function from the problem primitives.

    Args:
        lam: IR (individual rationality) constraint multiplier.
        mu: FOC (first-order condition) constraint multiplier.
        mu_hat: IC (incentive compatibility) constraint multipliers, shape (m,).
        s0: Score function evaluated at a0 on the grid, shape (n,).
        R: Ratio matrix from the cache, shape (m, n).
        problem: The MoralHazardProblem instance containing primitives.

    Returns:
        The optimal contract v evaluated on the grid, shape (n,).
    """
    g = problem.g
    z = lam + mu * s0 + (mu_hat @ R)
    v = g(z)
    return v


def _constraints(
    v: np.ndarray,
    *,    
    cache: Dict[str, Any],
    problem: "MoralHazardProblem",
    Ubar: float,
) -> Dict[str, Any]:
    """Evaluate all constraints and expected wage for a given contract.

    Uses precomputed arrays from the cache plus primitive inputs from
    the problem and reservation utility.

    Args:
        v: Contract values on the internal grid, shape (n,).
        cache: Precomputed cache from _make_cache.
        problem: The MoralHazardProblem instance containing primitives.
        Ubar: Agent's reservation utility.

    Returns:
        Dictionary with constraint values:
            - U0: Agent's expected utility at a0, float.
            - IR: IR constraint violation (Ubar - U0), float.
            - FOC: FOC constraint violation, float.
            - Uhat: Agent's expected utility at each a_hat, shape (m,).
            - IC: IC constraint violations (Uhat - U0), shape (m,).
            - Ewage: Expected wage E[k(v)], float.
    """
    k = problem.k_func
    wf0 = cache["wf0"]
    wf0s0 = cache["wf0s0"]
    weighted_D = cache["weighted_D"]
    C0 = cache["C0"]
    Cprime0 = cache["Cprime0"]
    C_hat = cache["C_hat"]

    # U0 = ∫ v f0 - C(a0)
    U0 = wf0 @ v - C0

    # FOC = ∫ v s0 f0 - C'(a0)
    FOC = wf0s0 @ v - Cprime0

    # Uhat (m,) and IC = Uhat - U0
    Uhat = weighted_D @ v - C_hat  # (m,)
    IC = Uhat - U0

    # IR
    IR = Ubar - U0

    # Expected wage: ∫ k(v) f0
    Ewage = wf0 @ k(v)

    return {"U0": U0, "IR": IR, "FOC": FOC, "Uhat": Uhat, "IC": IC, "Ewage": Ewage}


def _compute_expected_utility(
    v: np.ndarray,
    a: float | np.ndarray,
    problem: "MoralHazardProblem",
) -> float | np.ndarray:
    """Compute agent's expected utility for given contract and action(s).

    Evaluates U(a) = ∫ v(y) f(y|a) dy - C(a) using numerical integration
    on the problem's internal grid.

    Args:
        v: Contract values on the grid, must have shape (n,) matching
            problem.y_grid.shape.
        a: Action(s) to evaluate. Can be a scalar or 1D array.
        problem: The MoralHazardProblem instance containing primitives.

    Returns:
        Expected utility. Returns a scalar if a is scalar, otherwise
        returns an array with the same shape as a.
    """
    y_grid = problem.y_grid
    w = problem.w
    f = problem.f
    C = problem.C
    
    # f(y_grid[:, None], a) works for both scalar and array a due to broadcasting
    f_a = f(y_grid[:, None], a)
    integrals = w @ (v[:, None] * f_a)  # shape (m,) for array a, scalar for scalar a
    costs = C(a)  # shape (m,) for array a, scalar for scalar a
    result = integrals - costs
    
    # Return result as-is (numpy scalar for scalar input, array for array input)
    return result

def _compute_expected_utility_and_grad(
    v: np.ndarray,
    a: float | np.ndarray,
    problem: "MoralHazardProblem",
) -> Tuple[float | np.ndarray, float | np.ndarray]:
    """Compute agent's expected utility and its gradient with respect to action.

    Evaluates U(a) = ∫ v(y) f(y|a) dy - C(a) and its derivative dU/da
    using numerical integration on the problem's internal grid.

    Args:
        v: Contract values on the grid, must have shape (n,) matching
            problem.y_grid.shape.
        a: Action(s) to evaluate. Can be a scalar or numpy array.
        problem: The MoralHazardProblem instance containing primitives.

    Returns:
        A tuple (utility, gradient) where:
            - utility: Expected utility value(s).
            - gradient: Derivative dU/da.
        Both are scalars if a is scalar, otherwise arrays matching a's shape.
    """
    y_grid = problem.y_grid
    w = problem.w
    f = problem.f
    C = problem.C
    Cprime = problem.Cprime
    score = problem.score
    
    # f(y_grid[:, None], a) works for both scalar and array a due to broadcasting
    f_a = f(y_grid[:, None], a)
    s_a = score(y_grid[:, None], a)
    integrals = w @ (v[:, None] * f_a)  # shape (m,) for array a, scalar for scalar a
    integrals_grad = w @ (v[:, None] * f_a * s_a)  # shape (m,) for array a, scalar for scalar a
    costs = C(a)  # shape (m,) for array a, scalar for scalar a
    costs_grad = Cprime(a)  # shape (m,) for array a, scalar for scalar a

    result = integrals - costs
    result_grad = integrals_grad - costs_grad
    
    # Return result as-is (numpy scalar for scalar input, array for array input)
    return result, result_grad