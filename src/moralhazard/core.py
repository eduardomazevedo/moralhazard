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
    """
    Build the *per-solve* precomputations needed by the inner/outer problems.

    This cache now contains **only** derived arrays/scalars that are expensive or
    convenient to reuse. It does **not** include primitives (functions) or
    problem parameters/inputs like y_grid, w, a0, Ubar, or a_hat.

    Returns keys: (m dimension of a_hat, n dimension of y_grid)
      - f0, s0                 : (n,) density and score vectors at a0.
      - D, R                   : (m, n) density matrix at fixed comparison actions a_hat and (m, n) ratio matrix (n as last dimension for efficiency).
      - wf0, wf0s0             : (n,) weighted density and weighted density times score vectors.
      - weighted_D              : (m, n) = (w[None, :] * D) weighted density matrix (n as last dimension for efficiency).
      - C0, Cprime0, C_hat     : floats / (m,) cost function at a0, derivative of cost function at a0, and cost function at fixed comparison actions a_hat.
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
    """
    Canonical contract map v = g(λ + μ s0 + μ̂^T R).

    Parameters
    ----------
    lam: float
        IR multiplier
    mu: float
        FOC multiplier
    mu_hat: np.ndarray
        IC multipliers (m,)
    s0 : np.ndarray (n,)
    R  : np.ndarray (m, n)
    problem: MoralHazardProblem
        Problem instance containing primitives

    Returns
    -------
    v : np.ndarray (n,)
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
    """
    Compute U(a) = ∫ v(y) f(y|a) dy - C(a), evaluated on the Simpson grid.
    Also computes the gradient dU/da.

    Inputs:
      - v : must have shape equal to problem.y_grid.shape
      - a : a scalar or numpy array
      - problem: MoralHazardProblem instance

    Returns:
      - Tuple of scalar if a is scalar; 1D array otherwise
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