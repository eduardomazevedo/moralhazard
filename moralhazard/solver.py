from __future__ import annotations

import time
from typing import Dict, Any, Callable, Tuple
import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .types import SolveResults
from .core import _make_cache, _canonical_contract, _constraints, _compute_expected_utility


def _dual_value_and_grad(
    theta: np.ndarray,
    cache: Dict[str, Any],
    g: Callable[[np.ndarray], np.ndarray],
    k: Callable[[np.ndarray], np.ndarray],
    Ubar: float,
) -> Tuple[float, np.ndarray]:
    """
    Return (obj, grad) for a minimizer, where obj = -g_dual(θ)
    and grad = -∇g_dual(θ) with ∇ via Danskin on the inner optimum v*(θ).

    Assumes types already validated upstream.
    """
    m = cache["R"].shape[1]

    lam = theta[0]
    mu = theta[1]
    mu_hat = theta[2:]

    # Inner optimum v*(θ) via canonical map
    v = _canonical_contract(theta, cache["s0"], cache["R"], g)

    # Constraints at v
    cons = _constraints(v, cache, k=k, Ubar=float(Ubar))

    IR = cons["IR"]
    FOC = cons["FOC"]
    IC = cons["IC"]
    Ewage = cons["Ewage"]

    g_dual = Ewage + lam * IR - mu * FOC + mu_hat @ IC

    # ∇g
    grad = np.empty_like(theta, dtype=np.float64)
    grad[0] = IR
    grad[1] = -FOC
    if IC.size:
        grad[2:] = IC
    return -float(g_dual), -grad  # minimizer expects -g and -∇g


def _minimize_cost_a_hat(
    a0: float,
    Ubar: float,
    a_hat: np.ndarray,
    *,
    y_grid: np.ndarray,
    w: np.ndarray,
    f: Callable[[np.ndarray, float | np.ndarray], np.ndarray],
    score: Callable[[np.ndarray, float | np.ndarray], np.ndarray],
    C: Callable[[float | np.ndarray], float | np.ndarray],
    Cprime: Callable[[float | np.ndarray], float | np.ndarray],
    g: Callable[[np.ndarray], np.ndarray],
    k: Callable[[np.ndarray], np.ndarray],
    theta_init: np.ndarray | None = None,
    maxiter: int = 1000,
    ftol: float = 1e-8,
) -> tuple[SolveResults, Dict[str, Any], np.ndarray]:
    """
    Solve the dual at fixed action a0 and reservation utility Ubar.

    Returns:
      - SolveResults
      - cache used (precomputed arrays only)
      - theta_opt for warm-starting
    """
    # Build precomputed cache (no primitives or raw params stored)
    cache = _make_cache(
        float(a0),
        np.asarray(a_hat, dtype=np.float64),
        np.asarray(y_grid, dtype=np.float64),
        np.asarray(w, dtype=np.float64),
        f=f,
        score=score,
        C=C,
        Cprime=Cprime,
    )

    # Initialization policy
    m = int(cache["R"].shape[1])
    expected_shape = (2 + m,)
    warn_flags: list[str] = []

    def _select_x0() -> np.ndarray:
        # 1) user-provided theta_init
        if theta_init is not None:
            ti = np.asarray(theta_init, dtype=np.float64)
            if ti.shape == expected_shape and np.all(np.isfinite(ti)):
                return ti
            warn_flags.append("theta_init_shape_mismatch_or_nonfinite")

        # 2) default
        return np.zeros(expected_shape, dtype=np.float64)

    x0 = _select_x0()

    # Bounds: lam ∈ [0, ∞), mu ∈ (-∞, ∞), each mu_hat[j] ∈ [0, ∞)
    bounds = [(0.0, None)] + [(None, None)] + [(0.0, None)] * m

    # Solve
    t0 = time.time()
    res = minimize(
        fun=_dual_value_and_grad,
        x0=x0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(maxiter), "ftol": float(ftol)},
        args=(cache, g, k, float(Ubar)),
    )
    t1 = time.time()

    theta_opt = np.asarray(res.x, dtype=np.float64)
    grad_norm = float(np.linalg.norm(np.asarray(res.jac, dtype=np.float64))) if hasattr(res, "jac") else None

    state = {
        "method": "L-BFGS-B",
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "niter": int(getattr(res, "nit", -1)),
        "nfev": int(getattr(res, "nfev", -1)) if hasattr(res, "nfev") else None,
        "njev": int(getattr(res, "njev", -1)) if hasattr(res, "njev") else None,
        "time_sec": float(t1 - t0),
        "fun": float(res.fun),      # minimized value: -g_dual
        "grad_norm": grad_norm,
    }
    if warn_flags:
        state["warn_flags"] = warn_flags

    if not res.success:
        raise RuntimeError(f"Dual solver did not converge: {state['message']} (iter={state['niter']})")

    # Reconstruct v*(θ) and constraints for reporting
    v_star = _canonical_contract(theta_opt, cache["s0"], cache["R"], g)
    cons = _constraints(v_star, cache, k=k, Ubar=float(Ubar))

    # Multipliers
    lam = float(theta_opt[0])
    mu = float(theta_opt[1])
    mu_hat = np.asarray(theta_opt[2:], dtype=np.float64).reshape((m,))

    results = SolveResults(
        optimal_contract=np.asarray(v_star, dtype=np.float64),
        expected_wage=float(cons["Ewage"]),
        multipliers={"lam": lam, "mu": mu, "mu_hat": mu_hat},
        constraints={
            "U0": float(cons["U0"]),
            "IR": float(cons["IR"]),
            "FOC": float(cons["FOC"]),
            "Uhat": np.asarray(cons["Uhat"], dtype=np.float64),
            "IC": np.asarray(cons["IC"], dtype=np.float64),
            "Ewage": float(cons["Ewage"]),
        },
        solver_state=state,
    )
    return results, cache, theta_opt


def _minimize_cost_iterative(
    a0: float,
    Ubar: float,
    *,
    a_min: float,
    a_max: float,
    n_a_grid: int = 100,
    n_a_iterations: int = 1,
    y_grid: np.ndarray,
    w: np.ndarray,
    f: Callable[[np.ndarray, float | np.ndarray], np.ndarray],
    score: Callable[[np.ndarray, float | np.ndarray], np.ndarray],
    C: Callable[[float | np.ndarray], float | np.ndarray],
    Cprime: Callable[[float | np.ndarray], float | np.ndarray],
    g: Callable[[np.ndarray], np.ndarray],
    k: Callable[[np.ndarray], np.ndarray],
    theta_init: np.ndarray | None = None,
    maxiter: int = 1000,
    ftol: float = 1e-8,
) -> tuple[SolveResults, Dict[str, Any], np.ndarray]:
    """
    Solve the dual iteratively by updating a_hat based on expected utility maximization.
    
    This method iteratively solves the cost minimization problem by:
    1. Starting with a_current = (a_min + a_max) / 2
    2. Solving with a_hat = [0, a_current]
    3. Computing expected utility for all actions in the grid
    4. Updating a_current to maximize U + tie-breaking penalty
    5. Repeating for n_a_iterations
    
    Args:
        a0: intended action
        Ubar: reservation utility
        a_min: minimum action for grid search
        a_max: maximum action for grid search
        n_a_grid: number of grid points for action search
        n_a_iterations: number of iterations to perform (after the initial guess)
        y_grid: outcome grid
        w: Simpson weights
        f: density function
        score: score function
        C: cost function
        Cprime: derivative of cost function
        g: link function
        k: wage function
        theta_init: optional initial theta for warm-starting
        maxiter: maximum iterations for optimizer
        ftol: function tolerance for optimizer
        
    Returns:
        - SolveResults from final iteration
        - cache used (precomputed arrays only)
        - theta_opt for warm-starting
    """
    # Create action grid for utility evaluation
    a_grid = np.linspace(a_min, a_max, n_a_grid)
    
    # Initialize a_current as midpoint
    a_current = (a_min + a_max) / 2
    
    # Initialize theta for warm-starting across iterations
    current_theta = theta_init
    
    for iteration in range(n_a_iterations + 1):
        # Set a_hat for this iteration
        a_hat = np.array([0.0, a_current])
        
        # Solve the cost minimization problem
        results, cache, theta_opt = _minimize_cost_a_hat(
            a0=a0,
            Ubar=Ubar,
            a_hat=a_hat,
            y_grid=y_grid,
            w=w,
            f=f,
            score=score,
            C=C,
            Cprime=Cprime,
            g=g,
            k=k,
            theta_init=current_theta,
            maxiter=maxiter,
            ftol=ftol,
        )
        
        # Update theta for next iteration
        current_theta = theta_opt
        
        # Only update a_current if this isn't the last iteration
        if iteration < n_a_iterations:
            # Compute expected utility for all actions in the grid
            v_optimal = results.optimal_contract
            U_values = _compute_expected_utility(
                v=v_optimal,
                a=a_grid,
                y_grid=y_grid,
                w=w,
                f=f,
                C=C,
            )
            
            # Exclude actions within 1/10 of grid range if n_a_grid >= 20, no exclusion otherwise
            exclusion_radius = (a_max - a_min) / 10 * (n_a_grid >= 20)
            exclude_mask = np.abs(a_grid - a0) > exclusion_radius
            
            # If all actions are excluded, fall back to no exclusion
            if not np.any(exclude_mask):
                exclude_mask = np.ones_like(a_grid, dtype=bool)
            
            # Find the action that maximizes utility among non-excluded actions
            max_idx = np.argmax(U_values[exclude_mask])
            # Convert back to original grid index
            max_idx = np.where(exclude_mask)[0][max_idx]
            grid_opt_a = float(a_grid[max_idx])
            
            # Refine with 1D optimization starting from grid optimum
            def neg_utility(a):
                a_val = float(a)
                # Add penalty if action is too close to intended action
                if abs(a_val - a0) <= exclusion_radius:
                    return 1e6  # Large penalty
                return -float(_compute_expected_utility(
                    v=v_optimal,
                    a=a_val,
                    y_grid=y_grid,
                    w=w,
                    f=f,
                    C=C,
                ))
            
            # Use minimize_scalar with Brent's method for 1D optimization
            # Brent's method is more robust for 1D optimization than L-BFGS-B
            opt_result = minimize_scalar(
                neg_utility,
                bracket=(a_min, grid_opt_a, a_max),  # Provide bracket for Brent's method
                method='brent',
                options={'xtol': 1e-8}
            )
            
            if opt_result.success:
                a_current = float(opt_result.x)
            else:
                # Fall back to grid optimum if optimization fails
                a_current = grid_opt_a
    
    # Return the final results
    a_hat = np.array([0.0, a_current])
    return results, cache, current_theta, a_hat



