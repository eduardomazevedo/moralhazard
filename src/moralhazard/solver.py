from __future__ import annotations

import time
from typing import Dict, Any, Callable, Tuple
import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .types import SolveResults
from .core import _make_cache, _canonical_contract, _constraints, _compute_expected_utility


def _decode_theta(theta: np.ndarray) -> tuple[float, float, np.ndarray]:
    """
    Decode the theta array into lam, mu, and mu_hat.
    """
    lam = theta[0]
    mu = theta[1]
    mu_hat = theta[2:]
    return lam, mu, mu_hat

def _encode_theta(lam: float, mu: float, mu_hat: np.ndarray) -> np.ndarray:
    """
    Encode the lam, mu, and mu_hat into a theta array.
    
    Returns an array of shape (2 + m,) where m is the length of mu_hat.
    """
    return np.concatenate([np.array([lam, mu], dtype=np.float64), np.atleast_1d(mu_hat).astype(np.float64)])

def _pad_theta_for_warm_start(theta: np.ndarray, target_m: int) -> np.ndarray:
    """
    Pad a theta vector to match a target number of mu_hat components.
    
    If theta has shape (2,), pads with zeros to shape (2 + target_m,).
    If theta already has the correct shape, returns it unchanged.
    If theta has more components than needed, truncates (though this is unusual).
    
    Args:
        theta: Current theta vector, shape (2,) or (2 + m,)
        target_m: Target number of mu_hat components
        
    Returns:
        Padded theta vector of shape (2 + target_m,)
    """
    current_shape = theta.shape[0]
    target_shape = 2 + target_m
    
    if current_shape == target_shape:
        return theta
    elif current_shape == 2:
        # Pad with zeros for mu_hat components
        return np.concatenate([theta, np.zeros(target_m, dtype=np.float64)])
    elif current_shape > target_shape:
        # Truncate (unusual case)
        return theta[:target_shape]
    else:
        # This shouldn't happen, but handle gracefully
        # Pad with zeros
        padding = np.zeros(target_shape - current_shape, dtype=np.float64)
        return np.concatenate([theta, padding])

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

    lam, mu, mu_hat = _decode_theta(theta)

    # Inner optimum v*(θ) via canonical map
    v = _canonical_contract(lam, mu, mu_hat, cache["s0"], cache["R"], g)

    # Constraints at v
    cons = _constraints(v, cache=cache, k=k, Ubar=Ubar)

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
    return -g_dual, -grad  # minimizer expects -g and -∇g


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
    clip_ratio: float = 1e6,
) -> tuple[SolveResults, np.ndarray]:
    """
    Solve the dual at fixed action a0 and reservation utility Ubar.

    Returns:
      - SolveResults
      - theta_opt for warm-starting
    """
    # Build precomputed cache (no primitives or raw params stored)
    cache = _make_cache(
        a0,
        a_hat,
        y_grid=y_grid,
        w=w,
        f=f,
        score=score,
        C=C,
        Cprime=Cprime,
        clip_ratio=clip_ratio,
    )

    # Initialization policy
    m = int(cache["R"].shape[1])
    expected_shape = (2 + m,)
    warn_flags: list[str] = []

    def _select_x0() -> np.ndarray:
        # 1) user-provided theta_init
        if theta_init is not None:
            if np.all(np.isfinite(theta_init)):
                if theta_init.shape == expected_shape:
                    return theta_init
                else:
                    # Try to pad/truncate to match expected shape
                    # This allows warm starting lam and mu from an empty a_hat solve
                    padded = _pad_theta_for_warm_start(theta_init, m)
                    if padded.shape == expected_shape:
                        warn_flags.append("theta_init_shape_mismatch_padded")
                        return padded
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
        options={"maxiter": int(maxiter), "ftol": ftol},
        args=(cache, g, k, Ubar),
    )
    t1 = time.time()

    theta_opt = res.x
    grad_norm = np.linalg.norm(res.jac) if hasattr(res, "jac") and res.jac is not None else None

    state = {
        "method": "L-BFGS-B",
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "niter": int(getattr(res, "nit", -1)),
        "nfev": int(getattr(res, "nfev", -1)) if hasattr(res, "nfev") else None,
        "njev": int(getattr(res, "njev", -1)) if hasattr(res, "njev") else None,
        "time_sec": t1 - t0,
        "fun": res.fun,      # minimized value: -g_dual
        "grad_norm": grad_norm,
    }
    if warn_flags:
        state["warn_flags"] = warn_flags

    if not res.success:
        raise RuntimeError(f"Dual solver did not converge: {state['message']} (iter={state['niter']})")

    # Reconstruct v*(θ) and constraints for reporting
    lam_opt, mu_opt, mu_hat_opt = _decode_theta(theta_opt)
    v_star = _canonical_contract(lam_opt, mu_opt, mu_hat_opt, cache["s0"], cache["R"], g)
    cons = _constraints(v_star, cache=cache, k=k, Ubar=Ubar)

    # Multipliers (use decoded values)
    lam = lam_opt
    mu = mu_opt
    mu_hat = mu_hat_opt

    results = SolveResults(
        a0=a0,
        Ubar=Ubar,
        a_hat=a_hat,
        optimal_contract=v_star,
        expected_wage=cons["Ewage"],
        multipliers={"lam": lam, "mu": mu, "mu_hat": mu_hat},
        constraints={
            "U0": cons["U0"],
            "IR": cons["IR"],
            "FOC": cons["FOC"],
            "Uhat": cons["Uhat"],
            "IC": cons["IC"],
            "Ewage": cons["Ewage"],
        },
        solver_state=state,
    )
    return results, theta_opt


def _minimize_cost_iterative(
    a0: float,
    Ubar: float,
    *,
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
    clip_ratio: float = 1e6,
    a_ic_lb: float = -np.inf,
    a_ic_ub: float = np.inf,
    a_ic_initial: float = 0.0,
) -> tuple[SolveResults, np.ndarray]:
    """
    Solve the dual iteratively by updating a_hat based on expected utility maximization.
    
    This method iteratively solves the cost minimization problem by:
    1. Starting with a_current = a_ic_initial
    2. Solving with a_hat = [0, a_current]
    3. Finding the action that maximizes expected utility within the specified bounds
    4. If the arg max is close to a0, set a_hat = [0, 0]
    5. If the arg max is far from a0, set a_hat = [0, arg max]
    6. Repeating for n_a_iterations
    
    Args:
        a0: intended action
        Ubar: reservation utility
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
        a_ic_lb: lower bound for action search (default: -infinity)
        a_ic_ub: upper bound for action search (default: infinity)
        a_ic_initial: initial action value to start search from (default: 0.0)
        
    Returns:
        - SolveResults from final iteration
        - theta_opt for warm-starting
    """
    # Initialize a_current with the provided initial value
    a_current = a_ic_initial
    
    # Initialize theta for warm-starting across iterations
    # Note: warm-starting might not work well when a_hat shape changes
    current_theta = theta_init
    
    for iteration in range(n_a_iterations + 1):
        # Set a_hat for this iteration
        # Ensure a_current is a scalar (not numpy array) for array construction
        a_current_scalar = a_current.item() if isinstance(a_current, np.ndarray) else a_current
        a_hat = np.array([0.0, a_current_scalar])
        
        # Solve the cost minimization problem
        results, theta_opt = _minimize_cost_a_hat(
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
            clip_ratio=clip_ratio,
        )
        
        # Update theta for next iteration
        current_theta = theta_opt
        
        # Only update a_current if this isn't the last iteration
        if iteration < n_a_iterations:
            v_optimal = results.optimal_contract
            
            # Define negative utility function for maximization
            def neg_utility(a):
                utility = _compute_expected_utility(
                    v=v_optimal,
                    a=a,
                    y_grid=y_grid,
                    w=w,
                    f=f,
                    C=C,
                )
                return -utility
            
            # Find the action that maximizes utility within the specified bounds
            # Use the provided bounds, but ensure they're reasonable
            a_lower = max(a_ic_lb, -0.9 * abs(a0))  # Reasonable lower bound
            a_upper = min(a_ic_ub, 0.9 * abs(a0))    # Reasonable upper bound
            
            # Ensure we have a valid bracket for the optimizer
            if a_lower >= a_upper:
                a_lower = a_upper - 0.1  # Ensure lower < upper
            
            opt_result = minimize_scalar(
                neg_utility,
                bounds=(a_lower, a_upper),
                method='bounded',
                options={'xatol': 1e-8}
            )
            
            if opt_result.success:
                arg_max_a = opt_result.x
                
                # The optimizer now respects bounds, so arg_max_a is already within constraints
                # Check if arg max is close to a0
                # Use a small threshold relative to |a0| or absolute threshold
                threshold = max(0.1 * abs(a0), 0.01)
                
                if abs(arg_max_a - a0) <= threshold:
                    a_current = 0.0  # Set a_hat = [0] (more efficient than [0, 0])
                else:
                    a_current = arg_max_a  # Set a_hat = [0, arg_max]
            else:
                # If optimization fails, keep current value
                pass
    
    # Return the final results with consistent structure
    # Ensure a_current is a scalar (not numpy array) for array construction
    a_current_scalar = a_current.item() if isinstance(a_current, np.ndarray) else a_current
    a_hat = np.array([0.0, a_current_scalar])
    
    # Create a new SolveResults object with the final a_hat value
    final_results = SolveResults(
        a0=results.a0,
        Ubar=results.Ubar,
        a_hat=a_hat,
        optimal_contract=results.optimal_contract,
        expected_wage=results.expected_wage,
        multipliers=results.multipliers,
        constraints=results.constraints,
        solver_state=results.solver_state,
    )
    
    return final_results, current_theta



