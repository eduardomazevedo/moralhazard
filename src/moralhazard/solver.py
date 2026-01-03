"""Dual-based iterative solver implementing Algorithm 1 from Azevedo and Woff (2025).

Solves the cost minimization problem by maximizing the Lagrange dual and
iteratively adding binding global IC constraints.
"""
from __future__ import annotations

import time
import warnings
from typing import Dict, Any, Tuple, TYPE_CHECKING
import numpy as np
from scipy.optimize import minimize
from scipy.special import softplus, expit

from .types import DualMaximizerResults, CostMinimizationResults
from .core import _make_cache, _canonical_contract, _constraints, _compute_expected_utility
from .utils import _maximize_agent_utility
from .solver_cvxpy import _find_binding_ic_actions_cvxpy

if TYPE_CHECKING:
    from .problem import MoralHazardProblem


def _decode_theta(theta: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Decode the theta array into individual dual multipliers.

    Args:
        theta: Combined multiplier array of shape (2 + m,).

    Returns:
        A tuple (lam, mu, mu_hat) where:
            - lam: IR constraint multiplier (float).
            - mu: FOC constraint multiplier (float).
            - mu_hat: IC constraint multipliers, shape (m,).
    """
    lam = theta[0]
    mu = theta[1]
    mu_hat = theta[2:]
    return lam, mu, mu_hat

def _encode_theta(lam: float, mu: float, mu_hat: np.ndarray) -> np.ndarray:
    """Encode individual dual multipliers into a single theta array.

    Args:
        lam: IR constraint multiplier.
        mu: FOC constraint multiplier.
        mu_hat: IC constraint multipliers, shape (m,).

    Returns:
        Combined multiplier array of shape (2 + m,).
    """
    return np.concatenate([np.array([lam, mu], dtype=np.float64), np.atleast_1d(mu_hat).astype(np.float64)])

def _pad_theta_for_warm_start(theta: np.ndarray, target_m: int) -> np.ndarray:
    """Pad or truncate theta vector to match target IC constraint count.

    Used for warm-starting when the number of IC constraints changes
    between iterations.

    Args:
        theta: Current theta vector, shape (2,) or (2 + m,).
        target_m: Target number of mu_hat (IC) components.

    Returns:
        Adjusted theta vector of shape (2 + target_m,). Pads with zeros
        if too short, truncates if too long.
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
    problem: "MoralHazardProblem",
    Ubar: float,
) -> Tuple[float, np.ndarray]:
    """Compute negative dual objective and gradient for scipy minimizer.

    Returns (obj, grad) where obj = -g_dual(θ) and grad = -∇g_dual(θ).
    The gradient uses Danskin's theorem on the inner optimum v*(θ).

    Args:
        theta: Combined multiplier array of shape (2 + m,).
        cache: Precomputed arrays from _make_cache.
        problem: The MoralHazardProblem instance.
        Ubar: Agent's reservation utility.

    Returns:
        A tuple (objective, gradient) for use with scipy.optimize.minimize.
        Both are negated since scipy minimizes but we want to maximize.
    """
    m = cache["R"].shape[0]

    lam, mu, mu_hat = _decode_theta(theta)

    # Inner optimum v*(θ) via canonical map
    v = _canonical_contract(lam, mu, mu_hat, cache["s0"], cache["R"], problem)

    # Constraints at v
    cons = _constraints(v, cache=cache, problem=problem, Ubar=Ubar)

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


def _maximize_lagrange_dual(
    a0: float,
    Ubar: float,
    a_hat: np.ndarray,
    *,
    problem: "MoralHazardProblem",
    theta_init: np.ndarray | None = None,
    maxiter: int = 1000,
    ftol: float = 1e-8,
    clip_ratio: float = 1e6,
    reparametrize: str | None = None,
    raise_on_failure: bool = True,
) -> tuple[DualMaximizerResults, np.ndarray]:
    """Maximize the Lagrange dual for a fixed set of constraints.

    Solves the dual problem to find optimal multipliers and the corresponding
    contract for a given intended action a0, reservation utility Ubar, and
    set of IC constraint actions a_hat.
w
    Args:
        a0: The intended action to implement.
        Ubar: Agent's reservation utility.
        a_hat: Array of actions for global IC constraints, shape (m,).
        problem: The MoralHazardProblem instance.
        theta_init: Initial guess for multipliers. If None, uses zeros.
        maxiter: Maximum optimizer iterations. Defaults to 1000.
        ftol: Function tolerance for convergence. Defaults to 1e-8.
        clip_ratio: Ratio clipping for numerical stability. Defaults to 1e6.
        reparametrize: Reparametrization method for positivity constraints.
            One of None (L-BFGS-B bounds), "softplus", or "log".
        raise_on_failure: If True (default), raises RuntimeError on failure.
            If False, returns results with success=False for debugging.

    Returns:
        A tuple (results, theta_opt) where:
            - results: DualMaximizerResults with contract and diagnostics.
            - theta_opt: Optimal multiplier array of shape (2 + m,).

    Raises:
        RuntimeError: If solver fails and raise_on_failure is True.
    """

    # Build cache
    cache = _make_cache(
        a0,
        a_hat,
        problem=problem,
        clip_ratio=clip_ratio,
    )

    # Initialization
    m = int(cache["R"].shape[0])
    expected_shape = (2 + m,)
    warn_flags: list[str] = []

    def _select_x0() -> np.ndarray:
        if theta_init is not None:
            if np.all(np.isfinite(theta_init)):
                if theta_init.shape == expected_shape:
                    return theta_init
                else:
                    padded = _pad_theta_for_warm_start(theta_init, m)
                    if padded.shape == expected_shape:
                        warn_flags.append("theta_init_shape_mismatch_padded")
                        return padded
            warn_flags.append("theta_init_shape_mismatch_or_nonfinite")
        return np.zeros(expected_shape, dtype=np.float64)

    x0 = _select_x0()

    # Helper transforms
    def _id(x): 
        return x

    # Improved softplus inverse
    def _softplus_inv(y):
        y = np.maximum(y, 1e-12)        # <- improvement #2
        return np.log(np.expm1(y))

    # Setup reparametrization
    if reparametrize is None:
        f = f_inv = _id
        f_prime = lambda x: np.ones_like(x)
        use_reparam = False

    elif reparametrize == "log":
        f  = lambda x: np.exp(x)
        f_inv = lambda y: np.log(y)
        f_prime = lambda x: np.exp(x)
        use_reparam = True

    elif reparametrize == "softplus":
        f  = softplus
        f_inv = _softplus_inv
        f_prime = expit
        use_reparam = True

    else:
        raise ValueError("Invalid reparametrize option.")

    # indices where positivity enforced
    idx_pos = [0] + list(range(2, 2 + m))

    if use_reparam:
        # Construct φ0 from θ0 with safe handling of zeros/negatives
        phi0 = x0.copy()

        # --- CHANGE #1: ensure theta > 0 before applying f_inv
        theta_safe = phi0[idx_pos].copy()
        bad = (~np.isfinite(theta_safe)) | (theta_safe <= 0)
        if np.any(bad):
            theta_safe[bad] = 1e-2  # treat zeros / negatives as tiny positive
        phi0[idx_pos] = f_inv(theta_safe)
        # ---------------------------------------------------------

        def _wrapped(phi, cache, problem, Ubar):
            theta = phi.copy()
            theta[idx_pos] = f(theta[idx_pos])
            val, g = _dual_value_and_grad(theta, cache, problem, Ubar)
            g2 = g.copy()
            g2[idx_pos] *= f_prime(phi[idx_pos])
            return val, g2

        fun = _wrapped
        x0_used = phi0
        method_used = "BFGS"
        bounds_used = None

    else:
        fun = _dual_value_and_grad
        x0_used = x0
        method_used = "L-BFGS-B"
        bounds_used = [(0.0, None)] + [(None, None)] + [(0.0, None)] * m

    # Solve
    minimize_kwargs = {
        "fun": fun,
        "x0": x0_used,
        "jac": True,
        "method": method_used,
        "options": {"maxiter": int(maxiter), "ftol": ftol},
        "args": (cache, problem, Ubar),
    }
    if bounds_used is not None:
        minimize_kwargs["bounds"] = bounds_used

    t0 = time.time()
    res = minimize(**minimize_kwargs)
    t1 = time.time()

    # Check for solver failure
    solver_failed = not res.success
    if solver_failed:
        msg = f"Solver failed with reparametrize={reparametrize}, status={res.status}, message='{res.message}'"
        warnings.warn(msg, RuntimeWarning)
        if raise_on_failure:
            raise RuntimeError(msg)

    # Convert φ → θ for output
    theta_opt = res.x.copy()
    if use_reparam:
        theta_opt[idx_pos] = f(theta_opt[idx_pos])

    grad_norm = (
        np.linalg.norm(res.jac)
        if hasattr(res, "jac") and res.jac is not None
        else None
    )

    state = {
        "method": method_used,
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "niter": int(getattr(res, "nit", -1)),
        "nfev": int(getattr(res, "nfev", -1)) if hasattr(res, "nfev") else None,
        "njev": int(getattr(res, "njev", -1)) if hasattr(res, "njev") else None,
        "time_sec": t1 - t0,
        "fun": res.fun,
        "grad_norm": grad_norm,
    }
    if warn_flags:
        state["warn_flags"] = warn_flags

    # Decode & evaluate
    lam_opt, mu_opt, mu_hat_opt = _decode_theta(theta_opt)
    v_star = _canonical_contract(
        lam_opt, mu_opt, mu_hat_opt, cache["s0"], cache["R"], problem
    )
    cons = _constraints(v_star, cache=cache, problem=problem, Ubar=Ubar)

    results = DualMaximizerResults(
        optimal_contract=v_star,
        expected_wage=cons["Ewage"],
        multipliers={"lam": lam_opt, "mu": mu_opt, "mu_hat": mu_hat_opt},
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


def _maximize_lagrange_dual_with_fallback(
    a0: float,
    Ubar: float,
    a_hat: np.ndarray,
    *,
    problem: "MoralHazardProblem",
    theta_init: np.ndarray | None = None,
    maxiter: int = 1000,
    ftol: float = 1e-8,
    clip_ratio: float = 1e6,
    reparametrize_pecking_order: list[str] = [None, "softplus", "log"],
) -> tuple[DualMaximizerResults, np.ndarray]:
    """Maximize the Lagrange dual with automatic fallback on failure.

    Tries different reparametrization strategies in order until one succeeds.
    If all fail, issues a warning and returns the last result for debugging.

    Args:
        a0: The intended action to implement.
        Ubar: Agent's reservation utility.
        a_hat: Array of actions for global IC constraints, shape (m,).
        problem: The MoralHazardProblem instance.
        theta_init: Initial guess for multipliers. If None, uses zeros.
        maxiter: Maximum optimizer iterations. Defaults to 1000.
        ftol: Function tolerance for convergence. Defaults to 1e-8.
        clip_ratio: Ratio clipping for numerical stability. Defaults to 1e6.
        reparametrize_pecking_order: List of reparametrization methods to
            try in order. Defaults to [None, "softplus", "log"].

    Returns:
        A tuple (results, theta_opt) where:
            - results: DualMaximizerResults with contract and diagnostics.
            - theta_opt: Optimal multiplier array of shape (2 + m,).
    """
    last_result = None
    last_theta = None
    last_reparametrize = None
    
    for reparametrize in reparametrize_pecking_order:
        try:
            return _maximize_lagrange_dual(
                a0=a0,
                Ubar=Ubar,
                a_hat=a_hat,
                problem=problem,
                theta_init=theta_init,
                maxiter=maxiter,
                ftol=ftol,
                clip_ratio=clip_ratio,
                reparametrize=reparametrize,
                raise_on_failure=True,
            )
        except RuntimeError:
            # Get results from failed solver for debugging (without raising)
            last_result, last_theta = _maximize_lagrange_dual(
                a0=a0,
                Ubar=Ubar,
                a_hat=a_hat,
                problem=problem,
                theta_init=theta_init,
                maxiter=maxiter,
                ftol=ftol,
                clip_ratio=clip_ratio,
                reparametrize=reparametrize,
                raise_on_failure=False,
            )
            last_reparametrize = reparametrize
            continue
    
    # All solvers failed - warn and return last result for debugging
    warnings.warn(
        f"All reparametrizations failed for a0={a0}. "
        f"Returning result from reparametrize={last_reparametrize} for debugging.",
        RuntimeWarning
    )
    return last_result, last_theta



def _minimize_cost_internal(
    intended_action: float,
    reservation_utility: float,
    *,
    problem: "MoralHazardProblem",
    n_a_iterations: int = 10,
    theta_init: np.ndarray | None = None,
    clip_ratio: float = 1e6,
    a_ic_lb: float = 0,
    a_ic_ub: float = np.inf,
    a_always_check_global_ic: np.ndarray = np.array([0.0]),
) -> CostMinimizationResults:
    """Solve the cost minimization problem with iterative IC constraint addition.

    Implements the iterative algorithm:
        1. Solve relaxed problem (no global IC constraints).
        2. Find action with largest global IC violation.
        3. Add violating action to constraint set and re-solve.
        4. Repeat until no violations or max iterations reached.

    Uses warm-starting from previous iteration's solution.

    Args:
        intended_action: The action a0 to implement.
        reservation_utility: Agent's reservation utility Ubar.
        problem: The MoralHazardProblem instance.
        n_a_iterations: Maximum iterations for adding IC constraints.
            Set to 0 to solve only the relaxed problem. Defaults to 10.
        theta_init: Initial guess for multipliers. If None, uses zeros.
        clip_ratio: Ratio clipping for numerical stability. Defaults to 1e6.
        a_ic_lb: Lower bound for IC violation search. Defaults to 0.
        a_ic_ub: Upper bound for IC violation search. Defaults to inf.
        a_always_check_global_ic: Actions to always include in IC check
            on first violation. Defaults to [0.0].

    Returns:
        CostMinimizationResults containing optimal contract, expected wage,
        constraint diagnostics, and iteration traces.
    """
    # Initialize traces
    a_hat_trace: list[np.ndarray] = []
    multipliers_trace: list[dict] = []
    global_ic_violation_trace: list[float] = []
    best_action_distance_trace: list[float] = []
    best_action_trace: list[float] = []
    foa_flag = None if n_a_iterations == 0 else True
    cvxpy_fallback = False
    
    # Solve relaxed problem
    a_hat = np.array([], dtype=np.float64)
    results_dual, theta_optimal = _maximize_lagrange_dual_with_fallback(
        a0=intended_action,
        Ubar=reservation_utility,
        a_hat=a_hat,
        problem=problem,
        theta_init=theta_init,
        clip_ratio=clip_ratio,
    )
    theta_relaxed_optimal = theta_optimal
    
    # Add initial relaxed solve to traces
    a_hat_trace.append(a_hat.copy())
    multipliers_trace.append(results_dual.multipliers.copy())

    # Check for any global IC violations
    iterations = 0
    while iterations < n_a_iterations:
        a_best, utility_best = _maximize_agent_utility(
            v=results_dual.optimal_contract,
            a_left=a_ic_lb,
            a_right=a_ic_ub,
            problem=problem,
            n_intervals=5,
        )
        
        # Handle case where _maximize_agent_utility failed to find any valid action
        if a_best is None:
            warnings.warn(
                f"_maximize_agent_utility returned None for a0={intended_action}. "
                f"Breaking out of iteration loop.",
                RuntimeWarning
            )
            break
        
        global_ic_violation = utility_best - results_dual.constraints['U0']
        best_action_distance = np.abs(a_best - intended_action)
        
        # Add diagnostics to traces
        global_ic_violation_trace.append(global_ic_violation)
        best_action_distance_trace.append(best_action_distance)
        best_action_trace.append(a_best)

        global_ic_tolerance = 1e-3
        best_action_distance_tolerance = 1e-3

        if global_ic_violation > global_ic_tolerance and best_action_distance > best_action_distance_tolerance:
            if iterations == 0:
                a_hat = a_always_check_global_ic.copy()
            foa_flag = False
            iterations += 1
            
            # Check if we need CVXPY fallback:
            # 1. Solver failed (success=False) or
            # 2. Best action is repeated (already in a_hat, so we wouldn't grow)
            solver_failed = not results_dual.solver_state.get('success', True)
            best_action_repeated = a_best in a_hat
            
            if (solver_failed or best_action_repeated) and (cvxpy_fallback is False):
                # CVXPY fallback: find binding constraints using CVXPY
                reason = "solver failed" if solver_failed else "best action repeated"
                warnings.warn(
                    f"Triggering CVXPY fallback for a0={intended_action} ({reason}). "
                    f"Running CVXPY with 100 a_hats to find binding IC constraints.",
                    RuntimeWarning
                )
                cvxpy_fallback = True
                
                try:
                    binding_actions, cvxpy_result = _find_binding_ic_actions_cvxpy(
                        intended_action=intended_action,
                        reservation_utility=reservation_utility,
                        problem=problem,
                        a_ic_lb=a_ic_lb,
                        a_ic_ub=a_ic_ub if np.isfinite(a_ic_ub) else 100.0,
                        n_a_hat=100,
                        binding_tol=1e-4,
                    )
                    
                    if len(binding_actions) > 0:
                        warnings.warn(
                            f"CVXPY found {len(binding_actions)} binding IC constraints: "
                            f"{binding_actions[:5]}{'...' if len(binding_actions) > 5 else ''}",
                            RuntimeWarning
                        )
                        # Add binding actions to a_hat
                        for a_bind in binding_actions:
                            if a_bind not in a_hat:
                                a_hat = np.concatenate([a_hat, np.array([a_bind])])
                    else:
                        warnings.warn(
                            f"CVXPY found no binding IC constraints. Adding a_best={a_best} anyway.",
                            RuntimeWarning
                        )
                        if a_best not in a_hat:
                            a_hat = np.concatenate([a_hat, np.array([a_best])])
                            
                except Exception as e:
                    warnings.warn(
                        f"CVXPY fallback failed: {e}. Adding a_best={a_best} instead.",
                        RuntimeWarning
                    )
                    if a_best not in a_hat:
                        a_hat = np.concatenate([a_hat, np.array([a_best])])
            else:
                # Normal case: just add the best action
                a_hat = np.concatenate([a_hat, np.array([a_best])])
            
            results_dual, theta_optimal = _maximize_lagrange_dual_with_fallback(
                a0=intended_action,
                Ubar=reservation_utility,
                a_hat=a_hat,
                problem=problem,
                theta_init=theta_optimal,
                clip_ratio=clip_ratio,
            )

            # Add this iteration to traces
            a_hat_trace.append(a_hat.copy())
            multipliers_trace.append(results_dual.multipliers.copy())
        else:
            break

    # Build CostMinimizationResults
    cost_results = CostMinimizationResults(
        optimal_contract=results_dual.optimal_contract,
        expected_wage=results_dual.expected_wage,
        a_hat=a_hat,
        multipliers=results_dual.multipliers,
        constraints=results_dual.constraints,
        solver_state=results_dual.solver_state,
        n_outer_iterations=iterations,
        first_order_approach_holds=foa_flag,
        a_hat_trace=a_hat_trace,
        multipliers_trace=multipliers_trace,
        global_ic_violation_trace=global_ic_violation_trace,
        best_action_distance_trace=best_action_distance_trace,
        best_action_trace=best_action_trace,
    )
    
    return cost_results

