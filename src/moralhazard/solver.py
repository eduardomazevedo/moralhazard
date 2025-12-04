from __future__ import annotations

import time
from typing import Dict, Any, Tuple, TYPE_CHECKING
import numpy as np
from scipy.optimize import minimize
from scipy.special import softplus, expit

from .types import DualMaximizerResults, CostMinimizationResults
from .core import _make_cache, _canonical_contract, _constraints, _compute_expected_utility

if TYPE_CHECKING:
    from .problem import MoralHazardProblem


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
    problem: "MoralHazardProblem",
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
) -> tuple[DualMaximizerResults, np.ndarray]:
    """
    Maximizes the Lagrange dual given a0, Ubar, and a_hat.
    """

    # Build cache
    cache = _make_cache(
        a0,
        a_hat,
        problem=problem,
        clip_ratio=clip_ratio,
    )

    # Initialization
    m = int(cache["R"].shape[1])
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
    """
    Maximizes the Lagrange dual given a0, Ubar, and a_hat.
    Simple wrapper to handle fallbacks if the solver fails.
    """
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
            )
        except RuntimeError:
            continue
    raise RuntimeError("Failed to solve Lagrange dual with any reparametrization.")



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
    """
    Internal method to solve cost minimization problem. Calls maximize_lagrange_dual as needed.
    First solves relaxed problem (a_hat = []).
    Then finds largest global IC violation, and iteratively adds biggest constraint violation to a_hat. Warm starts contract from the relaxed optimal.
    """
    # Initialize traces
    a_hat_trace: list[np.ndarray] = []
    multipliers_trace: list[dict] = []
    global_ic_violation_trace: list[float] = []
    best_action_distance_trace: list[float] = []
    best_action_trace: list[float] = []
    foa_flag = None if n_a_iterations == 0 else True
    
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
        def objective(a: float | np.ndarray) -> float | np.ndarray:
            return -_compute_expected_utility(results_dual.optimal_contract, a, problem=problem)

        results_minimize = minimize(
            fun=objective,
            x0=a_ic_lb,
            bounds=((a_ic_lb, np.min([a_ic_ub, intended_action])),),
            method='L-BFGS-B'
        )

        utility_best = -results_minimize.fun
        a_best = results_minimize.x.item()

        global_ic_violation = utility_best - results_dual.constraints['U0']
        best_action_distance = np.abs(a_best - intended_action)
        
        # Add diagnostics to traces
        global_ic_violation_trace.append(global_ic_violation)
        best_action_distance_trace.append(best_action_distance)
        best_action_trace.append(a_best)

        if global_ic_violation > 1e-6 and best_action_distance > 1e-6:
            foa_flag = False
            iterations += 1
            a_hat = np.concatenate([a_always_check_global_ic, np.array([a_best])])
            results_dual, theta_optimal = _maximize_lagrange_dual_with_fallback(
                a0=intended_action,
                Ubar=reservation_utility,
                a_hat=a_hat,
                problem=problem,
                theta_init=theta_relaxed_optimal,
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

