from __future__ import annotations

import time
from typing import Dict, Any, Callable, Tuple
import numpy as np
from scipy.optimize import minimize

from .types import SolveResults
from .core import _make_cache, _canonical_contract, _constraints


def _dual_value_and_grad(theta: np.ndarray, cache: Dict[str, Any]) -> Tuple[float, np.ndarray]:
    """
    Return (obj, grad) for a minimizer, where obj = -g_dual(θ)
    and grad = -∇g_dual(θ) with ∇ via Danskin on the inner optimum v*(θ).

    Assumes types already validated upstream.
    """
    n = cache["y_grid"].shape[0]
    m = cache["a_hat"].shape[0]

    lam = float(theta[0])
    mu = float(theta[1])
    mu_hat = theta[2:]

    if mu_hat.shape != (m,):
        raise ValueError(f"theta[2:] must have shape {(m,)}; got {mu_hat.shape}")

    # Inner optimum v*(θ) via canonical map
    vm = _canonical_contract(theta, cache)
    v = vm["v"]

    # Constraints at v
    cons = _constraints(v, cache)

    IR = cons["IR"]
    FOC = cons["FOC"]
    IC = cons["IC"]
    Ewage = cons["Ewage"]

    g_dual = Ewage + lam * IR - mu * FOC + (mu_hat @ IC if IC.size else 0.0)

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
    primitives: Dict[str, Any],
    theta_init: np.ndarray | None = None,
    last_theta: np.ndarray | None = None,
    maxiter: int = 1000,
    ftol: float = 1e-8,
) -> tuple[SolveResults, Dict[str, Any], np.ndarray]:
    """
    Solve the dual at fixed action a0 and reservation utility Ubar.

    Returns:
      - SolveResults
      - cache used
      - theta_opt for warm-starting
    """
    # Build cache
    cache = _make_cache(float(a0), float(Ubar), np.asarray(a_hat, dtype=np.float64), y_grid, w, primitives)

    # Initialization policy with warm-start
    m = int(cache["a_hat"].shape[0])
    expected_shape = (2 + m,)
    warn_flags: list[str] = []

    def _select_x0() -> np.ndarray:
        # 1) user-provided theta_init
        if theta_init is not None:
            ti = np.asarray(theta_init, dtype=np.float64)
            if ti.shape == expected_shape and np.all(np.isfinite(ti)):
                return ti
            warn_flags.append("theta_init_shape_mismatch_or_nonfinite")

        # 2) class-level last_theta
        if last_theta is not None:
            lt = np.asarray(last_theta, dtype=np.float64)
            if lt.shape == expected_shape and np.all(np.isfinite(lt)):
                return lt
            warn_flags.append("class_warm_start_shape_mismatch_or_nonfinite")

        # 3) default
        vec = np.zeros(expected_shape, dtype=np.float64)
        vec[0] = 100.0  # lam0 ≥ 0
        vec[1] = 100.0  # mu0 free (unbounded)
        # mu_hat already zeros (≥ 0)
        return vec

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
        args=(cache,),
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
    vm = _canonical_contract(theta_opt, cache)
    v_star = vm["v"]
    cons = _constraints(v_star, cache)

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


def _make_expected_wage_fun(
    *,
    y_grid: np.ndarray,
    w: np.ndarray,
    primitives: Dict[str, Any],
    Ubar: float,
    a_hat: np.ndarray,
    warm_start: bool,
    last_theta_seed: np.ndarray | None,
) -> Callable[[float], float]:
    """
    Factory returning F(a) = E[w(v*(a))] with an optional warm start across calls.

    Attributes on returned callable (kept live across calls):
      - .last_theta: np.ndarray | None
      - .call_count: int
      - .reset(): None
    """
    last_theta_ref: np.ndarray | None = (
        np.asarray(last_theta_seed, dtype=np.float64) if last_theta_seed is not None else None
    )
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
            primitives=primitives,
            theta_init=theta_init,
            last_theta=last_theta_ref,
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
