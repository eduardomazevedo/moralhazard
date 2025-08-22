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

    g_dual(θ) = E[w] + λ·IR - μ·FOC + μ̂^T IC
    ∇ g_dual(θ) = [ IR, -FOC, IC[:] ]
    """
    # Inner canonical v*(θ)
    v = _canonical_contract(theta, cache)["v"]

    # Constraints at v*
    cons = _constraints(v, cache)

    lam, mu, mu_hat = float(theta[0]), float(theta[1]), np.asarray(theta[2:], dtype=np.float64)
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


def _run_solver(
    theta_init: np.ndarray | None,
    cache: Dict[str, Any],
    last_theta: np.ndarray | None = None,
    maxiter: int = 1000,
    ftol: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """
    L-BFGS-B bridge with v0 bounds policy:
      lam ∈ [0, +∞), mu ∈ (-∞, +∞), mu_hat[j] ∈ [0, +∞)

    Warm-start preference:
      1) user-provided theta_init (if shape matches)
      2) class-level last_theta (if shape matches)
      3) default [100.0, 100.0, zeros(m)]
    """
    m = int(cache["a_hat"].shape[0])
    expected_shape = (2 + m,)

    warn_flags: list[str] = []

    def _init_vector():
        nonlocal warn_flags
        # Option 1: user-provided
        if theta_init is not None:
            ti = np.asarray(theta_init, dtype=np.float64)
            if ti.shape == expected_shape and np.all(np.isfinite(ti)):
                return ti
            warn_flags.append("theta_init_shape_mismatch_or_nonfinite")

        # Option 2: class warm-start
        if last_theta is not None:
            lt = np.asarray(last_theta, dtype=np.float64)
            if lt.shape == expected_shape and np.all(np.isfinite(lt)):
                return lt
            warn_flags.append("class_warm_start_shape_mismatch_or_nonfinite")

        # Option 3: default
        vec = np.zeros(expected_shape, dtype=np.float64)
        vec[0] = 100.0  # lam0
        vec[1] = 100.0  # mu0
        # mu_hat already zeros
        return vec

    x0 = _init_vector()

    # Bounds in SciPy: use None for unbounded ends
    bounds = [(0.0, None)]  # lam
    bounds += [(None, None)]  # mu
    bounds += [(0.0, None)] * m  # mu_hat[j]

    t0 = time.perf_counter()
    res = minimize(
        fun=lambda th: _dual_value_and_grad(th, cache),
        x0=x0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(maxiter), "ftol": float(ftol)},
    )
    t1 = time.perf_counter()

    theta_opt = np.asarray(res.x, dtype=np.float64)
    grad_final = np.asarray(res.jac, dtype=np.float64) if res.jac is not None else None
    grad_norm = float(np.max(np.abs(grad_final))) if grad_final is not None else float("nan")

    state = {
        "method": "L-BFGS-B",
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
        # v0 error text is part of the spec
        raise RuntimeError(f"Dual solver did not converge: {state['message']} (iter={state['niter']})")

    return theta_opt, state


def _solve_fixed_a(
    a0: float,
    Ubar: float,
    a_hat: np.ndarray,
    theta_init: np.ndarray | None,
    *,
    y_grid: np.ndarray,
    w: np.ndarray,
    primitives: Dict[str, Any],
    last_theta: np.ndarray | None,
) -> tuple[SolveResults, Dict[str, Any], np.ndarray]:
    """
    Internal wrapper: build cache, solve for θ*, form v*, compute constraints,
    and package SolveResults. Returns (results, cache, theta_opt).
    """
    cache = _make_cache(a0, Ubar, a_hat, y_grid, w, primitives)
    theta_opt, solver_state = _run_solver(theta_init, cache, last_theta=last_theta)

    v = _canonical_contract(theta_opt, cache)["v"]
    cons = _constraints(v, cache)

    results = SolveResults(
        optimal_contract=v,
        expected_wage=cons["Ewage"],
        multipliers={"lam": float(theta_opt[0]), "mu": float(theta_opt[1]), "mu_hat": np.asarray(theta_opt[2:], dtype=np.float64)},
        constraints=cons,
        solver_state=solver_state,
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

    Attributes on returned callable:
      - .last_theta: np.ndarray | None
      - .call_count: int
      - .reset(): None
    """
    last_theta_ref: np.ndarray | None = np.asarray(last_theta_seed, dtype=np.float64) if last_theta_seed is not None else None
    call_count = 0

    def F(a: float) -> float:
        nonlocal last_theta_ref, call_count
        theta_init = last_theta_ref if warm_start else None
        results, _cache, theta_opt = _solve_fixed_a(
            float(a), float(Ubar), np.asarray(a_hat, dtype=np.float64), theta_init,
            y_grid=y_grid, w=w, primitives=primitives, last_theta=last_theta_ref
        )
        if warm_start:
            last_theta_ref = theta_opt
        call_count += 1
        return float(results.expected_wage)

    # Lightweight attributes
    def _reset():
        nonlocal last_theta_ref, call_count
        last_theta_ref = None
        call_count = 0

    F.last_theta = last_theta_ref
    F.call_count = call_count
    F.reset = _reset  # type: ignore[attr-defined]
    return F
