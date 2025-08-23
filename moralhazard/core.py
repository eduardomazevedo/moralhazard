from __future__ import annotations

from typing import Dict, Any, Callable
import numpy as np


def _make_cache(
    a0: float,
    a_hat: np.ndarray,
    y_grid: np.ndarray,
    w: np.ndarray,
    *,
    f: Callable[[np.ndarray, float | np.ndarray], np.ndarray],
    score: Callable[[np.ndarray, float | np.ndarray], np.ndarray],
    C: Callable[[float | np.ndarray], float | np.ndarray],
    Cprime: Callable[[float | np.ndarray], float | np.ndarray],
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
    y_grid = np.asarray(y_grid, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    a_hat = np.asarray(a_hat, dtype=np.float64)

    # Baseline density and score at a0 on the fixed grid
    f0 = np.asarray(f(y_grid, float(a0)), dtype=np.float64)           # (n,)
    s0 = np.asarray(score(y_grid, float(a0)), dtype=np.float64)       # (n,)

    # Density matrix at fixed comparison actions a_hat: (n, m)
    D = np.asarray(f(y_grid[:, None], a_hat[None, :]), dtype=np.float64)  # (n, m)

    # Cached weights/products
    wf0 = w * f0
    wf0s0 = wf0 * s0

    # Ratio for the global IC constraints: R = 1 - D / f0 (broadcast along columns)
    R = 1.0 - D / f0[:, None]  # (n, m)

    # Precompute C-related terms
    C0 = float(C(float(a0)))
    Cprime0 = float(Cprime(float(a0)))
    C_hat = np.asarray(C(a_hat), dtype=np.float64)  # (m,)

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
    theta: np.ndarray,
    s0: np.ndarray,
    R: np.ndarray,
    g: Callable[[np.ndarray], np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Canonical contract map v = g(λ + μ s0 + R μ̂).

    Parameters
    ----------
    theta : np.ndarray (2 + m,)
        [lam, mu, mu_hat...]
    s0 : np.ndarray (n,)
    R  : np.ndarray (n, m)
    g  : link function

    Returns
    -------
    v : np.ndarray (n,)
    """
    lam = float(theta[0])
    mu = float(theta[1])
    mu_hat = np.asarray(theta[2:], dtype=np.float64)

    z = lam + mu * s0 + R @ mu_hat
    v = np.asarray(g(z), dtype=np.float64)
    return v


def _constraints(
    v: np.ndarray,
    cache: Dict[str, Any],
    *,
    k: Callable[[np.ndarray], np.ndarray],
    Ubar: float,
) -> Dict[str, Any]:
    """
    Evaluate all constraints and E[wage] given a contract v on the internal grid.

    Uses only precomputed arrays from `cache` plus primitive inputs `k` and `Ubar`.

    Returns:
      - U0, IR, FOC : floats
      - Uhat, IC    : np.ndarray (m,)
      - Ewage       : float
    """
    v = np.asarray(v, dtype=np.float64)
    wf0 = cache["wf0"]
    expected = wf0.shape
    if v.shape != expected:
        raise ValueError(f"v must have shape {expected}; got {v.shape}")

    wf0s0 = cache["wf0s0"]
    WD_T = cache["WD_T"]
    C0 = cache["C0"]
    Cprime0 = cache["Cprime0"]
    C_hat = cache["C_hat"]

    # U0 = ∫ v f0 - C(a0)
    U0 = float(wf0 @ v - C0)

    # FOC = ∫ v s0 f0 - C'(a0)
    FOC = float(wf0s0 @ v - Cprime0)

    # Uhat (m,) and IC = Uhat - U0
    Uhat = np.asarray(WD_T @ v, dtype=np.float64) - C_hat  # (m,)
    IC = Uhat - U0

    # IR
    IR = float(Ubar) - U0

    # Expected wage: ∫ k(v) f0
    Ewage = float(wf0 @ np.asarray(k(v), dtype=np.float64))

    return {"U0": U0, "IR": IR, "FOC": FOC, "Uhat": Uhat, "IC": IC, "Ewage": Ewage}
