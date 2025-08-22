from __future__ import annotations

from typing import Dict, Any
import numpy as np


def _make_cache(
    a0: float,
    Ubar: float,
    a_hat: np.ndarray,
    y_grid: np.ndarray,
    w: np.ndarray,
    primitives: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build all per-solve, vectorized arrays (immutable in practice).

    Assumes boundary layer already validated/coerced types.
    """
    a0 = float(a0)
    Ubar = float(Ubar)
    if a_hat.ndim != 1:
        raise ValueError(f"a_hat must be a 1D array; got shape {a_hat.shape}")

    n_shape = y_grid.shape

    f = primitives["f"]
    score = primitives["score"]

    # Baseline density and score
    f0 = f(y_grid, a0)  # (n,)
    if f0.shape != n_shape:
        raise ValueError(f"f(y_grid, a0) must return shape {n_shape} aligned with the internal grid; got {f0.shape}")

    if np.any(f0 <= 0.0):
        # policy: fail fast; user adjusts tails / grid
        raise RuntimeError("Encountered zero/near-zero baseline density on grid; adjust y_min/y_max or model tails")

    s0 = score(y_grid, a0)  # (n,)
    if s0.shape != n_shape:
        raise ValueError(f"score(y_grid, a0) must return shape {n_shape} aligned with the internal grid; got {s0.shape}")

    # Density matrix at fixed comparison actions a_hat: (n, m)
    if a_hat.size == 0:
        D = np.zeros((y_grid.shape[0], 0), dtype=np.float64)
    else:
        # Only require vectorization in y; build columns by looping over a_hat.
        D = np.column_stack([f(y_grid, float(a)) for a in a_hat])

    # Cached weights/products
    wf0 = w * f0
    wf0s0 = wf0 * s0

    # R = 1 - D / f0 (broadcast along columns)
    if D.size == 0:
        R = np.zeros((y_grid.shape[0], 0), dtype=np.float64)
    else:
        R = 1.0 - D / f0[:, None]

    return {
        "y_grid": y_grid,
        "w": w,
        "a0": a0,
        "Ubar": Ubar,
        "a_hat": a_hat,
        "f0": f0,
        "s0": s0,
        "wf0": wf0,
        "wf0s0": wf0s0,
        "D": D,
        "R": R,
        # function refs
        "g": primitives["g"],
        "k": primitives["k"],
        "C": primitives["C"],
        "Cprime": primitives["Cprime"],
    }


def _canonical_contract(theta: np.ndarray, cache: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Canonical contract map v = g(λ + μ s0 + R μ̂).

    Parameters
    ----------
    theta : np.ndarray (2 + m,)
        [lam, mu, mu_hat...]
    cache : dict
        Must contain s0, R, and g.

    Returns
    -------
    {"z": np.ndarray (n,), "v": np.ndarray (n,)}
    """
    lam = float(theta[0])
    mu = float(theta[1])
    mu_hat = theta[2:]

    s0 = cache["s0"]
    R = cache["R"]
    g = cache["g"]

    z = lam + mu * s0
    if mu_hat.size:
        z = z + R @ mu_hat

    v = np.asarray(g(z), dtype=np.float64)
    return {"z": z, "v": v}


def _constraints(v: np.ndarray, cache: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate all constraints and Ewage on the internal grid.

    Returns a dict with:
      - U0, IR, FOC : floats
      - Uhat, IC    : np.ndarray (m,)
      - Ewage       : float
    """
    v = np.asarray(v, dtype=np.float64)
    expected = cache["y_grid"].shape
    if v.shape != expected:
        raise ValueError(f"v must have shape {expected}; got {v.shape}")

    w, wf0, wf0s0 = cache["w"], cache["wf0"], cache["wf0s0"]
    D = cache["D"]
    a0, Ubar = cache["a0"], cache["Ubar"]
    C, Cprime = cache["C"], cache["Cprime"]
    k = cache["k"]

    # U0 = ∫ v f0 - C(a0)
    U0 = float(wf0 @ v - C(a0))

    # FOC = ∫ v s0 f0 - C'(a0)
    FOC = float(wf0s0 @ v - Cprime(a0))

    # Uhat (m,) and IC = Uhat - U0
    if D.size == 0:
        Uhat = np.zeros((0,), dtype=np.float64)
    else:
        Uhat = (w[:, None] * D).T @ v - C(cache["a_hat"])
    IC = Uhat - U0

    # IR
    IR = float(Ubar - U0)

    # Expected wage
    Ewage = float(wf0 @ k(v))

    return {"U0": U0, "IR": IR, "FOC": FOC, "Uhat": Uhat, "IC": IC, "Ewage": Ewage}
