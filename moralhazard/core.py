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
    f = primitives["f"]
    score = primitives["score"]

    # Baseline density and score
    f0 = f(y_grid, a0)  # (n,)

    s0 = score(y_grid, a0)  # (n,)

    # Density matrix at fixed comparison actions a_hat: (n, m)
    D = f(y_grid[:, None], a_hat[None, :])

    # Cached weights/products
    wf0 = w * f0
    wf0s0 = wf0 * s0

    # R = 1 - D / f0 (broadcast along columns)
    # Ratio for the global ic constraints
    R = 1.0 - D / f0[:, None]  # (n, m)

    C = primitives["C"]
    Cprime = primitives["Cprime"]
    # Cache C0, Cprime0, C_hat, Cprime_hat
    C0 = C(a0)
    Cprime0 = Cprime(a0)
    C_hat = C(a_hat)

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
        "C0": C0,
        "Cprime0": Cprime0,
        "C_hat": C_hat,
        # function refs
        "g": primitives["g"],
        "k": primitives["k"],
        "C": C,
        "Cprime": Cprime,
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
    lam = theta[0]
    mu = theta[1]
    mu_hat = theta[2:]

    s0 = cache["s0"]
    R = cache["R"]
    g = cache["g"]

    z = lam + mu * s0 + R @ mu_hat

    v = g(z)
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
    Ubar = cache["Ubar"]
    C0, Cprime0 = cache["C0"], cache["Cprime0"]
    C_hat = cache["C_hat"]
    k = cache["k"]

    # U0 = ∫ v f0 - C(a0)
    U0 = wf0 @ v - C0

    # FOC = ∫ v s0 f0 - C'(a0)
    FOC = wf0s0 @ v - Cprime0

    # Uhat (m,) and IC = Uhat - U0
    Uhat = (w[:, None] * D).T @ v - C_hat
    IC = Uhat - U0

    # IR
    IR = Ubar - U0

    # Expected wage
    Ewage = wf0 @ k(v)

    return {"U0": U0, "IR": IR, "FOC": FOC, "Uhat": Uhat, "IC": IC, "Ewage": Ewage}
