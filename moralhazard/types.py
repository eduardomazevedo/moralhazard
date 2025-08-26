from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SolveResults:
    """
    Immutable container for a single dual solve at fixed a0.

    Fields follow the v0 interface:
      - a0              : float, the intended action
      - Ubar            : float, the reservation utility
      - a_hat           : np.ndarray, the action grid for the solve
      - optimal_contract: v*(y) on the internal grid; shape (n,)
      - expected_wage   : float, ∫ k(v*(y)) f(y|a0) dy
      - multipliers     : dict { "lam": float, "mu": float, "mu_hat": np.ndarray }
      - constraints     : dict { "U0": float, "IR": float, "FOC": float,
                                 "Uhat": np.ndarray, "IC": np.ndarray, "Ewage": float }
      - solver_state    : dict with method/status/iterations/timing/grad_norm metadata
    """
    a0: float
    Ubar: float
    a_hat: np.ndarray
    optimal_contract: np.ndarray
    expected_wage: float
    multipliers: dict
    constraints: dict
    solver_state: dict
