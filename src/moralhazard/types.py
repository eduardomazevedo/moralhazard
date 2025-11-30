from __future__ import annotations

from dataclasses import dataclass, asdict
from pprint import pformat
import numpy as np
from typing import Callable, Optional, Dict, Any


# ----------------------------------------------------------------------
# Utility: recursively convert NumPy scalars to Python scalars for repr
# ----------------------------------------------------------------------
def _clean(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(_clean(x) for x in obj)
    return obj


# ----------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class DualMaximizerResults:
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

    def __repr__(self):
        data = _clean(asdict(self))
        return f"{self.__class__.__name__}(\n{pformat(data, indent=4)}\n)"


@dataclass(frozen=True)
class CostMinimizationResults:
    """
    Immutable container for the cost minimization results. Same as for the ahat solver, but with iterations information.
    """
    a0: float
    Ubar: float
    a_hat: np.ndarray
    optimal_contract: np.ndarray
    expected_wage: float
    multipliers: dict
    constraints: dict
    solver_state: dict
    n_outer_iterations: int
    a_hat_trace: list[np.ndarray]
    multipliers_trace: list[dict]
    global_ic_violation_trace: list[float]
    best_action_distance_trace: list[float]
    best_action_trace: list[float]

    def __repr__(self):
        data = _clean(asdict(self))
        return f"{self.__class__.__name__}(\n{pformat(data, indent=4)}\n)"


@dataclass(frozen=True)
class PrincipalSolveResults:
    """
    Immutable container for the principal's outer problem.

    - a_min, a_max, a_init        : action search bounds and initializer
    - revenue_function            : reference to the revenue function
    - Ubar                        : reservation utility
    - profit                      : profit at optimal action
    - optimal_action              : argmax a*
    - a_hat                       : action grid used by the inner cost-minimization solve (if any)
    - optimal_contract            : v*(y) from the inner solve at a*
    - multipliers                 : dual multipliers from inner solve
    - constraints                 : constraint diagnostics from inner solve
    - solver_state_outer          : metadata from the line search (minimize_scalar)
    - solver_state_inner          : metadata from the inner cost-minimization solve
    """
    a_min: float
    a_max: float
    a_init: float
    revenue_function: Callable[[float], float]
    Ubar: float
    profit: float
    optimal_action: float
    a_hat: Optional[np.ndarray]
    optimal_contract: np.ndarray
    multipliers: Dict[str, Any]
    constraints: Dict[str, Any]
    solver_state_outer: Dict[str, Any]
    solver_state_inner: Dict[str, Any]

    def __repr__(self):
        data = _clean(asdict(self))
        return f"{self.__class__.__name__}(\n{pformat(data, indent=4)}\n)"
