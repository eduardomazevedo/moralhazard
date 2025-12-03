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
      - optimal_contract: v*(y) on the internal grid; shape (n,)
      - expected_wage   : float, ∫ k(v*(y)) f(y|a0) dy
      - multipliers     : dict { "lam": float, "mu": float, "mu_hat": np.ndarray }
      - constraints     : dict { "U0": float, "IR": float, "FOC": float,
                                 "Uhat": np.ndarray, "IC": np.ndarray, "Ewage": float }
      - solver_state    : dict with method/status/iterations/timing/grad_norm metadata
    """
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

    - profit                      : profit at optimal action
    - optimal_action              : argmax a*
    - cmp_result                  : CostMinimizationResults from the inner solve at optimal action
    """
    profit: float
    optimal_action: float
    cmp_result: CostMinimizationResults

    def __repr__(self):
        data = _clean(asdict(self))
        return f"{self.__class__.__name__}(\n{pformat(data, indent=4)}\n)"
