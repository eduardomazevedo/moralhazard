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


def _summarize_array(arr: np.ndarray, max_elements: int = 10) -> str:
    """Summarize a numpy array for pretty printing."""
    if arr.size <= max_elements:
        return str(arr.tolist())
    return f"array(shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f})"


def _summarize_arrays_recursive(obj, max_elements: int = 10, key: str = None):
    """Recursively summarize numpy arrays named 'optimal_contract' in nested data structures."""
    if isinstance(obj, np.ndarray):
        # Only summarize if this is an optimal_contract array
        if key == "optimal_contract":
            return _summarize_array(obj, max_elements)
        return obj
    elif isinstance(obj, dict):
        return {k: _summarize_arrays_recursive(v, max_elements, key=k) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(_summarize_arrays_recursive(x, max_elements, key=key) for x in obj)
    else:
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
        # Summarize large arrays recursively for better readability
        data = _summarize_arrays_recursive(data)
        return f"{self.__class__.__name__}(\n{pformat(data, indent=4)}\n)"


@dataclass(frozen=True)
class CostMinimizationResults:
    """
    Immutable container for the cost minimization results. Same as for the ahat solver, but with iterations information.
    """
    optimal_contract: np.ndarray
    expected_wage: float
    a_hat: np.ndarray
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
        # Summarize large arrays recursively for better readability
        data = _summarize_arrays_recursive(data)
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
        # Summarize large arrays recursively for better readability
        data = _summarize_arrays_recursive(data)
        return f"{self.__class__.__name__}(\n{pformat(data, indent=4)}\n)"
