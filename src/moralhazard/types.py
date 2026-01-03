"""Dataclass definitions for solver results.

Provides immutable containers for dual maximizer, cost minimization,
and principal problem results.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pprint import pformat
import numpy as np
from typing import Callable, Optional, Dict, Any


def _clean(obj):
    """Recursively convert NumPy scalars to Python scalars for repr.

    Args:
        obj: Any object, potentially containing NumPy scalars or nested
            structures with NumPy scalars.

    Returns:
        The object with NumPy scalars converted to Python scalars.
        Arrays, dicts, lists, and tuples are processed recursively.
    """
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
    """Create a compact string summary of a numpy array.

    Args:
        arr: The numpy array to summarize.
        max_elements: If array has more elements than this, show statistics
            instead of full contents. Defaults to 10.

    Returns:
        String representation: full list if small, otherwise summary
        with shape, dtype, min, max, and mean.
    """
    if arr.size <= max_elements:
        return str(arr.tolist())
    return f"array(shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f})"


def _summarize_arrays_recursive(obj, max_elements: int = 10, key: str = None):
    """Recursively summarize 'optimal_contract' arrays in nested structures.

    Only arrays with key 'optimal_contract' are summarized to keep repr
    output readable while preserving other array details.

    Args:
        obj: Any object, potentially containing nested arrays.
        max_elements: Threshold for array summarization. Defaults to 10.
        key: Current key name for dict traversal (internal use).

    Returns:
        The object with 'optimal_contract' arrays replaced by summaries.
    """
    if isinstance(obj, np.ndarray):
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


@dataclass(frozen=True)
class DualMaximizerResults:
    """Immutable container for dual maximization solve results.

    Holds the output from a single dual solve at a fixed intended action a0.

    Attributes:
        optimal_contract: Optimal contract v*(y) on the internal grid,
            shape (n,).
        expected_wage: Expected wage E[k(v*)], float.
        multipliers: Dictionary of dual multipliers with keys:
            - 'lam': IR constraint multiplier (float).
            - 'mu': FOC constraint multiplier (float).
            - 'mu_hat': IC constraint multipliers (np.ndarray).
        constraints: Dictionary of constraint values with keys:
            - 'U0': Agent utility at a0 (float).
            - 'IR': IR constraint violation (float).
            - 'FOC': FOC constraint violation (float).
            - 'Uhat': Agent utility at each a_hat (np.ndarray).
            - 'IC': IC constraint violations (np.ndarray).
            - 'Ewage': Expected wage (float).
        solver_state: Dictionary with solver metadata including method,
            status, iterations, timing, and gradient norm.
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
    """Immutable container for cost minimization problem results.

    Contains the solution from the iterative IC constraint addition
    algorithm, including traces of the iteration history.

    Attributes:
        optimal_contract: Optimal contract v*(y) on the internal grid,
            shape (n,).
        expected_wage: Minimum expected wage E[k(v*)], float.
        a_hat: Final set of IC constraint actions, shape (m,).
        multipliers: Dictionary of final dual multipliers.
        constraints: Dictionary of final constraint values.
        solver_state: Dictionary with final solver metadata.
        n_outer_iterations: Number of IC constraint addition iterations.
        first_order_approach_holds: True if FOA is valid (no IC violations
            found), False if global IC constraints needed, None if not checked.
        a_hat_trace: List of a_hat arrays at each iteration.
        multipliers_trace: List of multiplier dicts at each iteration.
        global_ic_violation_trace: List of max IC violation at each iteration.
        best_action_distance_trace: List of |a_best - a0| at each iteration.
        best_action_trace: List of utility-maximizing actions found.
    """
    optimal_contract: np.ndarray
    expected_wage: float
    a_hat: np.ndarray
    multipliers: dict
    constraints: dict
    solver_state: dict
    n_outer_iterations: int
    first_order_approach_holds: bool | None
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
    """Immutable container for the principal's problem results.

    Contains the solution to the principal's profit maximization problem,
    including the optimal action and the full cost minimization results
    at that action.

    Attributes:
        profit: Maximum profit (revenue - expected wage) at optimal action.
        optimal_action: The profit-maximizing action a*.
        cmp_result: Full CostMinimizationResults from the inner solve
            at the optimal action.
    """
    profit: float
    optimal_action: float
    cmp_result: CostMinimizationResults

    def __repr__(self):
        data = _clean(asdict(self))
        # Summarize large arrays recursively for better readability
        data = _summarize_arrays_recursive(data)
        return f"{self.__class__.__name__}(\n{pformat(data, indent=4)}\n)"
