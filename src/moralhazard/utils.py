"""Utility functions for agent optimization.

Provides multi-start optimization for finding the agent's utility-maximizing action.
"""
from __future__ import annotations

from typing import Callable, Optional, Dict, Any, TYPE_CHECKING
import numpy as np
from scipy.optimize import minimize_scalar, minimize

from moralhazard.core import _compute_expected_utility, _compute_expected_utility_and_grad

if TYPE_CHECKING:
    from .problem import MoralHazardProblem
    

import numpy as np
from typing import Optional, Tuple, Any
from scipy.optimize import minimize_scalar


def _maximize_agent_utility(
    v: np.ndarray,
    a_left: float,
    a_right: float,
    problem: Any,
    n_intervals: Optional[int] = 1,
) -> Tuple[Optional[float], float]:
    """Find the action that maximizes agent's expected utility.

    Uses multi-start bounded optimization by splitting the action domain
    into subintervals and running L-BFGS-B on each.

    Args:
        v: Contract values on the outcome grid, shape (n,).
        a_left: Lower bound for feasible actions.
        a_right: Upper bound for feasible actions.
        problem: MoralHazardProblem instance with primitives for
            expected utility computation.
        n_intervals: Number of subintervals for multi-start search.
            Defaults to 1.

    Returns:
        A tuple (best_action, best_utility) where:
            - best_action: Utility-maximizing action if found, else None.
            - best_utility: Maximum expected utility. May be -inf if all
              subinterval optimizations failed.
    """
    # Objective (minimize negative utility)
    def neg_expected_utility(a: float) -> float:
        return -_compute_expected_utility(v, a, problem)
    
    def neg_expected_utility_and_grad(a: float) -> Tuple[float, float]:
        eu, deu = _compute_expected_utility_and_grad(v, a, problem)
        return -eu, -deu

    interval_endpoints = np.linspace(a_left, a_right, n_intervals + 1)

    best_action = None
    best_utility = -np.inf

    for i in range(n_intervals):
        lb = interval_endpoints[i]
        ub = interval_endpoints[i + 1]

        result = minimize(
            fun=neg_expected_utility_and_grad,
            x0=(lb + ub) / 2,
            jac=True,
            method="L-BFGS-B",
            bounds=[(lb, ub)],
        )

        if result.success:
            utility = -result.fun
            if utility > best_utility:
                best_utility = utility
                best_action = result.x.item()

    return best_action, best_utility
