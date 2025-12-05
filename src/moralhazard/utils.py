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
    """
    Maximize the agent's expected utility over actions by splitting the
    action domain into subintervals and applying bounded scalar optimization
    on each.

    Parameters
    ----------
    v : np.ndarray
        Contract values at different output levels.
    a_left : float
        Lower bound for feasible actions.
    a_right : float
        Upper bound for feasible actions.
    problem : Any
        Object with parameters needed for expected utility computation.
    n_intervals : int, optional
        Number of subintervals to search over. Default is 1.

    Returns
    -------
    (best_action, best_utility) : (Optional[float], float)
        best_action : float or None
            Optimal action if at least one subproblem succeeded, else None.
        best_utility : float
            Max expected utility found. May be -inf if all failed.
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
