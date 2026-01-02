# solver_cvxpy.py
"""
CVXPY-based solvers for the moral hazard cost minimization problem.

These solvers formulate the problem as a convex optimization problem and solve
it directly using CVXPY, rather than using the dual approach in solver.py.

Requirements:
- The inverse utility function k must accept an `xp` argument to specify the
  module (numpy or cvxpy), e.g.: def k(utils, xp=np): return xp.exp(utils) - x0
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any
import numpy as np

try:
    import cvxpy as cp
except ImportError:
    cp = None  # Will raise informative error if functions are called

from .core import _make_cache

if TYPE_CHECKING:
    from .problem import MoralHazardProblem


def _check_cvxpy_available():
    """Raise ImportError if cvxpy is not installed."""
    if cp is None:
        raise ImportError(
            "cvxpy is required for CVXPY-based solvers. "
            "Install it with: pip install cvxpy"
        )


def _solve_cost_minimization_cvxpy(
    intended_action: float,
    reservation_utility: float,
    *,
    problem: "MoralHazardProblem",
    a_hat: np.ndarray = None,
    v_lb: float = None,
    v_ub: float = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Internal: Solve the cost minimization problem using CVXPY.
    
    Minimizes E[k(v(y))] subject to:
      - IR: E[v|a0] - C(a0) >= Ubar
      - Local IC (FOC): E[v * score|a0] = C'(a0)
      - Global IC: U(a0) >= U(a_hat) for each action in a_hat
      - Optional: v_lb <= v <= v_ub
    
    Parameters
    ----------
    intended_action : float
        The action a0 to implement.
    reservation_utility : float
        The agent's reservation utility Ubar.
    problem : MoralHazardProblem
        The moral hazard problem instance.
        The k function must accept xp argument: k(v, xp=np).
    a_hat : np.ndarray, optional
        Array of alternative actions for global IC constraints.
        If None or empty, only IR and FOC constraints are enforced.
    v_lb : float, optional
        Lower bound on v(y). If None, inferred from u(0).
    v_ub : float, optional
        Upper bound on v(y). Required for CARA/CRRA γ>1 (typically v_ub=0).
    verbose : bool, default False
        If True, print solver output.
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'status': solver status string
        - 'optimal_contract': v(y) array if optimal, else None
        - 'expected_wage': E[k(v)] if optimal, else None
        - 'agent_utility': U(a0) if optimal, else None
        - 'objective_value': raw objective value from CVXPY
    """
    _check_cvxpy_available()
    
    a0 = intended_action
    Ubar = reservation_utility
    
    # Convert a_hat to numpy array
    if a_hat is None:
        a_hat = np.array([])
    else:
        a_hat = np.asarray(a_hat)
    
    # Infer v_lb from u(0) if not provided
    u = problem._primitives["u"]
    if v_lb is None:
        v_lb = u(0.0)
    
    # Build cache with precomputed integration weights and densities
    cache = _make_cache(a0, a_hat, problem=problem)
    
    # Extract arrays from cache
    n = len(problem.y_grid)
    m = len(a_hat)
    
    wf0 = cache["wf0"]             # (n,) = w * f(y|a0)
    wf0s0 = cache["wf0s0"]         # (n,) = w * f(y|a0) * score(y, a0)
    weighted_D = cache["weighted_D"]  # (m, n) = w * f(y|a_hat[i])
    C0 = cache["C0"]               # C(a0)
    Cprime0 = cache["Cprime0"]     # C'(a0)
    C_hat = cache["C_hat"]         # (m,) C(a_hat)
    
    # Decision variable: v(y) on the grid
    v = cp.Variable(n)
    
    # Objective: minimize E[k(v)]
    k = problem.k_func
    objective = cp.Minimize(wf0 @ k(v, xp=cp))
    
    # Constraints
    constraints = []
    
    # IR: U(a0) >= Ubar  where U(a0) = wf0 @ v - C(a0)
    constraints.append(wf0 @ v >= Ubar + C0)
    
    # FOC (local IC): dU/da|_{a=a0} = 0  =>  wf0s0 @ v = C'(a0)
    constraints.append(wf0s0 @ v == Cprime0)
    
    # Global IC: U(a0) >= U(a_hat[i]) for each i
    # Vectorized: wf0 @ v (scalar, broadcast) - weighted_D @ v (m,) >= C0 - C_hat (m,)
    if m > 0:
        constraints.append(wf0 @ v - weighted_D @ v >= C0 - C_hat)
    
    # Bounds on v
    if v_lb is not None:
        constraints.append(v >= v_lb)
    if v_ub is not None:
        constraints.append(v <= v_ub)
    
    # Solve
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose)
    
    # Prepare results
    result = {
        'status': prob.status,
        'optimal_contract': None,
        'expected_wage': None,
        'agent_utility': None,
        'objective_value': prob.value,
    }
    
    if prob.status == cp.OPTIMAL:
        v_opt = v.value
        result['optimal_contract'] = v_opt
        result['agent_utility'] = float(wf0 @ v_opt - C0)
        result['expected_wage'] = prob.value
        
        # Also return binding IC constraint info
        if m > 0:
            # IC slack: U(a0) - U(a_hat[i]) - (C0 - C_hat[i]) for each i
            # A constraint is binding if slack ≈ 0
            U_a0 = wf0 @ v_opt
            U_a_hat = weighted_D @ v_opt  # (m,)
            ic_slack = (U_a0 - C0) - (U_a_hat - C_hat)  # should be >= 0
            result['ic_slack'] = ic_slack
            result['a_hat'] = a_hat
        else:
            result['ic_slack'] = np.array([])
            result['a_hat'] = np.array([])
    
    return result


def _find_binding_ic_actions_cvxpy(
    intended_action: float,
    reservation_utility: float,
    *,
    problem: "MoralHazardProblem",
    a_ic_lb: float = 0.0,
    a_ic_ub: float = 100.0,
    n_a_hat: int = 100,
    binding_tol: float = 1e-4,
    v_lb: float = None,
    v_ub: float = None,
) -> tuple[np.ndarray, dict]:
    """
    Run CVXPY solver with many a_hats and find binding IC constraints.
    
    Parameters
    ----------
    intended_action : float
        The action a0 to implement.
    reservation_utility : float
        The agent's reservation utility.
    problem : MoralHazardProblem
        The problem instance.
    a_ic_lb, a_ic_ub : float
        Range for a_hat actions.
    n_a_hat : int
        Number of a_hat points to use.
    binding_tol : float
        Tolerance for considering a constraint binding.
    v_lb, v_ub : float, optional
        Bounds on v(y).
    
    Returns
    -------
    binding_actions : np.ndarray
        Actions where IC constraint is binding (slack < binding_tol).
    cvxpy_result : dict
        Full result from CVXPY solver.
    """
    _check_cvxpy_available()
    
    # Create dense grid of a_hats
    a_hat = np.linspace(a_ic_lb, a_ic_ub, n_a_hat)
    
    # Run CVXPY solver
    result = _solve_cost_minimization_cvxpy(
        intended_action=intended_action,
        reservation_utility=reservation_utility,
        problem=problem,
        a_hat=a_hat,
        v_lb=v_lb,
        v_ub=v_ub,
        verbose=False,
    )
    
    if result['status'] != 'optimal':
        return np.array([]), result
    
    # Find binding constraints
    ic_slack = result['ic_slack']
    binding_mask = ic_slack < binding_tol
    binding_actions = a_hat[binding_mask]
    
    return binding_actions, result


def _minimum_cost_cvxpy(
    intended_actions: np.ndarray,
    reservation_utility: float,
    a_hat: np.ndarray,
    *,
    problem: "MoralHazardProblem",
    v_lb: float = None,
    v_ub: float = None,
) -> np.ndarray:
    """
    Internal: Compute minimum expected wage for multiple intended actions using CVXPY.
    
    Uses CVXPY Parameters for efficiency when solving multiple similar problems.
    Requires intended_actions to be a subset of a_hat.
    
    Parameters
    ----------
    intended_actions : np.ndarray
        1D array of actions to implement. Must be a subset of a_hat.
    reservation_utility : float
        The agent's reservation utility Ubar.
    a_hat : np.ndarray
        1D array of all actions for global IC constraints.
    problem : MoralHazardProblem
        The moral hazard problem instance.
    v_lb : float, optional
        Lower bound on v(y). If None, inferred from u(0).
    v_ub : float, optional
        Upper bound on v(y). Required for CARA/CRRA γ>1.
    
    Returns
    -------
    np.ndarray
        1D array of minimum expected wages, same length as intended_actions.
    """
    _check_cvxpy_available()
    
    intended_actions = np.asarray(intended_actions).ravel()  # 1D
    a_hat = np.asarray(a_hat).ravel()  # 1D
    Ubar = reservation_utility
    p = len(intended_actions)
    m = len(a_hat)
    
    # Find indices of intended_actions in a_hat (vectorized)
    matches = np.isclose(intended_actions[:, np.newaxis], a_hat[np.newaxis, :])  # (p, m)
    if not np.all(matches.any(axis=1)):
        missing = intended_actions[~matches.any(axis=1)]
        raise ValueError(f"intended_actions {missing} not found in a_hat")
    a0_indices = np.argmax(matches, axis=1)  # (p,) index of first match
    
    # Infer v_lb from u(0) if not provided
    u = problem._primitives["u"]
    if v_lb is None:
        v_lb = u(0.0)
    
    # Extract problem components
    y_grid = problem.y_grid
    n = len(y_grid)
    w = problem.w
    f = problem.f
    score_fn = problem.score
    C = problem.C
    Cprime = problem.Cprime
    k = problem.k_func
    
    # =========================================================================
    # Pre-compute all matrices (y_grid as last dimension for efficiency)
    # =========================================================================
    
    # wf[i, j] = w[j] * f(y_grid[j], a_hat[i])  shape: (m, n)
    wf = w[np.newaxis, :] * f(y_grid[np.newaxis, :], a_hat[:, np.newaxis])  # (m, n)
    
    # wfs[j, k] = w[k] * f(y_grid[k], a0[j]) * score(y_grid[k], a0[j])  shape: (p, n)
    wfs = (w[np.newaxis, :] 
           * f(y_grid[np.newaxis, :], intended_actions[:, np.newaxis]) 
           * score_fn(y_grid[np.newaxis, :], intended_actions[:, np.newaxis]))  # (p, n)
    
    # Pre-compute costs (vectorized)
    C_all = C(a_hat)                 # (m,)
    Cprime_all = Cprime(intended_actions)  # (p,)
    C0_arr = C(intended_actions)     # (p,)
    
    # =========================================================================
    # Build CVXPY problem with Parameters for efficiency
    # =========================================================================
    
    # Decision variable
    v = cp.Variable(n)
    
    # Parameters that change per intended action
    # Mark wf0_param as nonneg=True so CVXPY knows wf0 @ exp(v) is convex
    wf0_param = cp.Parameter(n, nonneg=True)  # w * f(y | a0) >= 0
    wfs0_param = cp.Parameter(n)               # w * f(y | a0) * score(y | a0)
    C0_param = cp.Parameter()                  # C(a0)
    Cprime0_param = cp.Parameter()             # C'(a0)
    
    # Objective: minimize E[k(v)]
    objective = cp.Minimize(wf0_param @ k(v, xp=cp))
    
    # Constraints
    constraints = []
    
    # IR: wf0 @ v >= Ubar + C0
    constraints.append(wf0_param @ v >= Ubar + C0_param)
    
    # FOC: wfs0 @ v == Cprime0
    constraints.append(wfs0_param @ v == Cprime0_param)
    
    # Global IC: (wf0 - wf[i]) @ v >= C0 - C_hat[i] for each i
    # Vectorized: wf0_param @ v (scalar, broadcast) - wf @ v (m,) >= C0_param - C_all (m,)
    if m > 0:
        constraints.append(wf0_param @ v - wf @ v >= C0_param - C_all)
    
    # Bounds on v
    if v_lb is not None:
        constraints.append(v >= v_lb)
    if v_ub is not None:
        constraints.append(v <= v_ub)
    
    # Build problem once
    prob = cp.Problem(objective, constraints)
    
    # =========================================================================
    # Solve for each intended action
    # =========================================================================
    
    results = np.zeros(p)
    
    for j in range(p):
        idx = a0_indices[j]
        
        # Update parameters
        wf0_param.value = wf[idx, :]
        wfs0_param.value = wfs[j, :]
        C0_param.value = C0_arr[j]
        Cprime0_param.value = Cprime_all[j]
        
        # Solve
        prob.solve(verbose=False)
        
        if prob.status == cp.OPTIMAL:
            results[j] = prob.value
        else:
            results[j] = np.nan
    
    return results
