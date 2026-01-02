# moral_hazard/problem.py
from __future__ import annotations

from typing import Dict, Any, Callable
import numpy as np
from scipy.optimize import minimize_scalar

from .types import CostMinimizationResults, PrincipalSolveResults
from .grids import _make_grid
from .solver import _minimize_cost_internal
from .solver_cvxpy import _solve_cost_minimization_cvxpy, _minimum_cost_cvxpy
from .core import _compute_expected_utility


class MoralHazardProblem:
    """
    Public entry point.

    Construction
    ------------
    mhp = MoralHazardProblem(cfg)

    Required cfg structure:
      cfg = {
          "problem_params": {
              "u": callable,           # utility function (from dollars -> utils, like u(x) = log(x0 + x))
              "k": callable,           # k function (cost of compensation from utils -> dollars, like k(x) = exp(utils) - x0)
              "link_function": callable, # link function as in the paper, eg np.log(np.maximum(z, x0))
              "C": callable,           # cost function, eg C(a) = a^2
              "Cprime": callable,      # derivative of cost function, eg Cprime(a) = 2*a
              "f": callable,           # f function, eg f(y|a) = normal(y|a, sigma)
              "score": callable        # score function, eg score(y|a) = (y - a) / sigma^2
          },
          "computational_params": {
              "distribution_type": str, # either "continuous" or "discrete"
              "y_min": float,          # minimum outcome value
              "y_max": float,          # maximum outcome value
              # For continuous: "n": int (number of grid points, must be odd)
              # For discrete: "step_size": float (step size that perfectly divides y_max - y_min)
          }
      }

    Validates:
      - cfg is a dict with 'problem_params' and 'computational_params'
      - required callables exist and are callable
      - computational_params must include distribution_type, y_min, y_max
      - For continuous: must include n (odd)
      - For discrete: must include step_size that perfectly divides y_max - y_min

    Side effects:
      - stores _y_grid (length n) and Simpson weights _w (length n)
      - stores primitives in _primitives (dict with keys u, k, g, C, Cprime, f, score)
    """

    def __init__(self, cfg: dict) -> None:
        if not isinstance(cfg, dict) or "problem_params" not in cfg or "computational_params" not in cfg:
            raise TypeError("cfg must be a dict with 'problem_params' and 'computational_params'")

        p = cfg["problem_params"]
        required = ["u", "k", "link_function", "C", "Cprime", "f", "score"]
        for name in required:
            if name not in p or not callable(p[name]):
                raise KeyError(f"problem_params['{name}'] is required and must be callable")

        comp = cfg["computational_params"]
        
        # Validate that distribution_type is present
        if "distribution_type" not in comp:
            raise KeyError("computational_params['distribution_type'] is required")

        y_grid, w = _make_grid(comp["distribution_type"], comp)

        # Store primitives with names expected by internals
        self._primitives: Dict[str, Any] = {
            "u": p["u"],
            "k": p["k"],
            "g": p["link_function"],
            "C": p["C"],
            "Cprime": p["Cprime"],
            "f": p["f"],
            "score": p["score"],
        }

        self._y_grid = y_grid
        self._w = w


    # ---- Convenience passthroughs / properties --------------------------------
    @property
    def y_grid(self) -> np.ndarray:
        """Read-only outcome grid used internally (shape (n,))."""
        return self._y_grid

    @property
    def w(self) -> np.ndarray:
        """Read-only Simpson weights used internally (shape (n,))."""
        return self._w

    @property
    def f(self) -> Callable:
        """Density function f(y|a)."""
        return self._primitives["f"]

    @property
    def score(self) -> Callable:
        """Score function score(y|a)."""
        return self._primitives["score"]

    @property
    def C(self) -> Callable:
        """Cost function C(a)."""
        return self._primitives["C"]

    @property
    def Cprime(self) -> Callable:
        """Derivative of cost function C'(a)."""
        return self._primitives["Cprime"]

    @property
    def g(self) -> Callable:
        """Link function g(z)."""
        return self._primitives["g"]

    @property
    def k_func(self) -> Callable:
        """Wage function k(v)."""
        return self._primitives["k"]

    def k(self, v: np.ndarray) -> np.ndarray:
        """Convenience passthrough to problem_params['k']."""
        return self._primitives["k"](v)


    # ---- Public API ------------------------------------------------------------
    def solve_cost_minimization_problem(
        self,
        intended_action: float,
        reservation_utility: float,
        a_ic_lb: float,
        a_ic_ub: float,
        n_a_iterations: int = 10,
        theta_init: np.ndarray | None = None,
        clip_ratio: float = 1e6,
        a_always_check_global_ic: np.ndarray = np.array([0.0])
    ) -> CostMinimizationResults:
        """
        Solve the dual for the cost-minimizing contract at a given intended action a0.

        Args:
            intended_action: The intended action a0
            reservation_utility: The reservation utility Ubar
            n_a_iterations: Number of iterations for iterative solver. Defaults to 10. Set to 0 to solve the relaxed problem with no global IC constraints.
            theta_init: Optional initial theta for warm-starting.
            clip_ratio: Maximum absolute value for ratio clipping in cache construction. Defaults to 1e6.
            a_ic_lb: Lower bound for action search when using iterative solver (default: 0)
            a_ic_ub: Upper bound for action search when using iterative solver (default: infinity)
            a_always_check_global_ic: Vector of a values where we always check global IC violation. Defaults to [0].

        Returns:
            SolveResults object.
        """

        return _minimize_cost_internal(
            intended_action,
            reservation_utility,
            problem=self,
            n_a_iterations=n_a_iterations,
            theta_init=theta_init,
            clip_ratio=clip_ratio,
            a_ic_lb=a_ic_lb,
            a_ic_ub=a_ic_ub,
            a_always_check_global_ic=a_always_check_global_ic,
        )
    

    def minimum_cost(
        self,
        intended_action: float | np.ndarray,
        reservation_utility: float,
        a_ic_lb: float,
        a_ic_ub: float,
        n_a_iterations: int = 10,
        theta_init: np.ndarray | None = None,
        clip_ratio: float = 1e6,
        a_always_check_global_ic: np.ndarray = np.array([0.0]),
    ) -> float | np.ndarray:
        """
        Compute the minimum expected wage E[w(v*(a))] for given action(s).
        
        Args:
            intended_action: Action(s) to evaluate. Can be a float or numpy array of any shape.
            reservation_utility: The reservation utility Ubar
            a_ic_lb: Lower bound for action search in iterative solver
            a_ic_ub: Upper bound for action search in iterative solver
            n_a_iterations: Number of iterations for iterative solver. Defaults to 10.
            theta_init: Optional initial theta for warm-starting.
            clip_ratio: Maximum absolute value for ratio clipping in cache construction. Defaults to 1e6.
            a_always_check_global_ic: Vector of a values where we always check global IC violation. Defaults to [0.0].

        Returns:
            Expected wage (float if action is float, numpy array with same shape if action is array).
        """
        # Handle scalar vs array input
        is_scalar = isinstance(intended_action, (float, int))
        if is_scalar:
            result = self.solve_cost_minimization_problem(
                intended_action=float(intended_action),
                reservation_utility=reservation_utility,
                a_ic_lb=a_ic_lb,
                a_ic_ub=a_ic_ub,
                n_a_iterations=n_a_iterations,
                theta_init=theta_init,
                clip_ratio=clip_ratio,
                a_always_check_global_ic=a_always_check_global_ic,
            ).expected_wage
            return float(result)
        
        # Array input: flatten, compute, then reshape
        actions = np.asarray(intended_action, dtype=np.float64)
        original_shape = actions.shape
        actions_flat = actions.flatten()
        
        # Compute expected wage for each action
        expected_wages_flat = np.array([
            self.solve_cost_minimization_problem(
                intended_action=float(a),
                reservation_utility=reservation_utility,
                a_ic_lb=a_ic_lb,
                a_ic_ub=a_ic_ub,
                n_a_iterations=n_a_iterations,
                theta_init=theta_init,
                clip_ratio=clip_ratio,
                a_always_check_global_ic=a_always_check_global_ic,
            ).expected_wage
            for a in actions_flat
        ])
        
        # Reshape to original shape
        return expected_wages_flat.reshape(original_shape)

    def solve_cost_minimization_problem_cvxpy(
        self,
        intended_action: float,
        reservation_utility: float,
        a_hat: np.ndarray = None,
        v_lb: float = None,
        v_ub: float = None,
        verbose: bool = False,
    ) -> dict:
        """
        Solve the cost minimization problem using CVXPY convex optimization.
        
        This is an alternative to solve_cost_minimization_problem() that uses
        direct convex optimization rather than the dual approach. Requires the
        k function to accept an xp argument: k(v, xp=np).
        
        Args:
            intended_action: The action a0 to implement.
            reservation_utility: The agent's reservation utility Ubar.
            a_hat: Array of actions for global IC constraints. If None/empty,
                   only IR and FOC constraints are enforced.
            v_lb: Lower bound on v(y). If None, inferred from u(0).
            v_ub: Upper bound on v(y). Required for CARA/CRRA γ>1 (typically 0).
            verbose: If True, print solver output.
        
        Returns:
            dict with keys:
            - 'status': solver status string
            - 'optimal_contract': v(y) array if optimal
            - 'expected_wage': E[k(v)] if optimal
            - 'agent_utility': U(a0) if optimal
            - 'objective_value': raw objective value
        """
        return _solve_cost_minimization_cvxpy(
            intended_action=intended_action,
            reservation_utility=reservation_utility,
            problem=self,
            a_hat=a_hat,
            v_lb=v_lb,
            v_ub=v_ub,
            verbose=verbose,
        )

    def minimum_cost_cvxpy(
        self,
        intended_actions: np.ndarray,
        reservation_utility: float,
        a_hat: np.ndarray,
        v_lb: float = None,
        v_ub: float = None,
    ) -> np.ndarray:
        """
        Compute minimum expected wage for multiple actions using CVXPY.
        
        Efficient batch computation using CVXPY Parameters. Requires 
        intended_actions to be a subset of a_hat.
        
        Args:
            intended_actions: 1D array of actions to implement. Must be subset of a_hat.
            reservation_utility: The agent's reservation utility Ubar.
            a_hat: 1D array of all actions for global IC constraints.
            v_lb: Lower bound on v(y). If None, inferred from u(0).
            v_ub: Upper bound on v(y). Required for CARA/CRRA γ>1 (typically 0).
        
        Returns:
            1D array of minimum expected wages, same length as intended_actions.
        """
        return _minimum_cost_cvxpy(
            intended_actions=intended_actions,
            reservation_utility=reservation_utility,
            a_hat=a_hat,
            problem=self,
            v_lb=v_lb,
            v_ub=v_ub,
        )

    def U(self, v: np.ndarray, a: float | np.ndarray) -> float | np.ndarray:
        """
        U(a) = ∫ v(y) f(y|a) dy - C(a), evaluated on the internal Simpson grid.

        Inputs:
          - v : must have shape equal to self.y_grid.shape
          - a : scalar or 1D array

        Returns:
          - scalar if a is scalar; 1D array otherwise
        """
        # Entry point: convert to numpy arrays
        v_arr = np.asarray(v, dtype=np.float64)
        if isinstance(a, (float, int)):
            a_val = a
        else:
            a_val = np.asarray(a, dtype=np.float64)

        return _compute_expected_utility(
            v=v_arr,
            a=a_val,
            problem=self,
        )

    def solve_principal_problem(
        self,
        revenue_function: "Callable[[float], float]",
        reservation_utility: float,
        a_min: float,
        a_max: float,
        *,
        # options forwarded to expected_wage_fun(...)
        a_ic_lb: float,
        a_ic_ub: float,
        n_a_iterations: int = 10,
        clip_ratio: float = 1e6,
        a_always_check_global_ic: np.ndarray = np.array([0.0]),
        # options forwarded to the outer line search
        minimize_scalar_options: dict | None = None,
    ) -> PrincipalSolveResults:
        """
        Public API: principal's outer problem via line search over actions.

        1) Build expected_wage_fun(a) for the given Ubar and solver params.
        2) Line-search a ∈ [a_min, a_max] to maximize revenue(a) - E[w(a)].
        3) Solve the inner cost-minimization problem at the optimal action.
        4) Return a PrincipalSolveResults bundle.
        """
        # 1) Construct E[w(a)] wrapper function using minimum_cost
        def Ew(a: float) -> float:
            return self.minimum_cost(
                intended_action=a,
                reservation_utility=reservation_utility,
                a_ic_lb=a_ic_lb,
                a_ic_ub=a_ic_ub,
                n_a_iterations=n_a_iterations,
                clip_ratio=clip_ratio,
                a_always_check_global_ic=a_always_check_global_ic,
            )

        # 2) Outer line search      
        results_outer = minimize_scalar(
            fun=lambda a: Ew(a) - revenue_function(a),
            bounds=(a_min, a_max),
            method='bounded',
            options=minimize_scalar_options,
        )
        a_star = results_outer.x
        profit = -results_outer.fun

        # 3) Inner solve at a*
        results_inner = self.solve_cost_minimization_problem(
            intended_action=a_star,
            reservation_utility=reservation_utility,
            a_ic_lb=a_ic_lb,
            a_ic_ub=a_ic_ub,
            n_a_iterations=n_a_iterations,
            clip_ratio=clip_ratio,
            a_always_check_global_ic=a_always_check_global_ic,
        )

        # 4) Pack results
        return PrincipalSolveResults(
            profit=profit,
            optimal_action=a_star,
            cmp_result=results_inner,
        )

    def solve_principal_problem_cvxpy(
        self,
        revenue_function: "Callable[[float], float]",
        reservation_utility: float,
        discretized_a_grid: np.ndarray,
        v_lb: float = None,
        v_ub: float = None,
    ) -> PrincipalSolveResults:
        """
        Principal's problem using CVXPY with full discretization.
        
        Simple approach:
        1. Use discretized_a_grid as both intended actions AND a_hat for IC
        2. Compute minimum cost for all actions using batch CVXPY solver
        3. Find action with highest profit = revenue(a) - E[w(a)]
        
        Parameters
        ----------
        revenue_function : callable
            R(a) -> revenue from action a
        reservation_utility : float
            Agent's reservation utility Ubar
        discretized_a_grid : np.ndarray
            Grid of actions (used for both intended actions and IC constraints)
        v_lb, v_ub : float, optional
            Bounds on contract v(y). If None, inferred from u.
            
        Returns
        -------
        PrincipalSolveResults
            Optimal action, profit, and cost minimization result at optimum.
        """
        a_grid = np.asarray(discretized_a_grid)
        
        # 1. Compute minimum expected wage for all actions using CVXPY
        # Use a_grid as both intended_actions and a_hat
        expected_wages = self.minimum_cost_cvxpy(
            intended_actions=a_grid,
            reservation_utility=reservation_utility,
            a_hat=a_grid,
            v_lb=v_lb,
            v_ub=v_ub,
        )
        
        # 2. Compute revenue for each action
        revenues = np.array([revenue_function(a) for a in a_grid])
        
        # 3. Find action with highest profit
        profits = revenues - expected_wages
        idx_star = np.argmax(profits)
        a_star = float(a_grid[idx_star])
        profit_star = float(profits[idx_star])
        
        # 4. Get full solution at optimal action
        cvxpy_result = self.solve_cost_minimization_problem_cvxpy(
            intended_action=a_star,
            reservation_utility=reservation_utility,
            a_hat=a_grid,
            v_lb=v_lb,
            v_ub=v_ub,
        )
        
        # 5. Convert to CostMinimizationResults for consistency
        cmp_result = CostMinimizationResults(
            optimal_contract=cvxpy_result['optimal_contract'],
            expected_wage=cvxpy_result['expected_wage'],
            a_hat=a_grid,
            multipliers={},  # CVXPY doesn't return dual multipliers in our interface
            constraints={'U0': cvxpy_result['agent_utility']},
            solver_state={'status': cvxpy_result['status'], 'method': 'cvxpy'},
            n_outer_iterations=0,
            first_order_approach_holds=None,
            a_hat_trace=[],
            multipliers_trace=[],
            global_ic_violation_trace=[],
            best_action_distance_trace=[],
            best_action_trace=[],
        )
        
        return PrincipalSolveResults(
            profit=profit_star,
            optimal_action=a_star,
            cmp_result=cmp_result,
        )
