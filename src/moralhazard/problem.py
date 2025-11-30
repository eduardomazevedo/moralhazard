# moral_hazard/problem.py
from __future__ import annotations

from typing import Dict, Any, Callable
import numpy as np

from .types import CostMinimizationResults, PrincipalSolveResults
from .grids import _make_grid
from .solver import _minimize_cost_internal
from .utils import _make_expected_wage_fun, _solve_principal_problem
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
        n_a_grid_points: int = 10,
        n_a_iterations: int = 1,
        theta_init: np.ndarray | None = None,
        clip_ratio: float = 1e6,
        a_always_check_global_ic: np.ndarray = np.array([0.0])
    ) -> CostMinimizationResults:
        """
        Solve the dual for the cost-minimizing contract at a given intended action a0.

        Args:
            intended_action: The intended action a0
            reservation_utility: The reservation utility Ubar
            n_a_iterations: Number of iterations for iterative solver. Defaults to 1.
            theta_init: Optional initial theta for warm-starting.
            clip_ratio: Maximum absolute value for ratio clipping in cache construction. Defaults to 1e6.
            a_ic_lb: Lower bound for action search when using iterative solver (default: 0)
            a_ic_ub: Upper bound for action search when using iterative solver (default: infinity)
            n_a_grid_points: Number of grid points for the action grid. Defaults to 10.
            a_always_check_global_ic: Vector of a values where we always check global IC violation. Defaults to [0].

        Returns:
            SolveResults object.
        """

        results = _minimize_cost_internal(
            intended_action,
            reservation_utility,
            problem=self,
            n_a_iterations=n_a_iterations,
            theta_init=theta_init,
            clip_ratio=clip_ratio,
            a_ic_lb=a_ic_lb,
            a_ic_ub=a_ic_ub,
            n_a_grid_points=n_a_grid_points,
            a_always_check_global_ic=a_always_check_global_ic,
        )

        return results

    def expected_wage_fun(
        self,
        reservation_utility: float,
        solver: str = "a_hat",
        a_hat: np.ndarray | None = None,
        n_a_iterations: int = 1,
        warm_start: bool = True,
        clip_ratio: float = 1e6,
        a_ic_lb: float = -np.inf,
        a_ic_ub: float = np.inf,
        a_ic_initial: float = 0.0,
    ) -> "Callable[[float], float]":
        """
        Returns F(a) = E[w(v*(a))] where v*(a) is the cost-minimizing contract
        at intended action a for the provided Ū and solver parameters.

        Args:
            reservation_utility: The reservation utility Ubar
            solver: Either "a_hat" (default) or "iterative"
            a_hat: Required when solver="a_hat". The action grid for the solve.
            n_a_iterations: Number of iterations for iterative solver. Defaults to 1.
            warm_start: When True, successive calls reuse the last θ* found
                       inside the returned function (does NOT mutate class-level warm start).
            clip_ratio: Maximum absolute value for ratio clipping in cache construction. Defaults to 1e6.
            a_ic_lb: Lower bound for a hat action search when using iterative solver (default: -infinity)
            a_ic_ub: Upper bound for a hat action search when using iterative solver (default: infinity)
            a_ic_initial: Initial a hat action value to start search from when using iterative solver (default: 0.0)

        Returns:
            Callable function F(a) that returns the expected wage for action a.
        """
        if solver not in ["a_hat", "iterative"]:
            raise ValueError(f"solver must be 'a_hat' or 'iterative', got '{solver}'")

        # Entry point: convert a_hat if provided
        a_hat_arr = np.asarray(a_hat, dtype=np.float64) if a_hat is not None else None

        F = _make_expected_wage_fun(
            problem=self,
            Ubar=reservation_utility,
            solver=solver,
            a_hat=a_hat_arr,
            n_a_iterations=int(n_a_iterations),
            warm_start=bool(warm_start),
            clip_ratio=clip_ratio,
            a_ic_lb=a_ic_lb,
            a_ic_ub=a_ic_ub,
            a_ic_initial=a_ic_initial,
        )

        return F

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
        a_init: float,
        *,
        # options forwarded to expected_wage_fun(...)
        solver: str = "a_hat",
        a_hat: np.ndarray | None = None,
        n_a_iterations: int = 1,
        warm_start: bool = True,
        clip_ratio: float = 1e6,
        a_ic_lb: float = -np.inf,
        a_ic_ub: float = np.inf,
        a_ic_initial: float = 0.0,
        # options forwarded to the outer line search
        minimize_scalar_options: dict | None = None,
        # options forwarded to the inner cost-minimization solver call
        theta_init: np.ndarray | None = None,
    ) -> PrincipalSolveResults:
        """
        Public API: principal's outer problem via line search over actions.

        1) Build expected_wage_fun(a) for the given Ubar and solver params.
        2) Line-search a ∈ [a_min, a_max] to maximize revenue(a) - E[w(a)].
        3) Solve the inner cost-minimization problem at the optimal action.
        4) Return a PrincipalSolveResults bundle.
        """
        # 1) Construct E[w(a)] using the class' primitives
        # Entry point: convert a_hat if provided
        a_hat_arr = np.asarray(a_hat, dtype=np.float64) if a_hat is not None else None

        Ew = self.expected_wage_fun(
            reservation_utility=reservation_utility,
            solver=solver,
            a_hat=a_hat_arr,
            n_a_iterations=n_a_iterations,
            warm_start=warm_start,
            clip_ratio=clip_ratio,
            a_ic_lb=a_ic_lb,
            a_ic_ub=a_ic_ub,
            a_ic_initial=a_ic_initial,
        )

        # 2) Outer line search
        outer = _solve_principal_problem(
            revenue_function=revenue_function,
            expected_wage_fun=Ew,
            a_min=a_min,
            a_max=a_max,
            a_init=a_init,
            minimize_scalar_options=minimize_scalar_options,
        )
        a_star = outer["optimal_action"]
        profit = outer["profit"]

        # 3) Inner solve at a*
        # Entry point: convert theta_init if provided
        theta_init_arr = np.asarray(theta_init, dtype=np.float64) if theta_init is not None else None

        inner: CostMinimizationResults = self.solve_cost_minimization_problem(
            intended_action=a_star,
            reservation_utility=reservation_utility,
            n_a_iterations=n_a_iterations,
            theta_init=theta_init_arr,
            clip_ratio=clip_ratio,
            a_ic_lb=a_ic_lb,
            a_ic_ub=a_ic_ub,
        )

        # 4) Pack results
        return PrincipalSolveResults(
            a_min=a_min,
            a_max=a_max,
            a_init=a_init,
            revenue_function=revenue_function,
            Ubar=reservation_utility,
            profit=profit,
            optimal_action=a_star,
            a_hat=inner.a_hat,
            optimal_contract=inner.optimal_contract,
            multipliers=inner.multipliers,
            constraints=inner.constraints,
            solver_state_outer=outer["outer_solver_state"],
            solver_state_inner=inner.solver_state,
        )
