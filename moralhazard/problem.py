# moral_hazard/problem.py
from __future__ import annotations

from typing import Dict, Any, Callable
import numpy as np

from .types import SolveResults
from .grids import _make_grid
from .solver import _minimize_cost_a_hat, _minimize_cost_iterative
from .utils import _make_expected_wage_fun
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
      - builds fixed y_grid (length n) and Simpson weights
      - stores primitive callables
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
        # Tiny fast checks: run user callables once at [0, 0]
        try:
            y0 = np.asarray([0.0], dtype=np.float64)
            a0 = float(0.0)
            v0 = np.zeros_like(y0)
            # Run each primitive once; ignore outputs except basic sanity where cheap
            _ = self._primitives["f"](y0, a0)
            _ = self._primitives["score"](y0, a0)
            _ = self._primitives["g"](v0)
            _ = self._primitives["k"](v0)
            _ = float(self._primitives["C"](a0))
            _ = float(self._primitives["Cprime"](a0))
        except Exception as e:
            raise TypeError(f"Primitive callable check failed on [0,0]: {e}")

        self._y_grid = y_grid
        self._w = w

    # ---- Convenience passthroughs / properties --------------------------------

    @property
    def y_grid(self) -> np.ndarray:
        """Read-only outcome grid used internally (shape (n,))."""
        return self._y_grid

    def k(self, v: np.ndarray) -> np.ndarray:
        """Convenience passthrough to problem_params['k']."""
        return np.asarray(self._primitives["k"](v), dtype=np.float64)

    # ---- Public API ------------------------------------------------------------

    def solve_cost_minimization_problem(
        self,
        intended_action: float,
        reservation_utility: float,
        solver: str = "a_hat",
        a_hat: np.ndarray | None = None,
        n_a_iterations: int = 1,
        theta_init: np.ndarray | None = None,
        clip_ratio: float = 1e6,
        a_search_lb: float = -np.inf,
        a_search_ub: float = np.inf,
        a_initial: float = 0.0,
    ) -> SolveResults:
        """
        Solve the dual for the cost-minimizing contract at a given intended action a0.

        Args:
            intended_action: The intended action a0
            reservation_utility: The reservation utility Ubar
            solver: Either "a_hat" (default) or "iterative"
            a_hat: Required when solver="a_hat". The action grid for the solve.
            n_a_iterations: Number of iterations for iterative solver. Defaults to 1.
            theta_init: Optional initial theta for warm-starting.
            clip_ratio: Maximum absolute value for ratio clipping in cache construction. Defaults to 1e6.
            a_search_lb: Lower bound for action search when using iterative solver (default: -infinity)
            a_search_ub: Upper bound for action search when using iterative solver (default: infinity)
            a_initial: Initial action value to start search from when using iterative solver (default: 0.0)

        Returns:
            SolveResults object.
        """
        if solver not in ["a_hat", "iterative"]:
            raise ValueError(f"solver must be 'a_hat' or 'iterative', got '{solver}'")

        if solver == "a_hat":
            if a_hat is None:
                raise ValueError("a_hat is required when solver='a_hat'")
            
            a_hat_arr = np.asarray(a_hat, dtype=np.float64)
            if a_hat_arr.ndim != 1:
                raise ValueError(f"a_hat must be a 1D array; got shape {a_hat_arr.shape}")

            results, theta_opt = _minimize_cost_a_hat(
                float(intended_action),
                float(reservation_utility),
                a_hat_arr,
                y_grid=self._y_grid,
                w=self._w,
                f=self._primitives["f"],
                score=self._primitives["score"],
                C=self._primitives["C"],
                Cprime=self._primitives["Cprime"],
                g=self._primitives["g"],
                k=self._primitives["k"],
                theta_init=theta_init,
                clip_ratio=clip_ratio,
            )
        else:  # solver == "iterative"
            results, theta_opt = _minimize_cost_iterative(
                a0=float(intended_action),
                Ubar=float(reservation_utility),
                n_a_iterations=int(n_a_iterations),
                y_grid=self._y_grid,
                w=self._w,
                f=self._primitives["f"],
                score=self._primitives["score"],
                C=self._primitives["C"],
                Cprime=self._primitives["Cprime"],
                g=self._primitives["g"],
                k=self._primitives["k"],
                theta_init=theta_init,
                clip_ratio=clip_ratio,
                a_search_lb=a_search_lb,
                a_search_ub=a_search_ub,
                a_initial=a_initial,
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
        a_search_lb: float = -np.inf,
        a_search_ub: float = np.inf,
        a_initial: float = 0.0,
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
            a_search_lb: Lower bound for action search when using iterative solver (default: -infinity)
            a_search_ub: Upper bound for action search when using iterative solver (default: infinity)
            a_initial: Initial action value to start search from when using iterative solver (default: 0.0)

        Returns:
            Callable function F(a) that returns the expected wage for action a.
        """
        if solver not in ["a_hat", "iterative"]:
            raise ValueError(f"solver must be 'a_hat' or 'iterative', got '{solver}'")

        F = _make_expected_wage_fun(
            y_grid=self._y_grid,
            w=self._w,
            f=self._primitives["f"],
            score=self._primitives["score"],
            C=self._primitives["C"],
            Cprime=self._primitives["Cprime"],
            g=self._primitives["g"],
            k=self._primitives["k"],
            Ubar=float(reservation_utility),
            solver=solver,
            a_hat=a_hat,
            n_a_iterations=int(n_a_iterations),
            warm_start=bool(warm_start),
            clip_ratio=clip_ratio,
            a_search_lb=a_search_lb,
            a_search_ub=a_search_ub,
            a_initial=a_initial,
        )

        return F

    def U(self, v: np.ndarray, a: float | np.ndarray) -> np.ndarray:
        """
        U(a) = ∫ v(y) f(y|a) dy - C(a), evaluated on the internal Simpson grid.

        Inputs:
          - v : must have shape equal to self.y_grid.shape
          - a : scalar or 1D array

        Returns:
          - scalar if a is scalar; 1D array otherwise
        """
        return _compute_expected_utility(
            v=v,
            a=a,
            y_grid=self._y_grid,
            w=self._w,
            f=self._primitives["f"],
            C=self._primitives["C"],
        )
