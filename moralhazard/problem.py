# moral_hazard/problem.py
from __future__ import annotations

from typing import Dict, Any, Callable
import numpy as np

from .types import SolveResults
from .grids import _make_grid
from .solver import _minimize_cost_a_hat, _make_expected_wage_fun


class MoralHazardProblem:
    """
    Public entry point.

    Construction
    ------------
    mhp = MoralHazardProblem(cfg)

    Validates:
      - cfg is a dict with 'problem_params' and 'computational_params'
      - required callables exist and are callable
      - computational_params must include y_min, y_max, n (odd)

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
        for key in ("y_min", "y_max", "n"):
            if key not in comp:
                raise KeyError(f"computational_params['{key}'] is required")

        y_min = float(comp["y_min"])
        y_max = float(comp["y_max"])
        n = int(comp["n"])

        y_grid, w = _make_grid(y_min, y_max, n)

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
        a_hat: np.ndarray,
        theta_init: np.ndarray | None = None,
    ) -> SolveResults:
        """
        Solve the dual for the cost-minimizing contract at a given intended action a0.

        Returns a SolveResults object.
        """
        a_hat_arr = np.asarray(a_hat, dtype=np.float64)
        if a_hat_arr.ndim != 1:
            raise ValueError(f"a_hat must be a 1D array; got shape {a_hat_arr.shape}")

        results, _cache, theta_opt = _minimize_cost_a_hat(
            float(intended_action),
            float(reservation_utility),
            a_hat_arr,
            y_grid=self._y_grid,
            w=self._w,
            primitives=self._primitives,
            theta_init=theta_init,
        )

        return results

    def expected_wage_fun(
        self,
        reservation_utility: float,
        a_hat: np.ndarray,
        warm_start: bool = True,
    ) -> "Callable[[float], float]":
        """
        Returns F(a) = E[w(v*(a))] where v*(a) is the cost-minimizing contract
        at intended action a for the provided Ū and a_hat.

        Warm-start policy:
          - when warm_start=True, successive calls reuse the last θ* found
            inside the returned function (does NOT mutate class-level warm start).
        """
        a_hat_arr = np.asarray(a_hat, dtype=np.float64)
        if a_hat_arr.ndim != 1:
            raise ValueError(f"a_hat must be a 1D array; got shape {a_hat_arr.shape}")

        F = _make_expected_wage_fun(
            y_grid=self._y_grid,
            w=self._w,
            primitives=self._primitives,
            Ubar=float(reservation_utility),
            a_hat=a_hat_arr,
            warm_start=bool(warm_start),
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
        # Check input types but don't convert
        if not isinstance(v, np.ndarray):
            raise TypeError(f"v must be a numpy array; got {type(v)}")
        if v.shape != self._y_grid.shape:
            raise ValueError(f"v must have shape {self._y_grid.shape}; got {v.shape}")
        
        if not isinstance(a, (float, int, np.ndarray)):
            raise TypeError(f"a must be scalar or numpy array; got {type(a)}")

        f = self._primitives["f"]
        C = self._primitives["C"]

        # Let NumPy broadcasting handle both scalar and array inputs
        if isinstance(a, np.ndarray) and a.ndim != 1:
            raise ValueError(f"a must be 1D array; got shape {a.shape}")
        
        # f(y_grid[:, None], a) works for both scalar and array a due to broadcasting
        f_a = f(self._y_grid[:, None], a)
        integrals = self._w @ (v[:, None] * f_a)  # shape (m,) for array a, scalar for scalar a
        costs = C(a)  # shape (m,) for array a, scalar for scalar a
        return integrals - costs
