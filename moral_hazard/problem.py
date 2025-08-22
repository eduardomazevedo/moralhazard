# moral_hazard/problem.py
from __future__ import annotations

from typing import Dict, Any, Callable
import numpy as np

from .types import SolveResults
from .grids import _make_grid
from .solver import _solve_fixed_a, _make_expected_wage_fun


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
      - initializes class-level warm start (None)
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
        self._y_grid = y_grid
        self._w = w
        self._last_theta: np.ndarray | None = None  # class-level warm start

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
        *,
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

        # Warm-start preference: passed theta_init; else class-level last_theta
        init = theta_init if theta_init is not None else self._last_theta

        results, _cache, theta_opt = _solve_fixed_a(
            float(intended_action),
            float(reservation_utility),
            a_hat_arr,
            init,
            y_grid=self._y_grid,
            w=self._w,
            primitives=self._primitives,
            last_theta=self._last_theta,
        )

        # Update class warm-start
        self._last_theta = np.asarray(theta_opt, dtype=np.float64)
        return results

    def expected_wage_fun(
        self,
        *,
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
            last_theta_seed=self._last_theta,  # seed with class warm-start (won't be mutated)
        )
        return F

    def U(self, v: np.ndarray, a: float | np.ndarray) -> np.ndarray:
        """
        U(a) = ∫ v(y) f(y|a) dy - C(a), evaluated on the internal Simpson grid.

        Args
        ----
        v : np.ndarray
            Contract in utility units evaluated on self.y_grid; shape (n,).
        a : float | array-like
            Scalar or array of actions.

        Returns
        -------
        np.ndarray with same shape as a, dtype float64.
        """
        v_arr = np.asarray(v, dtype=np.float64)
        expected = self._y_grid.shape
        if v_arr.shape != expected:
            raise ValueError(f"v must have shape {expected}; got {v_arr.shape}")

        f = self._primitives["f"]
        C = self._primitives["C"]

        def _u_of_scalar(a0: float) -> float:
            f_a = np.asarray(f(self._y_grid, float(a0)), dtype=np.float64)
            return float(self._w @ (v_arr * f_a) - C(float(a0)))

        if np.ndim(a) == 0:
            return np.array(_u_of_scalar(float(a)), dtype=np.float64)

        a_vec = np.asarray(a, dtype=np.float64).ravel()
        out = np.array([_u_of_scalar(float(a0)) for a0 in a_vec], dtype=np.float64)
        return out.reshape(np.shape(a))
