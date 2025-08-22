```
moral_hazard/                 # top-level package

├── __init__.py
│   └── exports:
│       - MoralHazardProblem
│       - SolveResults
│
├── types.py                   # dataclasses and typed containers
│   └── SolveResults
│       • optimal_contract: np.ndarray
│       • expected_wage: float
│       • multipliers: dict
│       • constraints: dict
│       • solver_state: dict
│
├── grids.py                   # grid construction
│   └── _make_grid(y_min: float) -> (y_grid, w)
│       • builds y_grid of length n (odd, step=1.0)
│       • builds Simpson weights w
│
├── core.py                    # mathematical core
│   └── _make_cache(a0, Ubar, a_hat, y_grid, w, primitives) -> dict
│       • f0, s0, wf0, wf0s0, D, R, g, k, C, Cprime, Ubar, a0, a_hat
│
│   └── _canonical_contract(theta, cache) -> dict
│       • returns {"z": (n,), "v": (n,)}
│
│   └── _constraints(v, cache) -> dict
│       • returns {"U0", "IR", "FOC", "Uhat", "IC", "Ewage"}
│
├── solver.py                  # dual problem + optimizer bridge
│   └── _dual_value_and_grad(theta, cache) -> (obj, grad)
│       • obj = -g_dual(θ)
│       • grad = -∇g_dual
│
│   └── _run_solver(theta_init, cache) -> (theta_opt, solver_state)
│       • packs bounds
│       • calls SciPy L-BFGS-B
│       • returns optimal θ and solver_state dict
│
│   └── _solve_fixed_a(a0, Ubar, a_hat, theta_init) -> (SolveResults, cache)
│       • wrapper: build cache, run solver, package results
│
│   └── _make_expected_wage_fun(Ubar, a_hat, warm_start) -> Callable
│       • closure: F(a) = E[w(v*(a))]
│       • attributes: .last_theta, .call_count, .reset()
│
├── problem.py                  # public class glue
│   └── class MoralHazardProblem:
│       • __init__(cfg)
│           - validate cfg
│           - build y_grid, weights
│           - store primitives
│           - init self._last_theta
│
│       • property y_grid
│       • def k(self, v) -> np.ndarray
│
│       • def solve_cost_minimization_problem(
│             intended_action, reservation_utility, a_hat,
│             theta_init=None
│         ) -> SolveResults
│
│       • def expected_wage_fun(
│             reservation_utility, a_hat, warm_start=True
│         ) -> Callable[[float], float]
│
│       • def U(self, v, a) -> np.ndarray
│
```