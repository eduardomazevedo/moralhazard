# Moral Hazard — Minimal Interface Spec (v0)

## Config (`cfg`: plain dict)

```python
cfg = {
  "problem_params": {
    # REQUIRED callables
    "u": callable(dollars: float | np.ndarray) -> float | np.ndarray,
    "k": callable(utils: float | np.ndarray) -> float | np.ndarray,          # inverse of u (in $-space)
    "link_function": callable(z: float | np.ndarray) -> float | np.ndarray,  # g(z) feeding into k
    "C": callable(a: float | np.ndarray) -> float | np.ndarray,
    "Cprime": callable(a: float | np.ndarray) -> float | np.ndarray,
    "f": callable(y: np.ndarray, a: float) -> np.ndarray,                    # density f(y|a), vectorized in y
    "score": callable(y: np.ndarray, a: float) -> np.ndarray,                # ∂_a log f(y|a), vectorized in y
  },
  "computational_params": {
    # MINIMAL policy (v0)
    "y_min": float,          # left endpoint of the outcome grid
    "y_max": float,
    "n": int # must be odd
  },
}
```

---

## Public class

### `MoralHazardProblem`

```python
class MoralHazardProblem:
    def __init__(self, cfg: dict) -> None
```

**Behavior**

* Validates required callables in `problem_params`.
* Builds and stores a fixed outcome grid:

  * `self.y_grid`: `np.ndarray` of shape `(n,)`, values `y_min + 0..200`.
  * `self.weights`: Simpson weights computed on `self.y_grid` (odd length required).
* Keeps direct references to `u, k, link_function (g), C, Cprime, f, score`.
* No I/O, no plotting, no RNG.

**Properties**

```python
@property
def y_grid(self) -> np.ndarray: ...
def k(self, v: np.ndarray) -> np.ndarray: ...  # convenience passthrough to problem_params["k"]
```

---

## Core methods

### `solve_cost_minimization_problem`

```python
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

    Args:
      intended_action: a0 (float).
      reservation_utility: Ū (float), already in utility units expected by the primitives.
      a_hat: 1D array of fixed comparison actions (IC constraints). Shape (m,).
      theta_init: optional warm-start vector θ = [λ, μ, μ_hat...].

    Returns:
      SolveResults (immutable container; see below).
    """
```

### `expected_wage_fun`

```python
def expected_wage_fun(
    self,
    *,
    reservation_utility: float,
    a_hat: np.ndarray,
    warm_start: bool = True,
) -> "Callable[[float], float]":
    """
    Returns a callable F so that F(a) = E[w(v*(a))], where v*(a) is the
    cost-minimizing contract at intended action a under the given Ū and a_hat.

    Notes:
      - If warm_start is True, successive calls to F(a) will reuse the last θ*
        as theta_init to accelerate solves across nearby 'a'.
      - The returned function exposes lightweight attributes:
          F.last_theta  -> np.ndarray | None
          F.call_count  -> int
          F.reset()     -> None (clears warm-start & counters)
    """
```

### `U` (agent utility under a contract)

```python
def U(self, v: np.ndarray, a: float | np.ndarray) -> np.ndarray:
    """
    Computes U(a) = ∫ v(y) f(y|a) dy - C(a) on the internal Simpson grid.

    Args:
      v: contract in utility units evaluated on self.y_grid; shape (n,).
      a: scalar or 1D array of actions.

    Returns:
      Array with same shape as 'a', dtype float64.
    """
```

---

## Return types

### `SolveResults`

```python
@dataclass(frozen=True)
class SolveResults:
    optimal_contract: np.ndarray         # v*(y), shape (n,)
    expected_wage: float                 # ∫ k(v*(y)) f(y|a0) dy
    multipliers: dict                    # {"lam": float, "mu": float, "mu_hat": np.ndarray}
    constraints: dict                    # {"U0": float, "IR": float, "FOC": float,
                                         #  "Uhat": np.ndarray, "IC": np.ndarray, "Ewage": float}
    solver_state: dict                   # method, status, niter, grad_norm, wall_time, etc.
```

---

## Errors & validation (messages are part of the spec)

* Config validation

  * Missing callable: `KeyError("problem_params['<name>'] is required and must be callable")`
  * Non-dict cfg: `TypeError("cfg must be a dict with 'problem_params' and 'computational_params'")`
  * Missing y\_min: `KeyError("computational_params['y_min'] is required (float)")`

* Shapes & domains

  * `a_hat` not 1D: `ValueError("a_hat must be a 1D array; got shape {a_hat.shape}")`
  * Contract length mismatch in `U`:
    `ValueError("v must have shape (n,); got {v.shape}")`

* Numerical sanity

  * Density zeros where prohibited for the canonical R term:
    `RuntimeError("Encountered zero/near-zero baseline density on grid; adjust y_min or model tails")`
  * Optimizer failure (non-convergence):
    `RuntimeError("Dual solver did not converge: {state['message']} (iter={state['niter']})")`

> v0 policy: we **do not** mutate the grid to “fix” these; the user moves `y_min` or tweaks primitives. Later we can add auto-expansion.

---

## Minimal usage (matches your flow)

```python
mhp = MoralHazardProblem(cfg)

# 1) Solve at a given intended action
results = mhp.solve_cost_minimization_problem(
    intended_action=70.0,
    reservation_utility=10.0,
    a_hat=np.array([0.0, 0.0]),
)
v = results.optimal_contract
expected_wage = results.expected_wage

plot(mhp.y_grid, v)
plot(mhp.y_grid, mhp.k(v))

# 2) U(a) profile under a fixed contract
a_grid = np.linspace(0, 120, 20)
u_profile = mhp.U(v, a_grid)   # plot U vs a_grid

# 3) User-driven optimal action search
F = mhp.expected_wage_fun(reservation_utility=10.0, a_hat=np.array([0.0, 0.0]))
optimal_a = fminbnd(lambda a: F(a) - a, 0.0, 120.0)

results_optimal_a = mhp.solve_cost_minimization_problem(
    intended_action=optimal_a,
    reservation_utility=10.0,
    a_hat=np.array([0.0, 0.0]),
)
v_optimal_a = results_optimal_a.optimal_contract
```

---

## File layout (v0)

```
moral_hazard/
  __init__.py                # exports MoralHazardProblem
  types.py                   # SolveResults dataclass
  core.py                    # cache builder, canonical map, constraints (private)
  solver.py                  # dual objective & optimizer bridge (private)
  grids.py                   # grid + Simpson weights using y_min, n=n, step=1.0 (private)
```

That’s it—lean and locked. If you want, I can add the precise docstrings as they’d appear in code comments (one-liners), still without implementations.

