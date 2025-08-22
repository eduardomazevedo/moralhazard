# Internal Development Notes — Moral Hazard (v0)

This document enumerates internals (names, signatures, shapes, invariants) to guide implementation. It mirrors the public spec but adds lower-level details. No code here—just contracts and responsibilities.

---

# Core Data & Grid Policy

* **Dtype:** `np.float64` everywhere unless explicitly documented.
* **Grid (fixed in v0):** `y_grid = np.linspace(y_min, y_max, n)`
* **Weights:** Simpson weights `w` on `y_grid`, shape `(201,)`, same dtype as `y_grid`.
* **Vectorization convention:** All functions that accept `y` must be vectorized over `y` and accept `np.ndarray` of shape `(201,)`. Functions that accept `a` must accept scalar floats; vectorization over `a` only where specified.

---

# Config & Primitives (inputs)

From `cfg["problem_params"]`:

* `u(d: float | np.ndarray) -> float | np.ndarray`
* `k(u: float | np.ndarray) -> float | np.ndarray`  (inverse wage map; used for expected wage & plotting)
* `link_function(z: float | np.ndarray) -> float | np.ndarray`  (denoted `g`)
* `C(a: float | np.ndarray) -> float | np.ndarray`
* `Cprime(a: float | np.ndarray) -> float | np.ndarray`
* `f(y: np.ndarray, a: float) -> np.ndarray`  (density `f(y|a)`, vectorized in `y`)
* `score(y: np.ndarray, a: float) -> np.ndarray` (∂ₐ log f(y|a), vectorized in `y`)

From `cfg["computational_params"]`:

* `y_min: float` (required)

---

# Internal Objects & Data Flow

## 1) Cache object (dict)

Built per solve; immutable in practice.

**Constructor**

```python
_make_cache(
  y_grid: np.ndarray,                  # (201,)
  w: np.ndarray,                       # (201,)
  a0: float,                           # intended action
  Ubar: float,                         # reservation utility
  a_hat: np.ndarray,                   # (m,)
  primitives: dict                     # references to g, k, C, Cprime, f, score
) -> dict
```

**Contents & shapes**

* Scalars: `a0: float`, `Ubar: float`
* Arrays on `(201,)`:

  * `f0 = f(y_grid, a0)` (baseline density)
  * `s0 = score(y_grid, a0)`
  * `wf0 = w * f0`
  * `wf0s0 = wf0 * s0`
* Matrices on `(201, m)`:

  * `D = f(y_grid[:, None], a_hat[None, :])`
  * `R = 1 - D / f0[:, None]` (requires `f0 > 0`; see Numerical Safeguards)
* Grid & weights:

  * `y_grid`, `w`
* Function refs:

  * `g`, `k`, `C`, `Cprime`
* Metadata: `a_hat` (1D, `(m,)`)

**Invariants**

* `f0[i] > 0` for all `i` (or epsilon policy applied consistently—see safeguards).
* `a_hat.ndim == 1`.
* `y_grid.shape == w.shape == (201,)`.

---

## 2) Canonical contract map

**Signature**

```python
_canonical_contract(theta: np.ndarray, cache: dict) -> dict
# θ layout: [lam, mu, mu_hat[0], ..., mu_hat[m-1]]
```

**Operations**

* Unpack `θ → (lam: float, mu: float, mu_hat: np.ndarray (m,))`.
* Compute `z = lam + mu * s0 + R @ mu_hat`  → `(201,)`.
* Compute `v = g(z)` → `(201,)`.

**Returns**

```python
{"z": np.ndarray (201,), "v": np.ndarray (201,)}
```

**Invariants**

* Broadcasts must preserve `(201,)`.

---

## 3) Constraint evaluation

**Signature**

```python
_constraints(v: np.ndarray, cache: dict) -> dict
```

**Definitions (all scalars unless noted)**

* `U0 = wf0 @ v - C(a0)`
* `FOC = wf0s0 @ v - Cprime(a0)`
* `Uhat = (w[:, None] * D).T @ v - C(a_hat)`  → `(m,)`
* `IC = Uhat - U0`                             → `(m,)`
* `IR = Ubar - U0`
* `Ewage = wf0 @ k(v)`

**Returns**

```python
{
  "U0": float, "IR": float, "FOC": float,
  "Uhat": np.ndarray (m,), "IC": np.ndarray (m,),
  "Ewage": float
}
```

---

## 4) Dual objective & gradient

**Signature**

```python
_dual_value_and_grad(theta: np.ndarray, cache: dict) -> tuple[float, np.ndarray]
```

**Procedure**

1. `v = _canonical_contract(theta, cache)["v"]`.
2. `cons = _constraints(v, cache)`.
3. Dual value:

   * `g_dual(θ) = Ewage + lam * IR - mu * FOC + mu_hat @ IC`.
4. Gradient of `g_dual` via Danskin:

   * `∇g_dual = [ IR, -FOC, IC[:] ]`.
5. Return objective for minimizer:

   * `obj = -g_dual(θ)`
   * `grad = -∇g_dual` (same layout as `θ`)

**Shapes**

* `theta.shape == (2 + m,)`
* `grad.shape == (2 + m,)`

---

## 5) Optimizer bridge

**Signature**

```python
_run_solver(
  theta_init: np.ndarray | None,       # if None, see warm-start policy
  cache: dict
) -> tuple[np.ndarray, dict]           # (theta_opt, solver_state)
```

**Responsibilities**

* Determine initial `θ`:

  * Prefer passed `theta_init`.
  * Else, use class-level warm-start (`_last_theta`) if available & shape matches `(2 + m,)`.
  * Else, default init (spec: lam0=100.0, mu0=100.0, mu\_hat0=zeros(m)).
* Bounds policy (v0): fixed internally as

  * `lam ∈ [0, +∞)`, `mu ∈ (-∞, +∞)`, `mu_hat[j] ∈ [0, +∞)`.
* Call SciPy L-BFGS-B (or compatible) with function returning `(obj, grad)`.
* Collect and return:

  ```python
  solver_state = {
    "method": "L-BFGS-B",
    "status": int,
    "message": str,
    "niter": int,
    "nfev": int,        # if available
    "njev": int,        # if available
    "time_sec": float,
    "fun": float,       # final obj (i.e., -g_dual)
    "grad_norm": float  # e.g., ℓ∞ norm of grad
  }
  ```

**Invariants**

* Returned `theta_opt.shape == (2 + m,)`.
* Do **not** mutate `cache`.

---

## 6) Solve wrapper (public method uses these)

**Signature**

```python
_solve_fixed_a(
  a0: float,
  Ubar: float,
  a_hat: np.ndarray,
  theta_init: np.ndarray | None
) -> tuple[SolveResults, dict]         # (results, cache)
```

**Behavior**

* Build `cache = _make_cache(...)`.
* `(theta_opt, state) = _run_solver(theta_init, cache)`.
* `v = _canonical_contract(theta_opt, cache)["v"]`.
* `cons = _constraints(v, cache)`.
* Package `SolveResults`:

  * `optimal_contract = v`
  * `expected_wage = cons["Ewage"]`
  * `multipliers = {"lam": theta_opt[0], "mu": theta_opt[1], "mu_hat": theta_opt[2:]}`
  * `constraints = cons`
  * `solver_state = state`
* Update class warm-start: `_last_theta = theta_opt` (documented policy).

---

## 7) Expected-wage function factory

**Signature**

```python
_make_expected_wage_fun(
  Ubar: float,
  a_hat: np.ndarray,
  warm_start: bool
) -> Callable[[float], float] with attributes:
  - .last_theta: np.ndarray | None
  - .call_count: int
  - .reset(): None
```

**Behavior**

* Closure captures `warm_start` and optional `theta_cache`:

  * On call `F(a)`:

    * Use `.last_theta` as `theta_init` if `warm_start` and shape matches current `(2 + m,)`.
    * Call `_solve_fixed_a(a, Ubar, a_hat, theta_init)` but **only** return Ewage (not `SolveResults`).
    * If `warm_start`, update `.last_theta` with optimal `θ`.
    * Increment `.call_count`.
* `.reset()` clears `.last_theta` and resets `.call_count = 0`.

**Constraints**

* Pure side-effects limited to attributes when `warm_start=True`.

---

# Public Class Glue

The public `MoralHazardProblem` uses the above internals:

* `__init__`:

  * Validates config; stores primitive callables; constructs `y_grid`, `w`.
  * Initializes `_last_theta: np.ndarray | None = None`.

* `solve_cost_minimization_problem(...)` → calls `_solve_fixed_a(...)` and returns `SolveResults`.

* `expected_wage_fun(...)` → returns closure from `_make_expected_wage_fun(...)`.

* `U(v, a)`:

  * If `a` is scalar: compute `w @ (v * f(y_grid, a)) - C(a)`.
  * If array-like: loop or vectorize over `a` to return same-shape array.
  * Validate `v.shape == (201,)`.

---

# Validation & Errors

* **Config presence & callability**

  * Missing/invalid primitives → `KeyError` with the exact primitive name.
* **`y_min` present** (float) → else `KeyError`.
* **Grid invariants** enforced at construction:

  * `y_grid.shape == (201,)`, `w.shape == (201,)`.
* **`a_hat` validation** in all relevant methods:

  * `np.asarray(a_hat, dtype=float)`; must be 1D, else `ValueError`.
* **Density positivity for `R`**:

  * If `np.any(f0 <= 0)` → raise `RuntimeError` with advice to adjust tails (`y_min`) or primitives.
* **Contract shape** in `U`:

  * If `v.shape != (201,)` → `ValueError`.

---

# Numerical Safeguards (v0)

* **No epsilon by default**: if `f0[i] <= 0` at any grid point, **raise** (fail fast).
* **No grid auto-expansion**: library does not mutate `y_min` or `n`; user adjusts config.
* **Bounds**: hard-coded as above; ensure L-BFGS-B inputs finite (replace `+∞` with `None`).

---

# Performance Notes

* **Warm-start** is the main speed lever across `a`.
* **Allocation discipline**: prefer reusing arrays inside `_make_cache` only if it doesn’t leak state; otherwise keep it simple and allocate (v0).
* **Avoid Python loops** on `(201,)` dimension; rely on vectorized ops and BLAS-level matvecs.

---

# Testing Checklist

1. **Shapes**

   * `v`, `z` `(201,)`; `D`, `R` `(201, m)`; `mu_hat` `(m,)`.
2. **Simple normal model smoke test**

   * Constant functions or trivial costs to ensure gradients wired correctly.
3. **Constraint gradients**

   * Finite-difference check: compare `_dual_value_and_grad` vs. numerical grad at random θ (small `m`, small grid subset for local test).
4. **Warm-start correctness**

   * Ensure `.last_theta` updates only when `warm_start=True`.
5. **Error paths**

   * Zero density on grid; wrong `a_hat` shape; wrong `v` shape in `U`.

---

# `SolveResults.solver_state` Keys (v0)

* `"method": "L-BFGS-B"`
* `"status": int` (scipy-style)
* `"message": str`
* `"niter": int`
* `"nfev": int | None`
* `"njev": int | None`
* `"time_sec": float`
* `"fun": float`      # final minimized value (i.e., -g\_dual)
* `"grad_norm": float`  # e.g., max(abs(grad))
* Optional extras: `"warn_flags": list[str]` (e.g., "density\_zero\_guard")

---

# File Layout & Visibility

```
moral_hazard/
  __init__.py            # exports MoralHazardProblem
  types.py               # public: SolveResults
  grids.py               # _make_grid(y_min) -> (y_grid, w)
  core.py                # _make_cache, _canonical_contract, _constraints
  solver.py              # _dual_value_and_grad, _run_solver, _solve_fixed_a, _make_expected_wage_fun
```

* Leading underscore on internal functions; only `MoralHazardProblem` and `SolveResults` are public in v0.

---

# Future Hooks (reserved, not used in v0)

* Allow `n`, `step` in computational params; auto-adjust to odd `n`.
* Epsilon policy for zero-density guard with logged annotation in `solver_state`.
* Alternative optimizers/backends; pluggable bounds.
* Optional `expected_wage_fun(..., return_results=False|True)` to surface intermediate `θ`/contracts for metalearning.
