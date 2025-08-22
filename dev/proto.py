#%%
import time
import numpy as np
from math import sqrt, pi
from scipy.optimize import minimize
import matplotlib.pyplot as plt  # optional


# ===========
# Parameters
# ===========
# Utility
x0 = 50.0   # 50k baseline consumption
u = lambda dollars: np.log(dollars + x0)
k = lambda utils: np.exp(utils) - x0
link_function_g = lambda x: np.log(np.maximum(x, x0))

# Distribution (normal pdf and related terms without SciPy)
sigma = 10.0
def _norm_pdf(z):
    return (1.0 / sqrt(2.0 * pi)) * np.exp(-0.5 * z * z)

def density(y, a):
    # pdf(y | mean=a, sd=sigma)
    z = (y - a) / sigma
    return _norm_pdf(z) / sigma

# Score s0(y) = (∂_a f / f)|_{a0} = (y-a0)/sigma^2 for Normal(a0, sigma^2)
def score(y, a):
    return (y - a) / (sigma ** 2)

# Cost
first_best_effort = 100
theta = 1.0 / first_best_effort / (first_best_effort + x0)
C       = lambda a: theta * a ** 2 / 2
Cprime  = lambda a: theta * a

# ==============================
# Grid and Simpson's rule setup
# ==============================
y_lower, y_upper, y_n = 0 - 3 * sigma, 120 + 3 * sigma, 201
y_grid = np.linspace(y_lower, y_upper, y_n)
y_grid_step_size = y_grid[1] - y_grid[0]
assert y_grid.shape[0] % 2 == 1  # Need odd number for Simpson's rule

# Simpson weights (vector form)
w = np.zeros_like(y_grid)
w[0] = 1.0
w[-1] = 1.0
w[1:-1:2] = 4.0
w[2:-2:2] = 2.0
w = w * (y_grid_step_size / 3.0)

# =================================
# Fixed a-hats (count only)
# =================================
n_a_hat = 2  # adjust as needed

# ======================================
# Cache builder in canonical (math) form
# ======================================
def make_cache(
    a0: float,
    reservation_utility: float,
    a_hat_vals: np.ndarray,
    *,
    link_g = link_function_g,
    k_fn   = k,
    C_fn   = C,
    Cprime_fn = Cprime
):
    """
    Build cached arrays per the vectorized canonical formulation.

    Returns keys:
      w, f0, s0, D, R, wf0, wf0s0, g, k, C, Cprime, Ubar, a0, a_hat
    """
    a0 = np.asarray(a0, dtype=y_grid.dtype)
    a_hat_vals = np.asarray(a_hat_vals, dtype=y_grid.dtype)
    assert a_hat_vals.ndim == 1 and a_hat_vals.shape[0] == n_a_hat, \
        f"Expected a_hat size {n_a_hat}, got {a_hat_vals.shape}"

    # Baseline density and score at a0
    f0 = density(y_grid, a0)            # (y_n,)
    s0 = score(y_grid, a0)              # (y_n,)

    # Density matrix for fixed a-hats
    D  = density(y_grid[:, None], a_hat_vals[None, :])  # (y_n, m)

    # Cached weights/products
    wf0   = w * f0                    # (y_n,)
    wf0s0 = wf0 * s0                  # (y_n,)
    # R = 1 - D / f0[:,None]
    # add tiny eps to avoid division issues if desired; f0>0 here for normal on R.
    R = 1.0 - D / f0[:, None]

    return {
        "w": w, "f0": f0, "s0": s0, "D": D, "R": R,
        "wf0": wf0, "wf0s0": wf0s0,
        "g": link_g, "k": k_fn, "C": C_fn, "Cprime": Cprime_fn,
        "Ubar": np.asarray(reservation_utility, dtype=y_grid.dtype),
        "a0": a0, "a_hat": a_hat_vals
    }

# ---------- Canonical contract (v = g(z)) ----------
def canonical_contract(multipliers, cache):
    """
    multipliers: tuple (lambda, mu, mu_hat) with mu_hat shape (m,)
    """
    lam, mu, mu_hat = multipliers
    # z = lambda + mu * s0 + R @ mu_hat
    z = lam + mu * cache["s0"] + cache["R"] @ mu_hat
    v = cache["g"](z)
    return {"z": z, "v": v}

# ---------- Constraints and expected wage ----------
def constraints(v, cache):
    """
    Returns:
      U0, IR, FOC, Uhat, IC, Ewage
    """
    wf0, wf0s0 = cache["wf0"], cache["wf0s0"]
    w_vec, D = cache["w"], cache["D"]
    C_fn, Cprime_fn = cache["C"], cache["Cprime"]
    a0, Ubar = cache["a0"], cache["Ubar"]
    k_fn = cache["k"]

    # U0 = ∫ v f0 - C(a0)
    U0  = wf0 @ v - C_fn(a0)

    # FOC = ∫ v s0 f0 - C'(a0)
    FOC = wf0s0 @ v - Cprime_fn(a0)

    # Uhat = [ (w * D)^T @ v ] - C(a_hat)
    # (w[:,None] * D) has shape (y_n, m), v has shape (y_n,)
    Uhat = (w_vec[:, None] * D).T @ v - C_fn(cache["a_hat"])

    # IC = Uhat - U0 (elementwise)
    IC = Uhat - U0

    # IR = Ubar - U0
    IR = Ubar - U0

    # Expected wage = ∫ k(v) f0
    Ewage = wf0 @ k_fn(v)

    return {"U0": U0, "IR": IR, "FOC": FOC,
            "Uhat": Uhat, "IC": IC, "Ewage": Ewage}

# ---------- Dual objective + gradient ----------
def objective_with_grad(multipliers, constraints_dict, cache):
    """
    g_dual(θ) = E[w] + λ·IR - μ·FOC + μ̂^T IC
    ∇ g_dual(θ) = (IR, -FOC, IC)
    """
    lam, mu, mu_hat = multipliers
    IR  = constraints_dict["IR"]
    FOC = constraints_dict["FOC"]
    IC  = constraints_dict["IC"]
    Ewage = constraints_dict["Ewage"]

    g_dual = Ewage + lam * IR - mu * FOC + mu_hat @ IC
    grad = (IR, -FOC, IC)
    return g_dual, grad

# ================
# Optimizer Bridge
# ================
# θ layout: [ λ, μ, μ_hat[0], ..., μ_hat[m-1] ]
def unpack_theta(theta_vec):
    lam = theta_vec[0]
    mu  = theta_vec[1]
    mu_hat = theta_vec[2:]
    return lam, mu, mu_hat

def pack_theta(lam, mu, mu_hat):
    return np.concatenate([np.array([lam, mu], dtype=float), np.atleast_1d(mu_hat).astype(float)])

def dual_value_and_grad(theta_vec, cache):
    """
    For use with minimizers: returns (obj, grad) for obj = -g_dual(θ).
    By Danskin, gradient of g_dual w.r.t. multipliers is constraint residuals at v*(θ).
    """
    lam, mu, mu_hat = unpack_theta(theta_vec)

    # Inner canonical v*(θ)
    v = canonical_contract((lam, mu, mu_hat), cache)["v"]

    # Constraints at v*
    cons = constraints(v, cache)

    # Dual value and gradient
    g_dual, grad_tuple = objective_with_grad((lam, mu, mu_hat), cons, cache)

    # We minimize obj = -g_dual
    obj = -g_dual
    IR, negFOC, IC = grad_tuple
    grad = np.concatenate([np.array([IR, negFOC], dtype=float), np.atleast_1d(IC).astype(float)])
    # grad here is ∇g; optimizer needs ∇obj = -∇g
    grad = -grad
    return obj, grad


def run_solver(theta_init, bounds, cache, maxiter=1000, tol=1e-8):
    # SciPy wants bounds as sequence of (low, high)
    lo, hi = bounds
    scipy_bounds = tuple((float(l), float(h)) for (l, h) in zip(lo, hi))
    def fun_and_jac(theta, cache):
        obj, grad = dual_value_and_grad(theta, cache)
        return float(obj), grad

    res = minimize(
        fun=lambda theta, cache: fun_and_jac(theta, cache),
        x0=np.asarray(theta_init, dtype=float),
        jac=True,
        args=(cache,),
        method="L-BFGS-B",
        bounds=scipy_bounds,
        options={"maxiter": maxiter, "ftol": tol}
    )
    return res.x, res  # mirror (params_opt, state)

# ===========================
# Initial values and bounds
# ===========================
lam0 = 100.0
mu0  = 100.0
mu_hats0 = np.zeros(n_a_hat)               # shape (n_a_hat,)
init = pack_theta(lam0, mu0, mu_hats0)

# Bounds: lam >= 0; mu free; mu_hats >= 0
lower_bounds = np.concatenate([
    np.array([0.0, -np.inf]),         # lam, mu
    np.full((n_a_hat,), 0.0)          # mu_hats
])
upper_bounds = np.concatenate([
    np.array([ np.inf,  np.inf]),     # lam, mu
    np.full((n_a_hat,),  np.inf)      # mu_hats
])
bounds = (lower_bounds, upper_bounds)

# ===========================
# Experiment configuration
# ===========================
BASE_INTENDED_ACTION = 80.0
NOISE_SD = 5.0            # std dev of the noise you add to 80
N_RUNS = 10
SEED = 123

rng = np.random.default_rng(SEED)
noisy_intended_actions = BASE_INTENDED_ACTION + rng.normal(loc=0.0, scale=NOISE_SD, size=N_RUNS)

# You can vary these two per run (values only; counts stay fixed):
reservation_wages = np.linspace(0.0, 80.0, N_RUNS)  # example
a_hat_experiments = np.zeros(n_a_hat)               # shape (n_a_hat, ), same values for all runs

# ===========================
# Warm-up "compile" (no-op here, but keep parity)
# ===========================
warm_res_utility = u(1.0)
warm_cache = make_cache(BASE_INTENDED_ACTION, warm_res_utility, a_hat_experiments)
_ = dual_value_and_grad(init, warm_cache)

# ===========================
# Timing loop across caches
# ===========================
times = []
solutions = []
actions_used = []

for i, (a0, rw) in enumerate(zip(noisy_intended_actions, reservation_wages), start=1):
    cache = make_cache(
        float(a0),
        reservation_utility=float(u(rw)),         # vary reservation utility here
        a_hat_vals=a_hat_experiments,             # vary fixed hats VALUES here (same count)
    )

    t0 = time.time()
    params_opt, state = run_solver(init, bounds, cache, maxiter=1000, tol=1e-8)
    t1 = time.time()

    dt = t1 - t0
    times.append(dt)
    solutions.append(params_opt)
    actions_used.append(a0)

    print(f"Run {i:02d} | intended_action = {a0:6.2f} | U0 from rw={rw:.2f} | time = {dt:.4f} s")

avg_time = sum(times) / len(times)
print(f"\nAverage solve time over {N_RUNS} runs: {avg_time:.4f} s")

# Unpack last solution for convenience
theta_star = solutions[-1]
def unpack_params(theta_vec):
    lam = theta_vec[0]
    mu  = theta_vec[1]
    mu_hats = theta_vec[2 : 2 + n_a_hat]
    return lam, mu, mu_hats

lam_star, mu_star, mu_hats_star = unpack_params(theta_star)


#%%
# ===========================
# Helpers for the new API
# ===========================
def expected_wage_from_cache(v, cache):
    """
    E[w] under the baseline density f0 (i.e., at the cache's a0).
    Uses the cached wf0 and k() consistent with the new make_cache().
    """
    return float(cache["wf0"] @ cache["k"](v))

def U_of_a(v, a, cache):
    """
    U(a) = ∫ v(y) f(y|a) dy - C(a), evaluated on the global y_grid
    using Simpson weights in cache["w"] and density() defined above.
    """
    f_a = density(y_grid, a)
    return float(cache["w"] @ (v * f_a) - cache["C"](a))

#%%
# ===========================
# New experiment: line search over intended action a
# Objective: maximize a - expected_wage(v*(a))
# ===========================
LINESEARCH_GRID_N = 121  # simple, robust grid (objective may be discontinuous)
a_grid = np.linspace(0.0, 130.0, LINESEARCH_GRID_N)

# Use the last run's reservation utility and fixed hats for the search
res_util = float(u(50.0) - 10)  # example reservation utility

best = {"a": None, "gap": -np.inf, "ew": None, "theta": None, "cache": None}
theta_init = theta_star  # warm start from last solve
obj_values = []

t0 = time.time()
n_solver_runs = 0
for a0 in a_grid:
    # build cache with the new function name/signature
    c = make_cache(float(a0), res_util, a_hat_experiments)
    theta_opt, state = run_solver(theta_init, bounds, c, maxiter=1000, tol=1e-8)
    n_solver_runs += 1

    # unpack and compute v*(θ) with the new canonical_contract API
    lam, mu, mu_hats = unpack_params(theta_opt)
    v_star = canonical_contract((lam, mu, mu_hats), c)["v"]

    # expected wage under f0 in cache 'c'
    ew = expected_wage_from_cache(v_star, c)

    gap = float(a0 - ew)
    obj_values.append(gap)

    if gap > best["gap"]:
        best.update({"a": float(a0), "gap": gap, "ew": ew, "theta": theta_opt, "cache": c})

    theta_init = theta_opt  # carry warm start along the grid
t1 = time.time()

print(
    f"\n[Line search] Best a in [{a_grid.min():.1f}, {a_grid.max():.1f}] is {best['a']:.4f} "
    f"with (a - E[w]) = {best['gap']:.6f} and E[w] = {best['ew']:.6f} "
    f"| time = {t1 - t0:.4f} s | solver runs = {n_solver_runs}"
)

# Expose best solution artifacts
a_star_linesearch = best["a"]
theta_star_linesearch = best["theta"]
cache_linesearch = best["cache"]

# Plot objective over grid
plt.figure(figsize=(6,4))
plt.plot(a_grid, obj_values, marker="o", markersize=3, linewidth=1)
plt.axvline(best["a"], linestyle="--", label=f"best a = {best['a']:.2f}")
plt.xlabel("Intended action a")
plt.ylabel("Objective = a - E[w]")
plt.title("Line search objective over a")
plt.legend()
plt.tight_layout()
plt.show()


#%%
# ===========================
# NEW experiment (right after line search):
# 1-D optimizer over the same interval to maximize a - E[w(v*(a))]
# ===========================
from scipy.optimize import minimize_scalar

# We'll use the same bounds as the line search grid
a_lo, a_hi = float(a_grid.min()), float(a_grid.max())

def _solve_and_eval(a0, theta_init, res_util, a_hat_vals):
    """
    For a given intended action a0, build cache, solve dual for θ,
    recover v*(θ), and return expected wage and objective gap a0 - E[w].
    """
    c = make_cache(float(a0), res_util, a_hat_vals)
    theta_opt, _state = run_solver(theta_init, bounds, c, maxiter=1000, tol=1e-8)
    lam, mu, mu_hats = unpack_params(theta_opt)
    v_star = canonical_contract((lam, mu, mu_hats), c)["v"]
    ew = expected_wage_from_cache(v_star, c)
    gap = float(a0 - ew)
    return theta_opt, c, ew, gap

# Keep a warm-start across objective evaluations
theta_last = theta_star  # start from the same warm start used for the line search
n_solver_runs_opt = 0
best_opt = {"a": None, "gap": -np.inf, "ew": None, "theta": None, "cache": None}

def neg_objective(a0):
    """Return - (a0 - E[w]) so we can use a minimizer."""
    global theta_last, n_solver_runs_opt, best_opt
    theta_opt, c, ew, gap = _solve_and_eval(a0, theta_last, res_util, a_hat_experiments)
    n_solver_runs_opt += 1
    # warm start next call
    theta_last = theta_opt
    # track best
    if gap > best_opt["gap"]:
        best_opt.update({"a": float(a0), "gap": gap, "ew": ew, "theta": theta_opt, "cache": c})
    return -gap

t0_opt = time.time()
res_1d = minimize_scalar(neg_objective, bounds=(a_lo, a_hi), method="bounded",
                         options={"xatol": 1e-2, "maxiter": 200})
t1_opt = time.time()

# Extract best found by the optimizer (we tracked it inside neg_objective)
a_star_opt = best_opt["a"]
gap_star_opt = best_opt["gap"]
ew_star_opt = best_opt["ew"]
time_opt = t1_opt - t0_opt

# Comparison table
print("\nComparison: Exhaustive grid search vs. 1-D optimizer on [a_min, a_max]")
hdr = f"{'Method':<22} {'a*':>10} {'Objective a - E[w]':>20} {'E[w]':>12} {'Time (s)':>12} {'Solver runs':>14}"
print(hdr)
print("-" * len(hdr))
row_grid = f"{'Grid (line search)':<22} {best['a']:>10.4f} {best['gap']:>20.6f} {best['ew']:>12.6f} {(t1 - t0):>12.4f} {n_solver_runs:>14d}"
row_opt  = f"{'1-D optimizer':<22} {a_star_opt:>10.4f} {gap_star_opt:>20.6f} {ew_star_opt:>12.6f} {time_opt:>12.4f} {n_solver_runs_opt:>14d}"
print(row_grid)
print(row_opt)


#%%
# ===========================
# Plot U(a) versus action for the best contract at best a
# ===========================
# Unpack best contract
lam_b, mu_b, mu_hats_b = unpack_params(theta_star_linesearch)
v_star_best = canonical_contract((lam_b, mu_b, mu_hats_b), cache_linesearch)["v"]

# Print lam, mu at optimal solution
print(f"Optimal solution: lam = {lam_b:.6f}, mu = {mu_b:.6f}")

# Print how many out of how many mu_hats_b are zero, and list a_hat entries where not zero
zero_mask = np.isclose(np.array(mu_hats_b), 0.0)
num_zero = np.sum(zero_mask)
num_total = len(mu_hats_b)
print(f"mu_hats_b: {num_zero} out of {num_total} are zero.")

if num_zero < num_total:
    nonzero_indices = np.where(~zero_mask)[0]
    a_hat_vals = np.array(cache_linesearch["a_hat"])[nonzero_indices]
    print("Nonzero mu_hats_b at a_hat entries:")
    for idx, val, mu_val in zip(nonzero_indices, a_hat_vals, np.array(mu_hats_b)[nonzero_indices]):
        print(f"  index {idx}: a_hat = {val:.4f}, mu_hat = {mu_val:.6f}")

# Evaluate utility over a grid of actions
a_eval_grid = np.linspace(0.0, 140.0, 200)
U_values = [U_of_a(v_star_best, float(a), cache_linesearch) for a in a_eval_grid]

plt.figure(figsize=(6,4))
plt.plot(a_eval_grid, U_values, linewidth=1.5)
plt.axvline(a_star_linesearch, linestyle="--", label=f"best intended a = {a_star_linesearch:.2f}")
plt.xlabel("Action a")
plt.ylabel("U(a)")
plt.title("Utility U(a) under best contract at best intended action")
plt.legend()
plt.tight_layout()
plt.show()

#%%
# ===========================
# Plot wage function k(v) as a function of y_grid for the optimal contract
# ===========================
wage_schedule = cache_linesearch["k"](v_star_best)  # k(v(y)) in dollars

plt.figure(figsize=(6,4))
plt.plot(np.asarray(y_grid), np.asarray(wage_schedule), linewidth=1.5)
plt.xlabel("Outcome y")
plt.ylabel("Wage k(v(y))")
plt.title("Wage schedule under optimal contract")
plt.tight_layout()
plt.show()

#%%
# ===========================
# Interactive experiment:
# Pick a_hat from the argmax of U(a) on the plotted grid
# ===========================
U_vec = np.array([U_of_a(v_star_best, float(a), cache_linesearch) for a in a_eval_grid])
idx = int(np.argmax(U_vec))
a_hat_from_U = float(np.asarray(a_eval_grid)[idx])

# Update the fixed hats to [0, a_hat_from_U]
a_hat_experiments_new = np.array([0.0, a_hat_from_U])
print(f"Chosen a_hat from U-argmax: {a_hat_from_U:.4f}")
print(f"New a_hat_experiments = {a_hat_experiments_new}")

# ===========================
# Re-solve at the same intended action using the new hats
# ===========================
cache_new = make_cache(float(a_star_linesearch), res_util, a_hat_experiments_new)
theta_opt_new, state = run_solver(theta_star_linesearch, bounds, cache_new, maxiter=1000, tol=1e-8)
lam_n, mu_n, mu_hats_n = unpack_params(theta_opt_new)
v_star_new = canonical_contract((lam_n, mu_n, mu_hats_n), cache_new)["v"]

# Print multipliers
print(f"\n[New solution] lam = {lam_n:.6f}, mu = {mu_n:.6f}")
print("mu_hats_n:", mu_hats_n)

# Also print which mu_hats are nonzero
zero_mask_new = np.isclose(np.array(mu_hats_n), 0.0)
num_zero_new = np.sum(zero_mask_new)
num_total_new = len(mu_hats_n)
print(f"{num_zero_new} out of {num_total_new} mu_hats are zero.")

if num_zero_new < num_total_new:
    nonzero_indices_new = np.where(~zero_mask_new)[0]
    a_hat_vals_new = np.array(cache_new["a_hat"])[nonzero_indices_new]
    print("Nonzero mu_hats at a_hat entries:")
    for idx, val, mu_val in zip(nonzero_indices_new, a_hat_vals_new, np.array(mu_hats_n)[nonzero_indices_new]):
        print(f"  index {idx}: a_hat = {val:.4f}, mu_hat = {mu_val:.6f}")

# ===========================
# Compare old vs new U(a)
# ===========================
U_values_new = [U_of_a(v_star_new, float(a), cache_new) for a in a_eval_grid]

plt.figure(figsize=(6,4))
plt.plot(a_eval_grid, U_values, linewidth=1.5, label="U(a) with old hats")
plt.plot(a_eval_grid, U_values_new, linewidth=1.5, linestyle="--", label="U(a) with new hats")
plt.axvline(a_hat_from_U, linestyle=":", label=f"a_hat* = {a_hat_from_U:.2f}")
plt.xlabel("Action a")
plt.ylabel("U(a)")
plt.title("Utility U(a): old vs. new a_hat (picked at U-argmax)")
plt.legend()
plt.tight_layout()
plt.show()
