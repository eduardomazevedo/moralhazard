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
x0 = 50.0   # meaning 50k per year in consumption
u = lambda dollars: np.log(dollars + x0)
k = lambda utils: np.exp(utils) - x0
k_prime = lambda utils: np.exp(utils)
k_prime_inverse = lambda utils: np.log(utils)
link_function_g = lambda x: np.log(np.maximum(x, x0))

# Distribution (normal pdf and related terms without SciPy)
sigma = 10.0
def _norm_pdf(z):
    return (1.0 / sqrt(2.0 * pi)) * np.exp(-0.5 * z * z)

def density(y, a):
    # pdf(y | mean=a, sd=sigma)
    z = (y - a) / sigma
    return _norm_pdf(z) / sigma

def d_density_d_a(y, a):
    # derivative wrt a
    return ((y - a) / (sigma ** 2)) * density(y, a)

score = lambda y, a: (y - a) / sigma ** 2

# Cost
first_best_effort = 100
theta = 1.0 / first_best_effort / (first_best_effort + x0)
cost_of_effort = lambda a: theta * a ** 2 / 2
marginal_cost_of_effort = lambda a: theta * a

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
n_a_hat = 2

# ======================================
# Cache builder for (a0, U0, fixed hats)
# ======================================
def build_cache(a0: float,
                reservation_utility: float,
                a_hat_vals: np.ndarray):
    """
    Build a small 'cache' of arrays that depend on the intended action a0,
    the reservation utility, and the values of the fixed a-hats.
    """
    a0 = np.asarray(a0, dtype=y_grid.dtype)
    a_hat_vals = np.asarray(a_hat_vals, dtype=y_grid.dtype)
    assert a_hat_vals.shape == (n_a_hat,), \
        f"Expected fixed hats of shape {(n_a_hat,)}, got {a_hat_vals.shape}"

    # Precompute at a0
    density_0 = density(y_grid, a0)                  # (y_n,)
    d_density_d_a_0 = d_density_d_a(y_grid, a0)      # (y_n,)
    score_0 = score(y_grid, a0)                      # (y_n,)

    cost_of_effort_0 = cost_of_effort(a0)
    marginal_cost_of_effort_0 = marginal_cost_of_effort(a0)

    # Weighted constants
    w_density0 = w * density_0
    w_d_density_da0 = w * d_density_d_a_0

    # Precompute density columns for the FIXED a-hats (used in canonical contract)
    dens_y_fixed = density(y_grid[:, None], a_hat_vals[None, :])  # (y_n, n_a_hat)

    return {
        "a0": a0,
        "reservation_utility": np.asarray(reservation_utility, dtype=y_grid.dtype),
        "a_hat": a_hat_vals,           # (n_a_hat,)
        "dens_y_fixed": dens_y_fixed,  # (y_n, n_a_hat)

        "density_0": density_0,                    # (y_n,)
        "d_density_d_a_0": d_density_d_a_0,        # (y_n,)
        "score_0": score_0,                        # (y_n,)
        "w_density0": w_density0,                  # (y_n,)
        "w_d_density_da0": w_d_density_da0,        # (y_n,)
        "cost_of_effort_0": cost_of_effort_0,      # scalar
        "marginal_cost_of_effort_0": marginal_cost_of_effort_0,  # scalar
    }

# =========================================
# Core functions parameterized by `cache`
# =========================================
def U(contract_vec, a):
    # a: NumPy array or scalar
    # contract_vec: shape (y_n,)
    a_arr = np.atleast_1d(a)  # shape (1,) or (n,)
    dens = density(y_grid[:, None], a_arr[None, :])  # (y_n, n_a)
    util = np.sum(w[:, None] * contract_vec[:, None] * dens, axis=0) - cost_of_effort(a_arr)
    return util.squeeze()

def U_0(contract_vec, cache):
    return np.vdot(cache["w_density0"], contract_vec) - cache["cost_of_effort_0"]

def expected_wage(contract_vec, cache):
    return np.vdot(cache["w_density0"], k(contract_vec))

def d_U_d_a(contract_vec, cache):
    return np.vdot(cache["w_d_density_da0"], contract_vec) - cache["marginal_cost_of_effort_0"]

# =========================================
# Canonical inner minimizer v*(λ, μ, μ_hat)
# =========================================
def canonical_contract_vec(lam, mu, mu_hats, cache, eps=1e-12):
    """
    Build v*(y) from multipliers using ONLY fixed a-hats.
    """
    dens_y_fixed = cache["dens_y_fixed"]                        # (y_n, n_a_hat)
    ratio = dens_y_fixed / np.maximum(cache["density_0"][:, None], eps)  # (y_n, n_a_hat)
    hat_term = (1.0 - ratio) @ mu_hats                          # (y_n,)
    z = lam + mu * cache["score_0"] + hat_term
    return link_function_g(z)

# =========================================
# Constraint maps (residual forms)
# =========================================
def c_ir(v, cache):
    # IR: U0 >= Ubar  <=>  Ubar - U0 <= 0
    return cache["reservation_utility"] - U_0(v, cache)

def c_ic(v, cache):
    # IC at sampled fixed points: U(a_hat_i) - U0 <= 0  (elementwise)
    U0 = U_0(v, cache)
    a_hats = cache["a_hat"]  # (n_a_hat,)
    U_vec = U(v, a_hats)     # (n_a_hat,)
    return U_vec - U0        # (n_a_hat,) <= 0

def h_foc(v, cache):
    # equality at a0: dU/da(a0) = 0
    return d_U_d_a(v, cache)

# =========================================
# Pack / unpack θ and analytic (value, grad)
# =========================================
# Layout: [ lam, mu, mu_hats (n_fixed) ]
def unpack_params(theta_vec):
    lam = theta_vec[0]
    mu  = theta_vec[1]
    mu_hats = theta_vec[2 : 2 + n_a_hat]
    return lam, mu, mu_hats

def pack_initial_params(lam0, mu0, mu_hats0):
    return np.concatenate([
        np.array([lam0, mu0]),
        mu_hats0
    ])

def dual_value_and_grad(theta_vec, cache):
    """
    g(θ) = L(v*(θ), θ) lagrange dual function.
    We MINIMIZE obj = -g, so return (obj, grad_obj).
    Danskin => ∂g/∂multiplier = constraint residual at v*(θ).
    """
    lam, mu, mu_hats = unpack_params(theta_vec)

    # inner argmin v*(θ) constructed analytically
    v_star = canonical_contract_vec(lam, mu, mu_hats, cache)

    # constraint residuals at v*
    cir  = c_ir(v_star, cache)           # scalar
    cic  = c_ic(v_star, cache)           # (n_fixed,)
    hfoc = h_foc(v_star, cache)          # scalar

    # dual value
    g = ( expected_wage(v_star, cache)
          + lam * cir
          - mu  * hfoc
          + np.dot(mu_hats, cic) )

    # analytic gradient of obj = -g
    grad_lam     = -cir
    grad_mu      =  hfoc
    grad_muhats  = -cic                   # (n_fixed,)

    obj = -g
    grad = np.concatenate([
        np.array([grad_lam, grad_mu]),
        np.atleast_1d(grad_muhats)
    ])
    return obj, grad

# SciPy wrappers: separate fun and jac, same bounds/behavior as before
def _fun(theta, cache):
    obj, _ = dual_value_and_grad(theta, cache)
    return float(obj)

def _jac(theta, cache):
    _, grad = dual_value_and_grad(theta, cache)
    return grad

def run_solver(theta_init, bounds, cache, maxiter=1000, tol=1e-8):
    # SciPy wants bounds as sequence of (low, high)
    lo, hi = bounds
    scipy_bounds = tuple((float(l), float(h)) for (l, h) in zip(lo, hi))
    res = minimize(
        fun=_fun,
        x0=np.asarray(theta_init, dtype=float),
        jac=_jac,
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
init = pack_initial_params(lam0, mu0, mu_hats0)

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
warm_cache = build_cache(BASE_INTENDED_ACTION, warm_res_utility, a_hat_experiments)
_ = dual_value_and_grad(init, warm_cache)

# ===========================
# Timing loop across caches
# ===========================
times = []
solutions = []
actions_used = []

for i, (a0, rw) in enumerate(zip(noisy_intended_actions, reservation_wages), start=1):
    cache = build_cache(
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
    c = build_cache(float(a0), res_util, a_hat_experiments)
    theta_opt, state = run_solver(theta_init, bounds, c, maxiter=1000, tol=1e-8)
    n_solver_runs += 1

    lam, mu, mu_hats = unpack_params(theta_opt)
    v_star = canonical_contract_vec(lam, mu, mu_hats, c)
    ew = float(expected_wage(v_star, c))
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
# Plot U(a) versus action for the best contract at best a
# ===========================
# Unpack best contract
lam_b, mu_b, mu_hats_b = unpack_params(theta_star_linesearch)
v_star_best = canonical_contract_vec(lam_b, mu_b, mu_hats_b, cache_linesearch)

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
U_values = [float(U(v_star_best, a)) for a in a_eval_grid]

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
wage_schedule = k(v_star_best)  # k(v(y)) in dollars

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
U_vec = U(v_star_best, np.asarray(a_eval_grid))
idx = int(np.argmax(U_vec))
a_hat_from_U = float(np.asarray(a_eval_grid)[idx])

# Update the fixed hats to [0, a_hat_from_U]
a_hat_experiments_new = np.array([0.0, a_hat_from_U])
print(f"Chosen a_hat from U-argmax: {a_hat_from_U:.4f}")
print(f"New a_hat_experiments = {a_hat_experiments_new}")

# ===========================
# Re-solve at the same intended action using the new hats
# ===========================
cache_new = build_cache(float(a_star_linesearch), res_util, a_hat_experiments_new)
theta_opt_new, state = run_solver(theta_star_linesearch, bounds, cache_new, maxiter=1000, tol=1e-8)
lam_n, mu_n, mu_hats_n = unpack_params(theta_opt_new)
v_star_new = canonical_contract_vec(lam_n, mu_n, mu_hats_n, cache_new)

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
U_values_new = [float(U(v_star_new, a)) for a in a_eval_grid]

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
