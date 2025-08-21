#%%
import time
import numpy as np

import jax.numpy as jnp
from jax import jit, vmap, lax
from jax.scipy.stats import norm
from jaxopt import ScipyBoundedMinimize
import matplotlib.pyplot as plt  # optional

# ===========
# Parameters
# ===========
# Utility
x0 = 50.0   # meaning 50k per year in consumption
u = lambda dollars: jnp.log(dollars + x0)
k = lambda utils: jnp.exp(utils) - x0
k_prime = lambda utils: jnp.exp(utils)
k_prime_inverse = lambda utils: jnp.log(utils)
link_function_g = lambda x: jnp.log(jnp.maximum(x, x0))

# Distribution
sigma = 10.0
density = lambda y, a: norm.pdf(y, loc=a, scale=sigma)
d_density_d_a = lambda y, a: ((y - a) / sigma ** 2) * density(y, a)
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
y_grid = jnp.linspace(y_lower, y_upper, y_n)
y_grid_step_size = y_grid[1] - y_grid[0]
assert y_grid.shape[0] % 2 == 1  # Need odd number for Simpson's rule

# Simpson weights (vector form)
w = jnp.zeros_like(y_grid)
w = w.at[0].set(1.0).at[-1].set(1.0)
w = w.at[1:-1:2].set(4.0)
w = w.at[2:-2:2].set(2.0)
w = w * (y_grid_step_size / 3.0)

def simpson_rule(y):
    return jnp.vdot(w, y)

# =================================
# Fixed/floating a-hats (counts)
# =================================
n_fixed_a_hat = 2
n_floating_a_hat = 1
n_total_a_hat = n_fixed_a_hat + n_floating_a_hat

# ======================================
# Cache builder for (a0, U0, fixed hats)
# ======================================
def build_cache(a0: float,
                reservation_utility: float,
                a_hat_fixed_vals: jnp.ndarray):
    """
    Build a small PyTree 'cache' of arrays that depend on the intended action a0,
    the reservation utility, and the values (NOT the count) of the fixed a-hats.

    Shapes/dtypes stay constant across runs, so JIT can be reused.
    """
    a0 = jnp.asarray(a0, dtype=y_grid.dtype)
    a_hat_fixed_vals = jnp.asarray(a_hat_fixed_vals, dtype=y_grid.dtype)
    assert a_hat_fixed_vals.shape == (n_fixed_a_hat,), \
        f"Expected fixed hats of shape {(n_fixed_a_hat,)}, got {a_hat_fixed_vals.shape}"

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
    dens_y_fixed = density(y_grid[:, None], a_hat_fixed_vals[None, :])  # (y_n, n_fixed_a_hat)

    return {
        "a0": a0,
        "reservation_utility": jnp.asarray(reservation_utility, dtype=y_grid.dtype),
        "a_hat_fixed": a_hat_fixed_vals,           # (n_fixed_a_hat,)
        "dens_y_fixed": dens_y_fixed,              # (y_n, n_fixed_a_hat)

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
    # independent of cache, uses full density(y, a)
    return jnp.vdot(w, contract_vec * density(y_grid, a)) - cost_of_effort(a)

def U_0(contract_vec, cache):
    return jnp.vdot(cache["w_density0"], contract_vec) - cache["cost_of_effort_0"]

def expected_wage(contract_vec, cache):
    return jnp.vdot(cache["w_density0"], k(contract_vec))

def d_U_d_a(contract_vec, cache):
    return jnp.vdot(cache["w_d_density_da0"], contract_vec) - cache["marginal_cost_of_effort_0"]

# =========================================
# Canonical inner minimizer v*(λ, μ, μ_hat)
# =========================================
def canonical_contract_vec(lam, mu, mu_hats, a_hats, cache, eps=1e-12):
    """
    Build v*(y) from multipliers. We reuse cached density columns for the fixed hats
    and compute density columns for the floating hats on the fly.
    """
    # Split a_hats into fixed/floating (counts are constant)
    a_hats_fixed = cache["a_hat_fixed"]  # (n_fixed_a_hat,)
    a_hats_floating = a_hats[n_fixed_a_hat:]  # (n_floating_a_hat,)

    # Densities: fixed part from cache; floating recomputed
    dens_y_fixed = cache["dens_y_fixed"]  # (y_n, n_fixed_a_hat)
    if n_floating_a_hat > 0:
        dens_y_float = density(y_grid[:, None], a_hats_floating[None, :])  # (y_n, n_floating_a_hat)
        dens_y_a = jnp.concatenate([dens_y_fixed, dens_y_float], axis=1)   # (y_n, n_total_a_hat)
    else:
        dens_y_a = dens_y_fixed

    ratio = dens_y_a / jnp.maximum(cache["density_0"][:, None], eps)  # (y_n, n_total_a_hat)
    hat_term = (1.0 - ratio) @ mu_hats                                # (y_n,)
    z = lam + mu * cache["score_0"] + hat_term
    return link_function_g(z)

# =========================================
# Constraint maps (residual forms)
# =========================================
def c_ir(v, cache):
    # IR: U0 >= Ubar  <=>  Ubar - U0 <= 0
    return cache["reservation_utility"] - U_0(v, cache)          # scalar <= 0

def c_ic(v, a_hats, cache):
    # IC at sampled points: U(a) - U0 <= 0  (elementwise)
    U0 = U_0(v, cache)
    U_vec = vmap(lambda a: U(v, a))(a_hats)                       # (n_total,)
    return U_vec - U0                                             # (n_total,) <= 0

def h_foc(v, cache):
    # equality at a0: dU/da(a0) = 0
    return d_U_d_a(v, cache)                                      # scalar = 0

# =========================================
# Pack / unpack θ and analytic (value, grad)
# =========================================
# Layout: [ lam, mu, mu_hats (n_total), a_hat_floating (n_floating) ]
def unpack_params(theta_vec, cache):
    lam = theta_vec[0]
    mu  = theta_vec[1]
    mu_hats = theta_vec[2 : 2 + n_total_a_hat]
    a_hat_floating = theta_vec[2 + n_total_a_hat : 2 + n_total_a_hat + n_floating_a_hat]
    a_hats = jnp.concatenate([cache["a_hat_fixed"], a_hat_floating])
    return lam, mu, mu_hats, a_hats

def pack_initial_params(lam0, mu0, mu_hats0, a_hat_floating0):
    return jnp.concatenate([
        jnp.array([lam0, mu0]),
        mu_hats0,
        a_hat_floating0
    ])

def dual_value_and_grad(theta_vec, cache):
    """
    g(θ) = L(v*(θ), θ)
    We MINIMIZE obj = -g, so return (obj, grad_obj).
    Danskin => ∂g/∂multiplier = constraint residual at v*(θ).
    """
    lam, mu, mu_hats, a_hats = unpack_params(theta_vec, cache)

    # inner argmin v*(θ) constructed analytically; do NOT backprop through it
    v_star = canonical_contract_vec(lam, mu, mu_hats, a_hats, cache)
    v_star = lax.stop_gradient(v_star)

    # constraint residuals at v*
    cir  = c_ir(v_star, cache)                 # scalar
    cic  = c_ic(v_star, a_hats, cache)         # (n_total,)
    hfoc = h_foc(v_star, cache)                # scalar

    # dual value
    g = ( expected_wage(v_star, cache)
          + lam * cir
          - mu  * hfoc
          + jnp.dot(mu_hats, cic) )

    # analytic gradient of obj = -g
    grad_lam     = -cir                          # scalar
    grad_mu      =  hfoc                         # scalar (since ∂g/∂μ = -h)
    grad_muhats  = -cic                          # (n_total,)

    # optional: gradients w.r.t floating a_hats
    if n_floating_a_hat > 0:
        a_float = a_hats[n_fixed_a_hat:]
        # ∂g/∂a_i = μ_i * ∂U(v*, a_i)/∂a_i ⇒ grad_obj = -∂g/∂a_i
        dU_da_at_ahat = vmap(
            lambda a: jnp.vdot(w, v_star * d_density_d_a(y_grid, a)) - marginal_cost_of_effort(a)
        )(a_float)
        grad_a_float = -(mu_hats[n_fixed_a_hat:] * dU_da_at_ahat)
    else:
        grad_a_float = jnp.zeros((0,), dtype=theta_vec.dtype)

    obj = -g
    grad = jnp.concatenate([jnp.array([grad_lam, grad_mu]), grad_muhats, grad_a_float])
    return obj, grad

dual_value_and_grad_jit = jit(dual_value_and_grad)

# ===========================
# Initial values and bounds
# ===========================
lam0 = 100.0
mu0  = 100.0
mu_hats0 = jnp.zeros(n_total_a_hat)               # shape (n_total_a_hat,)
a_hat_floating0 = jnp.full((n_floating_a_hat,), 0.0)
init = pack_initial_params(lam0, mu0, mu_hats0, a_hat_floating0)

# Bounds: lam >= 0; mu free; mu_hats >= 0; floating a_hats in [0, 120]
a_min, a_max = 0.0, 120.0
lower_bounds = jnp.concatenate([
    jnp.array([0.0, -jnp.inf]),                   # lam, mu
    jnp.full((n_total_a_hat,), 0.0),              # mu_hats
    jnp.full((n_floating_a_hat,), a_min)          # floating a_hats
])
upper_bounds = jnp.concatenate([
    jnp.array([ jnp.inf,  jnp.inf]),              # lam, mu
    jnp.full((n_total_a_hat,),  jnp.inf),         # mu_hats
    jnp.full((n_floating_a_hat,), a_max)          # floating a_hats
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
a_hat_fixed_sequences = np.tile([0.0, 50.0], (N_RUNS, 1))  # shape (N_RUNS, 2), same values for all runs

# ===========================
# Warm-up compile (once)
# ===========================
warm_res_utility = u(1.0)                          # pick a representative value
warm_fixed = jnp.array([0.0, 50.0])                # representative fixed hats (shape must match)
warm_cache = build_cache(BASE_INTENDED_ACTION, warm_res_utility, warm_fixed)
_ = dual_value_and_grad_jit(init, warm_cache)

# Create solver once; fun signature is (theta, cache)
solver = ScipyBoundedMinimize(
    fun=dual_value_and_grad_jit,
    method="l-bfgs-b",
    value_and_grad=True,
    maxiter=1000,
    tol=1e-8,   # slightly tighter now that gradients are analytic
)

# ===========================
# Timing loop across caches
# ===========================
times = []
solutions = []
actions_used = []

for i, (a0, rw, fixed_vals) in enumerate(zip(noisy_intended_actions,
                                             reservation_wages,
                                             a_hat_fixed_sequences), start=1):
    cache = build_cache(
        float(a0),
        reservation_utility=float(u(rw)),         # vary reservation utility here
        a_hat_fixed_vals=jnp.asarray(fixed_vals), # vary fixed hats VALUES here (same count)
    )

    t0 = time.time()
    params_opt, state = solver.run(init, bounds, cache)  # extra arg = cache
    t1 = time.time()

    dt = t1 - t0
    times.append(dt)
    solutions.append(params_opt)
    actions_used.append(a0)

    print(f"Run {i:02d} | intended_action = {a0:6.2f} | U0 from rw={rw:.2f} | fixed={np.array2string(np.array(fixed_vals), precision=2)} | time = {dt:.4f} s")

avg_time = sum(times) / len(times)
print(f"\nAverage solve time over {N_RUNS} runs: {avg_time:.4f} s")

# Unpack last solution for convenience
theta_star = solutions[-1]
lam_star, mu_star, mu_hats_star, a_hats_star = unpack_params(theta_star, cache)

#%%
# ===========================
# New experiment: line search over intended action a
# Objective: maximize a - expected_wage(v*(a))
# ===========================
LINESEARCH_GRID_N = 121  # simple, robust grid (objective may be discontinuous)
a_grid = np.linspace(a_min, a_max, LINESEARCH_GRID_N)

# Use the last run's reservation utility and fixed hats for the search
res_util = float(u(80))
fixed_vals = cache["a_hat_fixed"]

best = {"a": None, "gap": -np.inf, "ew": None, "theta": None, "cache": None}
theta_init = theta_star  # warm start from last solve
obj_values = []

t0 = time.time()
n_solver_runs = 0
for a0 in a_grid:
    c = build_cache(float(a0), res_util, fixed_vals)
    theta_opt, state = solver.run(theta_init, bounds, c)
    n_solver_runs += 1

    lam, mu, mu_hats, a_hats = unpack_params(theta_opt, c)
    v_star = canonical_contract_vec(lam, mu, mu_hats, a_hats, c)
    ew = float(expected_wage(v_star, c))
    gap = float(a0 - ew)
    obj_values.append(gap)

    if gap > best["gap"]:
        best.update({"a": float(a0), "gap": gap, "ew": ew, "theta": theta_opt, "cache": c})

    theta_init = theta_opt  # carry warm start along the grid
t1 = time.time()

print(
    f"\n[Line search] Best a in [{a_min:.1f}, {a_max:.1f}] is {best['a']:.4f} "
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
plt.axvline(best["a"], color="red", linestyle="--", label=f"best a = {best['a']:.2f}")
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
lam_b, mu_b, mu_hats_b, a_hats_b = unpack_params(theta_star_linesearch, cache_linesearch)
v_star_best = canonical_contract_vec(lam_b, mu_b, mu_hats_b, a_hats_b, cache_linesearch)

# Evaluate utility over a grid of actions
a_eval_grid = np.linspace(a_min, a_max, 200)
U_values = [float(U(v_star_best, a)) for a in a_eval_grid]

plt.figure(figsize=(6,4))
plt.plot(a_eval_grid, U_values, linewidth=1.5)
plt.axvline(a_star_linesearch, color="red", linestyle="--", label=f"best intended a = {a_star_linesearch:.2f}")
plt.xlabel("Action a")
plt.ylabel("U(a)")
plt.title("Utility U(a) under best contract at best intended action")
plt.legend()
plt.tight_layout()
plt.show()
