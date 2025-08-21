#%%
import time
import numpy as np

import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
from jax.scipy.stats import norm
from jaxopt import ScipyBoundedMinimize
import matplotlib.pyplot as plt  # optional

# ===========
# Parameters
# ===========
# Utility
x0 = 50.0   # meaning 50k per year in consumption
u = lambda dollars: jnp.log(dollars + x0)
u_0 = u(0)  # Utility at 0
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

# Reservation wage
reservation_wage = 1.0
reservation_utility = u(reservation_wage)

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

# Fixed a_hats
a_hat_fixed = jnp.array([0.0, 50.0])
n_fixed_a_hat = a_hat_fixed.shape[0]
n_floating_a_hat = 1
n_total_a_hat = n_fixed_a_hat + n_floating_a_hat

# ==============================
# Cache builder for a0
# ==============================
def build_cache(a0: float):
    """
    Build a small PyTree 'cache' of arrays that depend on the intended action a0.
    Keeps shapes/dtypes constant across runs so JIT can be reused.
    """
    a0 = jnp.asarray(a0, dtype=y_grid.dtype)
    density_0 = density(y_grid, a0)                  # (y_n,)
    d_density_d_a_0 = d_density_d_a(y_grid, a0)      # (y_n,)
    score_0 = score(y_grid, a0)                      # (y_n,)

    cost_of_effort_0 = cost_of_effort(a0)
    marginal_cost_of_effort_0 = marginal_cost_of_effort(a0)

    # Weighted constants
    w_density0 = w * density_0
    w_d_density_da0 = w * d_density_d_a_0

    return {
        "a0": a0,
        "density_0": density_0,
        "d_density_d_a_0": d_density_d_a_0,
        "score_0": score_0,
        "w_density0": w_density0,
        "w_d_density_da0": w_d_density_da0,
        "cost_of_effort_0": cost_of_effort_0,
        "marginal_cost_of_effort_0": marginal_cost_of_effort_0,
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

def canonical_contract_vec(lam, mu, mu_hats, a_hats, cache):
    # dens_y_a: (y_n, n_total_a_hat)
    dens_y_a = density(y_grid[:, None], a_hats[None, :])
    ratio = dens_y_a / cache["density_0"][:, None]
    hat_term = (1.0 - ratio) @ mu_hats  # (y_n,)
    z = lam + mu * cache["score_0"] + hat_term
    return link_function_g(z)

def lagrangian(contract_vec, lam, mu, mu_hats, a_hats, cache):
    U00 = U_0(contract_vec, cache)
    U_vec = vmap(lambda a: U(contract_vec, a))(a_hats)  # (n_total_a_hat,)
    return (
        expected_wage(contract_vec, cache)
        + lam * (reservation_utility - U00)
        - mu * d_U_d_a(contract_vec, cache)
        + jnp.dot(mu_hats, U_vec - U00)
    )

def lagrange_dual(lam, mu, mu_hats, a_hats, cache):
    v_star = canonical_contract_vec(lam, mu, mu_hats, a_hats, cache)
    return lagrangian(v_star, lam, mu, mu_hats, a_hats, cache)

# =========================================
# Objective that accepts (theta, cache)
# =========================================
# Layout: [ lam, mu, mu_hats (n_total), a_hat_floating (n_floating) ]
def unpack_params(theta):
    lam = theta[0]
    mu  = theta[1]
    mu_hats = theta[2 : 2 + n_total_a_hat]
    a_hat_floating = theta[2 + n_total_a_hat : 2 + n_total_a_hat + n_floating_a_hat]
    a_hats = jnp.concatenate([a_hat_fixed, a_hat_floating])
    return lam, mu, mu_hats, a_hats

def objective(theta, cache):
    lam, mu, mu_hats, a_hats = unpack_params(theta)
    return -lagrange_dual(lam, mu, mu_hats, a_hats, cache)

# JIT once; shapes of theta and cache contents remain constant across runs
objective_with_gradient_jit = jit(value_and_grad(objective))

# ===========================
# Initial values and bounds
# ===========================
def pack_initial_params(lam0, mu0, mu_hats0, a_hat_floating0):
    return jnp.concatenate([
        jnp.array([lam0, mu0]),
        mu_hats0,
        a_hat_floating0
    ])

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

# ===========================
# Warm-up compile (once)
# ===========================
warm_cache = build_cache(BASE_INTENDED_ACTION)
_ = objective_with_gradient_jit(init, warm_cache)

# Create solver once; fun signature is (theta, cache)
solver = ScipyBoundedMinimize(
    fun=objective_with_gradient_jit,
    method="l-bfgs-b",
    value_and_grad=True,
    maxiter=1000,
    tol=1e-6,
)

# ===========================
# Timing loop across caches
# ===========================
times = []
solutions = []
actions_used = []

for i, a0 in enumerate(noisy_intended_actions, start=1):
    cache = build_cache(float(a0))

    t0 = time.time()
    # ScipyBoundedMinimize.run accepts extra positional args after bounds
    params_opt, state = solver.run(init, bounds, cache)
    t1 = time.time()

    dt = t1 - t0
    times.append(dt)
    solutions.append(params_opt)
    actions_used.append(a0)

    print(f"Run {i:02d} | intended_action = {a0:6.2f} | time = {dt:.4f} s")

avg_time = sum(times) / len(times)
print(f"\nAverage solve time over {N_RUNS} runs: {avg_time:.4f} s")

# Unpack last solution for convenience
theta_star = solutions[-1]
lam_star, mu_star, mu_hats_star, a_hats_star = unpack_params(theta_star)
