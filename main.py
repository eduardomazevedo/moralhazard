#%% 
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax import jit, value_and_grad
from jaxopt import LBFGS, ProjectedGradient, ScipyBoundedMinimize
import matplotlib.pyplot as plt

#%% Parameters
# Utility function
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

# Reservation wage, intended action
reservation_wage = 1
reservation_utility = u(reservation_wage)
intended_action = 80

# Cost
# Aspirational parameter
first_best_effort = 100
theta = 1.0 / first_best_effort / (first_best_effort + x0)
cost_of_effort = lambda a: theta * a ** 2 / 2
marginal_cost_of_effort = lambda a: theta * a

#%% Construct grids for approximation
y_lower, y_upper, y_n = 0 - 3 * sigma, 120 + 3 * sigma, 201
y_grid = jnp.linspace(y_lower, y_upper, y_n)
y_grid_step_size = y_grid[1] - y_grid[0]
assert y_grid.shape[0] % 2 == 1 # Need odd number for simpson's rule

# Fixed a_hats
a_hat_fixed = jnp.array([0.0, 50.0])
n_fixed_a_hat = a_hat_fixed.shape[0]
n_floating_a_hat = 1
n_total_a_hat = n_fixed_a_hat + n_floating_a_hat

# Simpson rule helper
def simpson_rule(y):
    """
    Composite Simpson's rule for uniform grid, odd number of points.
    """
    s_odd  = jnp.sum(y[1:-1:2])   # indices 1,3,5,...
    s_even = jnp.sum(y[2:-2:2])   # indices 2,4,6,...
    return (y_grid_step_size / 3.0) * (y[0] + y[-1] + 4.0 * s_odd + 2.0 * s_even)


# Pre store vectors used over and over at intended action
cost_of_effort_0 = cost_of_effort(intended_action)
marginal_cost_of_effort_0 = marginal_cost_of_effort(intended_action)
density_0 = density(y_grid, intended_action)
d_density_d_a_0 = d_density_d_a(y_grid, intended_action)
score_0 = score(y_grid, intended_action)

#%% Convenience methods
# Expected utility of the agent
def U(contract_vec, a):
    z = contract_vec * density(y_grid, a)
    return simpson_rule(z) - cost_of_effort(a)

def U_0(contract_vec):
    z = contract_vec * density_0
    return simpson_rule(z) - cost_of_effort_0

# Expected wage at intended action
def expected_wage(contract_vec):
    z = k(contract_vec) * density_0
    return simpson_rule(z)

def d_U_d_a(contract_vec):
    z = contract_vec * d_density_d_a_0
    return  simpson_rule(z) - marginal_cost_of_effort_0


#%% Core functions (whole math is this thing see paper)
# Canonical contract from the paper
def canonical_contract_vec(lam, mu, mu_hats, a_hats):
    z = (
        lam
        + mu * score_0
        + sum(
            mu_hat * (1 - density(y_grid, a_hat) / density_0)
            for mu_hat, a_hat in zip(mu_hats, a_hats)
        ) # This term not in the paper
    )
    return link_function_g(z)

# Lagrangian from equation 2
def lagrangian(contract_vec, lam, mu, mu_hats, a_hats):
    U_00 = U_0(contract_vec)
    return (
        expected_wage(contract_vec)
        + lam * (reservation_utility - U_00)
        - mu * d_U_d_a(contract_vec)
        + sum(mu_hat * (U(contract_vec, a_hat) - U_00)
              for mu_hat, a_hat in zip(mu_hats, a_hats))
    )

def lagrange_dual(lam, mu, mu_hats, a_hats):
    v_star = canonical_contract_vec(lam, mu, mu_hats, a_hats)
    return lagrangian(v_star, lam, mu, mu_hats, a_hats)


#%% Objective function
# Helpers to pack/unpack a single flat parameter vector for the optimizer.
# Layout: [ lam, mu, mu_hats (n_total), a_hat_floating (n_floating) ]
def unpack_params(theta):
    lam = theta[0]
    mu  = theta[1]
    mu_hats = theta[2 : 2 + n_total_a_hat]
    a_hat_floating = theta[2 + n_total_a_hat : 2 + n_total_a_hat + n_floating_a_hat]
    a_hats = jnp.concatenate([a_hat_fixed, a_hat_floating])
    return lam, mu, mu_hats, a_hats

def pack_initial_params(lam0, mu0, mu_hats0, a_hat_floating0):
    return jnp.concatenate([
        jnp.array([lam0, mu0]),
        mu_hats0,
        a_hat_floating0
    ])

def objective(theta):
    lam, mu, mu_hats, a_hats = unpack_params(theta)
    return -lagrange_dual(lam, mu, mu_hats, a_hats)


objective_jit = jit(objective)
objective_with_gradient = value_and_grad(objective)
objective_with_gradient_jit = jit(objective_with_gradient)

#%% Initial values and bounds
# --- initial guesses ---
lam0 = 100.0
mu0  = 100.0
mu_hats0 = jnp.zeros(n_total_a_hat)                       # shape (n_total_a_hat,)
a_hat_floating0 = jnp.full((n_floating_a_hat,), 0.0)

init = pack_initial_params(lam0, mu0, mu_hats0, a_hat_floating0)

# --- bounds ---
# lam: free; mu: >= 0; mu_hats: free; floating a_hats: optional box [0, 120]
# (adjust box as needed)
a_min, a_max = 0.0, 120.0

lower_bounds = jnp.concatenate([
    jnp.array([0.0, -jnp.inf]),                   # lam, mu
    jnp.full((n_total_a_hat,), 0.0),         # mu_hats
    jnp.full((n_floating_a_hat,), a_min)          # floating a_hats
])
upper_bounds = jnp.concatenate([
    jnp.array([ jnp.inf,  jnp.inf]),              # lam, mu
    jnp.full((n_total_a_hat,),  jnp.inf),         # mu_hats
    jnp.full((n_floating_a_hat,), a_max)          # floating a_hats
])
bounds = (lower_bounds, upper_bounds)


#%% Solvers & timing (refactor: constrained only)

from time import perf_counter

# --- projection that matches your bounds exactly ---
# Layout of params: [ lam, mu, mu_hats (n_total_a_hat), a_hat_floating (n_floating_a_hat) ]
def _project_to_box(params, hyperparams=None):
    # lam >= 0
    lam = jnp.maximum(params[0], 0.0)
    # mu: free
    mu = params[1]

    # mu_hats >= 0
    start_mu = 2
    end_mu   = 2 + n_total_a_hat
    mu_hats = jnp.maximum(params[start_mu:end_mu], 0.0)

    # a_hat_floating in [a_min, a_max]
    start_a = end_mu
    end_a   = end_mu + n_floating_a_hat
    a_hat_float = jnp.clip(params[start_a:end_a], a_min, a_max)

    return jnp.concatenate([jnp.array([lam, mu]), mu_hats, a_hat_float])

# --- factory functions: fresh solver instances each call ---
def make_pg():
    return ProjectedGradient(
        fun=objective_with_gradient_jit,
        value_and_grad=True,
        projection=_project_to_box,
        maxiter=2000,        # PG typically needs more iterations
        tol=1e-6,
        stepsize=1e-1,       # tune as needed
    )

def make_scipy_bounded():
    return ScipyBoundedMinimize(
        fun=objective_with_gradient_jit,
        method="l-bfgs-b",
        value_and_grad=True,
        maxiter=1000,
        tol=1e-6,
    )

# --- timing helper ---
def _run_and_time(name, build_solver, *, warmup=True, **run_kwargs):
    """
    build_solver: () -> solver instance
    run_kwargs: passed to solver.run(...)
    Returns dict with method, params, state, time_sec.
    """
    solver = build_solver()

    # Optional warmup to exclude JIT compile time
    if warmup:
        _params_w, _state_w = solver.run(**run_kwargs)
        try:
            _ = _params_w.block_until_ready()
        except Exception:
            pass

    t0 = perf_counter()
    params, state = solver.run(**run_kwargs)
    try:
        _ = params.block_until_ready()
    except Exception:
        pass
    t1 = perf_counter()

    return {"method": name, "params": params, "state": state, "time_sec": t1 - t0}

# --- run solvers (constrained only) ---
results = []
results.append(_run_and_time("ProjectedGradient (boxed)", make_pg, warmup=True, init_params=init))
results.append(_run_and_time("ScipyBoundedMinimize (boxed)", make_scipy_bounded,
                             warmup=True, init_params=init, bounds=bounds))

# --- unpack helper for summaries ---
def _unpack_short(theta):
    lam = float(theta[0])
    mu  = float(theta[1])
    mu_vec = theta[2:2 + n_total_a_hat]
    a_flt  = theta[2 + n_total_a_hat : 2 + n_total_a_hat + n_floating_a_hat]
    return lam, mu, mu_vec, a_flt

# --- pretty print summary table ---
print("\nComparison of optimal parameters and runtimes (constrained methods):")
hdr = "{:<32} {:>12} {:>12} {:>10} {:>10} {:>10} {:>10} {:>12}"
row = "{:<32} {:>12.4f} {:>12.4f} {:>10.2e} {:>10.4f} {:>10.2f} {:>10.2f} {:>12.6f}"
print(hdr.format("Method", "lambda", "mu",
                 "||mû||₂", "μ̂_min", "â_min", "â_max", "time (s)"))
print("-" * 120)

for r in results:
    lam, mu, mu_vec, a_flt = _unpack_short(r["params"])
    mu_norm = float(jnp.linalg.norm(mu_vec))
    mu_min  = float(mu_vec.min()) if mu_vec.size else float('nan')
    a_min_v = float(a_flt.min()) if a_flt.size else float('nan')
    a_max_v = float(a_flt.max()) if a_flt.size else float('nan')
    t       = r["time_sec"]
    print(row.format(r["method"], lam, mu, mu_norm, mu_min, a_min_v, a_max_v, t))

# --- pick a constrained solution for downstream use (recommended) ---
_constrained = next(r for r in results if r["method"].startswith("ScipyBoundedMinimize"))
theta_star = _constrained["params"]
lam_star, mu_star, mu_hats_star, a_hats_star = unpack_params(theta_star)

#%% Plot U(a) at optimal contract
optimal_contract_pg = canonical_contract_vec(
    results[0]["params"][0], results[0]["params"][1],
    results[0]["params"][2:2 + n_total_a_hat],
    jnp.concatenate([a_hat_fixed, results[0]["params"][2 + n_total_a_hat : 2 + n_total_a_hat + n_floating_a_hat]])
)
optimal_contract_scipy = canonical_contract_vec(
    results[1]["params"][0], results[1]["params"][1],
    results[1]["params"][2:2 + n_total_a_hat],
    jnp.concatenate([a_hat_fixed, results[1]["params"][2 + n_total_a_hat : 2 + n_total_a_hat + n_floating_a_hat]])
)

a_grid = jnp.linspace(0, 120, 100)
U_grid_pg = jnp.array([U(optimal_contract_pg, a) for a in a_grid])
U_grid_scipy = jnp.array([U(optimal_contract_scipy, a) for a in a_grid])

plt.figure()
plt.plot(a_grid, U_grid_pg, label="ProjectedGradient (boxed)")
plt.plot(a_grid, U_grid_scipy, label="ScipyBoundedMinimize (boxed)")
plt.title("Expected Utility as a Function of Effort")
plt.xlabel("Effort level (a)")
plt.ylabel("Utility")
plt.grid(True)
plt.legend()
plt.show()

# Plot optimal wage function for both optimal contracts
wage_grid_pg = k(optimal_contract_pg)
wage_grid_scipy = k(optimal_contract_scipy)

plt.figure()
plt.plot(y_grid, wage_grid_pg, label="ProjectedGradient Wage")
plt.plot(y_grid, wage_grid_scipy, label="ScipyBoundedMinimize Wage")
plt.title("Optimal Wage as a Function of Effort")
plt.xlabel("Effort level (a)")
plt.ylabel("Wage")
plt.grid(True)
plt.legend()
plt.show()

# %%
