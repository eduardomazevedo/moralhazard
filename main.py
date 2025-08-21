#%%
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax import jit, value_and_grad
from jaxopt import ScipyBoundedMinimize
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
# lam >= 0; mu free; mu_hats >= 0; floating a_hats in [0, 120]
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

#%% Solver (ScipyBoundedMinimize only)
solver = ScipyBoundedMinimize(
    fun=objective_with_gradient_jit,
    method="l-bfgs-b",
    value_and_grad=True,
    maxiter=1000,
    tol=1e-6,
)

params_opt, state = solver.run(init_params=init, bounds=bounds)
theta_star = params_opt
lam_star, mu_star, mu_hats_star, a_hats_star = unpack_params(theta_star)

#%% Plot U(a) at optimal contract (Scipy-only)
optimal_contract = canonical_contract_vec(
    lam_star, mu_star, mu_hats_star, a_hats_star
)

a_grid = jnp.linspace(0, 120, 100)
U_grid = jnp.array([U(optimal_contract, a) for a in a_grid])

plt.figure()
plt.plot(a_grid, U_grid, label="ScipyBoundedMinimize")
plt.title("Expected Utility as a Function of Effort")
plt.xlabel("Effort level (a)")
plt.ylabel("Utility")
plt.grid(True)
plt.legend()
plt.show()

# Plot optimal wage function
wage_grid = k(optimal_contract)

plt.figure()
plt.plot(y_grid, wage_grid, label="Optimal Wage")
plt.title("Optimal Wage as a Function of Outcome y")
plt.xlabel("Outcome (y)")
plt.ylabel("Wage")
plt.grid(True)
plt.legend()
plt.show()

#%% Print optimal parameters and expected wage

print("\nOptimal parameters:")
print(f"lambda*: {lam_star:.4f}")
print(f"mu*: {mu_star:.4f}")
print(f"mu_hats*: {mu_hats_star}")
print(f"a_hats*: {a_hats_star}")

exp_wage_opt = expected_wage(optimal_contract)
print(f"\nAttained expected wage: {float(exp_wage_opt):.6f}")

# %%
