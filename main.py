#%% 
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax import jit
from jaxopt import LBFGS
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
reservation_wage = -1
reservation_utility = u(reservation_wage)
intended_action = 80

# Cost
# Aspirational parameter
first_best_effort = 100
theta = 1.0 / first_best_effort / (first_best_effort + x0)
cost_of_effort = lambda a: theta * a ** 2 / 2
marginal_cost_of_effort = lambda a: theta * a
cost_of_effort_at_intended_action = cost_of_effort(intended_action)
marginal_cost_of_effort_at_intended_action = marginal_cost_of_effort(intended_action)

#%% Construct grids for approximation
y_lower, y_upper, y_n = 0 - 3 * sigma, 120 + 3 * sigma, 201
y_grid = jnp.linspace(y_lower, y_upper, y_n)
y_grid_step_size = y_grid[1] - y_grid[0]
assert y_grid.shape[0] % 2 == 1 # Need odd number for simpson's rule

density_intended = density(y_grid, intended_action)
d_density_d_a_intended = d_density_d_a(y_grid, intended_action)

# Simpson rule helper
def simpson_rule(y):
    """
    Composite Simpson's rule for uniform grid, odd number of points.
    """
    s_odd  = jnp.sum(y[1:-1:2])   # indices 1,3,5,...
    s_even = jnp.sum(y[2:-2:2])   # indices 2,4,6,...
    return (y_grid_step_size / 3.0) * (y[0] + y[-1] + 4.0 * s_odd + 2.0 * s_even)


#%% Convenience methods
# Canonical contract from the paper
def canonical_contract_vec(lam, mu):
    return link_function_g(lam + mu * score(y_grid, intended_action))

# Expected utility of the agent
def U(contract_vec, a):
    z = contract_vec * density(y_grid, a)
    return simpson_rule(z) - cost_of_effort(a)

def U_intended(contract_vec):
    z = contract_vec * density_intended
    return simpson_rule(z) - cost_of_effort_at_intended_action

# Expected wage at intended action
def expected_wage(contract_vec):
    z = k(contract_vec) * density_intended
    return simpson_rule(z)

def d_U_d_a(contract_vec):
    z = contract_vec * d_density_d_a_intended
    return  simpson_rule(z) - marginal_cost_of_effort_at_intended_action

def lagrangian(contract_vec, lam, mu):
    return expected_wage(contract_vec) + lam * ( reservation_utility - U_intended(contract_vec)) - mu * d_U_d_a(contract_vec)

def lagrange_dual(lam, mu):
    v_star = canonical_contract_vec(lam, mu)
    return lagrangian(v_star, lam, mu)


#%% Objective function
def objective(multipliers):
    lam, mu = multipliers
    return -lagrange_dual(lam, mu)


def objective_with_gradient(multipliers):
    lam, mu = multipliers

    # Optimal contract for (lam, mu)
    v_star = canonical_contract_vec(lam, mu)

    # Convenience quantities at v*
    EU_int = U_intended(v_star)          # U(v*, a=intended_action)
    dUda   = d_U_d_a(v_star)             # ∂U/∂a at a=intended_action
    EW     = expected_wage(v_star)       # E[k(v*) | a=intended_action]

    # Lagrangian value at v*
    L = EW + lam * (reservation_utility - EU_int) - mu * dUda

    # Objective is -dual; gradient by envelope theorem:
    # ∂(-L*)/∂lam = EU_int - reservation_utility
    # ∂(-L*)/∂mu  = dUda
    obj_value = -L
    grad = jnp.array([EU_int - reservation_utility, dUda])

    return obj_value, grad

objective_jit = jit(objective)
objective_with_gradient_jit = jit(objective_with_gradient)

# %%
#%% LBFGS (JAXopt): unconstrained


# --- solvers ---
solver = LBFGS(fun=objective_with_gradient_jit, value_and_grad=True, maxiter=100, tol=1e-3)
init   = jnp.array([100, 100], dtype=float)
params_star, state = solver.run(init_params=init)


# %%
lambda_star, mu_star = params_star

optimal_contract = canonical_contract_vec(lambda_star, mu_star)
optimal_wage_function = k(optimal_contract)

print(f"Optimal multipliers: lambda = {lambda_star:.2f}, mu = {mu_star:.2f}")
print(f"Expected wage at intended action: {expected_wage(optimal_contract):.2f}")
print(f"Expected utility at intended action: {U_intended(optimal_contract):.2f}")

plt.plot(y_grid, optimal_wage_function)
plt.title("Optimal Wage Function")
plt.xlabel("Effort Level")
plt.ylabel("Wage")
plt.grid()
plt.show()

a_grid = jnp.linspace(0, 120, 20)
U_grid = jnp.array([U(optimal_contract, a) for a in a_grid])

plt.plot(a_grid, U_grid)
plt.title("Expected Utility Function")
plt.xlabel("Effort Level")
plt.ylabel("Utility")
plt.grid(True)
plt.show()

# %%
