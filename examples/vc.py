# Comparing VC vs public company CEO compensation
import numpy as np
import matplotlib.pyplot as plt
from moralhazard import MoralHazardProblem
from moralhazard.config_maker import make_utility_cfg, make_distribution_cfg

# Basics
base_company_value = 100000 # 100bn for a public company
ln_mu = 0 # y is a performance index 1 = median performance with perfect effort.
ln_sigma = 0.1 # Start with public company, small variance.
w0 = 10 # 10m consumption that a big shot CEO would have had in 10 years.

revenue_function = lambda a: base_company_value * np.exp(a + ln_sigma**2/2)


# Cost function
# We consider a function that would require infinite effort to achieve the target value.
# But elasticity of the gap with respect to marginal cost of effort is something reasonable.
alpha = 1 # Elasticity of the gap is 1 / alpha + 1, need alpha > 0

def c(x):
    """
    Power-barrier cost: c(x) = (1 - x)^(-alpha) - 1
    """
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (1.0 - x)**(-alpha) - 1.0

def mc(x):
    """
    Marginal cost: mc(x) = c'(x) = alpha * (1 - x)^(-alpha - 1)
    """
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return alpha * (1.0 - x)**(-alpha - 1.0)

# Create cfgs
utility_cfg = make_utility_cfg("log", w0=w0)
# Create distribution functions (gaussian with sigma)
dist_cfg = make_distribution_cfg("gaussian", sigma=ln_sigma)

# Reservation utility
u = utility_cfg["u"]
reservation_utility = u(2 * w0)

cfg = {
    "problem_params": {
        **utility_cfg,
        **dist_cfg,
        "C": c,
        "Cprime": mc,
    },
    "computational_params": {
        "distribution_type": "continuous",
        "y_min": - 4 * ln_sigma,
        "y_max": 1 + 4 * ln_sigma,
        "n": 201,  # must be odd
    }
}

mhp = MoralHazardProblem(cfg)

result_cmp = mhp.solve_cost_minimization_problem(
    intended_action=0.35,
    reservation_utility=reservation_utility,
    solver="iterative",
    n_a_iterations=1,
    a_ic_lb=0.0,
    a_ic_ub=0.35,
    a_ic_initial=0.0,
)

result_principal = mhp.solve_principal_problem(
    revenue_function=revenue_function,
    reservation_utility=reservation_utility,
    solver="iterative",
    a_min=0.0,
    a_max=0.35,
    a_init=0.0,
    a_ic_lb=0.0,
    a_ic_ub=0.35,
    a_ic_initial=0.0
)

print(result_cmp)
print(result_principal)

print("Optimal action: ", result_principal.optimal_action)
print("Expected wage: ",result_principal.constraints["Ewage"])
print("Optimal profits: ",result_principal.profit)

v = result_principal.optimal_contract
w = mhp.k(v)
density = mhp._primitives["f"](mhp.y_grid, result_principal.optimal_action)

# Plot results
plt.plot(mhp.y_grid, w)
plt.show()
plt.plot(mhp.y_grid, density)
plt.show()