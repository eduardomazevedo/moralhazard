# Comparing VC vs public company CEO compensation
import numpy as np
import matplotlib.pyplot as plt
from moralhazard import MoralHazardProblem
from moralhazard.config_maker import make_utility_cfg, make_distribution_cfg

# Basics
S = 1e1
ln_sigma = 2.3
w0 = 1
agent_max_impact = np.log(10) # At most multiplies the firm value by this

expected_firm_value = lambda a: S * np.exp(ln_sigma**2/2) * np.exp(agent_max_impact * a)
expected_revenue = lambda a: expected_firm_value(a) - expected_firm_value(0)
median_firm_value = lambda a: S * np.exp(agent_max_impact * a)


# Cost function
c = lambda a: - np.log(1 - a)
mc = lambda a: 1 / (1 - a)

# Create cfgs
utility_cfg = make_utility_cfg("log", w0=w0)
# Create distribution functions (gaussian with sigma)
dist_cfg = make_distribution_cfg("gaussian", sigma=ln_sigma)

# Reservation utility
u = utility_cfg["u"]
reservation_utility = u(w0)

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
        "y_max": agent_max_impact + 4 * ln_sigma,
        "n": 201,  # must be odd
    }
}

mhp = MoralHazardProblem(cfg)

result_principal = mhp.solve_principal_problem(
    revenue_function=expected_revenue,
    reservation_utility=reservation_utility,
    solver="a_hat",
    a_hat=[0.0],
    a_min=0.0,
    a_max=0.7,
    a_init=0.0
)

print(result_principal)

print("Optimal action: ", result_principal.optimal_action)
print("Expected wage: ",result_principal.constraints["Ewage"])
print("Optimal profits: ",result_principal.profit)

a0 = result_principal.optimal_action
v = result_principal.optimal_contract
w = mhp.k(v)
density = mhp._primitives["f"](mhp.y_grid, result_principal.optimal_action)

id_mid = abs(mhp.y_grid - agent_max_impact * a0) < 3 * ln_sigma
y_mid = mhp.y_grid[id_mid]
firm_val = S * np.exp(y_mid)

plt.plot(y_mid, w[id_mid])
plt.title("Wage vs y")
plt.show()

plt.plot(y_mid, S * np.exp(y_mid))
plt.title("Firm value vs y")
plt.show()

plt.plot(y_mid, density[id_mid])
plt.title("Density vs y")
plt.show()



plt.plot(firm_val, w[id_mid])
plt.title("Wage vs firm value")
plt.show()

plt.plot(firm_val, w[id_mid] / firm_val)
plt.xscale('log')
plt.title("Founder share vs firm value")
plt.show()

a_grid = np.linspace(0, 1, 100)
plt.plot(a_grid, mhp.U(v, a_grid))
plt.title("Utility vs action")
plt.show()