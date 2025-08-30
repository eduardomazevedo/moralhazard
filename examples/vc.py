# Comparing VC vs public company CEO compensation
import numpy as np
import matplotlib.pyplot as plt
from moralhazard import MoralHazardProblem
from moralhazard.config_maker import make_utility_cfg, make_distribution_cfg

# Basics
S_tilde = 1e3
ln_sigma = 0.94 # 0.94 vs 2.3 for 10 years of public company vs vc
S = S_tilde * np.exp(ln_sigma**2/2)
w0 = 50

a_left = -1
kappa = 0.1

# Cost function
def c(a):
    return kappa * (np.log(1 - np.exp(a_left)) - np.log(1 - np.exp(a)))

def mc(a):
    return kappa * (np.exp(a) / (1 - np.exp(a)))

revenue_function = lambda a: S * np.exp(a)

# Create cfgs
utility_cfg = make_utility_cfg("log", w0=w0)
# Create distribution functions (gaussian with sigma)
dist_cfg = make_distribution_cfg("gaussian", sigma=ln_sigma)

# Reservation utility
u = utility_cfg["u"]
reservation_utility = u(w0) # So agent could get another w0 besides her baseline consumption working somewhere else.

cfg = {
    "problem_params": {
        **utility_cfg,
        **dist_cfg,
        "C": c,
        "Cprime": mc,
    },
    "computational_params": {
        "distribution_type": "continuous",
        "y_min": a_left- 4 * ln_sigma,
        "y_max": 4 * ln_sigma,
        "n": 201,  # must be odd
    }
}

mhp = MoralHazardProblem(cfg)

result_principal = mhp.solve_principal_problem(
    revenue_function=revenue_function,
    reservation_utility=reservation_utility,
    solver="a_hat",
    a_hat=[a_left],
    a_min=a_left,
    a_max=0,
    a_init=a_left
)

print(result_principal)

print("Optimal action: ", result_principal.optimal_action)
print("Expected wage: ",result_principal.constraints["Ewage"])
print("Optimal profits: ",result_principal.profit)

a0 = result_principal.optimal_action
v = result_principal.optimal_contract
w = mhp.k(v)
density = mhp._primitives["f"](mhp.y_grid, result_principal.optimal_action)

id_mid = abs(mhp.y_grid - a0) < 2 * ln_sigma
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

a_grid = np.linspace(a_left, -0.1, 100)
plt.plot(a_grid, mhp.U(v, a_grid))
plt.title("Utility vs action")
plt.show()