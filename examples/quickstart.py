# Quickstart example
import numpy as np
import matplotlib.pyplot as plt
from moralhazard import MoralHazardProblem
from moralhazard.config_maker import make_utility_cfg, make_distribution_cfg

# ---- primitives (same as prototype Normal model) ----
x0 = 50
sigma = 10.0
first_best_effort = 100
theta = 1.0 / first_best_effort / (first_best_effort + x0)

def u(c): return np.log(x0 + c)
def k(utils): return np.exp(utils) - x0
def g(z): return np.log(np.maximum(z, x0))
def C(a): return theta * a ** 2 / 2
def Cprime(a): return theta * a
def f(y, a):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((y - a) ** 2) / (2 * sigma ** 2))
def score(y, a):
    return (y - a) / (sigma ** 2)

reservation_utility = u(0) - 5.0

# ---- configuration ----
cfg = {
    "problem_params": {
        "u": u,
        "k": k,
        "link_function": g,
        "C": C,
        "Cprime": Cprime,
        "f": f,
        "score": score,
    },
    "computational_params": {
        "distribution_type": "continuous",
        "y_min": 0.0   - 3 * sigma,
        "y_max": 180.0 + 3 * sigma,
        "n": 201,  # must be odd
    },
}

# ---- alternative: using config maker ----
# Create utility functions (log utility with baseline wealth x0)
utility_cfg = make_utility_cfg("log", w0=x0)
# Create distribution functions (gaussian with sigma)
dist_cfg = make_distribution_cfg("gaussian", sigma=sigma)

# Alternative configuration using config maker
cfg_alt = {
    "problem_params": {
        **utility_cfg,  # u, k, link_function
        **dist_cfg,     # f, score
        "C": C,
        "Cprime": Cprime,
    },
    "computational_params": cfg["computational_params"],
}

# ---- cost minimization problem ----
mhp = MoralHazardProblem(cfg)
results = mhp.solve_cost_minimization_problem(
    intended_action=80.0,
    reservation_utility=reservation_utility,
    a_ic_lb=0.0,
    a_ic_ub=100.0
)

print("Cost minimization problem results:")
print(results)

# ---- principal problem ----
results_principal = mhp.solve_principal_problem(
    revenue_function=lambda a: a,
    reservation_utility=reservation_utility,
    a_min=0.0,
    a_max=100.0,
    a_ic_lb=0.0,
    a_ic_ub=100.0
)

print("Principal problem results:")
print(results_principal)