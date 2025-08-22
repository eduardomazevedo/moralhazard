import numpy as np
import matplotlib.pyplot as plt
from moral_hazard import MoralHazardProblem

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
        "y_min": 0.0   - 3 * sigma,
        "y_max": 120.0 + 3 * sigma,
        "n": 201,  # must be odd
    },
}

# ---- solve once ----
mhp = MoralHazardProblem(cfg)
results = mhp.solve_cost_minimization_problem(
    intended_action=80.0,
    reservation_utility=u(50),
    a_hat=np.array([0.0, 0.0]),
)

print("Multipliers found:")
print(results.multipliers)

# ---- plots ----

# 1) Wage schedule k(v*(y)) vs y
y_grid = mhp.y_grid
v = results.optimal_contract            # utils on the grid
wage = mhp.k(v)                         # dollars on the grid
