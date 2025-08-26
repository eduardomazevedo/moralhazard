import numpy as np
import matplotlib.pyplot as plt
from moralhazard import MoralHazardProblem
from moralhazard.config_maker import make_utility_cfg, make_distribution_cfg

# ---- primitives (same as prototype Normal model) ----
x0 = 50
sigma = 10.0
first_best_effort = 100
theta = 1.0 / first_best_effort / (first_best_effort + x0)

# ---- create utility functions using config_maker ----
utility_cfg = make_utility_cfg("log", w0=x0)
u = utility_cfg["u"]
k = utility_cfg["k"]
g = utility_cfg["link_function"]

reservation_utility = u(0.0)

# ---- create distribution functions using config_maker ----
dist_cfg = make_distribution_cfg("gaussian", sigma=sigma)
f = dist_cfg["f"]
score = dist_cfg["score"]

# ---- cost functions (same as quickstart) ----
def C(a): return theta * a ** 2 / 2
def Cprime(a): return theta * a

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
        "y_max": 120.0 + 3 * sigma,
        "n": 201,  # must be odd
    },
}

# ---- solve once ----
mhp = MoralHazardProblem(cfg)
results = mhp.solve_cost_minimization_problem(
    intended_action=80.0,
    reservation_utility=reservation_utility,
    a_hat=np.array([0.0, 0.0]),
)

print("Multipliers found:")
print(results.multipliers)

# ---- plots ----

# 1) Wage schedule k(v*(y)) vs y
y_grid = mhp.y_grid
v = results.optimal_contract            # utils on the grid
wage = mhp.k(v)                         # dollars on the grid

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(y_grid, wage)
plt.xlabel('Output y')
plt.ylabel('Wage k(v*(y))')
plt.title('Optimal Wage Schedule')
plt.grid(True, alpha=0.3)

# 2) Utility schedule v*(y) vs y
plt.subplot(2, 2, 2)
plt.plot(y_grid, v)
plt.xlabel('Output y')
plt.ylabel('Utility v*(y)')
plt.title('Optimal Utility Schedule')
plt.grid(True, alpha=0.3)

# 3) Distribution f(y|a) for intended action
plt.subplot(2, 2, 3)
intended_action = 80.0
f_values = f(y_grid, intended_action)
plt.plot(y_grid, f_values)
plt.xlabel('Output y')
plt.ylabel('f(y|a)')
plt.title(f'Distribution f(y|a={intended_action})')
plt.grid(True, alpha=0.3)

# 4) Score function score(y, a) for intended action
plt.subplot(2, 2, 4)
score_values = score(y_grid, intended_action)
plt.plot(y_grid, score_values)
plt.xlabel('Output y')
plt.ylabel('score(y, a)')
plt.title(f'Score Function score(y, a={intended_action})')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nExample completed successfully!")
print(f"Used config_maker to create:")
print(f"  - Utility functions: {list(utility_cfg.keys())}")
print(f"  - Distribution functions: {list(dist_cfg.keys())}")
print(f"  - All functions are broadcastable and handle array inputs")
