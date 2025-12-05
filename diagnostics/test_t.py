"""
Testing the t distribution - Cost Minimization Problem
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
from moralhazard import MoralHazardProblem
from moralhazard.config_maker import make_utility_cfg, make_distribution_cfg

os.makedirs('diagnostics/figures', exist_ok=True)

# -------------------------------------------------------------
# Parameters
# -------------------------------------------------------------
initial_wealth = 50
sigma = 10.0
first_best_effort = 100
theta = 1.0 / first_best_effort / (first_best_effort + initial_wealth)
intended_action = 100.0
reservation_wage = 100.0

def C(a): return theta * a ** 2 / 2
def Cprime(a): return theta * a

utility_cfg = make_utility_cfg("log", w0=initial_wealth)
dist_cfg = make_distribution_cfg("Student_t", nu=2.0, sigma=sigma)

comp_cfg = {
    "distribution_type": "continuous",
    "y_min": - 1000.0,
    "y_max": 1000.0,
    "n": 2001,  # must be odd
}

cfg = {
    "problem_params": {**utility_cfg, **dist_cfg, "C": C, "Cprime": Cprime},
    "computational_params": comp_cfg
}

a_min, a_max = 0.0, 200.0
a_ic_lb, a_ic_ub = a_min, a_max
action_grid_plot = np.linspace(a_min, a_max, 100)

# -------------------------------------------------------------
# Solve Cost Minimization Problem
# -------------------------------------------------------------
mhp = MoralHazardProblem(cfg)
reservation_utility = utility_cfg["u"](reservation_wage)

sol = mhp.solve_cost_minimization_problem(
    intended_action=intended_action,
    reservation_utility=reservation_utility,
    a_ic_lb=a_ic_lb,
    a_ic_ub=a_ic_ub
)

# Print solution
print("\n" + "="*60)
print("COST MINIMIZATION PROBLEM SOLUTION")
print("="*60)
print(sol)
print("="*60 + "\n")

# -------------------------------------------------------------
# Plot Results
# -------------------------------------------------------------
y_grid = mhp._y_grid
optimal_contract = sol.optimal_contract
wage_function = mhp.k(optimal_contract)
agent_utility = mhp.U(optimal_contract, action_grid_plot)
utility_at_intended = mhp.U(optimal_contract, intended_action)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Wage function
ax1.plot(y_grid, wage_function, linewidth=2, color='steelblue')
ax1.set_xlabel("Output (y)")
ax1.set_ylabel("Wage")
ax1.set_title("Optimal Wage Function")
ax1.grid(True, alpha=0.3)

# Agent utility
ax2.plot(action_grid_plot, agent_utility, linewidth=2, color='steelblue')
ax2.scatter(intended_action, utility_at_intended, color='red', s=50, zorder=5, label='Intended action')
ax2.axvline(x=intended_action, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax2.axhline(y=utility_at_intended, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_xlabel("Action")
ax2.set_ylabel("Agent Utility")
ax2.set_title("Agent Utility vs Action")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig("diagnostics/figures/t_cmp.png", dpi=300)
plt.close()

print(f"Figure saved to: diagnostics/figures/t_cmp.png")
