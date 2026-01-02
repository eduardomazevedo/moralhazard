import os
import numpy as np
import matplotlib.pyplot as plt

from moralhazard import MoralHazardProblem
from moralhazard.config_maker import make_utility_cfg, make_distribution_cfg

# --------------------
# Primitives
# --------------------
initial_wealth = 50
sigma = 10.0
first_best_effort = 100
theta = 1.0 / first_best_effort / (first_best_effort + initial_wealth)

utility_cfg = make_utility_cfg("log", w0=initial_wealth)
dist_cfg = make_distribution_cfg("gaussian", sigma=sigma)

def C(a):
    return theta * a**2 / 2

def Cprime(a):
    return theta * a

computational_params = {
    "distribution_type": "continuous",
    "y_min": 0.0 - 3 * sigma,
    "y_max": 180.0 + 3 * sigma,
    "n": 201,
}

cfg = {
    "problem_params": {
        **utility_cfg,
        **dist_cfg,
        "C": C,
        "Cprime": Cprime,
    },
    "computational_params": computational_params,
}

mhp = MoralHazardProblem(cfg)
u_fun = cfg["problem_params"]["u"]

a_ic_lb = 0.0
a_ic_ub = 130.0

# --------------------
# Grids
# --------------------
# Action grid: integers from 0 to 100
a_grid = np.arange(0, 101, dtype=float)

# Reservation wage grid: integers from -1 to 10
reservation_wages = np.arange(-1, 11, dtype=float)

# --------------------
# Compute minimum_cost for each combination
# --------------------
print("Computing minimum_cost for each action and reservation wage...")
print(f"Actions: {len(a_grid)} values from {a_grid[0]} to {a_grid[-1]}")
print(f"Reservation wages: {len(reservation_wages)} values from {reservation_wages[0]} to {reservation_wages[-1]}")
print()

# Convert reservation wages to utilities
reservation_utilities = [u_fun(w) for w in reservation_wages]

# --------------------
# Create Plot
# --------------------
os.makedirs("./diagnostics/figures", exist_ok=True)

# Get colormap for reservation wages
w_min = float(reservation_wages[0])
w_max = float(reservation_wages[-1])
cmap = plt.cm.viridis

fig, ax = plt.subplots(figsize=(10, 6))

converged_count = 0
failed_count = 0
failed_cases = []

for i, (w, Ubar) in enumerate(zip(reservation_wages, reservation_utilities)):
    print(f"Computing for reservation wage {w:.0f} (Ubar = {Ubar:.4f})...")
    
    costs = []
    successful_actions = []
    
    # Compute for each action individually to track failures
    for a in a_grid:
        try:
            cost = mhp.minimum_cost(
                intended_action=a,
                reservation_utility=Ubar,
                a_ic_lb=a_ic_lb,
                a_ic_ub=a_ic_ub,
            )
            costs.append(cost)
            successful_actions.append(a)
            converged_count += 1
        except (RuntimeError, ValueError) as e:
            failed_count += 1
            failed_cases.append((w, a, str(e)))
            continue
    
    # Only plot if we have at least some successful actions
    if successful_actions:
        costs_array = np.array(costs)
        actions_array = np.array(successful_actions)
        color = cmap((w - w_min) / (w_max - w_min))
        ax.plot(actions_array, costs_array, color=color, alpha=0.7, linewidth=1.5, label=f"w={w:.0f}")

ax.set_xlabel("Action ($a$)")
ax.set_ylabel("Minimum Expected Wage $E[w(v^*(a))]$")
ax.set_title("Minimum Cost: Expected Wage vs Action for Different Reservation Wages")
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=8)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=w_min, vmax=w_max))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Reservation Wage", rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig("./diagnostics/figures/minimum_cost_vs_action.png", dpi=150, bbox_inches='tight')
print(f"\nPlot saved to ./diagnostics/figures/minimum_cost_vs_action.png")
plt.close()

print(f"\nSummary:")
print(f"  Successful computations: {converged_count}")
print(f"  Failed computations: {failed_count}")
total_combinations = len(reservation_wages) * len(a_grid)
print(f"  Total combinations: {total_combinations}")

if failed_cases:
    print(f"\nFailed cases (reservation wage, action, error):")
    for w, a, error in failed_cases:
        print(f"  (w={w:.0f}, a={a:.0f}): {error}")

print("\nDone!")
