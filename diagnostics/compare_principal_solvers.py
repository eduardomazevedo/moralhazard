# Deep dive: Compare principal problem solvers (Dual vs CVXPY)
import os
import numpy as np
import matplotlib.pyplot as plt

from moralhazard import MoralHazardProblem
from moralhazard.config_maker import make_utility_cfg, make_distribution_cfg

# ---- Setup (same as test) ----
x0 = 50.0
sigma = 40.0
first_best_effort = 100.0
theta = 1.0 / first_best_effort / (first_best_effort + x0)

utility_cfg = make_utility_cfg("log", w0=x0)
dist_cfg = make_distribution_cfg("gaussian", sigma=sigma)

def C(a):
    return theta * a ** 2 / 2

def Cprime(a):
    return theta * a

cfg = {
    "problem_params": {
        **utility_cfg,
        **dist_cfg,
        "C": C,
        "Cprime": Cprime,
    },
    "computational_params": {
        "distribution_type": "continuous",
        "y_min": 0.0 - 3 * sigma,
        "y_max": 100.0 + 3 * sigma,
        "n": 51,
    },
}

problem = MoralHazardProblem(cfg)
u_fun = problem._primitives["u"]
reservation_utility = u_fun(0) - 5.0

# ---- Action grid ----
a_grid = np.linspace(0.0, 100.0, 100)

print("=" * 60)
print("Comparing Principal Problem Solvers: Dual vs CVXPY")
print("=" * 60)
print(f"Reservation utility: {reservation_utility:.4f}")
print(f"Action grid: {len(a_grid)} points from {a_grid[0]} to {a_grid[-1]}")
print()

# ---- Compute expected wages for CVXPY ----
print("Computing CVXPY expected wages (batch)...")
expected_wages_cvxpy = problem.minimum_cost_cvxpy(
    intended_actions=a_grid,
    reservation_utility=reservation_utility,
    a_hat=a_grid,
)
print(f"  Done. Shape: {expected_wages_cvxpy.shape}")

# ---- Compute expected wages for Dual ----
print("Computing Dual expected wages (individual)...")
expected_wages_dual = []
for i, a in enumerate(a_grid):
    if i % 10 == 0:
        print(f"  Action {i+1}/{len(a_grid)}: a={a:.2f}")
    ew = problem.minimum_cost(
        intended_action=a,
        reservation_utility=reservation_utility,
        a_ic_lb=0.0,
        a_ic_ub=100.0,
        n_a_iterations=100,
    )
    expected_wages_dual.append(ew)
expected_wages_dual = np.array(expected_wages_dual)
print(f"  Done. Shape: {expected_wages_dual.shape}")

# ---- Revenue function ----
revenue = a_grid  # Linear revenue: R(a) = a

# ---- Compute profits ----
profits_cvxpy = revenue - expected_wages_cvxpy
profits_dual = revenue - expected_wages_dual

# ---- Find optimal actions ----
idx_cvxpy = np.argmax(profits_cvxpy)
idx_dual = np.argmax(profits_dual)

a_star_cvxpy = a_grid[idx_cvxpy]
a_star_dual = a_grid[idx_dual]
profit_star_cvxpy = profits_cvxpy[idx_cvxpy]
profit_star_dual = profits_dual[idx_dual]

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"CVXPY:")
print(f"  Optimal action: {a_star_cvxpy:.4f}")
print(f"  Profit: {profit_star_cvxpy:.4f}")
print(f"  Expected wage at optimum: {expected_wages_cvxpy[idx_cvxpy]:.4f}")
print()
print(f"Dual:")
print(f"  Optimal action: {a_star_dual:.4f}")
print(f"  Profit: {profit_star_dual:.4f}")
print(f"  Expected wage at optimum: {expected_wages_dual[idx_dual]:.4f}")
print()
print(f"Differences:")
print(f"  Action diff: {abs(a_star_cvxpy - a_star_dual):.4f}")
print(f"  Profit diff: {abs(profit_star_cvxpy - profit_star_dual):.4f}")

# ---- Also run the actual principal problem solvers for comparison ----
print()
print("Running actual principal problem solvers...")

result_cvxpy = problem.solve_principal_problem_cvxpy(
    revenue_function=lambda a: a,
    reservation_utility=reservation_utility,
    discretized_a_grid=a_grid,
)

result_dual = problem.solve_principal_problem(
    revenue_function=lambda a: a,
    reservation_utility=reservation_utility,
    a_min=0.0,
    a_max=100.0,
    a_ic_lb=0.0,
    a_ic_ub=100.0,
    n_a_iterations=100,
)

print(f"\nActual solver results:")
print(f"  CVXPY: a*={result_cvxpy.optimal_action:.4f}, profit={result_cvxpy.profit:.4f}")
print(f"  Dual:  a*={result_dual.optimal_action:.4f}, profit={result_dual.profit:.4f}")

# ---- Create plots ----
os.makedirs("./diagnostics/figures", exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Expected wage vs action
ax1 = axes[0, 0]
ax1.plot(a_grid, expected_wages_cvxpy, 'b-', label='CVXPY', linewidth=2)
ax1.plot(a_grid, expected_wages_dual, 'r--', label='Dual', linewidth=2)
ax1.axvline(a_star_cvxpy, color='b', linestyle=':', alpha=0.7, label=f'CVXPY a*={a_star_cvxpy:.1f}')
ax1.axvline(a_star_dual, color='r', linestyle=':', alpha=0.7, label=f'Dual a*={a_star_dual:.1f}')
ax1.set_xlabel('Action (a)')
ax1.set_ylabel('Expected Wage E[w]')
ax1.set_title('Expected Wage vs Action')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Revenue, wage, and profit
ax2 = axes[0, 1]
ax2.plot(a_grid, revenue, 'g-', label='Revenue R(a)=a', linewidth=2)
ax2.plot(a_grid, expected_wages_cvxpy, 'b-', label='E[w] CVXPY', linewidth=1.5)
ax2.plot(a_grid, expected_wages_dual, 'r--', label='E[w] Dual', linewidth=1.5)
ax2.axvline(a_star_cvxpy, color='b', linestyle=':', alpha=0.7)
ax2.axvline(a_star_dual, color='r', linestyle=':', alpha=0.7)
ax2.set_xlabel('Action (a)')
ax2.set_ylabel('Value')
ax2.set_title('Revenue and Expected Wage')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Profit vs action
ax3 = axes[1, 0]
ax3.plot(a_grid, profits_cvxpy, 'b-', label='Profit CVXPY', linewidth=2)
ax3.plot(a_grid, profits_dual, 'r--', label='Profit Dual', linewidth=2)
ax3.axvline(a_star_cvxpy, color='b', linestyle=':', alpha=0.7)
ax3.axvline(a_star_dual, color='r', linestyle=':', alpha=0.7)
ax3.scatter([a_star_cvxpy], [profit_star_cvxpy], color='b', s=100, zorder=5, marker='*')
ax3.scatter([a_star_dual], [profit_star_dual], color='r', s=100, zorder=5, marker='*')
ax3.set_xlabel('Action (a)')
ax3.set_ylabel('Profit = R(a) - E[w]')
ax3.set_title('Profit vs Action')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Wage difference (CVXPY - Dual)
ax4 = axes[1, 1]
wage_diff = expected_wages_cvxpy - expected_wages_dual
ax4.plot(a_grid, wage_diff, 'k-', linewidth=2)
ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax4.axvline(a_star_cvxpy, color='b', linestyle=':', alpha=0.7, label=f'CVXPY a*')
ax4.axvline(a_star_dual, color='r', linestyle=':', alpha=0.7, label=f'Dual a*')
ax4.set_xlabel('Action (a)')
ax4.set_ylabel('E[w]_CVXPY - E[w]_Dual')
ax4.set_title('Expected Wage Difference (CVXPY - Dual)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Add summary text
summary_text = (
    f"CVXPY: a*={a_star_cvxpy:.2f}, profit={profit_star_cvxpy:.4f}\n"
    f"Dual:  a*={a_star_dual:.2f}, profit={profit_star_dual:.4f}\n"
    f"Profit diff: {abs(profit_star_cvxpy - profit_star_dual):.4f}"
)
fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig('./diagnostics/figures/compare_principal_solvers.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to ./diagnostics/figures/compare_principal_solvers.png")
plt.close()

print("\nDone!")
