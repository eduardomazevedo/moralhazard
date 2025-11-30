import os
import numpy as np
import pandas as pd
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
reservation_wages = np.linspace(-1.0, 100.0, 5)

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
# Output DataFrames
# --------------------
cols = [
    "reservation_wage",
    "lam",
    "mu",
    "mu_hat",
    "cost",
]

records = []

# --------------------
# Solve for each reservation wage
# --------------------
for w in reservation_wages:
    # --- Cost-minimization problem ---
    cm = mhp.solve_cost_minimization_problem(
        intended_action=first_best_effort,
        reservation_utility=u_fun(w),
        a_ic_lb=a_ic_lb,
        a_ic_ub=a_ic_ub,
        n_a_iterations=5
    )

    cm_mult = cm.multipliers
    cm_lam = cm_mult["lam"]
    cm_mu = cm_mult["mu"]
    cm_mu_hat = cm_mult["mu_hat"]

    records.append({
        "results": cm,
        "reservation_wage": w,
        "cost": cm.expected_wage,
        "lam": cm_lam,
        "mu": cm_mu,
        "mu_hat": cm_mu_hat,
        "optimal_contract": cm.optimal_contract,
    })

# --------------------
# Save DataFrames
# --------------------
# Create Plots
# --------------------
os.makedirs("./diagnostics/figures", exist_ok=True)

# Get grids
y_grid = mhp.y_grid
a_grid = np.linspace(0, 140, 100)

# Get colormap for reservation wages
w_min = min(record["reservation_wage"] for record in records)
w_max = max(record["reservation_wage"] for record in records)
cmap = plt.cm.viridis

# Plot 1: Principal - Wage schedule k(v*(y)) vs y
fig, ax = plt.subplots(figsize=(8, 5))
for record in records:
    w = record["reservation_wage"]
    v = record["optimal_contract"]
    wage = mhp.k(v)
    color = cmap((w - w_min) / (w_max - w_min))
    ax.plot(y_grid, wage, color=color, alpha=0.6, linewidth=0.5)
ax.set_xlabel("Output ($y$)")
ax.set_ylabel("Wage")
ax.set_title("Cost Minimization: Wage Schedule $k(v^*(y))$ vs $y$")
plt.colorbar(
    plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=w_min, vmax=w_max)),
    ax=ax,
    label="Reservation Wage"
)
plt.tight_layout()
plt.savefig("./diagnostics/figures/cost_minimization_wage_schedule_vs_y.png")
plt.close()

# Plot 2: Principal - Utility function U(v, a) vs a
# Check IC constraint: max U(v, a) should not exceed U(v, a0)
print("Checking IC constraint violations...")
print("=" * 60)
problem_cases = []

fig, ax = plt.subplots(figsize=(8, 5))
for record in records:
    w = record["reservation_wage"]
    v = record["optimal_contract"]
    U_grid = mhp.U(v, a_grid)
    
    # Check IC constraint: U(v, a0) should be >= max U(v, a) for all a
    U_at_intended = float(np.asarray(mhp.U(v, first_best_effort)).item())
    U_max = float(np.max(U_grid))
    
    if U_max > U_at_intended:
        problem_cases.append({
            "reservation_wage": w,
            "U_at_intended": U_at_intended,
            "U_max": U_max,
            "violation": U_max - U_at_intended,
            "argmax_a": float(a_grid[np.argmax(U_grid)]),
            "results": record["results"]
        })
    
    color = cmap((w - w_min) / (w_max - w_min))
    ax.plot(a_grid, U_grid, color=color, alpha=0.6, linewidth=0.5)

# Print all problem cases
if problem_cases:
    print(f"Found {len(problem_cases)} cases with IC constraint violations:\n")
    for i, case in enumerate(problem_cases, 1):
        print(f"--- Problem Case {i} ---")
        print(f"Reservation wage: {case['reservation_wage']:.6f}")
        print(f"  U(v, a0={first_best_effort}): {case['U_at_intended']:.6f}")
        print(f"  max U(v, a): {case['U_max']:.6f}")
        print(f"  Violation: {case['violation']:.6e}")
        print(f"  Optimal action (argmax): {case['argmax_a']:.6f}")
        print()
        
        # Print full results
        results = case["results"]
        print("  Full solve results:")
        print(f"    a0 (intended action): {results.a0}")
        print(f"    Ubar (reservation utility): {results.Ubar}")
        print(f"    a_hat: {results.a_hat}")
        print(f"    Expected wage: {results.expected_wage:.6f}")
        print(f"    Multipliers:")
        print(f"      λ (lam): {results.multipliers['lam']:.6f}")
        print(f"      μ (mu): {results.multipliers['mu']:.6f}")
        print(f"      μ̂ (mu_hat): {results.multipliers['mu_hat']}")
        print(f"    Constraints:")
        print(f"      U0: {results.constraints['U0']:.6f}")
        print(f"      IR: {results.constraints['IR']:.6e}")
        print(f"      FOC: {results.constraints['FOC']:.6e}")
        print(f"      Uhat: {results.constraints['Uhat']}")
        print(f"      IC: {results.constraints['IC']}")
        print(f"      Ewage: {results.constraints['Ewage']:.6f}")
        print(f"    Solver state: {results.solver_state}")
        print()
else:
    print("No IC constraint violations found.")
print("=" * 60)
print()
ax.set_xlabel("Action ($a$)")
ax.set_ylabel("Utility")
ax.set_title("Cost Minimization: Utility $U(v, a)$ vs $a$")
plt.colorbar(
    plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=w_min, vmax=w_max)),
    ax=ax,
    label="Reservation Wage"
)
plt.tight_layout()
plt.savefig("./diagnostics/figures/cost_minimization_utility_vs_a.png")
plt.close()