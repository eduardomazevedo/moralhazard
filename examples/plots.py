import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from moralhazard import MoralHazardProblem

# -------------------------------
# Model primitives (Normal model)
# -------------------------------
x0 = 50
sigma = 10.0
first_best_effort = 100
theta = 1.0 / first_best_effort / (first_best_effort + x0)

def u(c): 
    return np.log(x0 + c)

Ubar = float(u(0.0) - 10)  # same reservation utility as quickstart
a_max = 150.0

def k(utils): 
    return np.exp(utils) - x0

def g(z): 
    return np.log(np.maximum(z, x0))

def C(a): 
    return theta * a ** 2 / 2

def Cprime(a): 
    return theta * a

def f(y, a):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((y - a) ** 2) / (2 * sigma ** 2))

def score(y, a):
    return (y - a) / (sigma ** 2)

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
        "y_min": 0.0 - 3 * sigma,
        "y_max": a_max + 3 * sigma,
        "n": 201,
    },
}

mhp = MoralHazardProblem(cfg)

# Principal revenue: original payoff was a - E[w], so revenue(a) = a
revenue = lambda a: float(a)

# Grids & shared objects
a_grid = np.linspace(0.0, a_max, 121)
a_hat = np.zeros(2)  # only needed for the a_hat solver
y = mhp.y_grid

# ---------------------------------------
# Solve principal's outer problem (a*)
# ---------------------------------------

# a_hat solver
res_a_hat = mhp.solve_principal_problem(
    revenue_function=revenue,
    reservation_utility=Ubar,
    a_min=0.0,
    a_max=a_max,
    a_init=first_best_effort,
    solver="a_hat",
    a_hat=a_hat,
)
a_star_a_hat = float(res_a_hat.optimal_action)
v_star_a_hat = res_a_hat.optimal_contract

# iterative solver
res_iter = mhp.solve_principal_problem(
    revenue_function=revenue,
    reservation_utility=Ubar,
    a_min=0.0,
    a_max=a_max,
    a_init=first_best_effort,
    solver="iterative",
    n_a_iterations=3,
    clip_ratio=1,
)
a_star_iter = float(res_iter.optimal_action)
v_star_iter = res_iter.optimal_contract

# Expected wage curves and payoff curves for plotting
F_a_hat = mhp.expected_wage_fun(
    reservation_utility=Ubar, solver="a_hat", a_hat=a_hat, warm_start=True
)
Ew_a_hat = np.array([F_a_hat(float(a)) for a in a_grid])
payoff_a_hat = a_grid - Ew_a_hat

F_iter = mhp.expected_wage_fun(
    reservation_utility=Ubar, solver="iterative", warm_start=True, n_a_iterations=3, clip_ratio=1
)
Ew_iter = np.array([F_iter(float(a)) for a in a_grid])
payoff_iter = a_grid - Ew_iter

# -----------------
# Output directory
# -----------------
out_dir = Path("examples/output/plots")
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------
# Plot 1: Wages
# ---------------
plt.figure(figsize=(10, 6))
plt.plot(y, mhp.k(v_star_a_hat), linewidth=2, label=f'a_hat solver (a* = {a_star_a_hat:.2f})', color='blue')
plt.plot(y, mhp.k(v_star_iter), linewidth=2, label=f'iterative solver (a* = {a_star_iter:.2f})', color='red', linestyle='--')
plt.xlabel("Outcome y")
plt.ylabel("Wage k(v*(y))")
plt.title("Optimal wage schedules comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "wages.png", dpi=200, bbox_inches="tight")
plt.close()

# -------------------
# Plot 2: Utilities
# -------------------
plt.figure(figsize=(10, 6))
Ua_a_hat = mhp.U(v_star_a_hat, a_grid)
Ua_iter = mhp.U(v_star_iter, a_grid)
plt.plot(a_grid, Ua_a_hat, linewidth=2, label=f'a_hat solver (a* = {a_star_a_hat:.2f})', color='blue')
plt.plot(a_grid, Ua_iter, linewidth=2, label=f'iterative solver (a* = {a_star_iter:.2f})', color='red', linestyle='--')
plt.axvline(a_star_a_hat, linestyle=":", color='blue', alpha=0.7, label=f'a_hat a* = {a_star_a_hat:.2f}')
plt.axvline(a_star_iter, linestyle=":", color='red', alpha=0.7, label=f'iterative a* = {a_star_iter:.2f}')
plt.xlabel("Action a")
plt.ylabel("U(a)")
plt.title("U(a) under optimal contracts comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "utilities.png", dpi=200, bbox_inches="tight")
plt.close()

# -------------------------
# Plot 3: Expected wages
# -------------------------
plt.figure(figsize=(10, 6))
plt.plot(a_grid, Ew_a_hat, linewidth=2, label=f'a_hat solver (a* = {a_star_a_hat:.2f})', color='blue')
plt.plot(a_grid, Ew_iter, linewidth=2, label=f'iterative solver (a* = {a_star_iter:.2f})', color='red', linestyle='--')
plt.axvline(a_star_a_hat, linestyle=":", color='blue', alpha=0.7, label=f'a_hat a* = {a_star_a_hat:.2f}')
plt.axvline(a_star_iter, linestyle=":", color='red', alpha=0.7, label=f'iterative a* = {a_star_iter:.2f}')
plt.xlabel("Action a")
plt.ylabel("Expected wage E[w]")
plt.title("Expected wages comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "expected_wages.png", dpi=200, bbox_inches="tight")
plt.close()

# ------------------------------
# Plot 4: Principal's payoff
# ------------------------------
plt.figure(figsize=(10, 6))
plt.plot(a_grid, payoff_a_hat, linewidth=2, label=f'a_hat solver (a* = {a_star_a_hat:.2f})', color='blue')
plt.plot(a_grid, payoff_iter, linewidth=2, label=f'iterative solver (a* = {a_star_iter:.2f})', color='red', linestyle='--')
plt.axvline(a_star_a_hat, linestyle=":", color='blue', alpha=0.7, label=f'a_hat a* = {a_star_a_hat:.2f}')
plt.axvline(a_star_iter, linestyle=":", color='red', alpha=0.7, label=f'iterative a* = {a_star_iter:.2f}')
plt.xlabel("Action a")
plt.ylabel("Principal payoff: a - E[w]")
plt.title("Principal's payoff comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "payoff.png", dpi=200, bbox_inches="tight")
plt.close()
