import numpy as np
import matplotlib.pyplot as plt
from moralhazard import MoralHazardProblem

# ---- primitives (same as prototype Normal model) ----
x0 = 50
sigma = 10.0
first_best_effort = 100
theta = 1.0 / first_best_effort / (first_best_effort + x0)

def u(c): return np.log(x0 + c)

Ubar = float(u(0) - 10)  # same reservation utility as quickstart
a_max = 150.0

def k(utils): return np.exp(utils) - x0
def g(z): return np.log(np.maximum(z, x0))
def C(a): return theta * a ** 2 / 2
def Cprime(a): return theta * a
def f(y, a):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((y - a) ** 2) / (2 * sigma ** 2))
def score(y, a):
    return (y - a) / (sigma ** 2)

cfg = {
    "problem_params": {"u": u, "k": k, "link_function": g, "C": C, "Cprime": Cprime, "f": f, "score": score},
    "computational_params": {"y_min": 0.0 - 3 * sigma, "y_max": a_max + 3 * sigma, "n": 201},
}

mhp = MoralHazardProblem(cfg)

# --- shared inputs ---
a_hat = np.zeros(2)
a_grid = np.linspace(0.0, a_max, 121)

# --- a_hat solver ---
print("=== a_hat solver ===")
F_a_hat = mhp.expected_wage_fun(reservation_utility=Ubar, solver="a_hat", a_hat=a_hat, warm_start=True)
Ew_a_hat = np.array([F_a_hat(float(a)) for a in a_grid])
payoff_a_hat = a_grid - Ew_a_hat
a_star_a_hat = float(a_grid[np.argmax(payoff_a_hat)])

# Solve once at the optimal action to get v*(·)
res_a_hat = mhp.solve_cost_minimization_problem(
    intended_action=a_star_a_hat,
    reservation_utility=Ubar,
    solver="a_hat",
    a_hat=a_hat,
)
v_star_a_hat = res_a_hat.optimal_contract
print(f"a* = {a_star_a_hat:.4f}")

# --- iterative solver ---
print("\n=== iterative solver ===")
F_iterative = mhp.expected_wage_fun(reservation_utility=Ubar, solver="iterative", a_max=50, warm_start=True, n_a_iterations=3)
Ew_iterative = np.array([F_iterative(float(a)) for a in a_grid])
payoff_iterative = a_grid - Ew_iterative
a_star_iterative = float(a_grid[np.argmax(payoff_iterative)])

# Solve once at the optimal action to get v*(·)
res_iterative = mhp.solve_cost_minimization_problem(
    intended_action=a_star_iterative,
    reservation_utility=Ubar,
    solver="iterative",
    a_max=50,
)
v_star_iterative = res_iterative.optimal_contract
print(f"a* = {a_star_iterative:.4f}")


y = mhp.y_grid

# --- Plots ---

# 1) Optimal wage schedules comparison
plt.figure(figsize=(10, 6))
plt.plot(y, mhp.k(v_star_a_hat), linewidth=2, label=f'a_hat solver (a* = {a_star_a_hat:.2f})', color='blue')
plt.plot(y, mhp.k(v_star_iterative), linewidth=2, label=f'iterative solver (a* = {a_star_iterative:.2f})', color='red', linestyle='--')
plt.xlabel("Outcome y")
plt.ylabel("Wage k(v*(y))")
plt.title("Optimal wage schedules comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 2) Utility functions comparison
plt.figure(figsize=(10, 6))
Ua_a_hat = mhp.U(v_star_a_hat, a_grid)
Ua_iterative = mhp.U(v_star_iterative, a_grid)
plt.plot(a_grid, Ua_a_hat, linewidth=2, label=f'a_hat solver (a* = {a_star_a_hat:.2f})', color='blue')
plt.plot(a_grid, Ua_iterative, linewidth=2, label=f'iterative solver (a* = {a_star_iterative:.2f})', color='red', linestyle='--')
plt.axvline(a_star_a_hat, linestyle=":", color='blue', alpha=0.7, label=f'a_hat a* = {a_star_a_hat:.2f}')
plt.axvline(a_star_iterative, linestyle=":", color='red', alpha=0.7, label=f'iterative a* = {a_star_iterative:.2f}')
plt.xlabel("Action a")
plt.ylabel("U(a)")
plt.title("U(a) under optimal contracts comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 3) Expected wages comparison
plt.figure(figsize=(10, 6))
plt.plot(a_grid, Ew_a_hat, linewidth=2, label=f'a_hat solver (a* = {a_star_a_hat:.2f})', color='blue')
plt.plot(a_grid, Ew_iterative, linewidth=2, label=f'iterative solver (a* = {a_star_iterative:.2f})', color='red', linestyle='--')
plt.axvline(a_star_a_hat, linestyle=":", color='blue', alpha=0.7, label=f'a_hat a* = {a_star_a_hat:.2f}')
plt.axvline(a_star_iterative, linestyle=":", color='red', alpha=0.7, label=f'iterative a* = {a_star_iterative:.2f}')
plt.xlabel("Action a")
plt.ylabel("Expected wage E[w]")
plt.title("Expected wages comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 4) Principal's payoff comparison
plt.figure(figsize=(10, 6))
plt.plot(a_grid, payoff_a_hat, linewidth=2, label=f'a_hat solver (a* = {a_star_a_hat:.2f})', color='blue')
plt.plot(a_grid, payoff_iterative, linewidth=2, label=f'iterative solver (a* = {a_star_iterative:.2f})', color='red', linestyle='--')
plt.axvline(a_star_a_hat, linestyle=":", color='blue', alpha=0.7, label=f'a_hat a* = {a_star_a_hat:.2f}')
plt.axvline(a_star_iterative, linestyle=":", color='red', alpha=0.7, label=f'iterative a* = {a_star_iterative:.2f}')
plt.xlabel("Action a")
plt.ylabel("Principal payoff: a - E[w]")
plt.title("Principal's payoff comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

print(f"\n=== Summary ===")
print(f"a_hat solver a*: {a_star_a_hat:.4f}")
print(f"iterative solver a*: {a_star_iterative:.4f}")
print(f"Difference in a*: {abs(a_star_a_hat - a_star_iterative):.6f}")
print(f"a_hat solver max payoff: {np.max(payoff_a_hat):.6f}")
print(f"iterative solver max payoff: {np.max(payoff_iterative):.6f}")
print(f"Difference in max payoff: {abs(np.max(payoff_a_hat) - np.max(payoff_iterative)):.6f}")

plt.show()
