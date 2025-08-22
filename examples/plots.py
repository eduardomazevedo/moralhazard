import numpy as np
import matplotlib.pyplot as plt
from moral_hazard import MoralHazardProblem

# ---- primitives (same as prototype Normal model) ----
x0 = 50
sigma = 10.0
first_best_effort = 100
theta = 1.0 / first_best_effort / (first_best_effort + x0)

def u(c): return np.log(x0 + c)

Ubar = float(u(0))  # same reservation utility as quickstart
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

# 1) Solve for optimal action using a grid
F = mhp.expected_wage_fun(reservation_utility=Ubar, a_hat=a_hat, warm_start=True)
a_grid = np.linspace(0.0, a_max, 121)
Ew = np.array([F(float(a)) for a in a_grid])
payoff = a_grid - Ew
a_star = float(a_grid[np.argmax(payoff)])

# 2) Solve once at the optimal action to get v*(·)
res = mhp.solve_cost_minimization_problem(
    intended_action=a_star,
    reservation_utility=Ubar,
    a_hat=a_hat,
)
v_star = res.optimal_contract
y = mhp.y_grid

# --- Plots ---
plt.figure(figsize=(6,4))
plt.plot(y, mhp.k(v_star), linewidth=1.6)
plt.xlabel("Outcome y")
plt.ylabel("Wage k(v*(y))")
plt.title("Optimal wage schedule at a*")
plt.tight_layout()

plt.figure(figsize=(6,4))
Ua = mhp.U(v_star, a_grid)
plt.plot(a_grid, Ua, linewidth=1.6)
plt.axvline(a_star, linestyle="--", label=f"a* = {a_star:.2f}")
plt.xlabel("Action a")
plt.ylabel("U(a)")
plt.title("U(a) under the optimal contract v*(·)")
plt.legend()
plt.tight_layout()

plt.figure(figsize=(6,4))
plt.plot(a_grid, Ew, linewidth=1.6)
plt.axvline(a_star, linestyle="--", label=f"a* = {a_star:.2f}")
plt.xlabel("Action a")
plt.ylabel("Expected wage E[w]")
plt.title("Expected wage of optimal contract vs. action")
plt.legend()
plt.tight_layout()

plt.figure(figsize=(6,4))
plt.plot(a_grid, payoff, linewidth=1.6)
plt.axvline(a_star, linestyle="--", label=f"a* = {a_star:.2f}")
plt.xlabel("Action a")
plt.ylabel("Principal payoff: a - E[w]")
plt.title("Principal's payoff vs. action")
plt.legend()
plt.tight_layout()

plt.show()
