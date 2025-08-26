import time
import numpy as np
from scipy.optimize import minimize_scalar
from moralhazard import MoralHazardProblem

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

cfg = {
    "problem_params": {"u": u, "k": k, "link_function": g, "C": C, "Cprime": Cprime, "f": f, "score": score},
    "computational_params": {"y_min": 0.0 - 3 * sigma, "y_max": 120.0 + 3 * sigma, "n": 201},
}

# --- setup shared inputs ---
Ubar = float(u(50))        # same reservation utility as quickstart
a_hat = np.zeros(2)

# --- grid search over actions ---
mhp_grid = MoralHazardProblem(cfg)
F_grid = mhp_grid.expected_wage_fun(reservation_utility=Ubar, a_hat=a_hat, warm_start=True)

a_grid = np.linspace(0.0, 140.0, 121)
t0 = time.perf_counter()
ews_grid = np.array([F_grid(float(a)) for a in a_grid])
payoff_grid = a_grid - ews_grid
idx = int(np.argmax(payoff_grid))
a_star_grid = float(a_grid[idx])
payoff_star_grid = float(payoff_grid[idx])
t1 = time.perf_counter()

print("[Grid search]")
print(f"  a* = {a_star_grid:.4f}")
print(f"  payoff a - E[w] = {payoff_star_grid:.6f}")
print(f"  time (s) = {t1 - t0:.4f}")

# --- 1-D optimization on the same interval ---
# Create a fresh MoralHazardProblem instance to avoid any warm start bias
mhp_opt = MoralHazardProblem(cfg)
F_opt = mhp_opt.expected_wage_fun(reservation_utility=Ubar, a_hat=a_hat, warm_start=True)

def neg_objective(a):
    return -(float(a) - F_opt(float(a)))

# Use midpoint of action range as initial guess for fair comparison
a_min, a_max = a_grid.min(), a_grid.max()
initial_guess = (a_min + a_max) / 2

t2 = time.perf_counter()
res = minimize_scalar(neg_objective, bounds=(a_min, a_max), method="bounded",
                      options={"xatol": 1e-2, "maxiter": 200})
t3 = time.perf_counter()

a_star_opt = float(res.x)
payoff_star_opt = -float(res.fun)
print("\n[1-D optimizer]")
print(f"  a* = {a_star_opt:.4f}")
print(f"  payoff a - E[w] = {payoff_star_opt:.6f}")
print(f"  time (s) = {t3 - t2:.4f}")
print(f"  initial guess: {initial_guess:.4f}")

# --- comparison summary ---
print(f"\n[Comparison]")
print(f"  Grid search time: {t1 - t0:.4f}s")
print(f"  1D optimizer time: {t3 - t2:.4f}s")
print(f"  Speedup: {(t1 - t0) / (t3 - t2):.2f}x")
print(f"  Difference in optimal action: {abs(a_star_grid - a_star_opt):.6f}")
print(f"  Difference in payoff: {abs(payoff_star_grid - payoff_star_opt):.6f}")
