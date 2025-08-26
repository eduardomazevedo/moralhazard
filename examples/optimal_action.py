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

# --- experiment config ---
a_max = 140.0

cfg = {
    "problem_params": {"u": u, "k": k, "link_function": g, "C": C, "Cprime": Cprime, "f": f, "score": score},
    "computational_params": {"distribution_type": "continuous", "y_min": 0.0 - 3 * sigma, "y_max": a_max + 3 * sigma, "n": 201},
}

# --- setup shared inputs ---
Ubar = float(u(50))        # same reservation utility as quickstart
a_hat = np.zeros(2)

# --- grid search over actions (a_hat solver) ---
print("=== a_hat solver ===")
mhp_grid_a_hat = MoralHazardProblem(cfg)
F_grid_a_hat = mhp_grid_a_hat.expected_wage_fun(reservation_utility=Ubar, solver="a_hat", a_hat=a_hat, warm_start=True)

a_grid = np.linspace(0.0, a_max, 121)
t0 = time.perf_counter()
ews_grid_a_hat = np.array([F_grid_a_hat(float(a)) for a in a_grid])
payoff_grid_a_hat = a_grid - ews_grid_a_hat
idx_a_hat = int(np.argmax(payoff_grid_a_hat))
a_star_grid_a_hat = float(a_grid[idx_a_hat])
payoff_star_grid_a_hat = float(payoff_grid_a_hat[idx_a_hat])
t1 = time.perf_counter()

print("[Grid search]")
print(f"  a* = {a_star_grid_a_hat:.4f}")
print(f"  payoff a - E[w] = {payoff_star_grid_a_hat:.6f}")
print(f"  time (s) = {t1 - t0:.4f}")

# --- 1-D optimization on the same interval (a_hat solver) ---
# Create a fresh MoralHazardProblem instance to avoid any warm start bias
mhp_opt_a_hat = MoralHazardProblem(cfg)
F_opt_a_hat = mhp_opt_a_hat.expected_wage_fun(reservation_utility=Ubar, solver="a_hat", a_hat=a_hat, warm_start=True)

def neg_objective_a_hat(a):
    return -(float(a) - F_opt_a_hat(float(a)))

# Use midpoint of action range as initial guess for fair comparison
a_min, a_max_opt = a_grid.min(), a_grid.max()
initial_guess = (a_min + a_max_opt) / 2

t2 = time.perf_counter()
res_a_hat = minimize_scalar(neg_objective_a_hat, bounds=(a_min, a_max_opt), method="bounded",
                      options={"xatol": 1e-2, "maxiter": 200})
t3 = time.perf_counter()

a_star_opt_a_hat = float(res_a_hat.x)
payoff_star_opt_a_hat = -float(res_a_hat.fun)
print("\n[1-D optimizer]")
print(f"  a* = {a_star_opt_a_hat:.4f}")
print(f"  payoff a - E[w] = {payoff_star_opt_a_hat:.6f}")
print(f"  time (s) = {t3 - t2:.4f}")
print(f"  initial guess: {initial_guess:.4f}")

# --- comparison summary (a_hat solver) ---
print(f"\n[Comparison - a_hat solver]")
print(f"  Grid search time: {t1 - t0:.4f}s")
print(f"  1D optimizer time: {t3 - t2:.4f}s")
print(f"  Speedup: {(t1 - t0) / (t3 - t2):.2f}x")
print(f"  Difference in optimal action: {abs(a_star_grid_a_hat - a_star_opt_a_hat):.6f}")
print(f"  Difference in payoff: {abs(payoff_star_grid_a_hat - payoff_star_opt_a_hat):.6f}")

# --- grid search over actions (iterative solver) ---
print(f"\n=== iterative solver ===")
mhp_grid_iterative = MoralHazardProblem(cfg)
F_grid_iterative = mhp_grid_iterative.expected_wage_fun(reservation_utility=Ubar, solver="iterative", warm_start=True)

t4 = time.perf_counter()
ews_grid_iterative = np.array([F_grid_iterative(float(a)) for a in a_grid])
payoff_grid_iterative = a_grid - ews_grid_iterative
idx_iterative = int(np.argmax(payoff_grid_iterative))
a_star_grid_iterative = float(a_grid[idx_iterative])
payoff_star_grid_iterative = float(payoff_grid_iterative[idx_iterative])
t5 = time.perf_counter()

print("[Grid search]")
print(f"  a* = {a_star_grid_iterative:.4f}")
print(f"  payoff a - E[w] = {payoff_star_grid_iterative:.6f}")
print(f"  time (s) = {t5 - t4:.4f}")

# --- 1-D optimization on the same interval (iterative solver) ---
# Create a fresh MoralHazardProblem instance to avoid any warm start bias
mhp_opt_iterative = MoralHazardProblem(cfg)
F_opt_iterative = mhp_opt_iterative.expected_wage_fun(reservation_utility=Ubar, solver="iterative", warm_start=True)

def neg_objective_iterative(a):
    return -(float(a) - F_opt_iterative(float(a)))

t6 = time.perf_counter()
res_iterative = minimize_scalar(neg_objective_iterative, bounds=(a_min, a_max_opt), method="bounded",
                      options={"xatol": 1e-2, "maxiter": 200})
t7 = time.perf_counter()

a_star_opt_iterative = float(res_iterative.x)
payoff_star_opt_iterative = -float(res_iterative.fun)
print("\n[1-D optimizer]")
print(f"  a* = {a_star_opt_iterative:.4f}")
print(f"  payoff a - E[w] = {payoff_star_opt_iterative:.6f}")
print(f"  time (s) = {t7 - t6:.4f}")
print(f"  initial guess: {initial_guess:.4f}")

# --- comparison summary (iterative solver) ---
print(f"\n[Comparison - iterative solver]")
print(f"  Grid search time: {t5 - t4:.4f}s")
print(f"  1D optimizer time: {t7 - t6:.4f}s")
print(f"  Speedup: {(t5 - t4) / (t7 - t6):.2f}x")
print(f"  Difference in optimal action: {abs(a_star_grid_iterative - a_star_opt_iterative):.6f}")
print(f"  Difference in payoff: {abs(payoff_star_grid_iterative - payoff_star_opt_iterative):.6f}")

# --- cross-solver comparison ---
print(f"\n=== Cross-solver comparison ===")
print(f"  Grid search - a_hat vs iterative:")
print(f"    a* difference: {abs(a_star_grid_a_hat - a_star_grid_iterative):.6f}")
print(f"    payoff difference: {abs(payoff_star_grid_a_hat - payoff_star_grid_iterative):.6f}")
print(f"    time ratio (iterative/a_hat): {(t5 - t4) / (t1 - t0):.2f}x")
print(f"  1D optimizer - a_hat vs iterative:")
print(f"    a* difference: {abs(a_star_opt_a_hat - a_star_opt_iterative):.6f}")
print(f"    payoff difference: {abs(payoff_star_opt_a_hat - payoff_star_opt_iterative):.6f}")
print(f"    time ratio (iterative/a_hat): {(t7 - t6) / (t3 - t2):.2f}x")
