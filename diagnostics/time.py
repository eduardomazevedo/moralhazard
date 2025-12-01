# Timing cost minimization problem solvers (a_hat and iterative)
import time
import numpy as np
from moralhazard import MoralHazardProblem

# ---- primitives (same as prototype Normal model) ----
initial_wealth = 50
sigma = 10.0
first_best_effort = 100
theta = 1.0 / first_best_effort / (first_best_effort + initial_wealth)

a_ic_lb = 0.0
a_ic_ub = 130.0

def u(c): return np.log(initial_wealth + c)
def k(utils): return np.exp(utils) - initial_wealth
def g(z): return np.log(np.maximum(z, initial_wealth))
def C(a): return theta * a ** 2 / 2
def Cprime(a): return theta * a
def f(y, a):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((y - a) ** 2) / (2 * sigma ** 2))
def score(y, a):
    return (y - a) / (sigma ** 2)

cfg = {
    "problem_params": {"u": u, "k": k, "link_function": g, "C": C, "Cprime": Cprime, "f": f, "score": score},
    "computational_params": {"distribution_type": "continuous", "y_min": a_ic_lb - 3 * sigma, "y_max": a_ic_ub + 3 * sigma, "n": 201},
}

mhp = MoralHazardProblem(cfg)

# Cost minimization problem options
intended_action = first_best_effort
reservation_utility = u(0)
n_a_grid_points = 10
n_a_iterations = 10
a_always_check_global_ic = np.array([0.0])

print("=== Timing cost minimization problem ===")
times_cost_minimization = []
for i in range(n_a_iterations):
    t0 = time.perf_counter()
    _ = mhp.solve_cost_minimization_problem(
        intended_action=intended_action,
        reservation_utility=reservation_utility,
        a_ic_lb=a_ic_lb,
        a_ic_ub=a_ic_ub,
        n_a_grid_points=n_a_grid_points,
        n_a_iterations=n_a_iterations,
        a_always_check_global_ic=a_always_check_global_ic,
    )
    t1 = time.perf_counter()
    dt = t1 - t0
    times_cost_minimization.append(dt)
    print(f"Iteration {i:02d}: time={dt:.4f}s")

print(f"Cost minimization problem mean time (s): {np.mean(times_cost_minimization):.4f}")


print("=== Timing cost minimization relaxed problem - force 0 iterations ===")
times_cost_minimization = []
for i in range(n_a_iterations):
    t0 = time.perf_counter()
    _ = mhp.solve_cost_minimization_problem(
        intended_action=intended_action,
        reservation_utility=reservation_utility,
        a_ic_lb=a_ic_lb,
        a_ic_ub=a_ic_ub,
        n_a_grid_points=n_a_grid_points,
        n_a_iterations=0,
        a_always_check_global_ic=a_always_check_global_ic,
    )
    t1 = time.perf_counter()
    dt = t1 - t0
    times_cost_minimization.append(dt)
    print(f"Iteration {i:02d}: time={dt:.4f}s")

print(f"Cost minimization relaxed problem mean time (s): {np.mean(times_cost_minimization):.4f}")