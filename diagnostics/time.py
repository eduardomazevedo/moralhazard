# Timing cost minimization problem solvers and principal problem
import time
import os
import numpy as np
import pandas as pd
from moralhazard import MoralHazardProblem
from moralhazard.config_maker import make_utility_cfg, make_distribution_cfg

# ---- primitives (same as prototype Normal model) ----
initial_wealth = 50
sigma_gaussian = 10.0
first_best_effort = 100
theta = 1.0 / first_best_effort / (first_best_effort + initial_wealth)

a_ic_lb = 0.0
a_ic_ub = 130.0

def u(c): return np.log(initial_wealth + c)
def k(utils): return np.exp(utils) - initial_wealth
def g(z): return np.log(np.maximum(z, initial_wealth))
def C(a): return theta * a ** 2 / 2
def Cprime(a): return theta * a

# Cost minimization problem options
intended_action = first_best_effort
n_a_iterations = 100
a_always_check_global_ic = np.array([])

# Setup utility config (used for all cases)
utility_cfg = make_utility_cfg("log", w0=initial_wealth)
u_fun = utility_cfg["u"]

# Cases with their problem configurations
cases = []

# Gaussian cases (easy and hard)
sigma = sigma_gaussian
def f_gaussian(y, a):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((y - a) ** 2) / (2 * sigma ** 2))
def score_gaussian(y, a):
    return (y - a) / (sigma ** 2)

cfg_gaussian = {
    "problem_params": {"u": u, "k": k, "link_function": g, "C": C, "Cprime": Cprime, "f": f_gaussian, "score": score_gaussian},
    "computational_params": {"distribution_type": "continuous", "y_min": a_ic_lb - 3 * sigma, "y_max": a_ic_ub + 3 * sigma, "n": 201},
}
mhp_gaussian = MoralHazardProblem(cfg_gaussian)

cases.append(("gaussian-easy", mhp_gaussian, u(10)))
cases.append(("gaussian-hard", mhp_gaussian, u(-10)))

# t-distribution case (sigma=20, nu=1.15, reservation_utility=u(0))
sigma_t = 20.0
a_min_t = 0.0
a_max_t = 100.0
dist_cfg_t = make_distribution_cfg("student_t", nu=1.15, sigma=sigma_t)
comp_cfg_t = {
    "distribution_type": "continuous",
    "y_min": a_min_t - 10 * sigma_t,
    "y_max": a_max_t + 10 * sigma_t,
    "n": 201,  # must be odd
}
cfg_t = {
    "problem_params": {**utility_cfg, **dist_cfg_t, "C": C, "Cprime": Cprime},
    "computational_params": comp_cfg_t
}
mhp_t = MoralHazardProblem(cfg_t)

cases.append(("t", mhp_t, u(0)))

# Storage for results
results = []

print("=== Timing benchmarks ===\n")

for case_name, mhp, reservation_utility in cases:
    print(f"Case: {case_name} (reservation_utility = {reservation_utility:.4f})")
    
    # Use appropriate bounds for each case
    if case_name == "t":
        a_lb = a_min_t
        a_ub = a_max_t
        a_pp_min = a_min_t
        a_pp_max = a_max_t
    else:
        a_lb = a_ic_lb
        a_ub = a_ic_ub
        a_pp_min = 0.0
        a_pp_max = 100.0
    
    # 1. Cost minimization relaxed problem (0 iterations)
    print("  Timing relaxed CMP...")
    times_relaxed = []
    for i in range(n_a_iterations):
        t0 = time.perf_counter()
        _ = mhp.solve_cost_minimization_problem(
            intended_action=intended_action,
            reservation_utility=reservation_utility,
            a_ic_lb=a_lb,
            a_ic_ub=a_ub,
            n_a_iterations=0,
            a_always_check_global_ic=a_always_check_global_ic,
        )
        t1 = time.perf_counter()
        times_relaxed.append(t1 - t0)
    relaxed_cmp_time = np.mean(times_relaxed) * 1000  # Convert to ms
    print(f"    Mean time: {relaxed_cmp_time:.2f}ms")
    
    # 2. Cost minimization problem (with iterations)
    print("  Timing CMP...")
    times_cmp = []
    for i in range(n_a_iterations):
        t0 = time.perf_counter()
        _ = mhp.solve_cost_minimization_problem(
            intended_action=intended_action,
            reservation_utility=reservation_utility,
            a_ic_lb=a_lb,
            a_ic_ub=a_ub,
            n_a_iterations=n_a_iterations,
            a_always_check_global_ic=a_always_check_global_ic,
        )
        t1 = time.perf_counter()
        times_cmp.append(t1 - t0)
    cmp_time = np.mean(times_cmp) * 1000  # Convert to ms
    print(f"    Mean time: {cmp_time:.2f}ms")
    
    # 3. Principal problem
    print("  Timing principal problem...")
    times_principal = []
    for i in range(n_a_iterations):
        t0 = time.perf_counter()
        _ = mhp.solve_principal_problem(
            revenue_function=lambda a: a,
            reservation_utility=reservation_utility,
            a_min=a_pp_min,
            a_max=a_pp_max,
            a_ic_lb=a_lb,
            a_ic_ub=a_ub,
        )
        t1 = time.perf_counter()
        times_principal.append(t1 - t0)
    principal_time = np.mean(times_principal) * 1000  # Convert to ms
    print(f"    Mean time: {principal_time:.2f}ms")
    
    results.append({
        "case": case_name,
        "relaxed_cmp": relaxed_cmp_time,
        "cmp": cmp_time,
        "principals_problem": principal_time,
    })
    print()

# Create DataFrame
df = pd.DataFrame(results)

# Print table
print("=== Results Table ===")
print(df.to_string(index=False))
print()

# Save CSV
output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "timing_results.csv")
df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")
