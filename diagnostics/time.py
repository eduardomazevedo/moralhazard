# Timing cost minimization problem solvers (a_hat and iterative)
import time
import numpy as np
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
BASE_INTENDED_ACTION = 80.0
NOISE_SD = 5.0
N_RUNS = 10
SEED = 123
a_max = 140.0

# Update grid to accommodate a_max
cfg = {
    "problem_params": {"u": u, "k": k, "link_function": g, "C": C, "Cprime": Cprime, "f": f, "score": score},
    "computational_params": {"distribution_type": "continuous", "y_min": 0.0 - 3 * sigma, "y_max": a_max + 3 * sigma, "n": 201},
}

mhp = MoralHazardProblem(cfg)

rng = np.random.default_rng(SEED)
a_list = BASE_INTENDED_ACTION + rng.normal(0.0, NOISE_SD, N_RUNS)
rw_list = np.linspace(0.0, 80.0, N_RUNS)        # vary reservation wage; pass u(rw)
a_hat = np.zeros(2)

print("=== Timing a_hat solver ===")
times_a_hat = []
for i, (a0, rw) in enumerate(zip(a_list, rw_list), 1):
    t0 = time.perf_counter()
    _ = mhp.solve_cost_minimization_problem(
        intended_action=float(a0),
        reservation_utility=float(u(rw)),
        solver="a_hat",
        a_hat=a_hat,
    )
    t1 = time.perf_counter()
    dt = t1 - t0
    times_a_hat.append(dt)
    print(f"Run {i:02d}: a={a0:6.2f}, Ubar=u({rw:5.2f})  time={dt:.4f}s")

print("\n=== Timing iterative solver ===")
times_iterative = []
for i, (a0, rw) in enumerate(zip(a_list, rw_list), 1):
    t0 = time.perf_counter()
    _ = mhp.solve_cost_minimization_problem(
        intended_action=float(a0),
        reservation_utility=float(u(rw)),
        solver="iterative"
    )
    t1 = time.perf_counter()
    dt = t1 - t0
    times_iterative.append(dt)
    print(f"Run {i:02d}: a={a0:6.2f}, Ubar=u({rw:5.2f})  time={dt:.4f}s")

print("\n=== Timing a_hat solver (empty a_hat) ===")
a_hat_empty = np.array([])
times_a_hat_empty = []
for i, (a0, rw) in enumerate(zip(a_list, rw_list), 1):
    t0 = time.perf_counter()
    _ = mhp.solve_cost_minimization_problem(
        intended_action=float(a0),
        reservation_utility=float(u(rw)),
        solver="a_hat",
        a_hat=a_hat_empty,
    )
    t1 = time.perf_counter()
    dt = t1 - t0
    times_a_hat_empty.append(dt)
    print(f"Run {i:02d}: a={a0:6.2f}, Ubar=u({rw:5.2f})  time={dt:.4f}s")

print("\n=== Summary ===")
print("a_hat solver times (s):", ", ".join(f"{t:.4f}" for t in times_a_hat))
print(f"a_hat solver mean time (s): {np.mean(times_a_hat):.4f}")
print("iterative solver times (s):", ", ".join(f"{t:.4f}" for t in times_iterative))
print(f"iterative solver mean time (s): {np.mean(times_iterative):.4f}")
print("a_hat solver (empty) times (s):", ", ".join(f"{t:.4f}" for t in times_a_hat_empty))
print(f"a_hat solver (empty) mean time (s): {np.mean(times_a_hat_empty):.4f}")
print(f"Speedup (iterative/a_hat): {np.mean(times_iterative)/np.mean(times_a_hat):.2f}x")
print(f"Speedup (iterative/a_hat_empty): {np.mean(times_iterative)/np.mean(times_a_hat_empty):.2f}x")
print(f"Speedup (a_hat_empty/a_hat): {np.mean(times_a_hat_empty)/np.mean(times_a_hat):.2f}x")
