# examples/timing_10_runs.py
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

cfg = {
    "problem_params": {"u": u, "k": k, "link_function": g, "C": C, "Cprime": Cprime, "f": f, "score": score},
    "computational_params": {"y_min": 0.0 - 3 * sigma, "y_max": 120.0 + 3 * sigma, "n": 201},
}

mhp = MoralHazardProblem(cfg)

# --- experiment config (prototype-like) ---
BASE_INTENDED_ACTION = 80.0
NOISE_SD = 5.0
N_RUNS = 10
SEED = 123
rng = np.random.default_rng(SEED)
a_list = BASE_INTENDED_ACTION + rng.normal(0.0, NOISE_SD, N_RUNS)
rw_list = np.linspace(0.0, 80.0, N_RUNS)        # vary reservation wage; pass u(rw)
a_hat = np.zeros(2)

times = []
for i, (a0, rw) in enumerate(zip(a_list, rw_list), 1):
    t0 = time.perf_counter()
    _ = mhp.solve_cost_minimization_problem(
        intended_action=float(a0),
        reservation_utility=float(u(rw)),
        a_hat=a_hat,
    )
    t1 = time.perf_counter()
    dt = t1 - t0
    times.append(dt)
    print(f"Run {i:02d}: a={a0:6.2f}, Ubar=u({rw:5.2f})  time={dt:.4f}s")

print("\nAll times (s):", ", ".join(f"{t:.4f}" for t in times))
print(f"Mean time (s): {np.mean(times):.4f}")
