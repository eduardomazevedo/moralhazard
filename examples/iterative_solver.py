import numpy as np
import matplotlib.pyplot as plt
from moralhazard import MoralHazardProblem
from moralhazard.solver import _minimize_cost_iterative

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
    "computational_params": {"distribution_type": "continuous", "y_min": 0.0 - 3 * sigma, "y_max": a_max + 3 * sigma, "n": 201},
}

mhp = MoralHazardProblem(cfg)

# --- shared inputs ---
a0 = 100.0  # intended action
a_min, a_max = 0.0, 150.0
n_a_grid = 100

# Test different numbers of iterations
iteration_counts = [0, 1, 2, 3]
results_by_iteration = {}

print("=== Iterative Solver Demo ===\n")
print(f"Intended action: {a0}")
print(f"Grid range: [{a_min}, {a_max}] with {n_a_grid} points")
print()

for n_iterations in iteration_counts:
    print(f"--- {n_iterations} iterations ---")
    
    # Solve using the iterative method
    results, theta_opt = _minimize_cost_iterative(
        a0=a0,
        Ubar=Ubar,
        a_min=a_min,
        a_max=a_max,
        n_a_grid=n_a_grid,
        n_a_iterations=n_iterations,
        y_grid=mhp.y_grid,
        w=mhp._w,
        f=mhp._primitives["f"],
        score=mhp._primitives["score"],
        C=mhp._primitives["C"],
        Cprime=mhp._primitives["Cprime"],
        g=mhp._primitives["g"],
        k=mhp._primitives["k"],
    )
    
    # Extract a_current from a_hat (a_hat = [0.0, a_current])
    a_current = results.a_hat[1] if len(results.a_hat) > 1 else 0.0
    
    # Extract multipliers and constraints
    lam = results.multipliers["lam"]
    mu = results.multipliers["mu"]
    mu_hat = results.multipliers["mu_hat"]

    print(f"  a_current: {a_current:.6f}")
    print(f"  λ (IR multiplier): {lam:.6f}")
    print(f"  μ (FOC multiplier): {mu:.6f}")
    print(f"  μ̂ (IC multipliers): {mu_hat}")
    print(f"  Expected wage: {results.expected_wage:.6f}")
    print(f"  IR constraint: {results.constraints['IR']:.2e}")
    print(f"  FOC constraint: {results.constraints['FOC']:.2e}")
    print(f"  IC constraints: {results.constraints['IC']}")
    
    # Store results for plotting
    results_by_iteration[n_iterations] = {
        'results': results,
        'theta_opt': theta_opt,
        'a_current': a_current
    }
    
    print()

# --- Plotting ---
plt.figure(figsize=(12, 8))

# Create action grid for utility evaluation
a_grid = np.linspace(a_min, a_max, n_a_grid)

# Plot utility curves for different iteration counts
colors = ['blue', 'red', 'green', 'orange']
for i, n_iterations in enumerate(iteration_counts):
    results = results_by_iteration[n_iterations]['results']
    v_optimal = results.optimal_contract
    
    # Compute utility for all actions
    U_values = mhp.U(v_optimal, a_grid)
    
    plt.plot(a_grid, U_values, 
             color=colors[i], 
             linewidth=2, 
             label=f'{n_iterations} iterations')

# Add vertical line for intended action
plt.axvline(a0, color='black', linestyle='--', alpha=0.7, label=f'Intended action a₀ = {a0}')

plt.xlabel('Action a')
plt.ylabel('U(a)')
plt.title('Utility curves under optimal contracts from iterative solver')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# --- Additional analysis ---
print("=== Final Analysis ===")
print(f"Intended action: {a0}")
print(f"Grid range: [{a_min}, {a_max}] with {n_a_grid} points")
print()

for n_iterations in iteration_counts:
    results = results_by_iteration[n_iterations]['results']
    a_current = results_by_iteration[n_iterations]['a_current']
    v_optimal = results.optimal_contract
    
    # Compute utility at the intended action
    U_at_intended = float(mhp.U(v_optimal, a0))
    
    print(f"{n_iterations} iterations:")
    print(f"  a_current: {a_current:.6f}")
    print(f"  Utility at intended action: {U_at_intended:.6f}")
    print(f"  Expected wage: {results.expected_wage:.6f}")
    print(f"  λ (IR multiplier): {results.multipliers['lam']:.6f}")
    print(f"  μ (FOC multiplier): {results.multipliers['mu']:.6f}")
    print(f"  μ̂ (IC multipliers): {results.multipliers['mu_hat']}")
    print()

plt.show()
