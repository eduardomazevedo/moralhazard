# Quickstart example
import numpy as np
import matplotlib.pyplot as plt
from moralhazard import MoralHazardProblem
from moralhazard.config_maker import make_utility_cfg, make_distribution_cfg

# ---- primitives (same as prototype Normal model) ----
x0 = 50
sigma = 10.0
first_best_effort = 100
theta = 1.0 / first_best_effort / (first_best_effort + x0)

def u(c): return np.log(x0 + c)
def k(utils, xp=np): return xp.exp(utils) - x0 # Specify module to be able to use cvxpy later.
def g(z): return np.log(np.maximum(z, x0))
def C(a): return theta * a ** 2 / 2
def Cprime(a): return theta * a
def f(y, a):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((y - a) ** 2) / (2 * sigma ** 2))
def score(y, a):
    return (y - a) / (sigma ** 2)

reservation_utility = u(0) - 5.0

# ---- configuration ----
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
        "y_min": 0.0   - 3 * sigma,
        "y_max": 180.0 + 3 * sigma,
        "n": 201,  # must be odd
    },
}

# ---- cost minimization problem ----
import time

mhp = MoralHazardProblem(cfg)

t0_dual = time.perf_counter()
results = mhp.solve_cost_minimization_problem(
    intended_action=80.0,
    reservation_utility=reservation_utility,
    a_ic_lb=0.0,
    a_ic_ub=100.0
)
t1_dual = time.perf_counter()
dual_time = t1_dual - t0_dual

print("Cost minimization problem results:")
print(results)
print(f"Dual solver runtime: {dual_time*1000:.2f} ms")

# ---- CVXPY-based principal-agent problem ----
# Solve the cost minimization problem directly using convex optimization.
# For log utility: u(c) = log(x0 + c), k(v) = exp(v) - x0 is convex in v.
# The objective E[k(v)] and all constraints are expressible in DCP form.

import cvxpy as cp
from moralhazard.core import _make_cache, _compute_expected_utility

# Setup parameters
a0 = 80.0  # intended action (same as above)
Ubar = reservation_utility

# Finite set of actions to check global IC constraints
a_hat = np.linspace(0.0, 100.0, 100)  # 100 points from 0 to 100

# Build cache with precomputed integration weights and densities
cache = _make_cache(a0, a_hat, problem=mhp)

# Extract arrays from cache
y_grid = mhp.y_grid
n = len(y_grid)
m = len(a_hat)

wf0 = cache["wf0"]             # (n,) = w * f(y|a0)
wf0s0 = cache["wf0s0"]         # (n,) = w * f(y|a0) * score(y, a0)
weighted_D = cache["weighted_D"]  # (m, n) = w * f(y|a_hat[i])
C0 = cache["C0"]               # C(a0)
Cprime0 = cache["Cprime0"]     # C'(a0)
C_hat = cache["C_hat"]         # (m,) C(a_hat)

# Decision variable: v(y) on the grid
v = cp.Variable(n)

# Objective: minimize E[k(v)] = E[exp(v) - x0] = wf0 @ exp(v) - x0
objective = cp.Minimize(wf0 @ cp.exp(v) - x0 * cp.sum(wf0))

# Constraints
constraints = []

# IR: U(a0) >= Ubar  where U(a0) = wf0 @ v - C(a0)
constraints.append(wf0 @ v >= Ubar + C0)

# FOC (local IC): dU/da|_{a=a0} = 0  =>  wf0s0 @ v = C'(a0)
constraints.append(wf0s0 @ v == Cprime0)

# Global IC: U(a0) >= U(a_hat[i]) for each i
# Rewritten: (wf0 - weighted_D[i]) @ v >= C0 - C_hat[i]
for i in range(m):
    constraints.append((wf0 - weighted_D[i]) @ v >= C0 - C_hat[i])

# Non-negative consumption: c = k(v) = exp(v) - x0 >= 0  =>  v >= log(x0)
constraints.append(v >= np.log(x0))

# Solve
prob = cp.Problem(objective, constraints)
t0_cvx = time.perf_counter()
prob.solve(verbose=False)
t1_cvx = time.perf_counter()
cvx_time = t1_cvx - t0_cvx

print("\n" + "="*60)
print("CVXPY Principal-Agent Problem (Cost Minimization)")
print("="*60)
print(f"Status: {prob.status}")
print(f"Runtime: {cvx_time*1000:.2f} ms")

if prob.status == cp.OPTIMAL:
    v_opt = v.value
    
    # Compute values at solution
    U0_cvx = wf0 @ v_opt - C0
    Ewage_cvx = wf0 @ (np.exp(v_opt) - x0)
    FOC_cvx = wf0s0 @ v_opt - Cprime0
    
    print(f"Optimal E[wage]: {Ewage_cvx:.6f}")
    print(f"U0 (agent utility): {U0_cvx:.6f}")
    print(f"IR slack (U0 - Ubar): {U0_cvx - Ubar:.6f}")
    print(f"FOC residual: {FOC_cvx:.2e}")
    
    # Check global IC on fine grid
    a_fine = np.linspace(0.0, 100.0, 1001)
    U_fine = _compute_expected_utility(v_opt, a_fine, mhp)
    max_ic_violation = np.max(U_fine) - U0_cvx
    worst_action = a_fine[np.argmax(U_fine)]
    print(f"Max IC violation (fine grid): {max_ic_violation:.2e} at a={worst_action:.2f}")
    
    print(f"\nv(y) range: [{np.min(v_opt):.4f}, {np.max(v_opt):.4f}]")
    print(f"c(y) range: [{np.min(np.exp(v_opt) - x0):.4f}, {np.max(np.exp(v_opt) - x0):.4f}]")
    
    # Compare with dual-based solver
    print("\n--- Comparison with dual-based solver ---")
    print(f"Dual E[wage]:  {results.expected_wage:.6f}  ({dual_time*1000:.2f} ms)")
    print(f"CVXPY E[wage]: {Ewage_cvx:.6f}  ({cvx_time*1000:.2f} ms)")
    print(f"Difference:    {abs(results.expected_wage - Ewage_cvx):.6f}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(y_grid, v_opt, label='CVXPY', linewidth=2)
    axes[0].plot(y_grid, results.optimal_contract, '--', label='Dual solver', linewidth=2)
    axes[0].set_xlabel('y (output)')
    axes[0].set_ylabel('v(y) (utility)')
    axes[0].set_title('Optimal Contract: v(y)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    c_cvx = np.exp(v_opt) - x0
    c_dual = np.exp(results.optimal_contract) - x0
    axes[1].plot(y_grid, c_cvx, label='CVXPY', linewidth=2)
    axes[1].plot(y_grid, c_dual, '--', label='Dual solver', linewidth=2)
    axes[1].axhline(0, color='k', linestyle=':', alpha=0.5)
    axes[1].set_xlabel('y (output)')
    axes[1].set_ylabel('c(y) = k(v(y)) (wage)')
    axes[1].set_title('Optimal Contract: c(y)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cvxpy_comparison.png', dpi=100)
    print("\nPlot saved to cvxpy_comparison.png")


# ============================================================================
# Function version of CVXPY cost minimization solver
# ============================================================================

def solve_cost_minimization_cvxpy(
    problem: MoralHazardProblem,
    intended_action: float,
    reservation_utility: float,
    a_hat=None,
    v_lb: float = None,
    v_ub: float = None,
    verbose: bool = False,
):
    """
    Solve the cost minimization problem using CVXPY.
    
    Minimizes E[k(v(y))] subject to:
      - IR: E[v|a0] - C(a0) >= Ubar
      - Local IC (FOC): E[v * score|a0] = C'(a0)
      - Global IC: U(a0) >= U(a_hat) for each action in a_hat
      - Optional: v_lb <= v <= v_ub
    
    The inverse utility function k is taken from problem.k. It must accept
    an `xp` argument to specify the module (numpy or cvxpy), e.g.:
        def k(utils, xp=np): return xp.exp(utils) - x0
    
    Parameters
    ----------
    problem : MoralHazardProblem
        The moral hazard problem instance with grids and primitives.
        Must have problem.k(v, xp=cp) work with cvxpy expressions.
    intended_action : float
        The action a0 to implement.
    reservation_utility : float
        The agent's reservation utility Ubar.
    a_hat : array-like, optional
        Array of alternative actions to check global IC constraints against.
        If empty or None, only IR and FOC constraints are enforced.
    v_lb : float, optional
        Lower bound on v(y). If None, inferred from u(0) (non-negative consumption).
        Set to -np.inf to disable.
    v_ub : float, optional
        Upper bound on v(y). Required for CARA and CRRA with γ>1 where 
        utility is bounded above (in these cases typically v_ub=0). Not inferred automatically.
    verbose : bool, default False
        If True, print solver output.
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'status': solver status string
        - 'optimal_contract': v(y) array if optimal, else None
        - 'expected_wage': E[k(v)] if optimal, else None
        - 'agent_utility': U(a0) if optimal, else None
        - 'objective_value': raw objective value from CVXPY
    """
    a0 = intended_action
    Ubar = reservation_utility
    
    # Convert a_hat to numpy array
    if a_hat is None:
        a_hat = np.array([])
    else:
        a_hat = np.asarray(a_hat)
    
    # Infer v_lb from u(0) if not provided
    # v_lb = u(0) ensures non-negative consumption (c = k(v) >= 0)
    u = problem._primitives["u"]
    if v_lb is None:
        v_lb = u(0.0)
    
    # Build cache with precomputed integration weights and densities
    cache = _make_cache(a0, a_hat, problem=problem)
    
    # Extract arrays from cache
    y_grid = problem.y_grid
    n = len(y_grid)
    m = len(a_hat)
    
    wf0 = cache["wf0"]             # (n,) = w * f(y|a0)
    wf0s0 = cache["wf0s0"]         # (n,) = w * f(y|a0) * score(y, a0)
    weighted_D = cache["weighted_D"]  # (m, n) = w * f(y|a_hat[i])
    C0 = cache["C0"]               # C(a0)
    Cprime0 = cache["Cprime0"]     # C'(a0)
    C_hat = cache["C_hat"]         # (m,) C(a_hat)
    
    # Decision variable: v(y) on the grid
    v = cp.Variable(n)
    
    # Objective: minimize E[k(v)]
    # problem.k_func(v, xp=cp) returns a cvxpy expression; wf0 @ k(v) is E[k(v)]
    k = problem.k_func
    objective = cp.Minimize(wf0 @ k(v, xp=cp))
    
    # Constraints
    constraints = []
    
    # IR: U(a0) >= Ubar  where U(a0) = wf0 @ v - C(a0)
    constraints.append(wf0 @ v >= Ubar + C0)
    
    # FOC (local IC): dU/da|_{a=a0} = 0  =>  wf0s0 @ v = C'(a0)
    constraints.append(wf0s0 @ v == Cprime0)
    
    # Global IC: U(a0) >= U(a_hat[i]) for each i
    # Vectorized: wf0 @ v (scalar, broadcast) - weighted_D @ v (m,) >= C0 - C_hat (m,)
    if m > 0:
        constraints.append(wf0 @ v - weighted_D @ v >= C0 - C_hat)
    
    # Bounds on v
    if v_lb is not None:
        constraints.append(v >= v_lb)
    if v_ub is not None:
        constraints.append(v <= v_ub)
    
    # Solve
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose)
    
    # Prepare results
    result = {
        'status': prob.status,
        'optimal_contract': None,
        'expected_wage': None,
        'agent_utility': None,
        'objective_value': prob.value,
    }
    
    if prob.status == cp.OPTIMAL:
        v_opt = v.value
        result['optimal_contract'] = v_opt
        result['agent_utility'] = wf0 @ v_opt - C0
        # Compute expected wage using numpy version of k
        # We need to evaluate k_cvx on the solution - but k_cvx returns cvxpy expr
        # Instead, use the objective value directly
        result['expected_wage'] = prob.value
    
    return result


# ============================================================================
# Efficient minimum_cost function using CVXPY with parameters
# ============================================================================

def minimum_cost_cvxpy(
    problem: MoralHazardProblem,
    intended_actions: np.ndarray,
    reservation_utility: float,
    a_hat: np.ndarray,
    v_lb: float = None,
    v_ub: float = None,
) -> np.ndarray:
    """
    Compute minimum expected wage E[k(v*)] for multiple intended actions using CVXPY.
    
    Uses CVXPY Parameters for efficiency when solving multiple similar problems.
    
    Parameters
    ----------
    problem : MoralHazardProblem
        The moral hazard problem instance.
    intended_actions : np.ndarray
        Array of actions to implement. Must be a subset of a_hat.
    reservation_utility : float
        The agent's reservation utility Ubar.
    a_hat : np.ndarray
        Array of all actions for global IC constraints.
        intended_actions must be a subset of a_hat.
    v_lb : float, optional
        Lower bound on v(y). If None, inferred from u(0).
    v_ub : float, optional
        Upper bound on v(y). Required for CARA/CRRA γ>1.
    
    Returns
    -------
    np.ndarray
        Array of minimum expected wages, same shape as intended_actions.
    """
    intended_actions = np.asarray(intended_actions).ravel()  # 1D
    a_hat = np.asarray(a_hat).ravel()  # 1D
    Ubar = reservation_utility
    p = len(intended_actions)
    m = len(a_hat)
    
    # Find indices of intended_actions in a_hat (vectorized)
    # For each intended action, find the index in a_hat where it matches
    # Using broadcasting: |intended_actions[:, None] - a_hat[None, :]| < tol
    matches = np.isclose(intended_actions[:, np.newaxis], a_hat[np.newaxis, :])  # (p, m)
    if not np.all(matches.any(axis=1)):
        missing = intended_actions[~matches.any(axis=1)]
        raise ValueError(f"intended_actions {missing} not found in a_hat")
    a0_indices = np.argmax(matches, axis=1)  # (p,) index of first match
    
    # Infer v_lb from u(0) if not provided
    u = problem._primitives["u"]
    if v_lb is None:
        v_lb = u(0.0)
    
    # Extract problem components
    y_grid = problem.y_grid
    n = len(y_grid)
    w = problem.w  # integration weights (Simpson's rule)
    f = problem.f
    score_fn = problem.score
    C = problem.C
    Cprime = problem.Cprime
    k = problem.k_func
    
    # =========================================================================
    # Pre-compute all matrices (y_grid as last dimension for efficiency)
    # =========================================================================
    
    # wf[i, j] = w[j] * f(y_grid[j], a_hat[i])  shape: (m, n)
    # Broadcasting: y_grid (n,) -> (1, n), a_hat (m,) -> (m, 1)
    wf = w[np.newaxis, :] * f(y_grid[np.newaxis, :], a_hat[:, np.newaxis])  # (m, n)
    
    # wfs[j, k] = w[k] * f(y_grid[k], a0[j]) * score(y_grid[k], a0[j])  shape: (p, n)
    # Broadcasting: y_grid (n,) -> (1, n), intended_actions (p,) -> (p, 1)
    wfs = (w[np.newaxis, :] 
           * f(y_grid[np.newaxis, :], intended_actions[:, np.newaxis]) 
           * score_fn(y_grid[np.newaxis, :], intended_actions[:, np.newaxis]))  # (p, n)
    
    # Pre-compute costs (vectorized)
    C_all = C(a_hat)                 # (m,)
    Cprime_all = Cprime(intended_actions)  # (p,)
    C0_arr = C(intended_actions)     # (p,)
    
    # =========================================================================
    # Build CVXPY problem with Parameters for efficiency
    # =========================================================================
    
    # Decision variable
    v = cp.Variable(n)
    
    # Parameters that change per intended action
    # Mark wf0_param as nonneg=True so CVXPY knows wf0 @ exp(v) is convex
    wf0_param = cp.Parameter(n, nonneg=True)  # w * f(y | a0) >= 0
    wfs0_param = cp.Parameter(n)               # w * f(y | a0) * score(y | a0)
    C0_param = cp.Parameter()                  # C(a0)
    Cprime0_param = cp.Parameter()             # C'(a0)
    
    # Objective: minimize E[k(v)]
    objective = cp.Minimize(wf0_param @ k(v, xp=cp))
    
    # Constraints
    constraints = []
    
    # IR: wf0 @ v >= Ubar + C0
    constraints.append(wf0_param @ v >= Ubar + C0_param)
    
    # FOC: wfs0 @ v == Cprime0
    constraints.append(wfs0_param @ v == Cprime0_param)
    
    # Global IC: (wf0 - wf[i]) @ v >= C0 - C_hat[i] for each i
    # Vectorized: wf0_param @ v (scalar, broadcast) - wf @ v (m,) >= C0_param - C_all (m,)
    if m > 0:
        constraints.append(wf0_param @ v - wf @ v >= C0_param - C_all)
    
    # Bounds on v
    if v_lb is not None:
        constraints.append(v >= v_lb)
    if v_ub is not None:
        constraints.append(v <= v_ub)
    
    # Build problem once
    prob = cp.Problem(objective, constraints)
    
    # =========================================================================
    # Solve for each intended action
    # =========================================================================
    
    results = np.zeros(p)
    
    for j in range(p):
        idx = a0_indices[j]
        
        # Update parameters
        wf0_param.value = wf[idx, :]
        wfs0_param.value = wfs[j, :]
        C0_param.value = C0_arr[j]
        Cprime0_param.value = Cprime_all[j]
        
        # Solve
        prob.solve(verbose=False)
        
        if prob.status == cp.OPTIMAL:
            results[j] = prob.value
        else:
            results[j] = np.nan
    
    return results  # 1D array matching intended_actions


# ---- Test the minimum_cost_cvxpy function ----
print("\n" + "="*60)
print("Testing minimum_cost_cvxpy function")
print("="*60)

# Test with multiple intended actions
a_hat_test = np.linspace(0.0, 100.0, 101)
intended_actions_test = np.linspace(0.0, 100.0, 101)

t0_mc = time.perf_counter()
min_costs = minimum_cost_cvxpy(
    problem=mhp,
    intended_actions=intended_actions_test,
    reservation_utility=reservation_utility,
    a_hat=a_hat_test,
)
t1_mc = time.perf_counter()
mc_time = t1_mc - t0_mc

print(f"Intended actions: {intended_actions_test}")
print(f"Minimum costs: {min_costs}")
print(f"Total runtime: {mc_time*1000:.2f} ms ({mc_time*1000/len(intended_actions_test):.2f} ms per action)")

# Compare with original solve_cost_minimization_cvxpy for one action
print("\nVerification against single-action solver:")
result_single = solve_cost_minimization_cvxpy(
    problem=mhp,
    intended_action=80.0,
    reservation_utility=reservation_utility,
    a_hat=a_hat_test,
)
# Index 80 corresponds to action 80.0 in linspace(0, 100, 101)
idx_80 = np.argmin(np.abs(intended_actions_test - 80.0))
print(f"  minimum_cost_cvxpy(80.0): {min_costs[idx_80]:.6f}")
print(f"  solve_cost_minimization_cvxpy(80.0): {result_single['expected_wage']:.6f}")
print(f"  Difference: {abs(min_costs[idx_80] - result_single['expected_wage']):.2e}")


# ---- Test the function version ----
print("\n" + "="*60)
print("Testing function version: solve_cost_minimization_cvxpy")
print("="*60)

t0_fn = time.perf_counter()
result_fn = solve_cost_minimization_cvxpy(
    problem=mhp,
    intended_action=80.0,
    reservation_utility=reservation_utility,
    a_hat=np.linspace(0.0, 100.0, 100),
    # v_lb and v_ub inferred from u(0) and u(∞)
)
t1_fn = time.perf_counter()
fn_time = t1_fn - t0_fn

print(f"Status: {result_fn['status']}")
print(f"Runtime: {fn_time*1000:.2f} ms")
print(f"Expected wage: {result_fn['expected_wage']:.6f}")
print(f"Agent utility: {result_fn['agent_utility']:.6f}")

# Verify it matches the procedural version
if result_fn['status'] == cp.OPTIMAL:
    diff = np.max(np.abs(result_fn['optimal_contract'] - v_opt))
    print(f"Max diff from procedural version: {diff:.2e}")
