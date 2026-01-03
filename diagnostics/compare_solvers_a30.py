# diagnostics/compare_solvers.py
"""
Compare dual and CVXPY solvers for actions where discrepancies occur.
"""
import numpy as np
import matplotlib.pyplot as plt
from moralhazard import MoralHazardProblem

# ---- Setup problem (same as tests) ----
x0 = 50.0
sigma = 10.0
first_best_effort = 100.0
theta = 1.0 / first_best_effort / (first_best_effort + x0)

def u(c):
    return np.log(x0 + c)

def k(utils, xp=np):
    return xp.exp(utils) - x0

def link_function(z):
    return np.log(np.maximum(z, x0))

def f(y, a):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((y - a) ** 2) / (2 * sigma ** 2))

def score(y, a):
    return (y - a) / (sigma ** 2)

def C(a):
    return theta * a ** 2 / 2

def Cprime(a):
    return theta * a

cfg = {
    "problem_params": {
        "u": u,
        "k": k,
        "link_function": link_function,
        "f": f,
        "score": score,
        "C": C,
        "Cprime": Cprime,
    },
    "computational_params": {
        "distribution_type": "continuous",
        "y_min": 0.0 - 3 * sigma,
        "y_max": 100.0 + 3 * sigma,
        "n": 101,
    },
}

problem = MoralHazardProblem(cfg)
reservation_utility = u(0) - 5.0

# ---- Action with discrepancy ----
action = 30.0

print(f"Comparing solvers for action a = {action}")
print("=" * 60)

# ---- Dual solver ----
print("\nRunning dual solver...")
result_dual = problem.solve_cost_minimization_problem(
    intended_action=action,
    reservation_utility=reservation_utility,
    a_ic_lb=0.0,
    a_ic_ub=100.0,
    n_a_iterations=10,
)
v_dual = result_dual.optimal_contract

print("\n" + "-" * 60)
print("DUAL SOLVER RESULT:")
print("-" * 60)
print(result_dual)

# ---- CVXPY solver ----
print("\nRunning CVXPY solver...")
a_hat = np.linspace(0.0, 100.0, 101)
result_cvxpy = problem.solve_cost_minimization_problem_cvxpy(
    intended_action=action,
    reservation_utility=reservation_utility,
    a_hat=a_hat,
)
v_cvxpy = result_cvxpy['optimal_contract']

print("\n" + "-" * 60)
print("CVXPY SOLVER RESULT:")
print("-" * 60)
for key, val in result_cvxpy.items():
    if key == 'optimal_contract':
        print(f"  {key}: array(shape={val.shape}, min={val.min():.4f}, max={val.max():.4f})")
    else:
        print(f"  {key}: {val}")

# ---- Compute derived quantities ----
y_grid = problem.y_grid
w_dual = k(v_dual)
w_cvxpy = k(v_cvxpy)

# Agent utility vs action
a_fine = np.linspace(0.0, 100.0, 201)
U_dual = problem.U(v_dual, a_fine)
U_cvxpy = problem.U(v_cvxpy, a_fine)

# Find best action for each contract
best_a_dual = a_fine[np.argmax(U_dual)]
best_a_cvxpy = a_fine[np.argmax(U_cvxpy)]

print("\n" + "-" * 60)
print("AGENT'S BEST ACTION:")
print("-" * 60)
print(f"  Dual solver - Best action for agent: {best_a_dual:.2f} (intended: {action})")
print(f"  CVXPY solver - Best action for agent: {best_a_cvxpy:.2f} (intended: {action})")

# ---- Find binding IC constraints for CVXPY solution ----
print("\n" + "-" * 60)
print("BINDING IC CONSTRAINTS (CVXPY solution):")
print("-" * 60)

U0_cvxpy = result_cvxpy['agent_utility']
U_at_a_hat = problem.U(v_cvxpy, a_hat)
ic_slack = U0_cvxpy - U_at_a_hat  # Positive means IC satisfied, zero means binding

# Find actions where IC is binding (slack < tolerance)
binding_tol = 1e-4
binding_mask = np.abs(ic_slack) < binding_tol
binding_actions = a_hat[binding_mask]
binding_slack = ic_slack[binding_mask]

print(f"  U(a0={action}) = {U0_cvxpy:.6f}")
print(f"  Number of binding IC constraints: {len(binding_actions)}")
if len(binding_actions) > 0:
    print(f"  Binding at actions: {binding_actions}")
    print(f"  IC slack at binding: {binding_slack}")

# Also show near-binding (within 0.01)
near_binding_tol = 0.01
near_binding_mask = (ic_slack >= 0) & (ic_slack < near_binding_tol) & (~binding_mask)
if np.any(near_binding_mask):
    near_binding_actions = a_hat[near_binding_mask]
    near_binding_slack = ic_slack[near_binding_mask]
    print(f"\n  Near-binding (slack < 0.01) at actions: {near_binding_actions}")
    print(f"  IC slack: {near_binding_slack}")

# Show where IC is violated (should be none for optimal solution)
violated_mask = ic_slack < -1e-6
if np.any(violated_mask):
    print(f"\n  WARNING: IC violated at actions: {a_hat[violated_mask]}")
    print(f"  Violation amount: {-ic_slack[violated_mask]}")

# ---- Create comparison plots ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Solver Comparison at a = {action}', fontsize=14, fontweight='bold')

# Plot 1: Optimal contract v(y)
ax = axes[0, 0]
ax.plot(y_grid, v_dual, 'b-', linewidth=2, label=f'Dual (E[w]={result_dual.expected_wage:.4f})')
ax.plot(y_grid, v_cvxpy, 'r--', linewidth=2, label=f'CVXPY (E[w]={result_cvxpy["expected_wage"]:.4f})')
ax.axvline(action, color='gray', linestyle=':', alpha=0.5, label=f'a={action}')
ax.set_xlabel('y (output)')
ax.set_ylabel('v(y) (utility)')
ax.set_title('Optimal Contract: v(y)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Wage function k(v(y))
ax = axes[0, 1]
ax.plot(y_grid, w_dual, 'b-', linewidth=2, label='Dual')
ax.plot(y_grid, w_cvxpy, 'r--', linewidth=2, label='CVXPY')
ax.axhline(0, color='k', linestyle=':', alpha=0.5)
ax.axvline(action, color='gray', linestyle=':', alpha=0.5, label=f'a={action}')
ax.set_xlabel('y (output)')
ax.set_ylabel('w(y) = k(v(y)) (wage)')
ax.set_title('Optimal Wage: w(y)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Agent utility vs action
ax = axes[1, 0]
ax.plot(a_fine, U_dual, 'b-', linewidth=2, label='Dual contract')
ax.plot(a_fine, U_cvxpy, 'r--', linewidth=2, label='CVXPY contract')
ax.axvline(action, color='green', linestyle='-', alpha=0.7, linewidth=2, label=f'Intended a={action}')
ax.axvline(best_a_dual, color='blue', linestyle=':', alpha=0.7, label=f'Dual best a={best_a_dual:.1f}')
ax.axvline(best_a_cvxpy, color='red', linestyle=':', alpha=0.7, label=f'CVXPY best a={best_a_cvxpy:.1f}')
ax.axhline(reservation_utility, color='gray', linestyle='--', alpha=0.5, label='Reservation utility')
# Mark binding constraints
if len(binding_actions) > 0:
    ax.scatter(binding_actions, problem.U(v_cvxpy, binding_actions), 
               color='orange', s=100, zorder=5, marker='o', label='Binding IC')
ax.set_xlabel('a (action)')
ax.set_ylabel('U(a) (agent utility)')
ax.set_title("Agent's Utility vs Action")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 4: IC slack for CVXPY solution
ax = axes[1, 1]
ax.plot(a_hat, ic_slack, 'g-', linewidth=2)
ax.axhline(0, color='k', linestyle='-', alpha=0.3)
ax.axvline(action, color='green', linestyle='-', alpha=0.7, linewidth=2, label=f'a={action}')
ax.fill_between(a_hat, ic_slack, 0, where=(ic_slack >= 0), alpha=0.3, color='green', label='IC satisfied')
ax.fill_between(a_hat, ic_slack, 0, where=(ic_slack < 0), alpha=0.3, color='red', label='IC violated')
if len(binding_actions) > 0:
    ax.scatter(binding_actions, np.zeros_like(binding_actions), 
               color='orange', s=100, zorder=5, marker='o', label='Binding')
ax.set_xlabel('a (alternative action)')
ax.set_ylabel('IC slack: U(a0) - U(a)')
ax.set_title('IC Constraint Slack (CVXPY solution)')
ax.legend()
ax.grid(True, alpha=0.3)

# Add text summary
summary_text = (
    f"Dual: E[w] = {result_dual.expected_wage:.4f}, FOA = {result_dual.first_order_approach_holds}\n"
    f"CVXPY: E[w] = {result_cvxpy['expected_wage']:.4f}, Binding IC at: {binding_actions}\n"
    f"Difference: {abs(result_dual.expected_wage - result_cvxpy['expected_wage']):.4f}"
)
fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0.06, 1, 0.96])
plt.savefig('diagnostics/figures/solver_comparison_a30.png', dpi=150)
plt.close(fig)
print(f"\nPlot saved to diagnostics/figures/solver_comparison_a30.png")

# ---- Also compare multiple actions ----
print("\n" + "=" * 60)
print("Comparing across multiple actions...")
print("=" * 60)

actions_to_compare = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]

fig2, axes2 = plt.subplots(len(actions_to_compare), 3, figsize=(15, 3 * len(actions_to_compare)))
fig2.suptitle('Solver Comparison Across Actions', fontsize=14, fontweight='bold')

for i, a0 in enumerate(actions_to_compare):
    # Dual
    res_dual = problem.solve_cost_minimization_problem(
        intended_action=a0,
        reservation_utility=reservation_utility,
        a_ic_lb=0.0,
        a_ic_ub=100.0,
        n_a_iterations=10,
    )
    
    # CVXPY
    res_cvxpy = problem.solve_cost_minimization_problem_cvxpy(
        intended_action=a0,
        reservation_utility=reservation_utility,
        a_hat=a_hat,
    )
    
    v_d = res_dual.optimal_contract
    v_c = res_cvxpy['optimal_contract']
    
    diff = abs(res_dual.expected_wage - res_cvxpy['expected_wage'])
    print(f"a={a0:5.1f}: Dual={res_dual.expected_wage:8.4f}, CVXPY={res_cvxpy['expected_wage']:8.4f}, Diff={diff:.4f}, FOA={res_dual.first_order_approach_holds}")
    
    # Contract v(y)
    axes2[i, 0].plot(y_grid, v_d, 'b-', linewidth=1.5, label='Dual')
    axes2[i, 0].plot(y_grid, v_c, 'r--', linewidth=1.5, label='CVXPY')
    axes2[i, 0].axvline(a0, color='gray', linestyle=':', alpha=0.5)
    axes2[i, 0].set_ylabel(f'a={a0}')
    if i == 0:
        axes2[i, 0].set_title('v(y)')
        axes2[i, 0].legend(fontsize=8)
    if i == len(actions_to_compare) - 1:
        axes2[i, 0].set_xlabel('y')
    
    # Wage w(y)
    axes2[i, 1].plot(y_grid, k(v_d), 'b-', linewidth=1.5)
    axes2[i, 1].plot(y_grid, k(v_c), 'r--', linewidth=1.5)
    axes2[i, 1].axhline(0, color='k', linestyle=':', alpha=0.3)
    axes2[i, 1].axvline(a0, color='gray', linestyle=':', alpha=0.5)
    if i == 0:
        axes2[i, 1].set_title('w(y) = k(v)')
    if i == len(actions_to_compare) - 1:
        axes2[i, 1].set_xlabel('y')
    
    # Agent utility
    U_d = problem.U(v_d, a_fine)
    U_c = problem.U(v_c, a_fine)
    axes2[i, 2].plot(a_fine, U_d, 'b-', linewidth=1.5)
    axes2[i, 2].plot(a_fine, U_c, 'r--', linewidth=1.5)
    axes2[i, 2].axvline(a0, color='green', linestyle='-', alpha=0.7, linewidth=2)
    axes2[i, 2].axhline(reservation_utility, color='gray', linestyle='--', alpha=0.5)
    if i == 0:
        axes2[i, 2].set_title('U(a)')
    if i == len(actions_to_compare) - 1:
        axes2[i, 2].set_xlabel('a')
    
    # Annotate with expected wage difference
    axes2[i, 2].annotate(f'Δ={diff:.3f}', xy=(0.95, 0.95), xycoords='axes fraction',
                         ha='right', va='top', fontsize=8,
                         bbox=dict(boxstyle='round', facecolor='yellow' if diff > 0.1 else 'lightgreen', alpha=0.7))

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('diagnostics/output/solver_comparison_multi.png', dpi=150)
plt.close('all')
print(f"\nMulti-action plot saved to diagnostics/output/solver_comparison_multi.png")
