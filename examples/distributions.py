import numpy as np
import matplotlib.pyplot as plt
import os
from moralhazard import MoralHazardProblem
from moralhazard.config_maker import make_utility_cfg, make_distribution_cfg

def solve_and_plot_distribution(
    dist_name: str,
    dist_cfg: dict,
    utility_cfg: dict,
    cost_functions: dict,
    reservation_utility: float,
    intended_action: float,
    n_subplots: int = 4
):
    """
    Solve moral hazard problem for a given distribution and create plots.
    
    Args:
        dist_name: Name of the distribution for plot titles
        dist_cfg: Distribution configuration from config_maker
        utility_cfg: Utility configuration from config_maker
        cost_functions: Dict with 'C' and 'Cprime' functions
        reservation_utility: Reservation utility level
        intended_action: Intended action for the distribution
        n_subplots: Number of subplots to create (default 4)
    
    Returns:
        tuple: (mhp, results, y_grid, f_values)
    """
    # Extract functions
    f = dist_cfg["f"]
    score = dist_cfg["score"]
    u = utility_cfg["u"]
    k = utility_cfg["k"]
    g = utility_cfg["link_function"]
    
    # Create distribution-specific cost functions for comparability
    if dist_name.lower() in ["binomial", "bernoulli"]:
        # For probability-based distributions, scale cost to be comparable
        # If binomial n=100 and a=0.8, mean = 80, so scale cost by 100
        scale_factor = dist_cfg.get("n", 100) if dist_name.lower() == "binomial" else 1.0
        def C_scaled(a): return cost_functions["C"](scale_factor * a)
        def Cprime_scaled(a): return scale_factor * cost_functions["Cprime"](scale_factor * a)
        C = C_scaled
        Cprime = Cprime_scaled
    else:
        # For continuous distributions, use original cost functions
        C = cost_functions["C"]
        Cprime = cost_functions["Cprime"]
    
    # Determine distribution type and grid parameters
    if dist_name.lower() in ["binomial", "bernoulli", "geometric"]:
        # Discrete distributions
        if dist_name.lower() == "binomial":
            n_trials = dist_cfg.get("n", 100)
            y_max = float(n_trials)
            action_range = (0.1, 0.9)  # probability range
        elif dist_name.lower() == "bernoulli":
            y_max = 1.0
            action_range = (0.1, 0.9)
        else:  # geometric
            y_max = 50.0
            action_range = (1.1, 10.0)
            
        cfg = {
            "problem_params": {
                "u": u, "k": k, "link_function": g,
                "C": C, "Cprime": Cprime, "f": f, "score": score,
            },
            "computational_params": {
                "distribution_type": "discrete",
                "y_min": 0.0,
                "y_max": y_max,
                "step_size": 1.0,
            },
        }
    elif dist_name.lower() == "poisson":
        # Poisson: only non-negative integers, need reasonable upper bound
        # For mean=80, 99.9% of mass is below mean + 3*sqrt(mean) ≈ 80 + 3*9 ≈ 107
        y_max = 120.0  # reasonable upper bound for Poisson with mean 80
        
        cfg = {
            "problem_params": {
                "u": u, "k": k, "link_function": g,
                "C": C, "Cprime": Cprime, "f": f, "score": score,
            },
            "computational_params": {
                "distribution_type": "discrete",
                "y_min": 0.0,
                "y_max": y_max,
                "step_size": 1.0,
            },
        }
    else:
        # Continuous distributions
        cfg = {
            "problem_params": {
                "u": u, "k": k, "link_function": g,
                "C": C, "Cprime": Cprime, "f": f, "score": score,
            },
            "computational_params": {
                "distribution_type": "continuous",
                "y_min": 0.0 - 3 * 10.0,  # assuming sigma=10 for continuous
                "y_max": 120.0 + 3 * 10.0,
                "n": 201,  # must be odd
            },
        }
    
    # Solve the problem
    mhp = MoralHazardProblem(cfg)
    results = mhp.solve_cost_minimization_problem(
        intended_action=intended_action,
        reservation_utility=reservation_utility,
        a_hat=np.array([0.0, 0.0]),
        solver="iterative"
    )
    
    print(f"{dist_name} - Multipliers found:")
    print(results.multipliers)
    
    # Create individual figure for this distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{dist_name} Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Create plots
    y_grid = mhp.y_grid
    v = results.optimal_contract
    wage = mhp.k(v)
    
    # Determine if this is a discrete distribution
    is_discrete = dist_name.lower() in ["binomial", "poisson", "bernoulli", "geometric"]
    
    # 1) Wage schedule
    ax = axes[0, 0]
    if is_discrete:
        ax.scatter(y_grid, wage, s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
    else:
        ax.plot(y_grid, wage)
    ax.set_xlabel('Output y')
    ax.set_ylabel('Wage k(v*(y))')
    ax.set_title('Optimal Wage Schedule')
    ax.grid(True, alpha=0.3)
    
    # 2) Agent's utility from optimal contract vs action
    ax = axes[0, 1]
    if dist_name.lower() in ["binomial", "bernoulli"]:
        a_grid = np.linspace(action_range[0], action_range[1], 100)
        ax.set_xlabel('Action a (probability)')
    elif dist_name.lower() == "geometric":
        a_grid = np.linspace(action_range[0], action_range[1], 100)
        ax.set_xlabel('Action a (mean)')
    else:
        a_grid = np.linspace(0.0, 110.0, 100)
        ax.set_xlabel('Action a')
    
    Ua = mhp.U(v, a_grid)
    ax.plot(a_grid, Ua)
    ax.set_ylabel('U(a)')
    ax.set_title('Agent Utility from Optimal Contract')
    ax.grid(True, alpha=0.3)
    
    # 3) Distribution f(y|a) for intended action
    ax = axes[1, 0]
    f_values = f(y_grid, intended_action)
    if is_discrete:
        ax.scatter(y_grid, f_values, s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
    else:
        ax.plot(y_grid, f_values)
    ax.set_xlabel('Output y')
    ax.set_ylabel('f(y|a)')
    ax.set_title(f'Distribution f(y|a={intended_action:.1f})')
    ax.grid(True, alpha=0.3)
    
    # 4) Score function score(y, a) for intended action
    ax = axes[1, 1]
    score_values = score(y_grid, intended_action)
    if is_discrete:
        ax.scatter(y_grid, score_values, s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
    else:
        ax.plot(y_grid, score_values)
    ax.set_xlabel('Output y')
    ax.set_ylabel('score(y, a)')
    ax.set_title(f'Score Function score(y, a={intended_action:.1f})')
    ax.grid(True, alpha=0.3)
    
    # Save the figure
    output_dir = "examples/output"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"distributions-{dist_name.lower()}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    print(f"Saved plot to: {filepath}")
    
    return mhp, results, y_grid, f_values

# ---- primitives (same as prototype Normal model) ----
x0 = 50
sigma = 10.0
first_best_effort = 100
theta = 1.0 / first_best_effort / (first_best_effort + x0)

# ---- create utility functions using config_maker ----
utility_cfg = make_utility_cfg("log", w0=x0)
reservation_utility = utility_cfg["u"](0.0) - 1.0

# ---- create cost functions ----
def C(a): return theta * a ** 2 / 2
def Cprime(a): return theta * a
cost_functions = {"C": C, "Cprime": Cprime}

# ---- create distribution configurations ----
distributions = {
    "Gaussian": {
        "cfg": make_distribution_cfg("gaussian", sigma=sigma),
        "intended_action": 80.0
    },
    "Binomial": {
        "cfg": make_distribution_cfg("binomial", n=100),
        "intended_action": 0.8
    },
    "Exponential": {
        "cfg": make_distribution_cfg("exponential"),
        "intended_action": 80.0  # mean = 80, same as gaussian
    },
    "Poisson": {
        "cfg": make_distribution_cfg("poisson"),
        "intended_action": 80.0  # mean = 80, same as gaussian
    }
}

# ---- solve and plot all distributions ----
print("Solving and plotting distributions...")

for dist_name, dist_info in distributions.items():
    print(f"\nProcessing {dist_name} distribution...")
    mhp, results, y_grid, f_values = solve_and_plot_distribution(
        dist_name=dist_name,
        dist_cfg=dist_info["cfg"],
        utility_cfg=utility_cfg,
        cost_functions=cost_functions,
        reservation_utility=reservation_utility,
        intended_action=dist_info["intended_action"]
    )

print(f"\nExample completed successfully!")
print(f"Used config_maker to create:")
print(f"  - Utility functions: {list(utility_cfg.keys())}")
print(f"  - Cost functions: {list(cost_functions.keys())}")
print(f"\nDistributions analyzed:")
for dist_name, dist_info in distributions.items():
    print(f"  - {dist_name}: {list(dist_info['cfg'].keys())}")
print(f"\nAll functions are broadcastable and handle array inputs")
print(f"To add more distributions, simply add them to the 'distributions' dictionary!")
print(f"\nPlots saved to: examples/output/")
