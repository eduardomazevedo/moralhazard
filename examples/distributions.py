import numpy as np
import matplotlib.pyplot as plt
import os

from moralhazard import MoralHazardProblem
from moralhazard.config_maker import make_utility_cfg, make_distribution_cfg

# =============================================================
# PART 1 — Build a plain list of specs to run (append items)
# =============================================================

specs_to_do = []

# ---- shared primitives ----
x0 = 50
sigma = 10.0
first_best_effort = 100
theta = 1.0 / first_best_effort / (first_best_effort + x0)

# utility (same across runs unless you change it here)
utility_cfg = make_utility_cfg("log", w0=x0)
reservation_utility = utility_cfg["u"](0.0) - 1.0

# default cost; override per-spec if you like
C  = lambda a: theta * a ** 2 / 2
Cprime = lambda a: theta * a

# ----- Append specs (procedural, nothing fancy) -----
# Each spec sets: name, dist_cfg, a0, cost functions, and optional y-grid overrides.

# Gaussian (continuous)
specs_to_do.append({
    "name": "Gaussian",
    "dist_cfg": make_distribution_cfg("gaussian", sigma=sigma),
    "a0": 80.0,
    "C": C, "Cprime": Cprime,
    # grid overrides (optional)
    "y_min": -30.0, "y_max": 150.0, "n": 201,  # n must be odd for continuous
})

# Binomial (discrete)
specs_to_do.append({
    "name": "Binomial",
    "dist_cfg": make_distribution_cfg("binomial", n=100),
    "a0": 0.8,  # probability
    "C": C, "Cprime": Cprime,
    # grid overrides (optional)
    "y_min": 0.0, "y_max": 100.0, "step_size": 1.0,
    # optional: scale cost by n so magnitudes are comparable (set to None to skip)
    "scale_cost_by": 100.0,
})

# Exponential (continuous)
specs_to_do.append({
    "name": "Exponential",
    "dist_cfg": make_distribution_cfg("exponential"),
    "a0": 80.0,
    "a_initial": 1.0,
    "a_search_lb": 1.0,
    "C": C, "Cprime": Cprime,
    "y_min": -30.0, "y_max": 300.0, "n": 201,
})

# Poisson (discrete)
specs_to_do.append({
    "name": "Poisson",
    "dist_cfg": make_distribution_cfg("poisson"),
    "a0": 80.0,  # mean
    "C": C, "Cprime": Cprime,
    "y_min": 0.0, "y_max": 120.0, "step_size": 1.0,
})

# =============================================================
# PART 2 — Single function that does all the work + plotting
# =============================================================

def solve_and_plot_distribution(*, spec: dict, utility_cfg: dict, reservation_utility: float, output_dir: str = "examples/output"):
    """Solve the problem for one spec and save the 2x2 plot.

    Required in `spec`:
        name (str), dist_cfg (dict from make_distribution_cfg), a0 (float), C, Cprime.
    Optional in `spec`:
        y_min, y_max, n (for continuous) OR step_size (for discrete), scale_cost_by (float).
    """
    dist_name = spec["name"]
    dist_cfg = spec["dist_cfg"]
    a0 = spec["a0"]

    # pull functions from configs
    f = dist_cfg["f"]
    score = dist_cfg["score"]
    u = utility_cfg["u"]
    k = utility_cfg["k"]
    g = utility_cfg["link_function"]

    # costs (optionally scaled for probability models)
    C = spec["C"]
    Cprime = spec["Cprime"]
    scale = spec.get("scale_cost_by")
    if scale is not None:
        C_orig, Cp_orig = C, Cprime
        C = lambda a, _C=C_orig, s=scale: _C(s * a)
        Cprime = lambda a, _Cp=Cp_orig, s=scale: s * _Cp(s * a)

    # auto-detect discrete vs continuous by distribution name (for labels only)
    name_lower = dist_name.lower()
    is_discrete = name_lower in ["binomial", "poisson", "bernoulli", "geometric"]

    # default grids (used to build the problem). Plotting will ALWAYS use mhp.y_grid.
    if is_discrete:
        if name_lower == "binomial":
            y_min, y_max, step = 0.0, float(dist_cfg.get("n", 100)), 1.0
        elif name_lower == "bernoulli":
            y_min, y_max, step = 0.0, 1.0, 1.0
        elif name_lower == "geometric":
            y_min, y_max, step = 0.0, 50.0, 1.0
        else:  # poisson
            y_min, y_max, step = 0.0, 120.0, 1.0
        # apply overrides
        y_min = spec.get("y_min", y_min)
        y_max = spec.get("y_max", y_max)
        step = spec.get("step_size", step)
        computational_params = {
            "distribution_type": "discrete",
            "y_min": float(y_min),
            "y_max": float(y_max),
            "step_size": float(step),
        }
    else:
        y_min, y_max, n = -30.0, 150.0, 201
        y_min = spec.get("y_min", y_min)
        y_max = spec.get("y_max", y_max)
        n = int(spec.get("n", n))
        if n % 2 == 0:  # enforce odd n if needed
            n += 1
        computational_params = {
            "distribution_type": "continuous",
            "y_min": float(y_min),
            "y_max": float(y_max),
            "n": int(n),
        }

    cfg = {
        "problem_params": {
            "u": u, "k": k, "link_function": g,
            "C": C, "Cprime": Cprime, "f": f, "score": score,
        },
        "computational_params": computational_params,
    }

    # ---- solve ----
    mhp = MoralHazardProblem(cfg)
    results = mhp.solve_cost_minimization_problem(
        intended_action=a0,
        reservation_utility=reservation_utility,
        a_hat=np.array([0.0, 0.0]),
        solver="iterative",
        a_initial=spec.get("a_initial", 0.0),
        a_search_lb=spec.get("a_search_lb", -np.inf),
    )

    print(f"{dist_name} - Multipliers found:")
    print(results.multipliers)

    # ---- PLOTTING (STRICTLY use mhp.y_grid) ----
    y = mhp.y_grid  # single source of truth for x-axis in y-based plots
    v = results.optimal_contract
    wage = mhp.k(v)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{dist_name} Distribution Analysis', fontsize=16, fontweight='bold')

    # 1) Wage schedule (values aligned to y)
    ax = axes[0, 0]
    if is_discrete:
        ax.scatter(y, wage, s=30, alpha=0.8, edgecolors='black', linewidth=0.5)
    else:
        ax.plot(y, wage)
    ax.set_xlabel('Output y')
    ax.set_ylabel('Wage k(v*(y))')
    ax.set_title('Optimal Wage Schedule')
    ax.grid(True, alpha=0.3)

    # 2) Agent's utility vs action (not y-based; still a continuous sweep over a)
    ax = axes[0, 1]
    if name_lower in ["binomial", "bernoulli"]:
        a_grid = np.linspace(0.1, 0.9, 100)
        ax.set_xlabel('Action a (probability)')
    elif name_lower == "geometric":
        a_grid = np.linspace(1.1, 10.0, 100)
        ax.set_xlabel('Action a (mean)')
    else:
        a_grid = np.linspace(0.0, 110.0, 100)
        ax.set_xlabel('Action a')
    Ua = mhp.U(v, a_grid)
    ax.plot(a_grid, Ua)
    ax.set_ylabel('U(a)')
    ax.set_title('Agent Utility from Optimal Contract')
    ax.grid(True, alpha=0.3)

    # 3) f(y|a0) sampled EXACTLY on mhp.y_grid
    ax = axes[1, 0]
    f_vals = f(y, a0)
    if is_discrete:
        ax.scatter(y, f_vals, s=30, alpha=0.8, edgecolors='black', linewidth=0.5)
    else:
        ax.plot(y, f_vals)
    ax.set_xlabel('Output y')
    ax.set_ylabel('f(y|a)')
    ax.set_title(f'Distribution f(y|a={a0:.1f})')
    ax.grid(True, alpha=0.3)

    # 4) score(y, a0) sampled EXACTLY on mhp.y_grid
    ax = axes[1, 1]
    score_vals = score(y, a0)
    if is_discrete:
        ax.scatter(y, score_vals, s=30, alpha=0.8, edgecolors='black', linewidth=0.5)
    else:
        ax.plot(y, score_vals)
    ax.set_xlabel('Output y')
    ax.set_ylabel('score(y, a)')
    ax.set_title(f'Score Function score(y, a={a0:.1f})')
    ax.grid(True, alpha=0.3)

    # save
    os.makedirs(output_dir, exist_ok=True)
    filename = f"distributions-{dist_name.lower()}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {filepath}")

    return mhp, results, y, f_vals


# =============================================================
# Run all specs
# =============================================================
if __name__ == "__main__":
    print("Solving and plotting distributions...")
    for spec in specs_to_do:
        print(f"Processing {spec['name']}...")
        solve_and_plot_distribution(spec=spec, utility_cfg=utility_cfg, reservation_utility=reservation_utility)

    print("All done. Plots saved under examples/output/")
