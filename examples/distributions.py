# examples/distribution_plots_refactor.py

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

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

# default cost; override per-spec if needed
C  = lambda a: theta * a ** 2 / 2
Cprime = lambda a: theta * a

# ----- Append specs -----

# Gaussian (continuous)
specs_to_do.append({
    "name": "Gaussian",
    "dist_cfg": make_distribution_cfg("gaussian", sigma=sigma),
    "a0": 80.0,
    "C": C, "Cprime": Cprime,
    "y_min": -30.0, "y_max": 150.0, "n": 201,
})

# Binomial (discrete)
specs_to_do.append({
    "name": "Binomial",
    "dist_cfg": make_distribution_cfg("binomial", n=100),
    "a0": 0.8,
    "C": C, "Cprime": Cprime,
    "y_min": 0.0, "y_max": 100.0, "step_size": 1.0,
    "scale_cost_by": 100.0,
})

# Exponential (continuous)
specs_to_do.append({
    "name": "Exponential",
    "dist_cfg": make_distribution_cfg("exponential"),
    "a0": 80.0,
    "a_ic_initial": 1.0,
    "a_ic_lb": 1.0,
    "C": C, "Cprime": Cprime,
    "y_min": -30.0, "y_max": 300.0, "n": 201,
    "plot_a_min": 1.0, "plot_a_max": 110.0,
})

# Poisson (discrete)
specs_to_do.append({
    "name": "Poisson",
    "dist_cfg": make_distribution_cfg("poisson"),
    "a0": 80.0,
    "C": C, "Cprime": Cprime,
    "y_min": 0.0, "y_max": 120.0, "step_size": 1.0,
})

# Bernoulli (discrete)
specs_to_do.append({
    "name": "Bernoulli",
    "dist_cfg": make_distribution_cfg("bernoulli"),
    "a0": 0.8,
    "C": C, "Cprime": Cprime,
    "y_min": 0.0, "y_max": 1.0, "step_size": 1.0,
    "scale_cost_by": 100.0,
})

# Geometric (discrete)
specs_to_do.append({
    "name": "Geometric",
    "dist_cfg": make_distribution_cfg("geometric"),
    "a0": 80.0,
    "a_ic_initial": 1.1,
    "a_ic_lb": 1.1,
    "C": C, "Cprime": Cprime,
    "y_min": 1.0, "y_max": 300.0, "step_size": 1.0,
    "plot_a_min": 1.1, "plot_a_max": 110.0 / 3.0,
})

# Gamma (continuous)
specs_to_do.append({
    "name": "Gamma",
    "dist_cfg": make_distribution_cfg("gamma", n=3.0),
    "a0": 80.0 / 3.0,
    "a_ic_initial": 1.0,
    "a_ic_lb": 1.0,
    "C": C, "Cprime": Cprime,
    "y_min": 0.0, "y_max": 200.0, "n": 201,
    "scale_cost_by": 3.0,
    "plot_a_min": 1.0, "plot_a_max": 110.0 / 3.0,
})

# Student's t (continuous)
specs_to_do.append({
    "name": "Student_t",
    "dist_cfg": make_distribution_cfg("student_t", nu=5.0, sigma=sigma),
    "a0": 80.0,
    "C": C, "Cprime": Cprime,
    "y_min": -50.0, "y_max": 200.0, "n": 201,
})

# =============================================================
# PART 2 — Function to solve + plot
# =============================================================

def solve_and_plot_distribution(*, spec: dict, utility_cfg: dict, reservation_utility: float, output_dir: str):
    dist_name = spec["name"]
    dist_cfg = spec["dist_cfg"]
    a0 = spec["a0"]

    # primitives
    f = dist_cfg["f"]
    score = dist_cfg["score"]
    u = utility_cfg["u"]
    k = utility_cfg["k"]
    g = utility_cfg["link_function"]

    # cost (maybe scaled)
    C = spec["C"]
    Cprime = spec["Cprime"]
    scale = spec.get("scale_cost_by")
    if scale is not None:
        C_orig, Cp_orig = C, Cprime
        C = lambda a, _C=C_orig, s=scale: _C(s * a)
        Cprime = lambda a, _Cp=Cp_orig, s=scale: s * _Cp(s * a)

    # continuous vs discrete grids
    name_lower = dist_name.lower()
    if name_lower in ["binomial", "poisson", "bernoulli", "geometric"]:
        computational_params = {
            "distribution_type": "discrete",
            "y_min": float(spec.get("y_min", 0.0)),
            "y_max": float(spec.get("y_max", 100.0)),
            "step_size": float(spec.get("step_size", 1.0)),
        }
    else:
        n = int(spec.get("n", 201))
        if n % 2 == 0:
            n += 1
        computational_params = {
            "distribution_type": "continuous",
            "y_min": float(spec.get("y_min", -30.0)),
            "y_max": float(spec.get("y_max", 150.0)),
            "n": n,
        }

    cfg = {
        "problem_params": {
            "u": u, "k": k, "link_function": g,
            "C": C, "Cprime": Cprime, "f": f, "score": score,
        },
        "computational_params": computational_params,
    }

    mhp = MoralHazardProblem(cfg)

    # solve principal problem
    revenue = lambda a: float(a)
    res = mhp.solve_principal_problem(
        revenue_function=revenue,
        reservation_utility=reservation_utility,
        a_min=spec.get("a_ic_lb", 0.0),
        a_max=spec.get("y_max", 150.0),
        a_init=a0,
        solver="iterative",
        a_ic_initial=spec.get("a_ic_initial", 0.0),
        a_ic_lb=spec.get("a_ic_lb", -np.inf),
    )
    a_star = float(res.optimal_action)
    v_star = res.optimal_contract

    # for plots
    y = mhp.y_grid
    wage = mhp.k(v_star)

    # action grid for utility plotting
    name_lower = dist_name.lower()
    if name_lower == "binomial":
        a_grid = np.linspace(spec.get("plot_a_min", 0.1), spec.get("plot_a_max", 0.9), 100)
    elif name_lower == "bernoulli":
        a_grid = np.linspace(spec.get("plot_a_min", 0.0), spec.get("plot_a_max", 1.0), 2)
    elif name_lower == "geometric":
        a_grid = np.linspace(spec.get("plot_a_min", 1.1), spec.get("plot_a_max", 10.0), 100)
    else:
        a_grid = np.linspace(spec.get("plot_a_min", 0.0), spec.get("plot_a_max", 110.0), 100)

    Ua = mhp.U(v_star, a_grid)
    f_vals = f(y, a0)
    score_vals = score(y, a0)

    # --- plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{dist_name} Distribution Analysis\nOptimal a* = {a_star:.2f}, Profit = {float(res.profit):.4f}',
                 fontsize=16, fontweight='bold')

    # 1) Wage schedule
    ax = axes[0, 0]
    if name_lower in ["binomial", "poisson", "bernoulli", "geometric"]:
        ax.scatter(y, wage, s=30, alpha=0.8, edgecolors='black', linewidth=0.5)
    else:
        ax.plot(y, wage)
    ax.set_xlabel("Output y")
    ax.set_ylabel("Wage k(v*(y))")
    ax.set_title("Optimal Wage Schedule")
    ax.grid(True, alpha=0.3)

    # 2) Utility vs action
    ax = axes[0, 1]
    ax.plot(a_grid, Ua)
    ax.axvline(a_star, linestyle=":", color="red", alpha=0.7, label=f"Optimal a* = {a_star:.2f}")
    ax.set_xlabel("Action a")
    ax.set_ylabel("U(a)")
    ax.set_title("Agent Utility from Optimal Contract")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3) Distribution f(y|a0)
    ax = axes[1, 0]
    if name_lower in ["binomial", "poisson", "bernoulli", "geometric"]:
        ax.bar(y, f_vals, alpha=0.7)
    else:
        ax.plot(y, f_vals)
    ax.set_xlabel("Output y")
    ax.set_ylabel("f(y|a)")
    ax.set_title(f"Distribution f(y|a={a0:.1f})")
    ax.grid(True, alpha=0.3)

    # 4) Score function
    ax = axes[1, 1]
    if name_lower in ["binomial", "poisson", "bernoulli", "geometric"]:
        ax.scatter(y, score_vals, s=30, alpha=0.8, edgecolors='black', linewidth=0.5)
    else:
        ax.plot(y, score_vals)
    ax.set_xlabel("Output y")
    ax.set_ylabel("score(y, a)")
    ax.set_title(f"Score Function score(y, a={a0:.1f})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # save
    out_dir = Path(output_dir) / "distributions"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{dist_name.lower()}.png"
    filepath = out_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    return mhp, res


# =============================================================
# PART 3 — Run all specs and save plots
# =============================================================

if __name__ == "__main__":
    out_dir = "output/distributions"
    os.makedirs(out_dir, exist_ok=True)

    for spec in specs_to_do:
        solve_and_plot_distribution(spec=spec, utility_cfg=utility_cfg, reservation_utility=reservation_utility, output_dir=out_dir)

    print("All distribution plots saved under output/distributions/")
