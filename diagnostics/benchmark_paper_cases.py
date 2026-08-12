"""Correctness and fallback benchmark using specifications from the paper.

These are deliberately difficult Student-t cases encountered while generating
Azevedo and Wolff's figures.  This is not a standardized timing benchmark:
wall time is recorded only to spot order-of-magnitude regressions.

Run from the package repository root:
    uv run python diagnostics/benchmark_paper_cases.py \
        --output diagnostics/paper_cases_current.json
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np

from moralhazard import MoralHazardProblem
from moralhazard.config_maker import make_distribution_cfg, make_utility_cfg
import moralhazard


CASES = [
    {
        "name": "student-t-s10-intermediate-principal-action",
        "sigma": 10.0,
        "intended_action": 111.24611797498108,
        "reservation_wage": 24.25,
    },
    {
        "name": "student-t-s20-intermediate-principal-action",
        "sigma": 20.0,
        "intended_action": 107.26009298535428,
        "reservation_wage": 24.25,
    },
    {
        "name": "student-t-s20-distant-principal-action",
        "sigma": 20.0,
        "intended_action": 137.50776405003785,
        "reservation_wage": 24.25,
    },
]


def make_problem(sigma: float) -> tuple[MoralHazardProblem, dict]:
    initial_wealth = 50.0
    theta = 1.0 / 100.0 / (100.0 + initial_wealth)
    utility = make_utility_cfg("log", w0=initial_wealth)
    distribution = make_distribution_cfg("Student_t", sigma=sigma, nu=1.15)

    def cost(action):
        return theta * action**2 / 2

    def cost_prime(action):
        return theta * action

    problem = MoralHazardProblem({
        "problem_params": {
            **utility,
            **distribution,
            "C": cost,
            "Cprime": cost_prime,
        },
        "computational_params": {
            "distribution_type": "continuous",
            "y_min": -10.0 * sigma,
            "y_max": 180.0 + 10.0 * sigma,
            "n": 201,
        },
    })
    return problem, utility


def source_hash() -> str:
    digest = hashlib.sha256()
    package_dir = Path(inspect.getfile(moralhazard)).resolve().parent
    for path in sorted(package_dir.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def package_revision() -> str | None:
    package_file = Path(inspect.getfile(moralhazard)).resolve()
    for parent in package_file.parents:
        if (parent / ".git").exists():
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=parent,
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
    return None


def run_case(case: dict, dense_points: int = 3601) -> dict:
    problem, utility = make_problem(case["sigma"])
    action = case["intended_action"]
    reservation_utility = float(utility["u"](case["reservation_wage"]))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.perf_counter()
        result = problem.solve_cost_minimization_problem(
            intended_action=action,
            reservation_utility=reservation_utility,
            a_ic_lb=0.0,
            a_ic_ub=180.0,
            n_a_iterations=100,
        )
        dual_seconds = time.perf_counter() - start

    cvxpy_actions = np.linspace(0.0, 180.0, 101)
    start = time.perf_counter()
    cvxpy = problem.solve_cost_minimization_problem_cvxpy(
        intended_action=action,
        reservation_utility=reservation_utility,
        a_hat=cvxpy_actions,
    )
    cvxpy_seconds = time.perf_counter() - start

    dense_actions = np.linspace(0.0, 180.0, dense_points)
    dense_utility = np.asarray(problem.U(result.optimal_contract, dense_actions))
    intended_utility = float(
        np.asarray(problem.U(result.optimal_contract, action)).item()
    )
    best_index = int(np.argmax(dense_utility))
    messages = [str(item.message) for item in caught]
    fallbacks = [message for message in messages if "Triggering CVXPY fallback" in message]
    master_failures = [
        message for message in messages
        if "All reparametrizations failed" in message
    ]

    return {
        **case,
        "expected_wage": float(result.expected_wage),
        "cvxpy_expected_wage": float(cvxpy["expected_wage"]),
        "wage_abs_error": float(abs(result.expected_wage - cvxpy["expected_wage"])),
        "dense_max_ic_violation": float(dense_utility[best_index] - intended_utility),
        "dense_best_action": float(dense_actions[best_index]),
        "foc_abs_residual": float(abs(result.constraints["FOC"])),
        "max_imposed_ic_violation": float(
            max(0.0, np.max(result.constraints["IC"], initial=-np.inf))
        ),
        "final_master_success": bool(result.solver_state["success"]),
        "outer_converged": result.solver_state.get("outer_converged"),
        "projected_grad_inf": result.solver_state.get("projected_grad_inf"),
        "primal_residual": result.solver_state.get("primal_residual"),
        "complementarity": result.solver_state.get("complementarity"),
        "outer_iterations": int(result.n_outer_iterations),
        "final_action_count": int(len(result.a_hat)),
        "restricted_master_failure_count": len(master_failures),
        "cvxpy_fallback_count": len(fallbacks),
        "fallback_reasons": [
            "solver_failed" if "solver failed" in message else "repeated_action"
            for message in fallbacks
        ],
        "warning_count": len(messages),
        "dual_seconds": dual_seconds,
        "cvxpy_seconds": cvxpy_seconds,
    }


def run(dense_points: int = 3601) -> dict:
    rows = []
    for case in CASES:
        row = run_case(case, dense_points)
        rows.append(row)
        print(
            f"{row['name']}: wage_error={row['wage_abs_error']:.2e} "
            f"dense_ic={row['dense_max_ic_violation']:.2e} "
            f"master_failures={row['restricted_master_failure_count']} "
            f"fallbacks={row['cvxpy_fallback_count']}"
        )
    return {
        "schema_version": 1,
        "package_revision": package_revision(),
        "source_sha256": source_hash(),
        "cases": rows,
        "summary": {
            "case_count": len(rows),
            "restricted_master_failure_count": sum(
                row["restricted_master_failure_count"] for row in rows
            ),
            "cvxpy_fallback_count": sum(row["cvxpy_fallback_count"] for row in rows),
            "max_wage_abs_error": max(row["wage_abs_error"] for row in rows),
            "max_dense_ic_violation": max(
                row["dense_max_ic_violation"] for row in rows
            ),
            "all_final_masters_successful": all(
                row["final_master_success"] for row in rows
            ),
            "all_outer_loops_converged": all(row["outer_converged"] for row in rows),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dense-points", type=int, default=3601)
    args = parser.parse_args()
    report = run(args.dense_points)
    print(json.dumps(report["summary"], indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
