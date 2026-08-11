"""Correctness and performance benchmark for the dual solver.

Run from the repository root with:
    uv run python diagnostics/benchmark_solver.py --output diagnostics/baseline.json \
        --contracts diagnostics/baseline_contracts.npz

The benchmark intentionally uses an independent dense action grid and a finite
CVXPY master.  It also records fallback warnings, optimizer work, wall time,
and fixed-master dual gradient/concavity diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moralhazard import MoralHazardProblem  # noqa: E402
from moralhazard.core import _make_cache  # noqa: E402
from moralhazard.solver import (  # noqa: E402
    _dual_value_and_grad,
    _maximize_lagrange_dual,
)


def make_gaussian_problem(n: int = 101) -> tuple[MoralHazardProblem, float]:
    x0 = 50.0
    sigma = 10.0
    theta = 1.0 / 100.0 / 150.0

    def u(c):
        return np.log(x0 + c)

    def k(v, xp=np):
        return xp.exp(v) - x0

    def link_function(z):
        return np.log(np.maximum(z, x0))

    def f(y, a):
        return np.exp(-((y - a) ** 2) / (2 * sigma**2)) / (
            np.sqrt(2 * np.pi) * sigma
        )

    def score(y, a):
        return (y - a) / sigma**2

    def C(a):
        return theta * a**2 / 2

    def Cprime(a):
        return theta * a

    problem = MoralHazardProblem(
        {
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
                "y_min": -30.0,
                "y_max": 130.0,
                "n": n,
            },
        }
    )
    return problem, float(u(0) - 5.0)


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip()


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src" / "moralhazard").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def fixed_master_diagnostics(problem, ubar) -> dict:
    """Numerically check the objective/gradient pair and convexity of -dual."""
    a0 = 20.0
    a_hat = np.array([0.0, 100.0])
    result, _ = _maximize_lagrange_dual(
        a0, ubar, a_hat, problem=problem, raise_on_failure=False
    )
    cache = _make_cache(a0, a_hat, problem=problem)
    # Check at a strictly interior point: central differences at a zero
    # inequality multiplier leave the feasible domain and cross floor kinks.
    theta = np.array([55.0, 8.0, 0.1, 0.1])
    value, analytic = _dual_value_and_grad(theta, cache, problem, ubar)
    finite_diff = np.empty_like(theta)
    for i in range(theta.size):
        step = 1e-5 * max(1.0, abs(theta[i]))
        plus = theta.copy()
        minus = theta.copy()
        plus[i] += step
        minus[i] -= step
        finite_diff[i] = (
            _dual_value_and_grad(plus, cache, problem, ubar)[0]
            - _dual_value_and_grad(minus, cache, problem, ubar)[0]
        ) / (2 * step)

    # For a convex negative dual, F(tx+(1-t)y) <= tF(x)+(1-t)F(y).
    rng = np.random.default_rng(20250308)
    max_jensen_violation = -np.inf
    for _ in range(100):
        x = theta + rng.normal(scale=[2.0, 2.0, 0.2, 0.2])
        y = theta + rng.normal(scale=[2.0, 2.0, 0.2, 0.2])
        x[[0, 2, 3]] = np.maximum(x[[0, 2, 3]], 0.0)
        y[[0, 2, 3]] = np.maximum(y[[0, 2, 3]], 0.0)
        t = rng.random()
        lhs = _dual_value_and_grad(t * x + (1 - t) * y, cache, problem, ubar)[0]
        rhs = t * _dual_value_and_grad(x, cache, problem, ubar)[0] + (
            1 - t
        ) * _dual_value_and_grad(y, cache, problem, ubar)[0]
        max_jensen_violation = max(max_jensen_violation, float(lhs - rhs))

    return {
        "a0": a0,
        "a_hat": a_hat.tolist(),
        "objective": float(value),
        "solver_success": bool(result.solver_state["success"]),
        "max_imposed_ic_violation": float(
            max(0.0, np.max(result.constraints["IC"], initial=-np.inf))
        ),
        "gradient_max_abs_error": float(np.max(np.abs(analytic - finite_diff))),
        "gradient_analytic": analytic.tolist(),
        "gradient_finite_difference": finite_diff.tolist(),
        "max_jensen_violation": max_jensen_violation,
    }


def run(actions: list[float], dense_points: int, iterations: int) -> tuple[dict, dict]:
    problem, ubar = make_gaussian_problem()
    action_grid = np.linspace(0.0, 100.0, 101)
    dense_grid = np.linspace(0.0, 100.0, dense_points)
    rows = []
    contracts = {}

    for a0 in actions:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            start = time.perf_counter()
            result = problem.solve_cost_minimization_problem(
                intended_action=a0,
                reservation_utility=ubar,
                a_ic_lb=0.0,
                a_ic_ub=100.0,
                n_a_iterations=iterations,
            )
            dual_seconds = time.perf_counter() - start

        start = time.perf_counter()
        cvx = problem.solve_cost_minimization_problem_cvxpy(
            intended_action=a0,
            reservation_utility=ubar,
            a_hat=action_grid,
        )
        cvx_seconds = time.perf_counter() - start
        dense_utility = np.asarray(problem.U(result.optimal_contract, dense_grid))
        intended_utility = float(np.asarray(problem.U(result.optimal_contract, a0)).item())
        dense_index = int(np.argmax(dense_utility))
        messages = [str(item.message) for item in caught]
        fallback_messages = [m for m in messages if "Triggering CVXPY fallback" in m]
        solver_failures = [m for m in messages if "Solver failed with" in m]
        repeated = [m for m in fallback_messages if "repeated" in m]

        rows.append(
            {
                "action": a0,
                "expected_wage": float(result.expected_wage),
                "cvxpy_expected_wage": float(cvx["expected_wage"]),
                "wage_abs_error": float(abs(result.expected_wage - cvx["expected_wage"])),
                "dense_max_ic_violation": float(dense_utility[dense_index] - intended_utility),
                "dense_best_action": float(dense_grid[dense_index]),
                "ir_violation": float(result.constraints["IR"]),
                "foc_abs_residual": float(abs(result.constraints["FOC"])),
                "max_imposed_ic_violation": float(
                    max(0.0, np.max(result.constraints["IC"], initial=-np.inf))
                ),
                "outer_iterations": int(result.n_outer_iterations),
                "final_master_iterations": int(result.solver_state["niter"]),
                "final_master_evaluations": int(result.solver_state["nfev"]),
                "final_master_success": bool(result.solver_state["success"]),
                "dual_seconds": dual_seconds,
                "cvxpy_seconds": cvx_seconds,
                "fallback_count": len(fallback_messages),
                "fallback_repeated_count": len(repeated),
                "optimizer_failure_warning_count": len(solver_failures),
                "warning_count": len(messages),
            }
        )
        contracts[f"a{a0:g}"] = result.optimal_contract
        print(
            f"a={a0:4.0f} wage={result.expected_wage:10.6f} "
            f"cvx_err={rows[-1]['wage_abs_error']:.2e} "
            f"dense_ic={rows[-1]['dense_max_ic_violation']:.2e} "
            f"fallbacks={len(fallback_messages)} time={dual_seconds:.3f}s"
        )

    report = {
        "schema_version": 1,
        "git_revision": _git_revision(),
        "source_sha256": _source_hash(),
        "actions": rows,
        "summary": {
            "fallback_count": sum(r["fallback_count"] for r in rows),
            "actions_with_fallback": sum(r["fallback_count"] > 0 for r in rows),
            "optimizer_failure_warning_count": sum(
                r["optimizer_failure_warning_count"] for r in rows
            ),
            "total_dual_seconds": sum(r["dual_seconds"] for r in rows),
            "max_wage_abs_error": max(r["wage_abs_error"] for r in rows),
            "max_dense_ic_violation": max(r["dense_max_ic_violation"] for r in rows),
            "max_imposed_ic_violation": max(
                r["max_imposed_ic_violation"] for r in rows
            ),
        },
        "fixed_master": fixed_master_diagnostics(problem, ubar),
    }
    return report, contracts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--contracts", type=Path)
    parser.add_argument("--actions", type=float, nargs="+", default=[20, 30, 40, 50, 60, 70, 80])
    parser.add_argument("--dense-points", type=int, default=2001)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    report, contracts = run(args.actions, args.dense_points, args.iterations)
    print(json.dumps(report["summary"], indent=2))
    print(json.dumps(report["fixed_master"], indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.contracts:
        args.contracts.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.contracts, **contracts)


if __name__ == "__main__":
    main()
