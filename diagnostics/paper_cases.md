# Paper fallback cases

`benchmark_paper_cases.py` exercises three difficult Student-t cost-minimization
problems encountered at intermediate actions during the paper's principal-problem
search. Unlike a timing benchmark, it emphasizes independent correctness checks:

- agreement with a 101-action CVXPY master;
- maximum IC violation on an independent 3,601-action grid;
- final restricted-master KKT diagnostics;
- restricted-master failure and CVXPY-fallback counts.

Run it with:

```bash
uv run python diagnostics/benchmark_paper_cases.py \
  --output diagnostics/paper_cases_current.json
```

## What “all reparametrizations failed” means

At each active-set iteration, the solver must optimize one restricted dual master.
It first uses direct nonnegative multiplier bounds and may then try softplus and
log transformations of those multipliers. The warning means that none of those
three numerical attempts passed the restricted master's acceptance checks at
that iteration. It does **not** mean that three different economic problems
failed, nor necessarily that the final cost-minimization result failed. The
outer solver can invoke CVXPY to obtain a better active set and then solve a
later restricted master successfully.

The benchmark distinguishes these transient restricted-master failures from the
status and certification of the final returned master.

## Recorded comparison

The JSON reports compare baseline commit `dd37704` with the current source.
Across these three cases:

| Metric | `dd37704` | Current |
|---|---:|---:|
| CVXPY fallbacks | 2 | 1 |
| Maximum wage difference from CVXPY | 0.4452 | 0.00077 |
| Maximum independent dense-grid IC violation | 0.00414 | 0.0000099 |
| Final masters passing the new checks | not recorded | 3/3 |
| Outer loops explicitly certified converged | not recorded | 3/3 |

One current Student-t case still has a transient restricted-master failure and
uses CVXPY fallback. Therefore the changes improve these comparisons, but do
not yet eliminate fallbacks throughout the paper's full principal search.
Wall times in the JSON are diagnostic only and should not be used for the
paper's performance claims.
