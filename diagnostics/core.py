# Diagnostics for core.py functions
import sys
import os
import numpy as np

# Add src directory to path to access internal modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from moralhazard.core import _make_cache, _canonical_contract, _constraints, _compute_expected_utility
from moralhazard.problem import MoralHazardProblem

# Redirect output to file
output_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'core.txt')
sys.stdout = open(output_path, 'w')

# Set numpy print options for better formatting
np.set_printoptions(precision=6, suppress=True, linewidth=100)

print("=" * 70)
print("Core Function Diagnostics")
print("=" * 70)
print()

# ---- Setup tiny test primitives ----
x0 = 50.0
sigma = 10.0
first_best_effort = 100.0
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

# ---- Create a MoralHazardProblem instance for testing ----
y_min = 0.0 - 3 * sigma
y_max = 100.0 + 3 * sigma
n = 11  # Small odd number for tiny example

cfg = {
    "problem_params": {"u": u, "k": k, "link_function": g, "C": C, "Cprime": Cprime, "f": f, "score": score},
    "computational_params": {"distribution_type": "continuous", "y_min": y_min, "y_max": y_max, "n": n},
}

mhp = MoralHazardProblem(cfg)
y_grid = mhp.y_grid
w = mhp.w

print(f"Grid setup:")
print(f"  y_min = {y_min:.2f}, y_max = {y_max:.2f}, n = {n}")
print(f"  y_grid shape: {y_grid.shape}")
print(f"  y_grid = {y_grid}")
print(f"  w shape: {w.shape}")
print(f"  w = {w}")
print()

# ---- Test 1: _make_cache ----
print("=" * 70)
print("Test 1: _make_cache")
print("=" * 70)

a0 = 80.0
a_hat = np.array([60.0, 100.0])

cache = _make_cache(
    a0=a0,
    a_hat=a_hat,
    problem=mhp,
    clip_ratio=1e6,
)

print(f"Inputs:")
print(f"  a0 = {a0}")
print(f"  a_hat = {a_hat}")
print()

print("Cache keys:", list(cache.keys()))
print()

print("Cache outputs:")
for key in ["f0", "s0", "wf0", "wf0s0"]:
    val = cache[key]
    print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
    print(f"    {key} = {val}")
    print()

for key in ["D", "R", "WD_T"]:
    val = cache[key]
    print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
    if val.ndim == 2:
        print(f"    {key} =")
        for i, row in enumerate(val):
            print(f"      [{i}] {row}")
    else:
        print(f"    {key} = {val}")
    print()

for key in ["C0", "Cprime0"]:
    val = cache[key]
    print(f"  {key}: {val} (type: {type(val).__name__})")
    print()

val = cache["C_hat"]
print(f"  C_hat: shape={val.shape}, dtype={val.dtype}")
print(f"    values: {val}")
print()

# ---- Test 1b: _make_cache with empty a_hat ----
print("=" * 70)
print("Test 1b: _make_cache with empty a_hat")
print("=" * 70)

a0_empty = 80.0
a_hat_empty = np.array([])  # Empty array

cache_empty = _make_cache(
    a0=a0_empty,
    a_hat=a_hat_empty,
    problem=mhp,
    clip_ratio=1e6,
)

print(f"Inputs:")
print(f"  a0 = {a0_empty}")
print(f"  a_hat = {a_hat_empty} (empty array, shape={a_hat_empty.shape})")
print()

print("Cache keys:", list(cache_empty.keys()))
print()

print("Cache outputs:")
for key in ["f0", "s0", "wf0", "wf0s0"]:
    val = cache_empty[key]
    print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
    print(f"    {key} = {val}")
    print()

for key in ["D", "R", "WD_T"]:
    val = cache_empty[key]
    print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
    if val.ndim == 2:
        if val.shape[1] == 0:
            print(f"    {key} = (empty matrix with shape {val.shape})")
        else:
            print(f"    {key} =")
            for i, row in enumerate(val):
                print(f"      [{i}] {row}")
    else:
        print(f"    {key} = {val}")
    print()

for key in ["C0", "Cprime0"]:
    val = cache_empty[key]
    print(f"  {key}: {val} (type: {type(val).__name__})")
    print()

val = cache_empty["C_hat"]
print(f"  C_hat: shape={val.shape}, dtype={val.dtype}")
if val.shape[0] == 0:
    print(f"    C_hat = (empty array)")
else:
    print(f"    values: {val}")
print()

# ---- Test 2: _canonical_contract ----
print("=" * 70)
print("Test 2: _canonical_contract")
print("=" * 70)

lam = 100.0
mu = 100.0
mu_hat = np.array([0.1, 0.2])
s0 = cache["s0"]
R = cache["R"]

v = _canonical_contract(
    lam=lam,
    mu=mu,
    mu_hat=mu_hat,
    s0=s0,
    R=R,
    problem=mhp,
)

print(f"Inputs:")
print(f"  lam = {lam}")
print(f"  mu = {mu}")
print(f"  mu_hat = {mu_hat}")
print(f"  s0 shape: {s0.shape}")
print(f"  R shape: {R.shape}")
print()

print(f"Output:")
print(f"  v shape: {v.shape}")
print(f"  v dtype: {v.dtype}")
print(f"  v = {v}")
print(f"  v min = {np.min(v):.6f}, max = {np.max(v):.6f}")
print()

# ---- Test 3: _constraints ----
print("=" * 70)
print("Test 3: _constraints")
print("=" * 70)

Ubar = 3.0

constraint_results = _constraints(
    v=v,
    cache=cache,
    problem=mhp,
    Ubar=Ubar,
)

print(f"Inputs:")
print(f"  v shape: {v.shape}")
print(f"  Ubar = {Ubar}")
print()

print("Constraint outputs:")
for key in ["U0", "IR", "FOC", "Ewage"]:
    val = constraint_results[key]
    print(f"  {key} = {val:.6f} (type: {type(val).__name__})")
    print()

for key in ["Uhat", "IC"]:
    val = constraint_results[key]
    print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
    print(f"    values: {val}")
    print()

# ---- Test 4: _compute_expected_utility (scalar a) ----
print("=" * 70)
print("Test 4: _compute_expected_utility (scalar action)")
print("=" * 70)

a_scalar = 75.0

U_scalar = _compute_expected_utility(
    v=v,
    a=a_scalar,
    problem=mhp,
)

print(f"Inputs:")
print(f"  v shape: {v.shape}")
print(f"  a = {a_scalar} (scalar)")
print()

print(f"Output:")
# Convert numpy scalar/array to Python float for formatting
U_scalar_val = float(np.asarray(U_scalar).item())
print(f"  U = {U_scalar_val:.6f} (type: {type(U_scalar).__name__})")
print()

# ---- Test 5: _compute_expected_utility (array a) ----
print("=" * 70)
print("Test 5: _compute_expected_utility (array of actions)")
print("=" * 70)

a_array = np.array([60.0, 75.0, 90.0, 100.0])

U_array = _compute_expected_utility(
    v=v,
    a=a_array,
    problem=mhp,
)

print(f"Inputs:")
print(f"  v shape: {v.shape}")
print(f"  a = {a_array}")
print()

print(f"Output:")
print(f"  U shape: {U_array.shape}")
print(f"  U dtype: {U_array.dtype}")
print(f"  U = {U_array}")
print()

# ---- Summary ----
print("=" * 70)
print("Summary")
print("=" * 70)
print("All core functions executed successfully!")
print(f"  - _make_cache: ✓")
print(f"  - _make_cache (empty a_hat): ✓")
print(f"  - _canonical_contract: ✓")
print(f"  - _constraints: ✓")
print(f"  - _compute_expected_utility (scalar): ✓")
print(f"  - _compute_expected_utility (array): ✓")
print()

sys.stdout.close()
sys.stdout = sys.__stdout__
print(f"Diagnostics written to {output_path}")
