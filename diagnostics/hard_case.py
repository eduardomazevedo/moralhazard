import os
import numpy as np
import traceback
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

from moralhazard import MoralHazardProblem
from moralhazard.config_maker import make_utility_cfg, make_distribution_cfg
from moralhazard.solver import _maximize_lagrange_dual, _minimize_cost_internal, _dual_value_and_grad, _decode_theta, _pad_theta_for_warm_start
from moralhazard.core import _make_cache, _canonical_contract, _constraints, _compute_expected_utility
from moralhazard.utils import _maximize_1d_robust

# --------------------
# Primitives (same as minimum_cost.py)
# --------------------
initial_wealth = 50
sigma = 10.0
first_best_effort = 100
theta = 1.0 / first_best_effort / (first_best_effort + initial_wealth)

utility_cfg = make_utility_cfg("log", w0=initial_wealth)
dist_cfg = make_distribution_cfg("gaussian", sigma=sigma)

def C(a):
    return theta * a**2 / 2

def Cprime(a):
    return theta * a

computational_params = {
    "distribution_type": "continuous",
    "y_min": 0.0 - 3 * sigma,
    "y_max": 180.0 + 3 * sigma,
    "n": 201,
}

cfg = {
    "problem_params": {
        **utility_cfg,
        **dist_cfg,
        "C": C,
        "Cprime": Cprime,
    },
    "computational_params": computational_params,
}

mhp = MoralHazardProblem(cfg)
u_fun = cfg["problem_params"]["u"]

a_ic_lb = 0.0
a_ic_ub = 130.0

# --------------------
# Hard case: w=0, a=4
# --------------------
reservation_wage = 0.0
intended_action = 4.0
reservation_utility = u_fun(reservation_wage)

print("=" * 80)
print("DIAGNOSTIC: Hard Case Analysis")
print("=" * 80)
print(f"Reservation wage: {reservation_wage}")
print(f"Intended action: {intended_action}")
print(f"Reservation utility (Ubar): {reservation_utility:.6f}")
print(f"a_ic_lb: {a_ic_lb}")
print(f"a_ic_ub: {a_ic_ub}")
print()

# --------------------
# Try to solve and catch error with detailed info
# --------------------
print("Attempting to solve cost minimization problem...")
print("-" * 80)

try:
    results = mhp.solve_cost_minimization_problem(
        intended_action=intended_action,
        reservation_utility=reservation_utility,
        a_ic_lb=a_ic_lb,
        a_ic_ub=a_ic_ub,
        n_a_grid_points=10,
        n_a_iterations=1,
    )
    print("SUCCESS: Problem solved!")
    print(results)
    
except Exception as e:
    print(f"\nERROR CAUGHT: {type(e).__name__}: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    print()
    
    # --------------------
    # Detailed diagnostics: try to solve relaxed problem manually
    # --------------------
    print("=" * 80)
    print("DETAILED DIAGNOSTICS: Relaxed Problem")
    print("=" * 80)
    
    try:
        # Build cache
        a_hat_empty = np.array([], dtype=np.float64)
        cache = _make_cache(
            intended_action,
            a_hat_empty,
            problem=mhp,
            clip_ratio=1e6,
        )
        
        # --------------------
        # Visualize Lagrange dual on grid
        # --------------------
        print("\nVisualizing Lagrange dual on (lam, mu) grid...")
        print("-" * 80)
        
        # Create grid for lam and mu
        lam_grid = np.linspace(-50, 100, 50)
        mu_grid = np.linspace(-50, 300, 50)
        Lam, Mu = np.meshgrid(lam_grid, mu_grid)
        
        # For relaxed problem, theta = [lam, mu] (no mu_hat)
        dual_values = np.full(Lam.shape, np.nan)
        FOC_values_dual = np.full(Lam.shape, np.nan)  # Also compute FOC for overlay
        
        print("Evaluating dual function and FOC constraint on grid...")
        for i in range(len(lam_grid)):
            for j in range(len(mu_grid)):
                lam = lam_grid[i]
                mu = mu_grid[j]
                theta = np.array([lam, mu], dtype=np.float64)
                
                try:
                    # Evaluate dual value (we want g_dual, not -g_dual)
                    neg_g_dual, _ = _dual_value_and_grad(theta, cache, mhp, reservation_utility)
                    g_dual = -neg_g_dual
                    dual_values[j, i] = g_dual
                    
                    # Also compute FOC for overlay
                    lam_val, mu_val, mu_hat_val = _decode_theta(theta)
                    v = _canonical_contract(lam_val, mu_val, mu_hat_val, cache["s0"], cache["R"], mhp)
                    cons = _constraints(v, cache=cache, problem=mhp, Ubar=reservation_utility)
                    FOC_values_dual[j, i] = cons["FOC"]
                except Exception:
                    # If evaluation fails, leave as NaN
                    pass
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(lam_grid)} columns")
        
        print("Creating plots...")
        os.makedirs("./diagnostics/figures", exist_ok=True)
        
        # 3D surface plot
        fig = plt.figure(figsize=(12, 5))
        
        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(Lam, Mu, dual_values, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
        ax1.set_xlabel('λ (lam)')
        ax1.set_ylabel('μ (mu)')
        ax1.set_zlabel('Lagrange Dual g(λ, μ)')
        ax1.set_title('Lagrange Dual: 3D Surface')
        fig.colorbar(surf, ax=ax1, shrink=0.5)
        
        # 2D color plot with FOC=0 overlay
        ax2 = fig.add_subplot(122)
        im = ax2.contourf(Lam, Mu, dual_values, levels=50, cmap='viridis')
        # Overlay FOC=0 line
        cs_foc_overlay = ax2.contour(Lam, Mu, FOC_values_dual, levels=[0], colors='black', linewidths=3, linestyles='solid')
        ax2.clabel(cs_foc_overlay, inline=True, fontsize=10, fmt='FOC=0')
        ax2.set_xlabel('λ (lam)')
        ax2.set_ylabel('μ (mu)')
        ax2.set_title('Lagrange Dual: Contour Plot (black line: FOC=0)')
        fig.colorbar(im, ax=ax2)
        
        plt.tight_layout()
        plt.savefig("./diagnostics/figures/hard_case_dual_visualization.png", dpi=150, bbox_inches='tight')
        print("  Saved: ./diagnostics/figures/hard_case_dual_visualization.png")
        plt.close()
        
        # Print some statistics
        valid_values = dual_values[np.isfinite(dual_values)]
        if len(valid_values) > 0:
            print(f"\nDual function statistics:")
            print(f"  Valid evaluations: {len(valid_values)}/{dual_values.size}")
            print(f"  Min value: {valid_values.min():.6e}")
            print(f"  Max value: {valid_values.max():.6e}")
            print(f"  Mean value: {valid_values.mean():.6e}")
            print(f"  NaN/Inf count: {np.sum(~np.isfinite(dual_values))}")
        else:
            print("\n  WARNING: No valid dual values computed!")
        
        print()
        
        # --------------------
        # Visualize IR and FOC constraints on grid
        # --------------------
        print("Visualizing IR and FOC constraints on (lam, mu) grid...")
        print("-" * 80)
        
        IR_values = np.full(Lam.shape, np.nan)
        FOC_values = np.full(Lam.shape, np.nan)
        
        print("Evaluating constraints on grid...")
        for i in range(len(lam_grid)):
            for j in range(len(mu_grid)):
                lam = lam_grid[i]
                mu = mu_grid[j]
                theta = np.array([lam, mu], dtype=np.float64)
                
                try:
                    # Decode theta (for relaxed problem, mu_hat is empty)
                    lam_val, mu_val, mu_hat_val = _decode_theta(theta)
                    
                    # Compute contract
                    v = _canonical_contract(lam_val, mu_val, mu_hat_val, cache["s0"], cache["R"], mhp)
                    
                    # Compute constraints
                    cons = _constraints(v, cache=cache, problem=mhp, Ubar=reservation_utility)
                    
                    IR_values[j, i] = cons["IR"]
                    FOC_values[j, i] = cons["FOC"]
                except Exception:
                    # If evaluation fails, leave as NaN
                    pass
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(lam_grid)} columns")
        
        print("Creating IR constraint plot...")
        
        # IR constraint plot
        fig = plt.figure(figsize=(14, 6))
        
        # 3D surface for IR
        ax1_3d = fig.add_subplot(121, projection='3d')
        surf_ir = ax1_3d.plot_surface(Lam, Mu, IR_values, cmap='RdBu_r', alpha=0.8, linewidth=0, antialiased=True)
        # Add zero plane
        ax1_3d.plot_surface(Lam, Mu, np.zeros_like(IR_values), alpha=0.3, color='black', linewidth=0.5)
        ax1_3d.set_xlabel('λ (lam)')
        ax1_3d.set_ylabel('μ (mu)')
        ax1_3d.set_zlabel('IR Constraint')
        ax1_3d.set_title('IR Constraint: 3D Surface (black plane at 0)')
        fig.colorbar(surf_ir, ax=ax1_3d, shrink=0.5)
        
        # 2D contour for IR with zero level highlighted
        ax2 = fig.add_subplot(122)
        im_ir = ax2.contourf(Lam, Mu, IR_values, levels=50, cmap='RdBu_r', extend='both')
        # Highlight zero level with a thick black line
        cs_ir = ax2.contour(Lam, Mu, IR_values, levels=[0], colors='black', linewidths=3, linestyles='solid')
        ax2.clabel(cs_ir, inline=True, fontsize=10, fmt='IR=0')
        ax2.set_xlabel('λ (lam)')
        ax2.set_ylabel('μ (mu)')
        ax2.set_title('IR Constraint: Contour Plot (black line at IR=0)')
        fig.colorbar(im_ir, ax=ax2)
        
        plt.tight_layout()
        plt.savefig("./diagnostics/figures/hard_case_IR_constraint.png", dpi=150, bbox_inches='tight')
        print("  Saved: ./diagnostics/figures/hard_case_IR_constraint.png")
        plt.close()
        
        print("Creating FOC constraint plot...")
        
        # FOC constraint plot
        fig = plt.figure(figsize=(14, 6))
        
        # 3D surface for FOC
        ax1_3d = fig.add_subplot(121, projection='3d')
        surf_foc = ax1_3d.plot_surface(Lam, Mu, FOC_values, cmap='RdBu_r', alpha=0.8, linewidth=0, antialiased=True)
        # Add zero plane
        ax1_3d.plot_surface(Lam, Mu, np.zeros_like(FOC_values), alpha=0.3, color='black', linewidth=0.5)
        ax1_3d.set_xlabel('λ (lam)')
        ax1_3d.set_ylabel('μ (mu)')
        ax1_3d.set_zlabel('FOC Constraint')
        ax1_3d.set_title('FOC Constraint: 3D Surface (black plane at 0)')
        fig.colorbar(surf_foc, ax=ax1_3d, shrink=0.5)
        
        # 2D contour for FOC with zero level highlighted
        ax2 = fig.add_subplot(122)
        im_foc = ax2.contourf(Lam, Mu, FOC_values, levels=50, cmap='RdBu_r', extend='both')
        # Highlight zero level with a thick black line
        cs_foc = ax2.contour(Lam, Mu, FOC_values, levels=[0], colors='black', linewidths=3, linestyles='solid')
        ax2.clabel(cs_foc, inline=True, fontsize=10, fmt='FOC=0')
        ax2.set_xlabel('λ (lam)')
        ax2.set_ylabel('μ (mu)')
        ax2.set_title('FOC Constraint: Contour Plot (black line at FOC=0)')
        fig.colorbar(im_foc, ax=ax2)
        
        plt.tight_layout()
        plt.savefig("./diagnostics/figures/hard_case_FOC_constraint.png", dpi=150, bbox_inches='tight')
        print("  Saved: ./diagnostics/figures/hard_case_FOC_constraint.png")
        plt.close()
        
        # Print constraint statistics
        valid_ir = IR_values[np.isfinite(IR_values)]
        valid_foc = FOC_values[np.isfinite(FOC_values)]
        
        if len(valid_ir) > 0:
            print(f"\nIR constraint statistics:")
            print(f"  Valid evaluations: {len(valid_ir)}/{IR_values.size}")
            print(f"  Min value: {valid_ir.min():.6e}")
            print(f"  Max value: {valid_ir.max():.6e}")
            print(f"  Values near zero (|IR| < 1e-3): {np.sum(np.abs(valid_ir) < 1e-3)}")
        
        if len(valid_foc) > 0:
            print(f"\nFOC constraint statistics:")
            print(f"  Valid evaluations: {len(valid_foc)}/{FOC_values.size}")
            print(f"  Min value: {valid_foc.min():.6e}")
            print(f"  Max value: {valid_foc.max():.6e}")
            print(f"  Values near zero (|FOC| < 1e-3): {np.sum(np.abs(valid_foc) < 1e-3)}")
        
        print()
        
        print("\nCache information:")
        print(f"  f0 shape: {cache['f0'].shape}")
        print(f"  s0 shape: {cache['s0'].shape}")
        print(f"  D shape: {cache['D'].shape}")
        print(f"  R shape: {cache['R'].shape}")
        print(f"  f0 min/max: {cache['f0'].min():.6e} / {cache['f0'].max():.6e}")
        print(f"  s0 min/max: {cache['s0'].min():.6e} / {cache['s0'].max():.6e}")
        print(f"  R min/max: {cache['R'].min():.6e} / {cache['R'].max():.6e}")
        print(f"  C0: {cache['C0']:.6e}")
        print(f"  Cprime0: {cache['Cprime0']:.6e}")
        print()
        
        # Try to solve relaxed problem with detailed output
        print("Attempting relaxed problem solve (a_hat = [])...")
        print("-" * 80)
        
        results_dual, theta_opt = _maximize_lagrange_dual(
            a0=intended_action,
            Ubar=reservation_utility,
            a_hat=a_hat_empty,
            problem=mhp,
            theta_init=None,
            maxiter=1000,
            ftol=1e-8,
            clip_ratio=1e6,
        )
        
        print("SUCCESS: Relaxed problem solved!")
        print(f"\nRelaxed problem results:")
        print(f"  Expected wage: {results_dual.expected_wage:.6e}")
        print(f"  Multipliers:")
        print(f"    λ (lam): {results_dual.multipliers['lam']:.6e}")
        print(f"    μ (mu): {results_dual.multipliers['mu']:.6e}")
        print(f"    μ̂ (mu_hat): {results_dual.multipliers['mu_hat']}")
        print(f"  Constraints:")
        print(f"    U0: {results_dual.constraints['U0']:.6e}")
        print(f"    IR: {results_dual.constraints['IR']:.6e}")
        print(f"    FOC: {results_dual.constraints['FOC']:.6e}")
        print(f"    Ewage: {results_dual.constraints['Ewage']:.6e}")
        print(f"  Solver state: {results_dual.solver_state}")
        print()
        
        # Check for global IC violations
        print("Checking for global IC violations...")
        print("-" * 80)
        
        def objective(a: float | np.ndarray) -> float | np.ndarray:
            return _compute_expected_utility(results_dual.optimal_contract, a, problem=mhp)
        
        a_best, utility_best = _maximize_1d_robust(
            objective=objective,
            lower_bound=a_ic_lb,
            upper_bound=a_ic_ub,
            n_grid_points=10,
        )
        
        global_ic_violation = utility_best - results_dual.constraints['U0']
        best_action_distance = np.abs(a_best - intended_action)
        
        print(f"  Best action found: {a_best:.6f}")
        print(f"  Utility at best action: {utility_best:.6e}")
        print(f"  Utility at intended action: {results_dual.constraints['U0']:.6e}")
        print(f"  Global IC violation: {global_ic_violation:.6e}")
        print(f"  Best action distance from intended: {best_action_distance:.6e}")
        print()
        
        if global_ic_violation > 1e-6 and best_action_distance > 1e-6:
            print("Global IC violation detected. Attempting to solve with a_hat = [a_best]...")
            print("-" * 80)
            
            a_always_check = np.array([0.0])
            a_hat = np.concatenate([a_always_check, np.array([a_best])])
            
            print(f"  a_hat: {a_hat}")
            print(f"  Using theta_init from relaxed problem: {theta_opt}")
            print()
            
            # Build cache for iterative solve
            cache_iter = _make_cache(
                intended_action,
                a_hat,
                problem=mhp,
                clip_ratio=1e6,
            )
            
            print("Cache for iterative solve:")
            print(f"  R shape: {cache_iter['R'].shape}")
            print(f"  R min/max: {cache_iter['R'].min():.6e} / {cache_iter['R'].max():.6e}")
            print()
            
            # Manual solve to capture detailed information
            m = int(cache_iter["R"].shape[1])
            expected_shape = (2 + m,)
            
            # Prepare initial guess
            if theta_opt.shape == expected_shape:
                x0 = theta_opt
            else:
                x0 = _pad_theta_for_warm_start(theta_opt, m)
                print(f"  Padded theta_init from shape {theta_opt.shape} to {x0.shape}")
            
            print(f"  Initial theta (x0): {x0}")
            print(f"  x0 min/max: {x0.min():.6e} / {x0.max():.6e}")
            print()
            
            # Bounds
            bounds = [(0.0, None)] + [(None, None)] + [(0.0, None)] * m
            print(f"  Bounds: lam ∈ [0, ∞), mu ∈ (-∞, ∞), mu_hat ∈ [0, ∞)^m (m={m})")
            print()
            
            # Evaluate initial point
            print("Evaluating initial point...")
            fun0, grad0 = _dual_value_and_grad(x0, cache_iter, mhp, reservation_utility)
            print(f"  Initial objective: {fun0:.6e}")
            print(f"  Initial gradient: {grad0}")
            print(f"  Initial gradient norm: {np.linalg.norm(grad0):.6e}")
            print(f"  Initial gradient min/max: {grad0.min():.6e} / {grad0.max():.6e}")
            print()
            
            # Solve
            print("Running L-BFGS-B optimizer...")
            t0 = time.time()
            res = minimize(
                fun=_dual_value_and_grad,
                x0=x0,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-8, "iprint": 1},
                args=(cache_iter, mhp, reservation_utility),
            )
            t1 = time.time()
            
            print(f"\nSolver finished in {t1 - t0:.4f} seconds")
            print(f"  success: {res.success}")
            print(f"  status: {res.status}")
            print(f"  message: {res.message}")
            print(f"  niter: {getattr(res, 'nit', 'N/A')}")
            print(f"  nfev: {getattr(res, 'nfev', 'N/A')}")
            print(f"  fun (final objective): {res.fun:.6e}")
            
            if hasattr(res, 'x'):
                print(f"  x (final theta): {res.x}")
                print(f"  x min/max: {res.x.min():.6e} / {res.x.max():.6e}")
                lam_final, mu_final, mu_hat_final = _decode_theta(res.x)
                print(f"  Decoded: lam={lam_final:.6e}, mu={mu_final:.6e}, mu_hat={mu_hat_final}")
            
            if hasattr(res, 'jac') and res.jac is not None:
                print(f"  jac (final gradient): {res.jac}")
                print(f"  jac norm: {np.linalg.norm(res.jac):.6e}")
                print(f"  jac min/max: {res.jac.min():.6e} / {res.jac.max():.6e}")
            
            if not res.success:
                print("\n  SOLVER FAILED - This is where the error occurs!")
                print("  Attempting to evaluate constraints at final point...")
                try:
                    lam_f, mu_f, mu_hat_f = _decode_theta(res.x)
                    v_final = _canonical_contract(lam_f, mu_f, mu_hat_f, cache_iter["s0"], cache_iter["R"], mhp)
                    cons_final = _constraints(v_final, cache=cache_iter, problem=mhp, Ubar=reservation_utility)
                    print(f"    U0: {cons_final['U0']:.6e}")
                    print(f"    IR: {cons_final['IR']:.6e}")
                    print(f"    FOC: {cons_final['FOC']:.6e}")
                    print(f"    Ewage: {cons_final['Ewage']:.6e}")
                except Exception as eval_e:
                    print(f"    Could not evaluate constraints: {eval_e}")
            
    except Exception as inner_e:
        print(f"\nERROR in detailed diagnostics: {type(inner_e).__name__}: {inner_e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print()
        
        # Try to inspect the minimize result if available
        # We need to modify the approach to capture the solver result before the error
        pass

print("\n" + "=" * 80)
print("END OF DIAGNOSTIC")
print("=" * 80)
