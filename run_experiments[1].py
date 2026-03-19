"""
run_experiments.py
==================
Runs all six experiments from the ALHA paper and saves results.

Usage:
    python experiments/run_experiments.py
    python experiments/run_experiments.py --problem ill_conditioned_quadratic
    python experiments/run_experiments.py --problem all --verbose
"""

import sys
import os
import argparse
import json
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from alha import alha, ALHAConfig
from baselines import run_gradient_descent, run_lbfgs, run_adam
from problems import get_all_problems


def run_single_problem(name: str, problem, verbose: bool = False) -> dict:
    """Run all optimizers on a single problem and collect results."""
    print(f"\n{'='*60}")
    print(f"Problem: {problem.name}")
    print(f"d = {problem.d}")
    print(f"{'='*60}")

    results = {}

    # --- ALHA ---
    print("\nRunning ALHA...")
    config = ALHAConfig(verbose=verbose, eps_tol=1e-8, max_iter=10000)
    result = alha(problem.f, problem.grad_f, problem.x0.copy(), config)
    results['ALHA'] = {
        'iterations': result.n_iter,
        'time': round(result.time_elapsed, 3),
        'final_grad_norm': float(result.grad_norm),
        'final_f': float(result.f_val),
        'converged': result.converged,
        'avg_rank': round(float(np.mean(result.rank_history)), 1),
        'f_history': [float(v) for v in result.f_history],
        'grad_norm_history': [float(v) for v in result.grad_norm_history],
        'rank_history': result.rank_history,
    }
    print(f"  Iterations: {result.n_iter} | Time: {result.time_elapsed:.3f}s | "
          f"||g||: {result.grad_norm:.2e} | Avg rank: {np.mean(result.rank_history):.1f}")

    # --- L-BFGS variants ---
    for m in [5, 10, 20]:
        print(f"\nRunning L-BFGS (m={m})...")
        result = run_lbfgs(problem.f, problem.grad_f, problem.x0.copy(),
                           m=m, max_iter=10000, eps_tol=1e-8, verbose=verbose)
        key = f'L-BFGS (m={m})'
        results[key] = {
            'iterations': result.n_iter,
            'time': round(result.time_elapsed, 3),
            'final_grad_norm': float(result.grad_norm),
            'final_f': float(result.f_val),
            'converged': result.converged,
            'avg_rank': m,
            'f_history': [float(v) for v in result.f_history],
            'grad_norm_history': [float(v) for v in result.grad_norm_history],
        }
        print(f"  Iterations: {result.n_iter} | Time: {result.time_elapsed:.3f}s | "
              f"||g||: {result.grad_norm:.2e}")

    # --- Adam ---
    print("\nRunning Adam...")
    result = run_adam(problem.f, problem.grad_f, problem.x0.copy(),
                      lr=1e-3, max_iter=10000, eps_tol=1e-8, verbose=verbose)
    results['Adam'] = {
        'iterations': result.n_iter,
        'time': round(result.time_elapsed, 3),
        'final_grad_norm': float(result.grad_norm),
        'final_f': float(result.f_val),
        'converged': result.converged,
        'avg_rank': None,
        'f_history': [float(v) for v in result.f_history],
        'grad_norm_history': [float(v) for v in result.grad_norm_history],
    }
    print(f"  Iterations: {result.n_iter} | Time: {result.time_elapsed:.3f}s | "
          f"||g||: {result.grad_norm:.2e}")

    # --- Gradient Descent ---
    print("\nRunning Gradient Descent...")
    # Estimate reasonable LR from initial Hessian approximation
    g0 = problem.grad_f(problem.x0)
    lr = min(0.01, 1.0 / (np.linalg.norm(g0) + 1e-8))
    result = run_gradient_descent(problem.f, problem.grad_f, problem.x0.copy(),
                                  lr=lr, max_iter=10000, eps_tol=1e-8, verbose=verbose)
    results['GD'] = {
        'iterations': result.n_iter,
        'time': round(result.time_elapsed, 3),
        'final_grad_norm': float(result.grad_norm),
        'final_f': float(result.f_val),
        'converged': result.converged,
        'avg_rank': None,
        'f_history': [float(v) for v in result.f_history],
        'grad_norm_history': [float(v) for v in result.grad_norm_history],
    }
    print(f"  Iterations: {result.n_iter} | Time: {result.time_elapsed:.3f}s | "
          f"||g||: {result.grad_norm:.2e}")

    return results


def print_summary_table(name: str, results: dict):
    """Print a formatted summary table like in the paper."""
    print(f"\n--- Summary: {name} ---")
    print(f"{'Method':<20} {'Iters':>8} {'Time(s)':>10} {'||grad||':>12} {'Avg Rank':>10}")
    print("-" * 65)
    for method, r in results.items():
        rank_str = f"{r['avg_rank']:.1f}" if r['avg_rank'] is not None else "--"
        converged_str = "" if r['converged'] else " (NC)"
        print(f"{method:<20} {r['iterations']:>8} {r['time']:>10.3f} "
              f"{r['final_grad_norm']:>12.3e} {rank_str:>10}{converged_str}")


def main():
    parser = argparse.ArgumentParser(description='Run ALHA experiments')
    parser.add_argument('--problem', type=str, default='all',
                        help='Problem to run (all, or specific name)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-iteration output')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_problems = get_all_problems()

    if args.problem == 'all':
        problems_to_run = all_problems
    else:
        if args.problem not in all_problems:
            print(f"Unknown problem: {args.problem}")
            print(f"Available: {list(all_problems.keys())}")
            sys.exit(1)
        problems_to_run = {args.problem: all_problems[args.problem]}

    all_results = {}
    t_total_start = time.time()

    for name, problem in problems_to_run.items():
        results = run_single_problem(name, problem, verbose=args.verbose)
        all_results[name] = results
        print_summary_table(name, results)

        # Save per-problem results
        out_path = os.path.join(args.output_dir, f'{name}.json')
        with open(out_path, 'w') as fp:
            json.dump(results, fp, indent=2)
        print(f"\nSaved results to {out_path}")

    # Save combined results
    combined_path = os.path.join(args.output_dir, 'all_results.json')
    with open(combined_path, 'w') as fp:
        json.dump(all_results, fp, indent=2)

    t_total = time.time() - t_total_start
    print(f"\n{'='*60}")
    print(f"All experiments complete. Total time: {t_total:.1f}s")
    print(f"Results saved to: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
