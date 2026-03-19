"""
generate_figures.py
===================
Generates all figures from the ALHA paper.

Usage:
    python experiments/generate_figures.py
    python experiments/generate_figures.py --results_dir results --output_dir figures
"""

import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not found. Install with: pip install matplotlib")


COLORS = {
    'ALHA':          '#2196F3',
    'L-BFGS (m=5)':  '#FF9800',
    'L-BFGS (m=10)': '#9C27B0',
    'L-BFGS (m=20)': '#4CAF50',
    'Adam':          '#F44336',
    'GD':            '#795548',
}

LINESTYLES = {
    'ALHA':          '-',
    'L-BFGS (m=5)':  '--',
    'L-BFGS (m=10)': '-.',
    'L-BFGS (m=20)': ':',
    'Adam':          (0, (3, 1, 1, 1)),
    'GD':            (0, (5, 2)),
}

LINEWIDTHS = {
    'ALHA': 2.5,
    'L-BFGS (m=5)': 1.5,
    'L-BFGS (m=10)': 1.5,
    'L-BFGS (m=20)': 1.5,
    'Adam': 1.5,
    'GD': 1.5,
}


def setup_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 150,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot_convergence(results: dict, problem_name: str, output_path: str,
                     metric: str = 'grad_norm_history', ylabel: str = r'$\|\nabla f\|$',
                     max_iters: int = None):
    """Plot convergence curves for all methods on one problem."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    for method, r in results.items():
        history = r.get(metric, [])
        if not history:
            continue
        iters = list(range(len(history)))
        if max_iters:
            iters = iters[:max_iters]
            history = history[:max_iters]

        color = COLORS.get(method, 'gray')
        ls = LINESTYLES.get(method, '-')
        lw = LINEWIDTHS.get(method, 1.5)

        ax.semilogy(iters, history, label=method, color=color,
                    linestyle=ls, linewidth=lw)

    ax.set_xlabel('Iteration')
    ax.set_ylabel(ylabel)
    ax.set_title(problem_name)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_rank_evolution(results: dict, problem_name: str, output_path: str,
                        max_iters: int = None):
    """Plot rank evolution for ALHA (Figure 1 from paper)."""
    if not HAS_MPL or 'ALHA' not in results:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    # Top: gradient norm convergence
    for method, r in results.items():
        history = r.get('grad_norm_history', [])
        if not history:
            continue
        iters = list(range(len(history)))
        if max_iters:
            iters, history = iters[:max_iters], history[:max_iters]
        ax1.semilogy(iters, history,
                     label=method,
                     color=COLORS.get(method, 'gray'),
                     linestyle=LINESTYLES.get(method, '-'),
                     linewidth=LINEWIDTHS.get(method, 1.5))

    ax1.set_ylabel(r'$\|\nabla f\|$')
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax1.set_title(f'{problem_name} — Convergence and Rank Evolution')

    # Bottom: ALHA rank evolution
    rank_history = results['ALHA'].get('rank_history', [])
    if rank_history:
        iters = list(range(len(rank_history)))
        if max_iters:
            iters, rank_history = iters[:max_iters], rank_history[:max_iters]
        ax2.plot(iters, rank_history, color=COLORS['ALHA'],
                 linewidth=2.5, label='ALHA rank $m_k$')

        # Reference lines for L-BFGS fixed ranks
        ax2.axhline(y=5, color=COLORS['L-BFGS (m=5)'], linestyle='--',
                    linewidth=1.2, alpha=0.7, label='L-BFGS m=5')
        ax2.axhline(y=20, color=COLORS['L-BFGS (m=20)'], linestyle=':',
                    linewidth=1.2, alpha=0.7, label='L-BFGS m=20')

        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Rank $m_k$')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_sensitivity(sensitivity_data: dict, output_path: str):
    """
    Plot sensitivity analysis (Table 8 from paper).
    sensitivity_data: dict mapping (eps_low, eps_high) -> {iterations, avg_rank, time}
    """
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    labels = [f"({k[0]},{k[1]})" for k in sensitivity_data.keys()]
    iterations = [v['iterations'] for v in sensitivity_data.values()]
    avg_ranks = [v['avg_rank'] for v in sensitivity_data.values()]
    times = [v['time'] for v in sensitivity_data.values()]

    x = np.arange(len(labels))

    axes[0].bar(x, iterations, color=COLORS['ALHA'], alpha=0.8, edgecolor='white')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha='right')
    axes[0].set_xlabel(r'$(\epsilon_{\rm low}, \epsilon_{\rm high})$')
    axes[0].set_ylabel('Iterations to convergence')
    axes[0].set_title('Sensitivity: Iterations')

    axes[1].bar(x, avg_ranks, color=COLORS['L-BFGS (m=20)'], alpha=0.8, edgecolor='white')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha='right')
    axes[1].set_xlabel(r'$(\epsilon_{\rm low}, \epsilon_{\rm high})$')
    axes[1].set_ylabel('Average rank')
    axes[1].set_title('Sensitivity: Average Rank')

    plt.suptitle('Threshold Sensitivity Analysis (Ill-Conditioned Quadratic)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate ALHA paper figures')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--output_dir', type=str, default='figures')
    args = parser.parse_args()

    if not HAS_MPL:
        print("ERROR: matplotlib required. Install: pip install matplotlib")
        sys.exit(1)

    setup_style()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from: {args.results_dir}/")
    print(f"Saving figures to:    {args.output_dir}/")

    # Load all results
    all_results_path = os.path.join(args.results_dir, 'all_results.json')
    if not os.path.exists(all_results_path):
        print(f"ERROR: {all_results_path} not found.")
        print("Run experiments first: python experiments/run_experiments.py")
        sys.exit(1)

    with open(all_results_path) as fp:
        all_results = json.load(fp)

    print("\nGenerating convergence plots...")

    problem_display_names = {
        'well_conditioned_quadratic': 'Well-conditioned Quadratic ($\\kappa=10$)',
        'ill_conditioned_quadratic':  'Ill-conditioned Quadratic ($\\kappa=10^4$)',
        'rosenbrock':                 'Rosenbrock Function',
        'logistic_mnist':             'Logistic Regression (MNIST)',
        'neural_network':             'Neural Network Training',
        'sparse_logistic':            'Sparse Logistic Regression',
    }

    for prob_name, results in all_results.items():
        display = problem_display_names.get(prob_name, prob_name)

        # Gradient norm convergence
        plot_convergence(
            results, display,
            os.path.join(args.output_dir, f'{prob_name}_convergence.pdf'),
            metric='grad_norm_history',
            ylabel=r'$\|\nabla f(x_k)\|$'
        )

        # Function value convergence
        plot_convergence(
            results, display,
            os.path.join(args.output_dir, f'{prob_name}_fval.pdf'),
            metric='f_history',
            ylabel=r'$f(x_k) - f^*$'
        )

    print("\nGenerating rank evolution plot (Figure 1)...")
    if 'ill_conditioned_quadratic' in all_results:
        plot_rank_evolution(
            all_results['ill_conditioned_quadratic'],
            'Ill-conditioned Quadratic ($\\kappa=10^4$)',
            os.path.join(args.output_dir, 'rank_evolution.pdf')
        )

    # Hardcoded sensitivity from paper Table 8
    print("\nGenerating sensitivity analysis plot...")
    sensitivity = {
        (0.05, 0.15): {'iterations': 812, 'avg_rank': 22.1, 'time': 6.9},
        (0.10, 0.30): {'iterations': 743, 'avg_rank': 18.7, 'time': 5.8},
        (0.15, 0.40): {'iterations': 698, 'avg_rank': 15.3, 'time': 5.2},
        (0.20, 0.50): {'iterations': 687, 'avg_rank': 12.8, 'time': 5.0},
    }
    plot_sensitivity(sensitivity, os.path.join(args.output_dir, 'sensitivity.pdf'))

    print("\nAll figures generated.")


if __name__ == '__main__':
    main()
