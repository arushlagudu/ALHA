# ALHA: Adaptive Low-Rank Hessian Approximation

Official code repository for:

> **Adaptive Low-Rank Hessian Approximation for Large-Scale Optimization: Theory, Algorithms, and Applications**
> Arush Rao Lagudu — Frisco Centennial High School
> arushlagudu@gmail.com

---

## Overview

ALHA is a quasi-Newton optimizer that **automatically adapts its memory parameter** at each iteration using a computable spectral quality metric. Unlike L-BFGS which fixes the memory parameter `m` throughout optimization, ALHA detects when more or less curvature information is needed and adjusts accordingly.

**Key results:**
- Up to **40% fewer iterations** than fixed-rank L-BFGS
- Linear convergence at rate O((1 − μ/L)^k) — provably optimal
- Automatically discovers the right rank — no manual tuning

---

## Repository Structure

```
alha_repo/
├── src/
│   ├── alha.py          # Core ALHA algorithm
│   ├── baselines.py     # GD, L-BFGS, Adam implementations
│   └── problems.py      # All 6 test problems
├── experiments/
│   ├── run_experiments.py   # Run all paper experiments
│   └── generate_figures.py  # Generate all paper figures
├── tests/
│   └── test_alha.py     # Unit tests
├── results/             # Experiment outputs (JSON)
├── figures/             # Generated figures (PDF)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/alha.git
cd alha
pip install -r requirements.txt
```

Requirements: numpy, scipy, matplotlib, pytest

---

## Quick Start

```python
import numpy as np
import sys
sys.path.insert(0, 'src')

from alha import alha, ALHAConfig

# Define your problem
def f(x):
    return 0.5 * np.dot(x, x)

def grad_f(x):
    return x.copy()

# Run ALHA
x0 = np.ones(100) * 5.0
config = ALHAConfig(m_min=2, m_max=30, eps_tol=1e-8)
result = alha(f, grad_f, x0, config)

print(f"Converged: {result.converged}")
print(f"Iterations: {result.n_iter}")
print(f"Final ||grad||: {result.grad_norm:.2e}")
print(f"Average rank: {sum(result.rank_history)/len(result.rank_history):.1f}")
```

---

## Reproducing Paper Experiments

### Run all experiments
```bash
python experiments/run_experiments.py
```

### Run a specific experiment
```bash
python experiments/run_experiments.py --problem ill_conditioned_quadratic --verbose
```

Available problems:
- `well_conditioned_quadratic` — κ = 10, d = 1000
- `ill_conditioned_quadratic`  — κ = 10^4, d = 1000
- `rosenbrock`                 — d = 100
- `logistic_mnist`             — logistic regression, d = 784
- `neural_network`             — 2-layer MLP, d = 79510
- `sparse_logistic`            — RCV1-style, d = 5000

### Generate figures
```bash
python experiments/generate_figures.py
```

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Algorithm Summary

ALHA adds three lines to L-BFGS:

```
Standard L-BFGS:
    dk ← −TwoLoop(gk, S, Y, mk)
    αk ← WolfeLineSearch(...)
    update x, g, store (sk, yk)

ALHA adds:
    Q̂k ← ComputeQuality(S, Y, sk, yk)   # one extra TwoLoop call
    if Q̂k < 1 − ε_high: mk += Δm+        # approximation bad → grow
    elif Q̂k > 1 − ε_low: mk -= Δm−       # approximation good → shrink
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `m_min`   | 2       | Minimum notebook size |
| `m_max`   | 50      | Maximum notebook size |
| `m_0`     | 5       | Initial notebook size |
| `eps_low` | 0.1     | Shrink threshold (Q̂ > 1−0.1 = 0.9 → shrink) |
| `eps_high`| 0.3     | Grow threshold (Q̂ < 1−0.3 = 0.7 → grow) |
| `delta_m_plus` | 2  | Entries to add when growing |
| `delta_m_minus`| 1  | Entries to remove when shrinking |

---

## Citation

```bibtex
@article{lagudu2025alha,
  title   = {Adaptive Low-Rank Hessian Approximation for Large-Scale Optimization:
             Theory, Algorithms, and Applications},
  author  = {Lagudu, Arush Rao},
  year    = {2026},
  note    = {Frisco Centennial High School}
}
```

---

## License

MIT License. See LICENSE file.
