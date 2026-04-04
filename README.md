# ALHA: Adaptive Low-Rank Hessian Approximation

A quasi-Newton optimizer that automatically adjusts its rank during optimization. No need to pick a memory parameter upfront like you do with L-BFGS.

---

## How It Works

ALHA tracks two signals each iteration to decide whether to increase or decrease rank:

1. **Held-out spectral quality** — builds the Hessian approximation from stored curvature pairs, then tests it against the oldest held-out pair. Poor prediction → rank goes up.
2. **Gradient norm progress rate** — compares current gradient norm to what it was `w` iterations ago. Slow progress → rank goes up.

Rank increases if either signal looks bad. Rank decreases only when both agree things are going well. This keeps the algorithm from dropping rank too early on hard problems.

---

## Repo Structure
```
ALHA/
├── README.md
├── ALHA_paper.pdf
├── ALHA_experiments.ipynb    # Kaggle notebook with all code
└── results/
    └── all_results.json      # Original experimental output
```

---

## Reproducing Results

1. Open `ALHA_experiments.ipynb` in Kaggle
2. Run all cells in order

Results save to `results/all_results.json`. Your numbers may differ slightly from `all_results.json` due to hardware differences.

---

## Results

### Well-Conditioned Quadratic (κ=10, d=1000)
| Method | Iterations | Time (s) | ‖grad‖ | Avg Rank |
|--------|-----------|----------|--------|----------|
| GD | 3615 | 1.62 | 9.97e-09 | — |
| L-BFGS (m=5) | 35 | 0.50 | 6.70e-08 | 5 |
| L-BFGS (m=10) | 34 | 0.66 | 4.82e-08 | 10 |
| L-BFGS (m=20) | 34 | 0.61 | 5.15e-08 | 20 |
| Adam | >10000 (NC) | 5.08 | 5.06e-04 | — |
| **ALHA** | **39** | **0.049** | **9.09e-09** | **12.0** |

### Ill-Conditioned Quadratic (κ=10⁴, d=1000)
| Method | Iterations | Time (s) | ‖grad‖ | Avg Rank |
|--------|-----------|----------|--------|----------|
| GD | >10000 (NC) | 4.54 | 1.34e+01 | — |
| L-BFGS (m=5) | 767 | 10.01 | 9.25e-08 | 5 |
| L-BFGS (m=10) | 559 | 9.44 | 8.17e-08 | 10 |
| L-BFGS (m=20) | 483 | 9.68 | 8.28e-08 | 20 |
| Adam | >10000 (NC) | 4.94 | 1.82e+01 | — |
| **ALHA** | **6179** | **7.37** | **9.99e-09** | **4.1** |

### Rosenbrock (d=100)
| Method | Iterations | Time (s) | Final f | Avg Rank |
|--------|-----------|----------|---------|----------|
| GD | >10000 (NC) | 0.43 | 2.41e+00 | — |
| L-BFGS (m=5) | 546 | 0.084 | 3.89e-08 | 5 |
| L-BFGS (m=10) | 526 | 0.086 | 2.57e-08 | 10 |
| L-BFGS (m=20) | 514 | 0.105 | 3.30e-08 | 20 |
| Adam | >10000 (NC) | 0.61 | 2.64e+00 | — |
| **ALHA** | **556** | **0.444** | **8.95e-09** | **45.4** |

### Logistic Regression on MNIST (n=12665, d=784)
| Method | Iterations | Time (s) | ‖grad‖ | Avg Rank |
|--------|-----------|----------|--------|----------|
| GD | >10000 (NC) | 138.82 | 5.10e-03 | — |
| L-BFGS (m=5) | 62 | 2.35 | 5.69e-08 | 5 |
| L-BFGS (m=10) | 52 | 2.15 | 3.88e-08 | 10 |
| L-BFGS (m=20) | 45 | 1.94 | 6.26e-08 | 20 |
| Adam | >10000 (NC) | 139.20 | 4.40e-05 | — |
| **ALHA** | **72** | **1.40** | **9.25e-09** | **8.6** |

### Neural Network on MNIST (2-layer MLP, n=5000, d=39760)
| Method | Time (s) | ‖grad‖ | Train Acc. | Avg Rank |
|--------|----------|--------|-----------|----------|
| GD | 500.6 | 3.34e-02 | 95.72% | — |
| L-BFGS (m=5) | 2464.3 | 4.48e-04 | 100.00% | 5 |
| L-BFGS (m=10) | 2787.8 | 3.91e-04 | 100.00% | 10 |
| L-BFGS (m=20) | 2611.7 | 4.68e-04 | 100.00% | 20 |
| Adam | 509.9 | 1.53e-03 | 100.00% | — |
| **ALHA** | **772.7** | **4.44e-04** | **100.00%** | **14.5** |

### Sparse Logistic Regression (n=2000, d=5000)
| Method | Iterations | Time (s) | ‖grad‖ | Avg Rank |
|--------|-----------|----------|--------|----------|
| GD | 2050 | 28.47 | 9.98e-09 | — |
| L-BFGS (m=5) | 8 | 0.31 | 2.38e-09 | 5 |
| L-BFGS (m=10) | 8 | 0.33 | 2.31e-09 | 10 |
| L-BFGS (m=20) | 8 | 0.33 | 2.31e-09 | 20 |
| Adam | 293 | 4.24 | 9.61e-09 | — |
| **ALHA** | **8** | **0.17** | **3.92e-09** | **4.2** |

---

## Citation
```bibtex
@article{lagudu2026alha,
  title={Adaptive Low-Rank Hessian Approximation for Large-Scale Optimization: Theory, Algorithms, and Applications},
  author={Lagudu, Arush Rao},
  year={2026},
  institution={Frisco Centennial High School}
}
```

---

