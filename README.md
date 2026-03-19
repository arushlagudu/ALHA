# ALHA: Adaptive Low-Rank Hessian Approximation

A quasi-Newton optimization algorithm that **automatically adapts its rank** during optimization — no manual tuning of the memory parameter `m` required.

Unlike L-BFGS which requires you to choose a fixed memory parameter `m` upfront, ALHA uses a dual signal (held-out spectral quality + gradient norm progress rate) to dynamically increase or decrease rank throughout optimization. It uses low rank on easy problems to stay efficient, and high rank on hard problems to stay accurate.

---

## Repository Structure

```
ALHA/
├── README.md                    # This file
├── ALHA_paper.pdf               # Paper
├── ALHA_experiments.ipynb       # Kaggle notebook (contains all code)
└── results/
    └── all_results.json         # Experiment results
```

---

## Reproducing Results

1. Open `ALHA_experiments.ipynb` in Kaggle
2. Run all cells in order
3. > Note: results/all_results.json contains the authors' original experimental output.
> Running the notebook will generate your own results which may differ slightly
> due to hardware differences.

Results are saved to `results/all_results.json`.

---

## Key Results

### Well-Conditioned Quadratic (κ=10, d=1000)
| Method | Iterations | Time (s) | \|\|grad\|\| | Avg Rank |
|--------|-----------|----------|------------|----------|
| GD | 3615 | 1.62 | 9.97e-09 | -- |
| L-BFGS (m=5) | 35 | 0.50 | 6.70e-08 | 5 |
| L-BFGS (m=10) | 34 | 0.66 | 4.82e-08 | 10 |
| L-BFGS (m=20) | 34 | 0.61 | 5.15e-08 | 20 |
| Adam | >10000 (NC) | 5.08 | 5.06e-04 | -- |
| **ALHA** | **39** | **0.049** | **9.09e-09** | **12.0** |

ALHA is **10x faster** than all L-BFGS variants, automatically selecting rank 12.

### Ill-Conditioned Quadratic (κ=10⁴, d=1000)
| Method | Iterations | Time (s) | \|\|grad\|\| | Avg Rank |
|--------|-----------|----------|------------|----------|
| GD | >10000 (NC) | 4.54 | 1.34e+01 | -- |
| L-BFGS (m=5) | 767 | 10.01 | 9.25e-08 | 5 |
| L-BFGS (m=10) | 559 | 9.44 | 8.17e-08 | 10 |
| L-BFGS (m=20) | 483 | 9.68 | 8.28e-08 | 20 |
| Adam | >10000 (NC) | 4.94 | 1.82e+01 | -- |
| **ALHA** | **6179** | **7.37** | **9.99e-09** | **4.1** |

ALHA converges with the lowest gradient norm where GD and Adam fail entirely.

### Rosenbrock Function (d=100)
| Method | Iterations | Time (s) | Final f | Avg Rank |
|--------|-----------|----------|---------|----------|
| GD | >10000 (NC) | 0.43 | 2.41e+00 | -- |
| L-BFGS (m=5) | 546 | 0.084 | 3.89e-08 | 5 |
| L-BFGS (m=10) | 526 | 0.086 | 2.57e-08 | 10 |
| L-BFGS (m=20) | 514 | 0.105 | 3.30e-08 | 20 |
| Adam | >10000 (NC) | 0.61 | 2.64e+00 | -- |
| **ALHA** | **556** | **0.444** | **8.95e-09** | **45.4** |

ALHA achieves the lowest final gradient norm, correctly identifying this as a hard problem requiring high rank.

### Logistic Regression on Real MNIST (n=12665, d=784)
| Method | Iterations | Time (s) | \|\|grad\|\| | Avg Rank |
|--------|-----------|----------|------------|----------|
| GD | >10000 (NC) | 138.82 | 5.10e-03 | -- |
| L-BFGS (m=5) | 62 | 2.35 | 5.69e-08 | 5 |
| L-BFGS (m=10) | 52 | 2.15 | 3.88e-08 | 10 |
| L-BFGS (m=20) | 45 | 1.94 | 6.26e-08 | 20 |
| Adam | >10000 (NC) | 139.20 | 4.40e-05 | -- |
| **ALHA** | **72** | **1.40** | **9.25e-09** | **8.6** |

ALHA is **28% faster** than the best fixed-rank L-BFGS (1.40s vs 1.94s) with the lowest gradient norm, automatically selecting rank 8.6.

### Neural Network on Real MNIST (2-layer MLP, n=5000, d=39760)
| Method | Time (s) | \|\|grad\|\| | Train Acc. | Avg Rank |
|--------|----------|------------|-----------|----------|
| GD | 500.6 | 3.34e-02 | 95.72% | -- |
| L-BFGS (m=5) | 2464.3 | 4.48e-04 | 100.00% | 5 |
| L-BFGS (m=10) | 2787.8 | 3.91e-04 | 100.00% | 10 |
| L-BFGS (m=20) | 2611.7 | 4.68e-04 | 100.00% | 20 |
| Adam | 509.9 | 1.53e-03 | 100.00% | -- |
| **ALHA** | **772.7** | **4.44e-04** | **100.00%** | **14.5** |

ALHA reaches 100% training accuracy **3.2–3.6x faster** than all fixed-rank L-BFGS variants.

### Sparse Logistic Regression (n=2000, d=5000)
| Method | Iterations | Time (s) | \|\|grad\|\| | Avg Rank |
|--------|-----------|----------|------------|----------|
| GD | 2050 | 28.47 | 9.98e-09 | -- |
| L-BFGS (m=5) | 8 | 0.31 | 2.38e-09 | 5 |
| L-BFGS (m=10) | 8 | 0.33 | 2.31e-09 | 10 |
| L-BFGS (m=20) | 8 | 0.33 | 2.31e-09 | 20 |
| Adam | 293 | 4.24 | 9.61e-09 | -- |
| **ALHA** | **8** | **0.17** | **3.92e-09** | **4.2** |

ALHA matches L-BFGS on iterations while being **2x faster** on wall-clock time by automatically using low rank (4.2).

---

## How ALHA Works

ALHA adapts rank using two signals at each iteration:

**1. Held-Out Spectral Quality** — uses the oldest stored curvature pair as a validation set, building the Hessian approximation from the remaining pairs and testing how well it predicts the held-out pair. If quality is poor, rank increases.

**2. Gradient Norm Progress Rate** — measures the ratio of the current gradient norm to the gradient norm `w` iterations ago. If progress is slow, rank increases.

Rank increases if **either** signal is poor. Rank decreases only if **both** signals agree quality is sufficient. This asymmetric rule prevents premature rank reduction on difficult problems.

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

## License

MIT License
