"""
Test problems used in the ALHA paper experiments.

Problems:
    1. Well-conditioned quadratic  (kappa = 10)
    2. Ill-conditioned quadratic   (kappa = 1e4)
    3. Rosenbrock function
    4. Logistic regression (MNIST)
    5. Neural network training (MNIST)
    6. Sparse logistic regression  (RCV1)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Callable


@dataclass
class Problem:
    """Container for an optimization problem."""
    name: str
    f: Callable
    grad_f: Callable
    x0: np.ndarray
    d: int
    description: str = ""


# ---------------------------------------------------------------------------
# Problem 1 & 2: Quadratic f(x) = 0.5 * x^T A x
# ---------------------------------------------------------------------------

def make_quadratic(d: int = 1000, kappa: float = 10.0, seed: int = 42) -> Problem:
    """
    Create a quadratic optimization problem f(x) = 0.5 * x^T A x.

    A is symmetric positive definite with eigenvalues in [1, kappa].

    Args:
        d:     Dimension
        kappa: Condition number (lambda_max / lambda_min)
        seed:  Random seed

    Returns:
        Problem instance
    """
    rng = np.random.RandomState(seed)

    # Random orthogonal matrix
    Q, _ = np.linalg.qr(rng.randn(d, d))

    # Eigenvalues uniformly distributed in [1, kappa]
    lambdas = np.linspace(1.0, kappa, d)

    # A = Q diag(lambdas) Q^T
    A = Q @ np.diag(lambdas) @ Q.T
    # Ensure symmetry
    A = 0.5 * (A + A.T)

    x0 = rng.randn(d)

    def f(x):
        return 0.5 * float(x @ A @ x)

    def grad_f(x):
        return A @ x

    name = f"Quadratic (kappa={kappa:.0e}, d={d})"
    return Problem(name=name, f=f, grad_f=grad_f, x0=x0, d=d,
                   description=f"Quadratic with condition number {kappa}")


# ---------------------------------------------------------------------------
# Problem 3: Rosenbrock function
# ---------------------------------------------------------------------------

def make_rosenbrock(d: int = 100, seed: int = 42) -> Problem:
    """
    Extended Rosenbrock function in d dimensions.

    f(x) = sum_{i=1}^{d-1} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]

    Minimum at x* = (1, 1, ..., 1) with f* = 0.
    """
    rng = np.random.RandomState(seed)
    x0 = rng.randn(d) * 0.5

    def f(x):
        return float(np.sum(
            100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2
        ))

    def grad_f(x):
        g = np.zeros_like(x)
        g[:-1] += -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
        g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
        return g

    return Problem(
        name=f"Rosenbrock (d={d})",
        f=f, grad_f=grad_f, x0=x0, d=d,
        description="Extended Rosenbrock function"
    )


# ---------------------------------------------------------------------------
# Problem 4: Logistic Regression (synthetic MNIST-like)
# ---------------------------------------------------------------------------

def make_logistic_regression(
    n: int = 12665,
    d: int = 784,
    lam: float = 1e-4,
    seed: int = 42
) -> Problem:
    """
    Binary logistic regression with L2 regularization.

    f(w) = (1/n) * sum log(1 + exp(-y_i * w^T x_i)) + (lambda/2) ||w||^2

    Uses synthetic data with same dimensions as MNIST binary (0 vs 1).
    Replace X, y with real MNIST data if available (see load_mnist_binary).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d) / np.sqrt(d)
    y = rng.choice([-1, 1], size=n)
    w0 = np.zeros(d)

    def f(w):
        margins = y * (X @ w)
        loss = np.mean(np.log1p(np.exp(-np.clip(margins, -500, 500))))
        reg = 0.5 * lam * np.dot(w, w)
        return float(loss + reg)

    def grad_f(w):
        margins = y * (X @ w)
        sigmoid = 1.0 / (1.0 + np.exp(np.clip(margins, -500, 500)))
        g = -(X.T @ (y * sigmoid)) / n + lam * w
        return g

    return Problem(
        name=f"Logistic Regression (n={n}, d={d})",
        f=f, grad_f=grad_f, x0=w0, d=d,
        description="Binary logistic regression with L2 regularization"
    )


# ---------------------------------------------------------------------------
# Problem 5: Two-layer neural network
# ---------------------------------------------------------------------------

def make_neural_network(
    n: int = 1000,
    d_in: int = 784,
    d_hidden: int = 100,
    d_out: int = 10,
    lam: float = 1e-4,
    seed: int = 42
) -> Problem:
    """
    Two-layer MLP with cross-entropy loss and L2 regularization.

    Architecture: Linear(d_in, d_hidden) -> ReLU -> Linear(d_hidden, d_out)
    Parameters: W1 (d_in x d_hidden), b1 (d_hidden,), W2 (d_hidden x d_out), b2 (d_out,)
    Total params: d_in*d_hidden + d_hidden + d_hidden*d_out + d_out
    """
    rng = np.random.RandomState(seed)

    # Synthetic data
    X = rng.randn(n, d_in) / np.sqrt(d_in)
    labels = rng.randint(0, d_out, size=n)
    Y = np.eye(d_out)[labels]  # one-hot

    # Parameter layout
    n_W1 = d_in * d_hidden
    n_b1 = d_hidden
    n_W2 = d_hidden * d_out
    n_b2 = d_out
    d_total = n_W1 + n_b1 + n_W2 + n_b2

    # Initialize with small random values
    theta0 = rng.randn(d_total) * 0.01

    def unpack(theta):
        W1 = theta[:n_W1].reshape(d_in, d_hidden)
        b1 = theta[n_W1:n_W1 + n_b1]
        W2 = theta[n_W1 + n_b1:n_W1 + n_b1 + n_W2].reshape(d_hidden, d_out)
        b2 = theta[n_W1 + n_b1 + n_W2:]
        return W1, b1, W2, b2

    def forward(theta):
        W1, b1, W2, b2 = unpack(theta)
        Z1 = X @ W1 + b1          # (n, d_hidden)
        A1 = np.maximum(0, Z1)    # ReLU
        Z2 = A1 @ W2 + b2         # (n, d_out)
        # Softmax
        Z2 -= Z2.max(axis=1, keepdims=True)
        exp_Z2 = np.exp(Z2)
        probs = exp_Z2 / exp_Z2.sum(axis=1, keepdims=True)
        return Z1, A1, probs

    def f(theta):
        _, _, probs = forward(theta)
        log_probs = np.log(np.clip(probs, 1e-15, 1.0))
        loss = -np.mean(np.sum(Y * log_probs, axis=1))
        reg = 0.5 * lam * np.dot(theta, theta)
        return float(loss + reg)

    def grad_f(theta):
        W1, b1, W2, b2 = unpack(theta)
        Z1, A1, probs = forward(theta)

        # Backprop
        delta2 = (probs - Y) / n                  # (n, d_out)
        dW2 = A1.T @ delta2                        # (d_hidden, d_out)
        db2 = delta2.sum(axis=0)                   # (d_out,)

        delta1 = (delta2 @ W2.T) * (Z1 > 0)       # ReLU grad
        dW1 = X.T @ delta1                         # (d_in, d_hidden)
        db1 = delta1.sum(axis=0)                   # (d_hidden,)

        g = np.concatenate([
            dW1.ravel(), db1, dW2.ravel(), db2
        ]) + lam * theta

        return g

    return Problem(
        name=f"Neural Network (n={n}, d={d_total})",
        f=f, grad_f=grad_f, x0=theta0, d=d_total,
        description="Two-layer MLP with ReLU and cross-entropy loss"
    )


# ---------------------------------------------------------------------------
# Problem 6: Sparse logistic regression with smooth L1
# ---------------------------------------------------------------------------

def make_sparse_logistic(
    n: int = 2000,
    d: int = 5000,
    lam: float = 1e-3,
    mu_smooth: float = 1e-3,
    seed: int = 42
) -> Problem:
    """
    Sparse logistic regression with smoothed L1 regularization.

    f(w) = (1/n) sum log(1 + exp(-y_i w^T x_i)) + lambda * sum phi(w_j)
    where phi(t) = sqrt(t^2 + mu^2) - mu  (smooth L1 approximation)
    """
    rng = np.random.RandomState(seed)

    # Sparse data
    nnz_per_row = max(1, d // 20)
    X = np.zeros((n, d))
    for i in range(n):
        idx = rng.choice(d, nnz_per_row, replace=False)
        X[i, idx] = rng.randn(nnz_per_row)
    X /= np.sqrt(nnz_per_row)

    y = rng.choice([-1, 1], size=n)
    w0 = np.zeros(d)

    def smooth_l1(w):
        return np.sum(np.sqrt(w ** 2 + mu_smooth ** 2) - mu_smooth)

    def grad_smooth_l1(w):
        return w / np.sqrt(w ** 2 + mu_smooth ** 2)

    def f(w):
        margins = y * (X @ w)
        loss = np.mean(np.log1p(np.exp(-np.clip(margins, -500, 500))))
        reg = lam * smooth_l1(w)
        return float(loss + reg)

    def grad_f(w):
        margins = y * (X @ w)
        sigmoid = 1.0 / (1.0 + np.exp(np.clip(margins, -500, 500)))
        g = -(X.T @ (y * sigmoid)) / n + lam * grad_smooth_l1(w)
        return g

    return Problem(
        name=f"Sparse Logistic (n={n}, d={d})",
        f=f, grad_f=grad_f, x0=w0, d=d,
        description="Sparse logistic regression with smooth L1 regularization"
    )


def get_all_problems() -> dict:
    """Return all six test problems from the paper."""
    return {
        "well_conditioned_quadratic": make_quadratic(d=1000, kappa=10.0),
        "ill_conditioned_quadratic":  make_quadratic(d=1000, kappa=1e4),
        "rosenbrock":                 make_rosenbrock(d=100),
        "logistic_mnist":             make_logistic_regression(n=5000, d=784),
        "neural_network":             make_neural_network(n=500, d_in=784),
        "sparse_logistic":            make_sparse_logistic(n=2000, d=5000),
    }
