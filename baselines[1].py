"""
Baseline optimizers for comparison with ALHA.
Includes: Gradient Descent, L-BFGS, Adam.
"""

import numpy as np
from scipy.optimize import minimize, line_search_wolfe2
from dataclasses import dataclass, field
from typing import Callable, Optional
import time


@dataclass
class BaselineResult:
    """Result object for baseline optimizers."""
    x: np.ndarray
    f_val: float
    grad_norm: float
    n_iter: int
    converged: bool
    time_elapsed: float
    f_history: list = field(default_factory=list)
    grad_norm_history: list = field(default_factory=list)


def run_gradient_descent(
    f: Callable,
    grad_f: Callable,
    x0: np.ndarray,
    lr: float = 0.01,
    max_iter: int = 10000,
    eps_tol: float = 1e-8,
    verbose: bool = False
) -> BaselineResult:
    """Gradient descent with fixed step size."""
    x = x0.copy().astype(float)
    f_history = [f(x)]
    grad_norm_history = [np.linalg.norm(grad_f(x))]

    t_start = time.time()
    converged = False
    n_iter = 0

    for k in range(max_iter):
        g = grad_f(x)
        grad_norm = np.linalg.norm(g)

        if grad_norm < eps_tol:
            converged = True
            n_iter = k
            break

        x = x - lr * g
        n_iter = k + 1
        f_history.append(f(x))
        grad_norm_history.append(np.linalg.norm(grad_f(x)))

        if verbose and k % 500 == 0:
            print(f"  GD Iter {k:5d} | f={f(x):.6e} | ||g||={grad_norm:.3e}")

    t_elapsed = time.time() - t_start
    g_final = grad_f(x)

    return BaselineResult(
        x=x, f_val=f(x), grad_norm=np.linalg.norm(g_final),
        n_iter=n_iter, converged=converged, time_elapsed=t_elapsed,
        f_history=f_history, grad_norm_history=grad_norm_history
    )


def run_lbfgs(
    f: Callable,
    grad_f: Callable,
    x0: np.ndarray,
    m: int = 10,
    max_iter: int = 10000,
    eps_tol: float = 1e-8,
    verbose: bool = False
) -> BaselineResult:
    """
    L-BFGS with fixed memory parameter m.
    Uses scipy's implementation wrapped to collect history.
    """
    x = x0.copy().astype(float)
    f_history = [f(x)]
    grad_norm_history = [np.linalg.norm(grad_f(x))]
    iter_count = [0]

    def callback(xk):
        iter_count[0] += 1
        fval = f(xk)
        gnorm = np.linalg.norm(grad_f(xk))
        f_history.append(fval)
        grad_norm_history.append(gnorm)
        if verbose and iter_count[0] % 100 == 0:
            print(f"  L-BFGS(m={m}) Iter {iter_count[0]:5d} | f={fval:.6e} | ||g||={gnorm:.3e}")

    t_start = time.time()

    result = minimize(
        fun=f,
        x0=x,
        jac=grad_f,
        method='L-BFGS-B',
        callback=callback,
        options={
            'maxiter': max_iter,
            'ftol': 0,
            'gtol': eps_tol,
            'maxcor': m,
            'maxfun': max_iter * 20,
        }
    )

    t_elapsed = time.time() - t_start

    return BaselineResult(
        x=result.x,
        f_val=float(result.fun),
        grad_norm=np.linalg.norm(grad_f(result.x)),
        n_iter=iter_count[0],
        converged=result.success,
        time_elapsed=t_elapsed,
        f_history=f_history,
        grad_norm_history=grad_norm_history
    )


def run_adam(
    f: Callable,
    grad_f: Callable,
    x0: np.ndarray,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    max_iter: int = 10000,
    eps_tol: float = 1e-8,
    verbose: bool = False
) -> BaselineResult:
    """Adam optimizer."""
    x = x0.copy().astype(float)
    m_adam = np.zeros_like(x)
    v_adam = np.zeros_like(x)
    f_history = [f(x)]
    grad_norm_history = [np.linalg.norm(grad_f(x))]

    t_start = time.time()
    converged = False
    n_iter = 0

    for k in range(1, max_iter + 1):
        g = grad_f(x)
        grad_norm = np.linalg.norm(g)

        if grad_norm < eps_tol:
            converged = True
            n_iter = k - 1
            break

        m_adam = beta1 * m_adam + (1 - beta1) * g
        v_adam = beta2 * v_adam + (1 - beta2) * g ** 2

        m_hat = m_adam / (1 - beta1 ** k)
        v_hat = v_adam / (1 - beta2 ** k)

        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        n_iter = k
        f_history.append(f(x))
        grad_norm_history.append(np.linalg.norm(grad_f(x)))

        if verbose and k % 500 == 0:
            print(f"  Adam Iter {k:5d} | f={f(x):.6e} | ||g||={grad_norm:.3e}")

    t_elapsed = time.time() - t_start

    return BaselineResult(
        x=x, f_val=f(x), grad_norm=np.linalg.norm(grad_f(x)),
        n_iter=n_iter, converged=converged, time_elapsed=t_elapsed,
        f_history=f_history, grad_norm_history=grad_norm_history
    )
