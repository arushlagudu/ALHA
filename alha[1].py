"""
ALHA: Adaptive Low-rank Hessian Approximation
=============================================
Core algorithm implementation.

Reference:
    Lagudu, A. R. (2025). Adaptive Low-Rank Hessian Approximation for
    Large-Scale Optimization: Theory, Algorithms, and Applications.
"""

import numpy as np
from scipy.optimize import line_search_wolfe2
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple
import time


@dataclass
class ALHAConfig:
    """Configuration for the ALHA optimizer."""
    # Rank bounds
    m_min: int = 2
    m_max: int = 50
    m_0: int = 5

    # Quality thresholds
    eps_low: float = 0.1
    eps_high: float = 0.3

    # Rank update steps
    delta_m_plus: int = 2
    delta_m_minus: int = 1

    # Wolfe line search constants
    c1: float = 1e-4
    c2: float = 0.9

    # Numerical stability
    eps_curv: float = 1e-8
    gamma_min: float = 1e-8
    gamma_max: float = 1e8
    q_clip_low: float = 0.01
    q_clip_high: float = 100.0

    # Convergence
    eps_tol: float = 1e-8
    max_iter: int = 10000

    # Verbosity
    verbose: bool = False
    log_interval: int = 100


@dataclass
class ALHAResult:
    """Result object returned by ALHA."""
    x: np.ndarray                    # Final iterate
    f_val: float                     # Final function value
    grad_norm: float                 # Final gradient norm
    n_iter: int                      # Number of iterations
    converged: bool                  # Whether convergence criterion was met
    time_elapsed: float              # Wall-clock time in seconds
    f_history: list = field(default_factory=list)
    grad_norm_history: list = field(default_factory=list)
    rank_history: list = field(default_factory=list)
    q_history: list = field(default_factory=list)


def two_loop_recursion(
    g: np.ndarray,
    S: list,
    Y: list,
    m: int,
    gamma: float = 1.0
) -> np.ndarray:
    """
    Two-loop recursion for computing H_k * g.

    Args:
        g:     Current gradient vector, shape (d,)
        S:     List of step vectors [s_{k-m+1}, ..., s_k]
        Y:     List of gradient-change vectors [y_{k-m+1}, ..., y_k]
        m:     Memory parameter (number of pairs to use)
        gamma: Scaling parameter gamma_k

    Returns:
        r: The product H_k * g, shape (d,)
    """
    ell = min(len(S), m)
    q = g.copy()

    # Precompute rho values
    rho = []
    alpha = []

    # First loop: backward pass (newest to oldest)
    for i in range(ell - 1, -1, -1):
        sy = np.dot(S[i], Y[i])
        if abs(sy) < 1e-15:
            rho.insert(0, 0.0)
            alpha.insert(0, 0.0)
            continue
        rho_i = 1.0 / sy
        alpha_i = rho_i * np.dot(S[i], q)
        q = q - alpha_i * Y[i]
        rho.insert(0, rho_i)
        alpha.insert(0, alpha_i)

    # Apply scaling
    r = gamma * q

    # Second loop: forward pass (oldest to newest)
    for i in range(ell):
        if rho[i] == 0.0:
            continue
        beta_i = rho[i] * np.dot(Y[i], r)
        r = r + (alpha[i] - beta_i) * S[i]

    return r


def compute_quality(
    S: list,
    Y: list,
    s: np.ndarray,
    y: np.ndarray,
    m: int,
    gamma: float,
    clip_low: float = 0.01,
    clip_high: float = 100.0
) -> float:
    """
    Compute the computable quality metric Q_hat_k.

    Q_hat_k = (H_k * y_k)^T * y_k / (s_k^T * y_k)

    Args:
        S, Y:  Current curvature pair storage
        s, y:  Most recent step and gradient change
        m:     Current memory parameter
        gamma: Current scaling parameter
        clip_low, clip_high: Clipping bounds for numerical stability

    Returns:
        Q_hat: Quality score (1.0 = perfect secant equation)
    """
    sy = np.dot(s, y)
    if abs(sy) < 1e-15:
        return 1.0

    # h = H_k * y_k (apply approximation to gradient change)
    h = two_loop_recursion(y, S, Y, len(S), gamma)
    hy = np.dot(h, y)

    q_hat = hy / sy
    # Clip for numerical stability
    q_hat = np.clip(q_hat, clip_low, clip_high)
    return float(q_hat)


def alha(
    f: Callable,
    grad_f: Callable,
    x0: np.ndarray,
    config: Optional[ALHAConfig] = None
) -> ALHAResult:
    """
    ALHA: Adaptive Low-rank Hessian Approximation optimizer.

    Args:
        f:       Objective function f(x) -> float
        grad_f:  Gradient function grad_f(x) -> np.ndarray
        x0:      Initial point, shape (d,)
        config:  ALHAConfig object (uses defaults if None)

    Returns:
        ALHAResult with final iterate and history
    """
    if config is None:
        config = ALHAConfig()

    x = x0.copy().astype(float)
    d = len(x)

    # Storage for curvature pairs
    S = []  # step vectors
    Y = []  # gradient change vectors

    # Initialize
    g = grad_f(x)
    m_k = config.m_0
    gamma = 1.0  # initial scaling

    # History tracking
    f_history = [f(x)]
    grad_norm_history = [np.linalg.norm(g)]
    rank_history = [m_k]
    q_history = [1.0]

    t_start = time.time()
    converged = False
    n_iter = 0

    for k in range(config.max_iter):
        grad_norm = np.linalg.norm(g)

        if config.verbose and k % config.log_interval == 0:
            print(f"  Iter {k:5d} | f={f(x):.6e} | ||g||={grad_norm:.3e} | m={m_k} | gamma={gamma:.3e}")

        # Check convergence
        if grad_norm < config.eps_tol:
            converged = True
            n_iter = k
            break

        # --- Compute search direction ---
        d_k = -two_loop_recursion(g, S, Y, m_k, gamma)

        # Safety check: ensure descent direction
        if np.dot(g, d_k) >= 0:
            d_k = -g  # fall back to gradient descent
            if config.verbose:
                print(f"  Warning: non-descent direction at iter {k}, falling back to GD")

        # --- Wolfe line search ---
        f_val = f(x)

        def f_for_ls(alpha):
            return f(x + alpha * d_k)

        def grad_for_ls(alpha):
            return np.dot(grad_f(x + alpha * d_k), d_k)

        alpha_k, _, _, f_new, _, _ = line_search_wolfe2(
            f_for_ls, grad_for_ls,
            0.0, f_val, np.dot(g, d_k),
            c1=config.c1, c2=config.c2
        )

        if alpha_k is None or alpha_k <= 0:
            # Line search failed, use small step
            alpha_k = 1e-4
            if config.verbose:
                print(f"  Warning: line search failed at iter {k}, using alpha={alpha_k}")

        # --- Take step ---
        x_new = x + alpha_k * d_k
        g_new = grad_f(x_new)

        s_k = x_new - x
        y_k = g_new - g

        # --- Curvature condition check ---
        sy = np.dot(s_k, y_k)
        curv_threshold = config.eps_curv * np.linalg.norm(s_k) * np.linalg.norm(y_k)

        if sy > curv_threshold:
            # Accept the curvature pair
            S.append(s_k.copy())
            Y.append(y_k.copy())

            # Update gamma from most recent pair
            yy = np.dot(y_k, y_k)
            if yy > 1e-15:
                gamma = np.clip(sy / yy, config.gamma_min, config.gamma_max)

            # Trim to current memory size
            while len(S) > m_k:
                S.pop(0)
                Y.pop(0)

        # --- Compute quality metric and adapt rank ---
        q_hat = 1.0
        if k >= 1 and len(S) > 0:
            q_hat = compute_quality(
                S, Y, s_k, y_k, m_k, gamma,
                config.q_clip_low, config.q_clip_high
            )

            # Adapt rank
            if q_hat < 1.0 - config.eps_high:
                m_k = min(m_k + config.delta_m_plus, config.m_max)
            elif q_hat > 1.0 - config.eps_low:
                m_k = max(m_k - config.delta_m_minus, config.m_min)
            # else: keep m_k unchanged

        # --- Update state ---
        x = x_new
        g = g_new
        n_iter = k + 1

        # Record history
        f_history.append(f(x))
        grad_norm_history.append(np.linalg.norm(g))
        rank_history.append(m_k)
        q_history.append(q_hat)

    t_elapsed = time.time() - t_start

    return ALHAResult(
        x=x,
        f_val=f(x),
        grad_norm=np.linalg.norm(g),
        n_iter=n_iter,
        converged=converged,
        time_elapsed=t_elapsed,
        f_history=f_history,
        grad_norm_history=grad_norm_history,
        rank_history=rank_history,
        q_history=q_history,
    )
