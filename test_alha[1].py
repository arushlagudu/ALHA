"""
tests/test_alha.py
==================
Unit tests for the ALHA algorithm.

Run with: python -m pytest tests/ -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from alha import alha, ALHAConfig, two_loop_recursion, compute_quality
from problems import make_quadratic, make_rosenbrock, make_logistic_regression


# ---------------------------------------------------------------------------
# Tests for two_loop_recursion
# ---------------------------------------------------------------------------

class TestTwoLoopRecursion:

    def test_empty_notebook_returns_gamma_g(self):
        """With no curvature pairs, H_k = gamma*I, so result = gamma * g."""
        g = np.array([1.0, 2.0, 3.0])
        gamma = 0.5
        r = two_loop_recursion(g, [], [], m=5, gamma=gamma)
        np.testing.assert_allclose(r, gamma * g)

    def test_single_pair_perfect_parabola(self):
        """For f(x) = x^2 + 2x + 1, Hessian = 2, so result should = 0.5 * g."""
        # Simulate one step from x=3
        x0, x1 = 3.0, 2.2
        s0 = x1 - x0           # -0.8
        y0 = (2*x1+2) - (2*x0+2)  # -1.6
        g = np.array([2*x1 + 2])   # gradient at x1

        S = [np.array([s0])]
        Y = [np.array([y0])]

        # gamma from this pair
        gamma = (s0 * y0) / (y0 * y0)  # 0.5

        r = two_loop_recursion(g, S, Y, m=1, gamma=gamma)
        expected = 0.5 * g  # H^{-1} = 1/2 for Hessian = 2
        np.testing.assert_allclose(r, expected, rtol=1e-10)

    def test_output_shape(self):
        """Output shape matches input gradient shape."""
        d = 50
        g = np.random.randn(d)
        S = [np.random.randn(d) for _ in range(3)]
        Y = [np.random.randn(d) for _ in range(3)]
        # Make S, Y satisfy curvature condition
        for i in range(3):
            Y[i] = S[i] * 2.0 + 0.1 * np.random.randn(d)

        r = two_loop_recursion(g, S, Y, m=3, gamma=0.5)
        assert r.shape == g.shape

    def test_memory_limit_respected(self):
        """With m=1, only the most recent pair should be used."""
        d = 10
        g = np.random.randn(d)
        # Two pairs, but m=1 means only use the most recent
        S = [np.ones(d), np.ones(d) * 2]
        Y = [np.ones(d) * 2, np.ones(d) * 4]
        r1 = two_loop_recursion(g, S[-1:], Y[-1:], m=1, gamma=1.0)
        r2 = two_loop_recursion(g, S, Y, m=1, gamma=1.0)
        np.testing.assert_allclose(r1, r2, rtol=1e-10)


# ---------------------------------------------------------------------------
# Tests for compute_quality
# ---------------------------------------------------------------------------

class TestComputeQuality:

    def test_perfect_secant_gives_one(self):
        """When H_k * y_k = s_k exactly, Q_hat = 1."""
        # 1D perfect parabola
        x0, x1 = 3.0, 2.2
        s = np.array([x1 - x0])
        y = np.array([(2*x1+2) - (2*x0+2)])

        S = [s]
        Y = [y]
        gamma = float((s * y) / (y * y))

        q = compute_quality(S, Y, s, y, m=1, gamma=gamma)
        assert abs(q - 1.0) < 1e-8

    def test_output_is_positive(self):
        """Quality metric should always be positive."""
        d = 20
        rng = np.random.RandomState(0)
        s = rng.randn(d)
        y = 2.0 * s + 0.1 * rng.randn(d)  # near-Hessian relationship

        S = [s]
        Y = [y]
        gamma = np.dot(s, y) / np.dot(y, y)

        q = compute_quality(S, Y, s, y, m=1, gamma=gamma)
        assert q > 0

    def test_clipping(self):
        """Quality metric should be clipped to [clip_low, clip_high]."""
        d = 5
        # Pathological case
        s = np.ones(d) * 1e-10
        y = np.ones(d)
        S = [s]
        Y = [y]
        q = compute_quality(S, Y, s, y, m=1, gamma=1.0,
                            clip_low=0.01, clip_high=100.0)
        assert 0.01 <= q <= 100.0


# ---------------------------------------------------------------------------
# Tests for ALHA convergence
# ---------------------------------------------------------------------------

class TestALHAConvergence:

    def test_converges_on_1d_parabola(self):
        """ALHA should converge on f(x) = (x-3)^2."""
        f = lambda x: (x[0] - 3.0) ** 2
        grad_f = lambda x: np.array([2.0 * (x[0] - 3.0)])
        x0 = np.array([0.0])

        config = ALHAConfig(eps_tol=1e-8, max_iter=1000, verbose=False)
        result = alha(f, grad_f, x0, config)

        assert result.converged
        np.testing.assert_allclose(result.x, [3.0], atol=1e-6)

    def test_converges_on_quadratic(self):
        """ALHA should converge on a well-conditioned quadratic."""
        problem = make_quadratic(d=50, kappa=10.0, seed=0)
        config = ALHAConfig(eps_tol=1e-8, max_iter=5000, verbose=False)
        result = alha(problem.f, problem.grad_f, problem.x0, config)

        assert result.converged or result.grad_norm < 1e-6

    def test_converges_on_rosenbrock(self):
        """ALHA should converge on Rosenbrock (d=10)."""
        problem = make_rosenbrock(d=10, seed=0)
        config = ALHAConfig(eps_tol=1e-6, max_iter=5000, verbose=False)
        result = alha(problem.f, problem.grad_f, problem.x0, config)

        assert result.converged or result.grad_norm < 1e-4

    def test_rank_stays_bounded(self):
        """Rank should always stay within [m_min, m_max]."""
        problem = make_quadratic(d=50, kappa=100.0, seed=1)
        config = ALHAConfig(m_min=2, m_max=15, m_0=5,
                            eps_tol=1e-6, max_iter=500, verbose=False)
        result = alha(problem.f, problem.grad_f, problem.x0, config)

        for m in result.rank_history:
            assert config.m_min <= m <= config.m_max, \
                f"Rank {m} outside [{config.m_min}, {config.m_max}]"

    def test_monotone_decrease(self):
        """Function values should be non-increasing."""
        problem = make_quadratic(d=30, kappa=10.0, seed=2)
        config = ALHAConfig(eps_tol=1e-8, max_iter=200, verbose=False)
        result = alha(problem.f, problem.grad_f, problem.x0, config)

        for i in range(1, len(result.f_history)):
            assert result.f_history[i] <= result.f_history[i-1] + 1e-10, \
                f"Non-monotone at iter {i}: {result.f_history[i-1]} -> {result.f_history[i]}"

    def test_result_object_fields(self):
        """Result object should have all expected fields."""
        problem = make_quadratic(d=10, kappa=5.0, seed=3)
        config = ALHAConfig(eps_tol=1e-4, max_iter=100, verbose=False)
        result = alha(problem.f, problem.grad_f, problem.x0, config)

        assert hasattr(result, 'x')
        assert hasattr(result, 'f_val')
        assert hasattr(result, 'grad_norm')
        assert hasattr(result, 'n_iter')
        assert hasattr(result, 'converged')
        assert hasattr(result, 'time_elapsed')
        assert hasattr(result, 'f_history')
        assert hasattr(result, 'grad_norm_history')
        assert hasattr(result, 'rank_history')
        assert hasattr(result, 'q_history')

        assert len(result.f_history) == result.n_iter + 1
        assert len(result.rank_history) == result.n_iter + 1

    def test_rank_adapts_on_ill_conditioned(self):
        """On ill-conditioned problem, ALHA should increase rank."""
        problem = make_quadratic(d=50, kappa=1e4, seed=4)
        config = ALHAConfig(m_min=2, m_max=30, m_0=3,
                            eps_tol=1e-6, max_iter=500, verbose=False)
        result = alha(problem.f, problem.grad_f, problem.x0, config)

        max_rank = max(result.rank_history)
        # Should have increased beyond starting rank on ill-conditioned problem
        assert max_rank > config.m_0, \
            f"Expected rank to increase beyond {config.m_0}, max was {max_rank}"


# ---------------------------------------------------------------------------
# Run tests directly
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
