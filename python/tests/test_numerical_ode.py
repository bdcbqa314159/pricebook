"""Tests for numerical._ode."""
import pytest, numpy as np
from pricebook.numerical._ode import euler, rk4, rk45, bdf

class TestEuler:
    def test_exponential(self):
        result = euler(lambda t, y: y, (0, 1), [1.0], n_steps=1000)
        assert abs(result.y[-1][0] - np.exp(1)) < 0.01

class TestRK4:
    def test_exponential(self):
        result = rk4(lambda t, y: y, (0, 1), [1.0], n_steps=100)
        assert abs(result.y[-1][0] - np.exp(1)) < 1e-6

class TestRK45:
    def test_exponential(self):
        result = rk45(lambda t, y: y, (0, 1), [1.0])
        assert abs(result.y[-1][0] - np.exp(1)) < 1e-4

class TestBDF:
    def test_stiff(self):
        result = bdf(lambda t, y: -100*np.array(y), (0, 0.1), [1.0])
        assert abs(result.y[-1][0] - np.exp(-10)) < 0.05
