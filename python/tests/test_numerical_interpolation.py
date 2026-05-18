"""Tests for numerical._interpolation: bilinear, bicubic, rbf."""
import pytest, numpy as np
from pricebook.numerical._interpolation import bilinear, bicubic, rbf_interpolate

class TestBilinear:
    def test_exact_at_grid(self):
        xs = np.array([0.0, 1.0])
        ys = np.array([0.0, 1.0])
        zs = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert abs(bilinear(0, 0, xs, ys, zs) - 1.0) < 1e-10
        assert abs(bilinear(1, 1, xs, ys, zs) - 4.0) < 1e-10

    def test_midpoint(self):
        xs = np.array([0.0, 1.0])
        ys = np.array([0.0, 1.0])
        zs = np.array([[0.0, 0.0], [0.0, 4.0]])
        assert abs(bilinear(0.5, 0.5, xs, ys, zs) - 1.0) < 1e-10

class TestRBF:
    def test_interpolates_data(self):
        centers = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=float)
        values = np.array([1, 2, 3, 4], dtype=float)
        result = rbf_interpolate(centers, values)
        pred = result.evaluate(centers)
        assert np.allclose(pred, values, atol=0.1)
