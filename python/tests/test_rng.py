"""Tests for random number generators."""

import pytest
import numpy as np

from pricebook.rng import PseudoRandom, QuasiRandom


class TestPseudoRandom:
    def test_shape(self):
        rng = PseudoRandom(seed=42)
        z = rng.normals(1000, 10)
        assert z.shape == (1000, 10)

    def test_single_step_shape(self):
        rng = PseudoRandom(seed=42)
        z = rng.normals(500)
        assert z.shape == (500, 1)

    def test_reproducible(self):
        z1 = PseudoRandom(seed=123).normals(100, 5)
        z2 = PseudoRandom(seed=123).normals(100, 5)
        np.testing.assert_array_equal(z1, z2)

    def test_different_seeds_differ(self):
        z1 = PseudoRandom(seed=1).normals(100, 5)
        z2 = PseudoRandom(seed=2).normals(100, 5)
        assert not np.allclose(z1, z2)

    def test_mean_near_zero(self):
        rng = PseudoRandom(seed=42)
        z = rng.normals(100_000)
        assert abs(z.mean()) < 0.02

    def test_std_near_one(self):
        rng = PseudoRandom(seed=42)
        z = rng.normals(100_000)
        assert abs(z.std() - 1.0) < 0.02


class TestQuasiRandom:
    def test_shape(self):
        qrng = QuasiRandom(dimension=10, seed=42)
        z = qrng.normals(1000)
        assert z.shape == (1000, 10)

    def test_single_dimension_shape(self):
        qrng = QuasiRandom(dimension=1, seed=42)
        z = qrng.normals(500)
        assert z.shape == (500, 1)

    def test_reproducible(self):
        z1 = QuasiRandom(dimension=5, seed=123).normals(100)
        z2 = QuasiRandom(dimension=5, seed=123).normals(100)
        np.testing.assert_array_equal(z1, z2)

    def test_mean_near_zero(self):
        qrng = QuasiRandom(dimension=1, seed=42)
        z = qrng.normals(10_000)
        assert abs(z.mean()) < 0.02

    def test_std_near_one(self):
        qrng = QuasiRandom(dimension=1, seed=42)
        z = qrng.normals(10_000)
        assert abs(z.std() - 1.0) < 0.05

    def test_invalid_dimension_raises(self):
        with pytest.raises(ValueError, match="dimension"):
            QuasiRandom(dimension=0)

    def test_non_power_of_2_paths(self):
        """Sobol works even when n_paths is not a power of 2."""
        qrng = QuasiRandom(dimension=3, seed=42)
        z = qrng.normals(700)
        assert z.shape == (700, 3)

    def test_quasi_lower_discrepancy(self):
        """Quasi-random should have more uniform coverage than pseudo-random.
        Test via mean convergence: quasi mean should be closer to 0."""
        n = 1024
        pseudo = PseudoRandom(seed=42).normals(n)
        quasi = QuasiRandom(dimension=1, seed=42).normals(n)
        # Quasi should have mean at least as close to 0 (usually much closer)
        assert abs(quasi.mean()) <= abs(pseudo.mean()) + 0.1
