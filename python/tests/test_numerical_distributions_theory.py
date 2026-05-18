"""Tests for numerical._distributions_theory: TemperedDistribution, Schwartz, Sobolev."""
import pytest, numpy as np
from pricebook.numerical._distributions_theory import (
    SchwartzTestFunction, TemperedDistribution, dirac_delta, sobolev_norm,
)

class TestSchwartz:
    def test_gaussian(self):
        phi = SchwartzTestFunction(sigma=1.0)
        assert abs(phi(0) - 1.0) < 1e-10  # e^0 = 1

class TestDirac:
    def test_sifting(self):
        delta = dirac_delta(0.0)
        phi = SchwartzTestFunction(sigma=1.0)
        assert abs(delta(phi) - 1.0) < 1e-10

class TestSobolev:
    def test_l2_positive(self):
        f = np.exp(-np.linspace(-5, 5, 500)**2)
        result = sobolev_norm(f, dx=10/500, s=1.0)
        assert result.h0 > 0
        assert result.h1 > 0
