"""Tests for numerical._distributions: Normal, StudentT, LogNormal, Uniform, Exponential."""
import pytest, math, numpy as np
from pricebook.numerical._distributions import Normal, StudentT, LogNormal, Uniform, Exponential

class TestNormal:
    def test_cdf_symmetry(self):
        n = Normal()
        assert abs(n.cdf(0) - 0.5) < 1e-10
    def test_ppf_inverse(self):
        n = Normal()
        assert abs(n.ppf(0.975) - 1.96) < 0.01
    def test_pdf_peak(self):
        n = Normal()
        assert abs(n.pdf(0) - 1/math.sqrt(2*math.pi)) < 1e-10
    def test_rvs(self):
        samples = Normal(5, 2).rvs(1000, rng=np.random.default_rng(42))
        assert abs(np.mean(samples) - 5) < 0.2

class TestStudentT:
    def test_heavier_tails(self):
        t3 = StudentT(df=3)
        n = Normal()
        assert t3.pdf(3.0) > n.pdf(3.0)  # heavier tails

class TestLogNormal:
    def test_positive(self):
        ln = LogNormal()
        assert all(x > 0 for x in ln.rvs(100, rng=np.random.default_rng(42)))

class TestUniform:
    def test_bounds(self):
        u = Uniform(2, 5)
        assert u.cdf(2) == 0.0
        assert u.cdf(5) == 1.0

class TestExponential:
    def test_memoryless(self):
        e = Exponential(rate=2.0)
        assert abs(e.mean() - 0.5) < 1e-10
