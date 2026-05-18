"""Tests for numerical._quadrature: gauss_jacobi, tanh_sinh, clenshaw_curtis."""
import pytest, math
from pricebook.numerical._quadrature import gauss_jacobi, tanh_sinh, clenshaw_curtis

class TestGaussJacobi:
    def test_polynomial(self):
        result = gauss_jacobi(lambda x: x**2, n=8, a=-1, b=1)
        assert abs(result.value - 2/3) < 1e-10

class TestTanhSinh:
    def test_smooth(self):
        result = tanh_sinh(lambda x: math.exp(x), a=0, b=1)
        assert abs(result.value - (math.e - 1)) < 0.01

class TestClenshawCurtis:
    def test_trig(self):
        result = clenshaw_curtis(lambda x: math.cos(x), a=0, b=math.pi)
        assert abs(result.value - 0.0) < 0.01  # ∫cos = sin(π)-sin(0) = 0
