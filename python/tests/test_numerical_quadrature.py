"""Tests for numerical integration — migrated from old quadrature API."""
import pytest, math
from pricebook.numerical._integrate import integrate, IntegrationMethod

class TestGaussLegendre:
    def test_polynomial(self):
        result = integrate(lambda x: x**2, -1, 1, IntegrationMethod.GAUSS_LEGENDRE, n=8)
        assert abs(result.value - 2/3) < 1e-10

class TestTanhSinh:
    def test_smooth(self):
        result = integrate(lambda x: math.exp(x), 0, 1, IntegrationMethod.TANH_SINH)
        assert abs(result.value - (math.e - 1)) < 0.01

class TestClenshawCurtis:
    def test_trig(self):
        result = integrate(lambda x: math.cos(x), 0, math.pi, IntegrationMethod.CLENSHAW_CURTIS)
        assert abs(result.value - 0.0) < 0.01
