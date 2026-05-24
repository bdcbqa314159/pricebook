"""Numerical integration — redirects to numerical._integrate.

Legacy module. Use pricebook.numerical._integrate directly.
"""

from pricebook.numerical._integrate import (
    integrate, IntegrationMethod, IntegrationResult,
)

# Backward-compat alias
QuadratureResult = IntegrationResult


def gauss_legendre(f, a=-1.0, b=1.0, n=16):
    return integrate(f, a, b, IntegrationMethod.GAUSS_LEGENDRE, n=n)


def gauss_laguerre(f, n=16):
    return integrate(f, 0, float('inf'), IntegrationMethod.GAUSS_LAGUERRE, n=n)


def gauss_hermite(f, n=16):
    return integrate(f, float('-inf'), float('inf'), IntegrationMethod.GAUSS_HERMITE, n=n)


def adaptive_simpson(f, a, b, tol=1e-8):
    return integrate(f, a, b, IntegrationMethod.ADAPTIVE, tol=tol)
