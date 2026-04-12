"""Tests for Laplace transform inversion."""

import math
import cmath

import pytest

from pricebook.laplace_inversion import (
    LaplaceInversionResult,
    euler_inversion,
    gaver_stehfest,
    invert,
    talbot_inversion,
)


# ---- Known Laplace pairs ----
# f(t) = exp(-at) → F(s) = 1/(s+a)
# f(t) = t^n → F(s) = n!/s^{n+1}
# f(t) = sin(ωt) → F(s) = ω/(s²+ω²)
# f(t) = 1 → F(s) = 1/s

def _exponential_F(a: float):
    """F(s) = 1/(s+a), f(t) = exp(-at)."""
    def F(s):
        return 1.0 / (s + a)
    return F


def _constant_F():
    """F(s) = 1/s, f(t) = 1."""
    def F(s):
        return 1.0 / s
    return F


def _sine_F(omega: float):
    """F(s) = ω/(s²+ω²), f(t) = sin(ωt)."""
    def F(s):
        return omega / (s * s + omega * omega)
    return F


# ---- Talbot ----

class TestTalbotInversion:
    def test_exponential(self):
        """L^{-1}[1/(s+2)](t=1) = exp(-2)."""
        F = _exponential_F(2.0)
        result = talbot_inversion(F, t=1.0, N=32)
        assert result == pytest.approx(math.exp(-2.0), rel=1e-6)

    def test_constant(self):
        """L^{-1}[1/s](t=1) = 1."""
        result = talbot_inversion(_constant_F(), t=1.0)
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_sine(self):
        """L^{-1}[ω/(s²+ω²)](t=1) = sin(ω)."""
        omega = 3.0
        result = talbot_inversion(_sine_F(omega), t=1.0)
        assert result == pytest.approx(math.sin(omega), rel=1e-4)

    def test_multiple_times(self):
        F = _exponential_F(1.0)
        for t in [0.5, 1.0, 2.0, 5.0]:
            result = talbot_inversion(F, t)
            assert result == pytest.approx(math.exp(-t), rel=1e-4), f"t={t}"

    def test_t_zero(self):
        assert talbot_inversion(_constant_F(), t=0.0) == 0.0


# ---- Euler (Abate-Whitt) ----

class TestEulerInversion:
    def test_exponential(self):
        result = euler_inversion(_exponential_F(2.0), t=1.0)
        assert result == pytest.approx(math.exp(-2.0), rel=1e-4)

    def test_constant(self):
        result = euler_inversion(_constant_F(), t=1.0)
        assert result == pytest.approx(1.0, rel=1e-3)

    def test_sine(self):
        omega = 2.0
        result = euler_inversion(_sine_F(omega), t=1.0)
        assert result == pytest.approx(math.sin(omega), rel=1e-3)


# ---- Gaver-Stehfest (real-valued) ----

class TestGaverStehfest:
    def test_exponential(self):
        """Stehfest with real F(s) = 1/(s+a)."""
        def F(s):
            return 1.0 / (s + 2.0)
        result = gaver_stehfest(F, t=1.0, N=8)
        assert result == pytest.approx(math.exp(-2.0), rel=1e-2)

    def test_constant(self):
        def F(s):
            return 1.0 / s
        result = gaver_stehfest(F, t=1.0, N=8)
        assert result == pytest.approx(1.0, rel=1e-2)

    def test_no_complex_needed(self):
        """Stehfest only evaluates F at real positive s."""
        calls = []
        def F(s):
            assert isinstance(s, float) and s > 0
            calls.append(s)
            return 1.0 / (s + 1.0)
        gaver_stehfest(F, t=1.0, N=8)
        assert len(calls) == 8


# ---- Unified interface ----

class TestInvert:
    def test_talbot(self):
        result = invert(_exponential_F(1.0), t=1.0, method="talbot")
        assert result.method == "talbot"
        assert result.value == pytest.approx(math.exp(-1.0), rel=1e-4)

    def test_euler(self):
        result = invert(_exponential_F(1.0), t=1.0, method="euler")
        assert result.method == "euler"
        assert result.value == pytest.approx(math.exp(-1.0), rel=1e-3)

    def test_stehfest(self):
        def F(s):
            return 1.0 / (s + 1.0)
        result = invert(F, t=1.0, method="stehfest", N=8)
        assert result.method == "stehfest"
        assert result.value == pytest.approx(math.exp(-1.0), rel=1e-2)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            invert(_constant_F(), t=1.0, method="unknown")

    def test_all_methods_agree(self):
        """All three methods should agree on a simple transform."""
        t = 1.5
        expected = math.exp(-2.0 * t)
        F_complex = _exponential_F(2.0)
        def F_real(s): return 1.0 / (s + 2.0)

        talbot = invert(F_complex, t, "talbot").value
        euler = invert(F_complex, t, "euler").value
        stehfest = invert(F_real, t, "stehfest", N=8).value

        assert talbot == pytest.approx(expected, rel=1e-3)
        assert euler == pytest.approx(expected, rel=1e-2)
        assert stehfest == pytest.approx(expected, rel=0.05)
