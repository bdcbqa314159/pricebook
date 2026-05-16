"""Guaranteed note tests — joint default, guarantee value, sensitivities."""

from __future__ import annotations

import math
from datetime import date

import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve
from pricebook.guaranteed_note import (
    GuaranteedNote, guarantee_value, guarantee_spread,
)


REF = date(2024, 7, 15)
END = date(2029, 7, 15)


def _disc():
    return DiscountCurve.flat(REF, 0.04)


def _risky_issuer():
    return SurvivalCurve.flat(REF, 0.03)


def _strong_guarantor():
    return SurvivalCurve.flat(REF, 0.005)


def _gn(correlation=0.30):
    return GuaranteedNote(
        REF, END, coupon_rate=0.05, notional=100.0,
        correlation=correlation,
    )


# ── Pricing ──

class TestGuaranteedNotePricing:

    def test_guaranteed_pv_above_unguaranteed(self):
        result = _gn().price(_disc(), _risky_issuer(), _strong_guarantor())
        assert result.pv > result.pv_unguaranteed

    def test_guarantee_value_positive(self):
        result = _gn().price(_disc(), _risky_issuer(), _strong_guarantor())
        assert result.guarantee_value > 0

    def test_guarantee_spread_positive(self):
        result = _gn().price(_disc(), _risky_issuer(), _strong_guarantor())
        assert result.guarantee_spread_bp > 0

    def test_no_guarantee_when_both_safe(self):
        """If both issuer and guarantor are risk-free, guarantee value ≈ 0."""
        safe = SurvivalCurve.flat(REF, 0.0001)
        result = _gn().price(_disc(), safe, safe)
        assert abs(result.guarantee_value) < 0.5  # near zero

    def test_guarantee_value_increases_with_issuer_risk(self):
        """Riskier issuer → more guarantee value."""
        low_risk = SurvivalCurve.flat(REF, 0.01)
        high_risk = SurvivalCurve.flat(REF, 0.05)
        guar = _strong_guarantor()
        v1 = _gn().price(_disc(), low_risk, guar).guarantee_value
        v2 = _gn().price(_disc(), high_risk, guar).guarantee_value
        assert v2 > v1


# ── Correlation ──

class TestCorrelation:

    def test_higher_correlation_reduces_guarantee(self):
        """Higher ρ → more joint defaults → less guarantee value."""
        low_rho = _gn(correlation=0.10)
        high_rho = _gn(correlation=0.80)
        d, i, g = _disc(), _risky_issuer(), _strong_guarantor()
        v_low = low_rho.price(d, i, g).guarantee_value
        v_high = high_rho.price(d, i, g).guarantee_value
        assert v_low > v_high

    def test_rho01_negative(self):
        """Higher correlation → lower PV → rho01 < 0."""
        rho01 = _gn().rho01(_disc(), _risky_issuer(), _strong_guarantor())
        assert rho01 < 0

    def test_zero_correlation(self):
        """ρ=0 → independent defaults."""
        gn = _gn(correlation=0.0)
        result = gn.price(_disc(), _risky_issuer(), _strong_guarantor())
        assert result.pv > 0
        assert math.isfinite(result.guarantee_value)


# ── Sensitivities ──

class TestSensitivities:

    def test_cs01_issuer_small(self):
        """Issuer CS01 small because guarantee protects."""
        cs01 = _gn().cs01_issuer(_disc(), _risky_issuer(), _strong_guarantor())
        assert math.isfinite(cs01)

    def test_cs01_guarantor_negative(self):
        """Guarantor getting riskier → note loses value."""
        cs01 = _gn().cs01_guarantor(_disc(), _risky_issuer(), _strong_guarantor())
        assert cs01 < 0

    def test_cs01_guarantor_larger_than_issuer(self):
        """Guarantee-sensitive: guarantor CS01 should dominate."""
        d, i, g = _disc(), _risky_issuer(), _strong_guarantor()
        cs01_i = abs(_gn().cs01_issuer(d, i, g))
        cs01_g = abs(_gn().cs01_guarantor(d, i, g))
        assert cs01_g > cs01_i


# ── Convenience functions ──

class TestConvenienceFunctions:

    def test_guarantee_value_function(self):
        val = guarantee_value(
            0.05, END, 100.0, _disc(),
            _risky_issuer(), _strong_guarantor(),
        )
        assert val > 0

    def test_guarantee_spread_function(self):
        spread = guarantee_spread(
            0.05, END, 100.0, _disc(),
            _risky_issuer(), _strong_guarantor(),
        )
        assert spread > 0


# ── Edge cases ──

class TestEdgeCases:

    def test_validation_dates(self):
        with pytest.raises(ValueError):
            GuaranteedNote(END, REF, 0.05)

    def test_validation_correlation(self):
        with pytest.raises(ValueError):
            GuaranteedNote(REF, END, 0.05, correlation=1.5)

    def test_perfect_correlation(self):
        """ρ=1 → both default together → guarantee worthless."""
        gn = _gn(correlation=0.99)
        result = gn.price(_disc(), _risky_issuer(), _strong_guarantor())
        # Guarantee still has some value because guarantor is stronger
        assert result.guarantee_value > 0
        # But less than low-correlation case
        low = _gn(correlation=0.10).price(_disc(), _risky_issuer(), _strong_guarantor())
        assert result.guarantee_value < low.guarantee_value
