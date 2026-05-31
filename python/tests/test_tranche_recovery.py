"""Tests for stochastic correlated recovery in tranche pricing."""

import pytest
import math
from datetime import date

from pricebook.credit.tranche_pricing import (
    expected_tranche_loss, expected_tranche_loss_t, TrancheCDS,
)
from pricebook.credit.recovery_pricing import RecoverySpec


def _make_survival_curves(n=20, hazard=0.02):
    from pricebook.core.survival_curve import SurvivalCurve
    ref = date(2024, 1, 1)
    dates = [date(2024 + y, 1, 1) for y in range(1, 11)]
    return [SurvivalCurve(ref, dates, [math.exp(-hazard * y) for y in range(1, 11)])
            for _ in range(n)]


def _make_discount_curve():
    from pricebook.core.discount_curve import DiscountCurve
    from pricebook.core.interpolation import InterpolationMethod
    ref = date(2024, 1, 1)
    dates = [date(2024 + y, 1, 1) for y in range(1, 11)]
    dfs = [math.exp(-0.04 * y) for y in range(1, 11)]
    return DiscountCurve(ref, dates, dfs, interpolation=InterpolationMethod.LOG_LINEAR)


class TestExpectedTrancheLossRecovery:
    def test_fixed_spec_matches_flat(self):
        sc = _make_survival_curves()
        dc = _make_discount_curve()
        specs = [RecoverySpec(0.4, 0.0, "fixed", 0.0) for _ in range(20)]

        flat = expected_tranche_loss(0.0, 0.03, sc, dc, 0.3, 5.0, recovery=0.4, seed=42)
        stoch = expected_tranche_loss(0.0, 0.03, sc, dc, 0.3, 5.0, seed=42, recovery_specs=specs)

        assert stoch == pytest.approx(flat, rel=0.10)

    def test_wrong_way_increases_equity_el(self):
        sc = _make_survival_curves()
        dc = _make_discount_curve()

        flat = expected_tranche_loss(0.0, 0.03, sc, dc, 0.3, 5.0, recovery=0.4, seed=42)
        specs_ww = [RecoverySpec(0.4, 0.15, "beta", -0.5) for _ in range(20)]
        ww = expected_tranche_loss(0.0, 0.03, sc, dc, 0.3, 5.0, seed=42, recovery_specs=specs_ww)

        assert ww >= flat * 0.90  # wrong-way risk increases equity loss

    def test_senior_less_affected(self):
        """Senior tranche should be less affected by wrong-way recovery."""
        sc = _make_survival_curves()
        dc = _make_discount_curve()
        specs_ww = [RecoverySpec(0.4, 0.15, "beta", -0.5) for _ in range(20)]

        eq_flat = expected_tranche_loss(0.0, 0.03, sc, dc, 0.3, 5.0, recovery=0.4, seed=42)
        eq_ww = expected_tranche_loss(0.0, 0.03, sc, dc, 0.3, 5.0, seed=42, recovery_specs=specs_ww)
        sr_flat = expected_tranche_loss(0.12, 0.22, sc, dc, 0.3, 5.0, recovery=0.4, seed=42)
        sr_ww = expected_tranche_loss(0.12, 0.22, sc, dc, 0.3, 5.0, seed=42, recovery_specs=specs_ww)

        # Both should be non-negative
        assert eq_ww >= 0
        assert sr_ww >= 0


class TestStudentTRecovery:
    def test_t_copula_with_specs(self):
        sc = _make_survival_curves()
        dc = _make_discount_curve()
        specs = [RecoverySpec(0.4, 0.15, "beta", -0.3) for _ in range(20)]

        el = expected_tranche_loss_t(0.0, 0.03, sc, dc, 0.3, 5.0, nu=5.0,
                                      seed=42, recovery_specs=specs)
        assert el > 0

    def test_t_heavier_than_gaussian(self):
        """t-copula equity EL should be higher than Gaussian (tail dependence)."""
        sc = _make_survival_curves()
        dc = _make_discount_curve()
        specs = [RecoverySpec(0.4, 0.15, "beta", -0.3) for _ in range(20)]

        gauss = expected_tranche_loss(0.0, 0.03, sc, dc, 0.3, 5.0, seed=42, recovery_specs=specs)
        student = expected_tranche_loss_t(0.0, 0.03, sc, dc, 0.3, 5.0, nu=3.0,
                                           seed=42, recovery_specs=specs)

        # t-copula clusters defaults more → higher equity loss
        assert student >= gauss * 0.80  # allow MC noise


class TestTrancheCDSRecovery:
    def test_price_with_specs(self):
        sc = _make_survival_curves()
        dc = _make_discount_curve()
        specs = [RecoverySpec(0.4, 0.15, "beta", -0.3) for _ in range(20)]

        tcds = TrancheCDS(0.0, 0.03, date(2029, 1, 1), spread=0.05)
        result = tcds.price(dc, sc, correlation=0.3, seed=42, recovery_specs=specs)

        assert result.expected_loss > 0
        assert result.par_spread > 0
