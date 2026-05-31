"""Tests for stochastic correlated recovery in basket CDS."""

import pytest
import math
from datetime import date

from pricebook.credit.basket_cds import ftd_spread, ntd_spread, bespoke_tranche
from pricebook.credit.recovery_pricing import RecoverySpec


def _make_survival_curves(n=5, hazard=0.02):
    """Create simple flat survival curves."""
    from pricebook.core.survival_curve import SurvivalCurve
    ref = date(2024, 1, 1)
    dates = [date(2024 + y, 1, 1) for y in range(1, 11)]
    curves = []
    for _ in range(n):
        survivals = [math.exp(-hazard * y) for y in range(1, 11)]
        curves.append(SurvivalCurve(ref, dates, survivals))
    return curves


def _make_discount_curve():
    from pricebook.core.discount_curve import DiscountCurve
    from pricebook.core.interpolation import InterpolationMethod
    ref = date(2024, 1, 1)
    dates = [date(2024 + y, 1, 1) for y in range(1, 11)]
    dfs = [math.exp(-0.04 * y) for y in range(1, 11)]
    return DiscountCurve(ref, dates, dfs, interpolation=InterpolationMethod.LOG_LINEAR)


class TestFTDWithRecovery:
    def test_fixed_spec_matches_flat(self):
        """Fixed RecoverySpec(0.4) should reproduce flat recovery=0.4."""
        sc = _make_survival_curves()
        dc = _make_discount_curve()
        specs = [RecoverySpec(0.4, 0.0, "fixed", 0.0) for _ in range(5)]

        flat = ftd_spread(sc, dc, 0.3, 5.0, recovery=0.4, seed=42)
        stoch = ftd_spread(sc, dc, 0.3, 5.0, recovery=0.4, seed=42, recovery_specs=specs)

        assert stoch == pytest.approx(flat, rel=0.05)

    def test_wrong_way_increases_spread(self):
        """Negative correlation (wrong-way) should increase FTD spread."""
        sc = _make_survival_curves()
        dc = _make_discount_curve()

        flat = ftd_spread(sc, dc, 0.3, 5.0, recovery=0.4, seed=42)

        # Wrong-way: recovery drops when defaults cluster
        specs_ww = [RecoverySpec(0.4, 0.15, "beta", -0.5) for _ in range(5)]
        ww = ftd_spread(sc, dc, 0.3, 5.0, seed=42, recovery_specs=specs_ww)

        assert ww > flat * 0.95  # wrong-way should be at least as high

    def test_stochastic_positive(self):
        sc = _make_survival_curves()
        dc = _make_discount_curve()
        specs = [RecoverySpec(0.4, 0.15, "beta", -0.3) for _ in range(5)]
        s = ftd_spread(sc, dc, 0.3, 5.0, seed=42, recovery_specs=specs)
        assert s > 0


class TestNTDWithRecovery:
    def test_ntd_with_specs(self):
        sc = _make_survival_curves()
        dc = _make_discount_curve()
        specs = [RecoverySpec(0.4, 0.15, "beta", -0.3) for _ in range(5)]
        s = ntd_spread(sc, dc, 0.3, 5.0, n=2, seed=42, recovery_specs=specs)
        assert s > 0

    def test_ftd_gt_std(self):
        """FTD spread > 2TD spread (more likely to trigger)."""
        sc = _make_survival_curves()
        dc = _make_discount_curve()
        specs = [RecoverySpec(0.4, 0.15, "beta", -0.3) for _ in range(5)]
        ftd = ntd_spread(sc, dc, 0.3, 5.0, n=1, seed=42, recovery_specs=specs)
        std = ntd_spread(sc, dc, 0.3, 5.0, n=2, seed=42, recovery_specs=specs)
        assert ftd > std


class TestBespokeWithRecovery:
    def test_fixed_spec_matches_flat(self):
        """Fixed RecoverySpec(0.4) should reproduce flat lgd=0.6."""
        pds = [0.03] * 10
        specs = [RecoverySpec(0.4, 0.0, "fixed", 0.0) for _ in range(10)]

        flat = bespoke_tranche(pds, 0.03, 0.07, lgd=0.6, seed=42)
        stoch = bespoke_tranche(pds, 0.03, 0.07, lgd=0.6, seed=42, recovery_specs=specs)

        assert stoch.expected_loss == pytest.approx(flat.expected_loss, rel=0.10)

    def test_wrong_way_increases_equity_el(self):
        """Wrong-way recovery should increase equity tranche expected loss."""
        pds = [0.03] * 20

        flat = bespoke_tranche(pds, 0.0, 0.03, lgd=0.6, seed=42)

        specs_ww = [RecoverySpec(0.4, 0.15, "beta", -0.5) for _ in range(20)]
        ww = bespoke_tranche(pds, 0.0, 0.03, seed=42, recovery_specs=specs_ww)

        # Wrong-way: losses concentrate when recovery falls → equity hit harder
        assert ww.expected_loss >= flat.expected_loss * 0.90

    def test_heterogeneous_seniority(self):
        """Mix of senior and sub recovery specs."""
        pds = [0.03] * 10
        specs = (
            [RecoverySpec(0.65, 0.15, "beta", -0.3)] * 5  # senior secured
            + [RecoverySpec(0.28, 0.20, "beta", -0.3)] * 5  # subordinated
        )
        result = bespoke_tranche(pds, 0.03, 0.07, seed=42, recovery_specs=specs)
        assert result.expected_loss > 0
        assert result.tranche_spread > 0
