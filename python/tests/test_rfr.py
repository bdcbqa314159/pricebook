"""Tests for RFR compounding, IBOR projection, spread curves, and fallback."""

import pytest
import math
from datetime import date

from pricebook.rfr import (
    compound_rfr,
    compound_rfr_from_curve,
    SpreadCurve,
    IBORProjection,
    bootstrap_spread_curve,
    StochasticBasis,
    FallbackConfig,
    ibor_fallback_rate,
)
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)


class TestCompoundRFR:
    def test_single_day(self):
        rate = compound_rfr([0.05], [1 / 360])
        # (1 + 0.05/360) - 1, annualised by /( 1/360) = 0.05
        assert rate == pytest.approx(0.05, rel=1e-6)

    def test_flat_rate(self):
        """Flat daily rate → compounded ≈ simple."""
        n = 90
        r = 0.05
        day_fracs = [1 / 360] * n
        rates = [r] * n
        result = compound_rfr(rates, day_fracs)
        # Compounded is slightly above simple due to daily compounding
        assert result == pytest.approx(r, rel=0.01)

    def test_compounding_effect(self):
        """Higher rates show compounding: compounded > simple."""
        n = 360
        r = 0.10
        day_fracs = [1 / 360] * n
        rates = [r] * n
        result = compound_rfr(rates, day_fracs)
        assert result > r  # compounding adds

    def test_empty(self):
        assert compound_rfr([], []) == 0.0

    def test_lockout(self):
        """Lockout: last N days use rate from day N-lockout."""
        rates = [0.05, 0.05, 0.05, 0.10, 0.10]  # spike at end
        day_fracs = [1/360] * 5
        # Without lockout: uses the spike
        no_lock = compound_rfr(rates, day_fracs, lockout_days=0)
        # With 2-day lockout: last 2 days use rate[2]=0.05 instead of 0.10
        locked = compound_rfr(rates, day_fracs, lockout_days=2)
        assert locked < no_lock

    def test_matches_curve_forward(self):
        """Compounded RFR from curve matches forward rate."""
        curve = make_flat_curve(REF, 0.05)
        start = date(2024, 4, 15)
        end = date(2024, 7, 15)
        rate = compound_rfr_from_curve(curve, start, end)
        assert rate == pytest.approx(0.05, rel=0.01)


class TestSpreadCurve:
    def test_flat_spread(self):
        sc = SpreadCurve(REF, [date(2025, 1, 15)], [0.002])
        assert sc.spread(date(2025, 1, 15)) == pytest.approx(0.002)
        assert sc.spread(date(2026, 1, 15)) == pytest.approx(0.002)  # extrap

    def test_term_structure(self):
        sc = SpreadCurve(
            REF,
            [date(2025, 1, 15), date(2029, 1, 15)],
            [0.001, 0.003],
        )
        # Short end: lower spread
        s_short = sc.spread(date(2025, 1, 15))
        s_long = sc.spread(date(2029, 1, 15))
        assert s_short < s_long

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            SpreadCurve(REF, [date(2025, 1, 15)], [0.001, 0.002])


class TestIBORProjection:
    def test_ibor_equals_rfr_plus_spread(self):
        rfr = make_flat_curve(REF, 0.04)
        sc = SpreadCurve(REF, [date(2025, 1, 15)], [0.002])
        proj = IBORProjection(rfr, sc)
        start = date(2024, 7, 15)
        end = date(2024, 10, 15)
        ibor_fwd = proj.forward_rate(start, end)
        rfr_fwd = rfr.forward_rate(start, end)
        assert ibor_fwd == pytest.approx(rfr_fwd + 0.002, rel=0.01)

    def test_zero_spread_equals_rfr(self):
        rfr = make_flat_curve(REF, 0.04)
        sc = SpreadCurve(REF, [date(2025, 1, 15)], [0.0])
        proj = IBORProjection(rfr, sc)
        start = date(2024, 7, 15)
        end = date(2024, 10, 15)
        assert proj.forward_rate(start, end) == pytest.approx(
            rfr.forward_rate(start, end), rel=1e-6)


class TestBootstrapSpread:
    def test_positive_spread(self):
        """IBOR rates > OIS rates → positive spread."""
        ois = make_flat_curve(REF, 0.04)
        ibor_quotes = [
            (date(2025, 1, 15), 0.042),
            (date(2027, 1, 15), 0.043),
            (date(2029, 1, 15), 0.044),
        ]
        sc = bootstrap_spread_curve(REF, ibor_quotes, ois)
        for d in [date(2025, 1, 15), date(2027, 1, 15), date(2029, 1, 15)]:
            assert sc.spread(d) > 0

    def test_zero_spread_when_equal(self):
        """IBOR = OIS → spread ≈ 0."""
        ois = make_flat_curve(REF, 0.04)
        # OIS par rate ≈ 0.04 for flat curve
        ibor_quotes = [(date(2029, 1, 15), 0.04)]
        sc = bootstrap_spread_curve(REF, ibor_quotes, ois)
        assert sc.spread(date(2029, 1, 15)) == pytest.approx(0.0, abs=0.002)


class TestStochasticBasis:
    def test_mean_reversion(self):
        sb = StochasticBasis(mean_spread=0.002, mean_reversion=1.0, vol=0.001, seed=42)
        paths = sb.simulate(s0=0.005, T=10.0, n_steps=100, n_paths=50_000)
        assert paths[:, -1].mean() == pytest.approx(0.002, rel=0.1)

    def test_stationary(self):
        sb = StochasticBasis(mean_spread=0.002, mean_reversion=2.0, vol=0.001)
        assert sb.stationary_mean() == 0.002
        assert sb.stationary_std() > 0

    def test_zero_vol_deterministic(self):
        sb = StochasticBasis(mean_spread=0.002, mean_reversion=1.0, vol=1e-10, seed=42)
        paths = sb.simulate(s0=0.002, T=5.0, n_steps=50, n_paths=100)
        # With near-zero vol, all paths should be ≈ mean_spread
        assert paths[:, -1].std() < 1e-8


class TestFallback:
    def test_post_cessation(self):
        config = FallbackConfig(spread_adjustment=0.0026161, cessation_date=date(2023, 6, 30))
        rfr_rate = 0.045
        fallback = ibor_fallback_rate(rfr_rate, config, fixing_date=date(2024, 1, 15))
        assert fallback == pytest.approx(rfr_rate + 0.0026161)

    def test_spread_adjustment_adds(self):
        config = FallbackConfig(spread_adjustment=0.005, cessation_date=date(2023, 1, 1))
        assert ibor_fallback_rate(0.03, config, date(2024, 1, 1)) == pytest.approx(0.035)

    def test_zero_spread(self):
        config = FallbackConfig(spread_adjustment=0.0, cessation_date=date(2023, 1, 1))
        assert ibor_fallback_rate(0.04, config, date(2024, 1, 1)) == pytest.approx(0.04)
