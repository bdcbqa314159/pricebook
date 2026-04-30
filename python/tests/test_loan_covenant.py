"""Tests for loan covenant analytics."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.loan_covenant import (
    Covenant, CovenantSchedule,
    covenant_cushion, cushion_trajectory, periods_to_breach,
    breach_probability, breach_probability_mc,
    waiver_cost, amendment_cost, equity_cure_cost,
    covenant_adjusted_pv, CovenantAdjustedResult,
)

REF = date(2026, 4, 28)


# ---- Covenant spec ----

class TestCovenant:

    def test_leverage_breach(self):
        cov = Covenant(metric="leverage", threshold=5.0, direction="max")
        assert not cov.is_breached(4.5)
        assert cov.is_breached(5.5)

    def test_coverage_breach(self):
        cov = Covenant(metric="coverage", threshold=2.0, direction="min")
        assert not cov.is_breached(2.5)
        assert cov.is_breached(1.5)

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="type"):
            Covenant(type="invalid")

    def test_round_trip(self):
        cov = Covenant(metric="leverage", threshold=4.5, direction="max")
        d = cov.to_dict()
        cov2 = Covenant.from_dict(d)
        assert cov2.threshold == 4.5


class TestCovenantSchedule:

    def test_cov_lite(self):
        schedule = CovenantSchedule(
            covenants=[Covenant(type="incurrence", metric="leverage", threshold=6.0)],
            test_dates=[REF],
        )
        assert schedule.is_cov_lite

    def test_not_cov_lite(self):
        schedule = CovenantSchedule(
            covenants=[Covenant(type="maintenance", metric="leverage", threshold=5.0)],
        )
        assert not schedule.is_cov_lite

    def test_round_trip(self):
        schedule = CovenantSchedule(
            covenants=[Covenant(metric="leverage", threshold=5.0)],
            test_dates=[REF, REF + timedelta(days=91)],
        )
        d = schedule.to_dict()
        s2 = CovenantSchedule.from_dict(d)
        assert len(s2.covenants) == 1
        assert len(s2.test_dates) == 2


# ---- Cushion ----

class TestCushion:

    def test_positive_cushion(self):
        c = covenant_cushion(actual=4.2, threshold=5.0, direction="max")
        assert c == pytest.approx(0.16)

    def test_zero_cushion(self):
        c = covenant_cushion(actual=5.0, threshold=5.0, direction="max")
        assert c == pytest.approx(0.0)

    def test_negative_cushion_breached(self):
        c = covenant_cushion(actual=5.5, threshold=5.0, direction="max")
        assert c < 0

    def test_coverage_cushion(self):
        c = covenant_cushion(actual=2.5, threshold=2.0, direction="min")
        assert c == pytest.approx(0.25)

    def test_trajectory_improving(self):
        t = cushion_trajectory([4.5, 4.3, 4.0, 3.8], threshold=5.0)
        assert t == "improving"

    def test_trajectory_deteriorating(self):
        t = cushion_trajectory([3.5, 4.0, 4.5, 4.8], threshold=5.0)
        assert t == "deteriorating"

    def test_periods_to_breach(self):
        p = periods_to_breach(current_ratio=4.0, threshold=5.0,
                               trend_per_period=0.25, direction="max")
        assert p == 4  # (5.0 - 4.0) / 0.25

    def test_periods_improving(self):
        p = periods_to_breach(current_ratio=4.0, threshold=5.0,
                               trend_per_period=-0.1, direction="max")
        assert p is None  # moving away


# ---- Breach probability ----

class TestBreachProbability:

    def test_high_cushion_low_prob(self):
        p = breach_probability(cushion=0.30, ebitda_vol=0.15, horizon=1.0)
        assert p < 0.10

    def test_low_cushion_high_prob(self):
        p = breach_probability(cushion=0.05, ebitda_vol=0.25, horizon=1.0)
        assert p > 0.30

    def test_zero_cushion(self):
        p = breach_probability(cushion=0.0, ebitda_vol=0.20)
        assert p == 1.0

    def test_zero_vol(self):
        p = breach_probability(cushion=0.10, ebitda_vol=0.0)
        assert p == 0.0

    def test_longer_horizon_higher_prob(self):
        p1 = breach_probability(cushion=0.15, ebitda_vol=0.20, horizon=1.0)
        p3 = breach_probability(cushion=0.15, ebitda_vol=0.20, horizon=3.0)
        assert p3 > p1

    def test_mc_consistent(self):
        """MC breach prob should be close to closed-form."""
        cushion = 0.15
        vol = 0.20
        p_cf = breach_probability(cushion, vol, horizon=1.0)
        # MC
        current_ebitda = 100
        debt = current_ebitda / (1 / (1 + cushion))  # leverage = 1/(1+cushion) of threshold
        threshold = debt / current_ebitda * (1 + cushion)
        p_mc = breach_probability_mc(current_ebitda, debt, threshold,
                                      vol, horizon=1.0, n_paths=100_000)
        # Should be in same ballpark (not exact due to model differences)
        assert abs(p_cf - p_mc) < 0.15


# ---- Costs ----

class TestCosts:

    def test_waiver_cost(self):
        wc = waiver_cost(notional=10_000_000, waiver_fee_bps=25)
        assert wc == pytest.approx(25_000)

    def test_amendment_cost(self):
        ac = amendment_cost(notional=10_000_000, amendment_fee_bps=50,
                            spread_change_bps=25, remaining_life=3.0)
        assert ac > waiver_cost(10_000_000)  # amendment more expensive

    def test_equity_cure(self):
        ec = equity_cure_cost(cure_amount=5_000_000, sponsor_required_return=0.15)
        assert ec > 0


# ---- Adjusted PV ----

class TestCovenantAdjustedPV:

    def test_adjusted_less_than_base(self):
        r = covenant_adjusted_pv(
            base_pv=10_000_000, notional=10_000_000,
            cushion=0.15, ebitda_vol=0.20)
        assert r.adjusted_pv < r.base_pv

    def test_high_cushion_small_adjustment(self):
        r = covenant_adjusted_pv(
            base_pv=10_000_000, notional=10_000_000,
            cushion=0.40, ebitda_vol=0.10)
        adjustment = r.base_pv - r.adjusted_pv
        assert adjustment < 20_000  # small for strong cushion

    def test_low_cushion_large_adjustment(self):
        r = covenant_adjusted_pv(
            base_pv=10_000_000, notional=10_000_000,
            cushion=0.05, ebitda_vol=0.30)
        adjustment = r.base_pv - r.adjusted_pv
        assert adjustment > 100_000

    def test_result_dict(self):
        r = covenant_adjusted_pv(
            base_pv=10_000_000, notional=10_000_000,
            cushion=0.15, ebitda_vol=0.20)
        d = r.to_dict()
        assert "breach_prob" in d
        assert "adjusted_pv" in d
