"""Tests for regulatory capital: SA-CCR and FRTB."""

import math
import pytest
from datetime import date

from pricebook.regulatory import (
    SaCcrTrade, sa_ccr_addon, sa_ccr_single,
    frtb_delta_charge, frtb_delta_charge_all_scenarios,
    IR_SUPERVISORY_FACTOR, FRTB_IR_RISK_WEIGHTS, FRTB_TENOR_LABELS,
)


# ---- SA-CCR ----

class TestSaCcrSingle:
    def test_hand_calculation(self):
        """Single swap: addon = SF * |delta * notional * MF|."""
        notional = 10_000_000
        maturity = 5.0
        direction = 1  # receiver
        addon = sa_ccr_single(notional, maturity, direction)
        # MF = sqrt(min(5, 1)) = 1.0
        # adjusted = 1 * 10M * 1.0 = 10M
        # addon = 0.005 * 10M = 50_000
        assert addon == pytest.approx(50_000)

    def test_short_maturity(self):
        """Short maturity → smaller maturity factor."""
        addon_long = sa_ccr_single(10_000_000, 5.0, 1)
        addon_short = sa_ccr_single(10_000_000, 0.25, 1)
        assert addon_short < addon_long

    def test_scales_with_notional(self):
        a1 = sa_ccr_single(10_000_000, 5.0, 1)
        a2 = sa_ccr_single(20_000_000, 5.0, 1)
        assert a2 == pytest.approx(2 * a1)

    def test_direction_doesnt_affect_single(self):
        """For a single trade, direction only affects sign, not addon."""
        a_recv = sa_ccr_single(10_000_000, 5.0, 1)
        a_pay = sa_ccr_single(10_000_000, 5.0, -1)
        assert a_recv == pytest.approx(a_pay)


class TestSaCcrNetting:
    def test_opposite_directions_net(self):
        """Offsetting trades should have smaller addon than single."""
        single = sa_ccr_addon([SaCcrTrade(10_000_000, 5.0, 1)])
        netted = sa_ccr_addon([
            SaCcrTrade(10_000_000, 5.0, 1),
            SaCcrTrade(10_000_000, 5.0, -1),
        ])
        assert netted.addon < single.addon

    def test_same_direction_additive(self):
        """Same-direction trades in same bucket should compound."""
        single = sa_ccr_addon([SaCcrTrade(10_000_000, 5.0, 1)])
        doubled = sa_ccr_addon([
            SaCcrTrade(10_000_000, 5.0, 1),
            SaCcrTrade(10_000_000, 5.0, 1),
        ])
        assert doubled.addon == pytest.approx(2 * single.addon)

    def test_cross_bucket_correlation(self):
        """Trades in different buckets partially offset."""
        same = sa_ccr_addon([
            SaCcrTrade(10_000_000, 3.0, 1),
            SaCcrTrade(10_000_000, 3.0, -1),
        ])
        cross = sa_ccr_addon([
            SaCcrTrade(10_000_000, 0.5, 1),  # bucket 0
            SaCcrTrade(10_000_000, 3.0, -1),  # bucket 1
        ])
        # Cross-bucket netting is less effective
        assert cross.addon > same.addon

    def test_trade_details(self):
        result = sa_ccr_addon([SaCcrTrade(10_000_000, 5.0, 1)])
        assert len(result.trade_level) == 1
        assert result.trade_level[0]["bucket"] == "1Y-5Y"
        assert result.trade_level[0]["notional"] == 10_000_000

    def test_perfect_offset_zero_addon(self):
        """Perfectly offsetting trades → zero addon."""
        result = sa_ccr_addon([
            SaCcrTrade(10_000_000, 5.0, 1),
            SaCcrTrade(10_000_000, 5.0, -1),
        ])
        assert result.addon == pytest.approx(0.0)


# ---- FRTB Delta ----

class TestFrtbDelta:
    def test_single_tenor(self):
        """Single sensitivity → charge = |s * RW|."""
        s = {"5Y": 100_000}  # 100k DV01 at 5Y
        result = frtb_delta_charge(s)
        # RW for 5Y = 0.011
        expected = abs(100_000 * 0.011)
        assert result.risk_charge == pytest.approx(expected)

    def test_scales_with_position(self):
        s1 = {"10Y": 50_000}
        s2 = {"10Y": 100_000}
        r1 = frtb_delta_charge(s1)
        r2 = frtb_delta_charge(s2)
        assert r2.risk_charge == pytest.approx(2 * r1.risk_charge)

    def test_diversification_benefit(self):
        """Spread position has lower charge than outright."""
        outright = frtb_delta_charge({"10Y": 100_000})
        spread = frtb_delta_charge({"5Y": 100_000, "10Y": -100_000})
        assert spread.risk_charge < outright.risk_charge

    def test_zero_sensitivities(self):
        result = frtb_delta_charge({})
        assert result.risk_charge == pytest.approx(0.0)

    def test_all_scenarios(self):
        s = {"5Y": 100_000, "10Y": -50_000}
        results = frtb_delta_charge_all_scenarios(s)
        assert "low" in results
        assert "medium" in results
        assert "high" in results
        # All should be positive
        for r in results.values():
            assert r.risk_charge >= 0

    def test_low_correlation_higher_for_spread(self):
        """Low correlation → less netting → higher charge for spread."""
        s = {"5Y": 100_000, "10Y": -100_000}
        low = frtb_delta_charge(s, "low")
        high = frtb_delta_charge(s, "high")
        # Low correlation means less netting between 5Y and 10Y
        assert low.risk_charge > high.risk_charge

    def test_weighted_sensitivities(self):
        s = {"3M": 200_000, "10Y": 100_000}
        result = frtb_delta_charge(s)
        # 3M is index 0, RW = 0.017
        # 10Y is index 6, RW = 0.011
        assert result.weighted_sensitivities[0] == pytest.approx(200_000 * 0.017)
        assert result.weighted_sensitivities[6] == pytest.approx(100_000 * 0.011)

    def test_scenario_label(self):
        result = frtb_delta_charge({"5Y": 100_000}, "high")
        assert result.scenario == "high"
