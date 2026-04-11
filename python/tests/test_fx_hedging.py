"""Tests for FX hedging."""

import pytest
from datetime import date

from pricebook.fx_hedging import (
    CrossHedgeResult,
    NDFSettlementResult,
    TriangularArbResult,
    fx_cross_hedge,
    fx_delta_hedge,
    ndf_settlement,
    triangular_arb_monitor,
)


class TestFXDeltaHedge:
    def test_basic(self):
        qty = fx_delta_hedge(10_000_000)
        assert qty == pytest.approx(-10_000_000)

    def test_zero_per_unit(self):
        assert fx_delta_hedge(10_000_000, 0.0) == 0.0


class TestFXCrossHedge:
    def test_perfect_correlation(self):
        result = fx_cross_hedge("NOK/USD", 10_000_000, "SEK/USD",
                                correlation=0.95, proxy_vol=0.10, target_vol=0.12)
        assert result.hedge_ratio == pytest.approx(0.95 * 0.12 / 0.10)
        assert result.proxy_quantity < 0  # sell proxy to hedge long target

    def test_hedged_near_zero(self):
        result = fx_cross_hedge("NOK/USD", 10_000_000, "SEK/USD",
                                correlation=1.0, proxy_vol=0.10, target_vol=0.10)
        # h = 1.0, qty = -10M
        assert result.proxy_quantity == pytest.approx(-10_000_000)

    def test_zero_vol(self):
        result = fx_cross_hedge("A", 10_000_000, "B", 0.9, 0.0, 0.10)
        assert result.hedge_ratio == 0.0


class TestTriangularArbMonitor:
    def test_no_arb(self):
        # EUR/USD=1.085, USD/JPY=148, EUR/JPY=160.58 (= 1.085×148)
        result = triangular_arb_monitor(
            "EUR/USD", 1.085, "USD/JPY", 148.0,
            "EUR/JPY", 1.085 * 148.0,
        )
        assert result.arb_bps == pytest.approx(0.0)
        assert result.is_arb is False

    def test_arb_detected(self):
        result = triangular_arb_monitor(
            "EUR/USD", 1.085, "USD/JPY", 148.0,
            "EUR/JPY", 162.0,  # should be 160.58
            threshold_bps=1.0,
        )
        assert abs(result.arb_bps) > 1.0
        assert result.is_arb is True

    def test_synthetic_rate(self):
        result = triangular_arb_monitor(
            "EUR/USD", 1.085, "USD/JPY", 148.0,
            "EUR/JPY", 160.58,
        )
        assert result.synthetic_rate == pytest.approx(1.085 * 148.0)


class TestNDFSettlement:
    def test_positive_settlement(self):
        """Test: NDF settlement matches manual calculation."""
        result = ndf_settlement(
            "USD/BRL", contracted_rate=5.0, fixing_rate=5.2,
            notional=10_000_000, fixing_date=date(2024, 1, 15),
        )
        # (5.2 − 5.0) × 10M = 2,000,000
        assert result.settlement_amount == pytest.approx(2_000_000)

    def test_negative_settlement(self):
        result = ndf_settlement(
            "USD/BRL", contracted_rate=5.0, fixing_rate=4.8,
            notional=10_000_000, fixing_date=date(2024, 1, 15),
        )
        assert result.settlement_amount == pytest.approx(-2_000_000)

    def test_settlement_date_t_plus_2(self):
        result = ndf_settlement(
            "USD/BRL", 5.0, 5.2, 10_000_000,
            fixing_date=date(2024, 1, 15),  # Monday
        )
        assert result.settlement_date == date(2024, 1, 17)  # Wednesday

    def test_settlement_skips_weekend(self):
        result = ndf_settlement(
            "USD/BRL", 5.0, 5.2, 10_000_000,
            fixing_date=date(2024, 1, 11),  # Thursday → T+2 skips weekend
        )
        assert result.settlement_date == date(2024, 1, 15)  # Monday
