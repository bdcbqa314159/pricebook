"""Tests for inflation desk: book, breakeven, carry, RV, regulatory."""

import pytest
from datetime import date

from pricebook.inflation_book import (
    InflationBook, InflationLimits, InflationPosition,
)
from pricebook.inflation_trading import (
    BreakevenTrade, CrossMarketInflationRV, InflationCapitalReport,
    InflationCarry, InflationRiskDecomposition, SeasonalCarrySignal,
    breakeven_basis_monitor, breakeven_term_structure,
    build_breakeven_trade, cross_market_inflation_rv,
    inflation_carry, inflation_frtb_capital,
    inflation_risk_decomposition, seasonal_carry,
)
from pricebook.swaption import Swaption
from pricebook.trade import Trade


def _trade(direction=1, trade_id="t"):
    instr = Swaption(date(2025, 1, 15), date(2030, 1, 15), strike=0.05, notional=1_000_000)
    return Trade(instr, direction=direction, trade_id=trade_id)


# ---- Slice 166: inflation book ----

class TestInflationBook:
    def test_empty(self):
        book = InflationBook("test")
        assert len(book) == 0

    def test_add_and_aggregate(self):
        book = InflationBook("test")
        book.add(_trade(), issuer="TIPS", product_type="linker",
                 notional=10_000_000, ie01=50.0, real_dv01=40.0)
        positions = book.positions_by_issuer()
        assert len(positions) == 1
        assert positions[0].net_notional == pytest.approx(10_000_000)
        assert positions[0].net_ie01 == pytest.approx(500.0)

    def test_ie01_aggregation(self):
        book = InflationBook("test")
        book.add(_trade(direction=1), issuer="TIPS", notional=10_000_000, ie01=50.0)
        book.add(_trade(direction=-1), issuer="TIPS", notional=5_000_000, ie01=50.0)
        assert book.net_ie01() == pytest.approx(250.0)

    def test_limits_ie01_breach(self):
        limits = InflationLimits(max_ie01=200.0)
        book = InflationBook("test", limits=limits)
        book.add(_trade(), issuer="TIPS", notional=10_000_000, ie01=50.0)
        breaches = book.check_limits()
        assert any(b.limit_type == "ie01" for b in breaches)

    def test_no_breaches(self):
        limits = InflationLimits(max_ie01=1000.0)
        book = InflationBook("test", limits=limits)
        book.add(_trade(), issuer="TIPS", notional=10_000_000, ie01=50.0)
        assert book.check_limits() == []


# ---- Slice 167: breakeven ----

class TestBreakeven:
    def test_term_structure(self):
        ts = breakeven_term_structure([
            ("5Y", 0.04, 0.015), ("10Y", 0.042, 0.018), ("30Y", 0.045, 0.02),
        ])
        assert len(ts) == 3
        assert ts[0].breakeven == pytest.approx(0.025)
        assert ts[1].breakeven == pytest.approx(0.024)

    def test_basis_monitor(self):
        sig = breakeven_basis_monitor("10Y", 0.024, 0.023,
                                      history_bps=[8, 10, 12, 9, 11] * 4)
        assert sig.basis_bps == pytest.approx(10.0)

    def test_dv01_neutral_trade(self):
        trade = build_breakeven_trade("10Y", 10_000_000, 80.0, 85.0)
        assert trade.is_dv01_neutral
        assert trade.net_rate_dv01 == pytest.approx(0.0, abs=1.0)


# ---- Slice 168: carry ----

class TestInflationCarry:
    def test_positive_carry(self):
        c = inflation_carry(10_000_000, real_yield=0.015,
                            breakeven_rolldown_bps=5.0, financing_rate=0.04,
                            linker_price=98.0, days=1)
        assert c.real_yield_carry > 0
        assert c.net_carry == pytest.approx(
            c.real_yield_carry + c.breakeven_rolldown - c.financing_cost
        )

    def test_seasonal_strong(self):
        factors = {1: 0.004, 7: -0.001, 12: 0.005}
        sig = seasonal_carry(12, factors, threshold=0.003)
        assert sig.signal == "strong"

    def test_seasonal_weak(self):
        factors = {1: 0.001, 7: -0.001}
        sig = seasonal_carry(7, factors, threshold=0.003)
        assert sig.signal == "weak"


# ---- Slice 169: RV + risk ----

class TestInflationRV:
    def test_cross_market(self):
        rv = cross_market_inflation_rv("US", 0.025, "UK", 0.035,
                                       history_bps=[-80, -90, -100, -85, -95] * 4)
        assert rv.spread_bps == pytest.approx(-100.0)

    def test_risk_decomposition(self):
        """IE01 + real DV01 = nominal DV01 (approximately)."""
        decomp = inflation_risk_decomposition(ie01=500, real_dv01=400, nominal_dv01=900)
        assert decomp.residual == pytest.approx(0.0)
        assert decomp.ie01 + decomp.real_dv01 == pytest.approx(decomp.nominal_dv01)

    def test_risk_with_residual(self):
        decomp = inflation_risk_decomposition(ie01=500, real_dv01=400, nominal_dv01=950)
        assert decomp.residual == pytest.approx(50.0)


# ---- Slice 170: regulatory ----

class TestInflationRegulatory:
    def test_capital_matches_manual(self):
        """Capital matches manual: 1M × 1.6% = 16K."""
        positions = [{"bucket": "USD", "sensitivity": 1_000_000}]
        report = inflation_frtb_capital(positions)
        assert report.inflation_capital == pytest.approx(16_000)

    def test_multiple_currencies(self):
        positions = [
            {"bucket": "USD", "sensitivity": 1_000_000},
            {"bucket": "EUR", "sensitivity": 500_000},
        ]
        report = inflation_frtb_capital(positions)
        assert report.inflation_capital > 0
        assert report.n_positions == 2

    def test_total_rwa(self):
        positions = [{"bucket": "USD", "sensitivity": 1_000_000}]
        report = inflation_frtb_capital(positions)
        assert report.total_rwa == pytest.approx(report.total_capital * 12.5)
