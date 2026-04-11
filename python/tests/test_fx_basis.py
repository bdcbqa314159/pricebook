"""Tests for FX basis strategies."""

import pytest

from pricebook.fx_basis import (
    BasisCurveTrade,
    BasisRVEntry,
    BasisSignal,
    basis_carry,
    basis_monitor,
    basis_term_structure,
    cross_market_basis_rv,
)


class TestBasisMonitor:
    def test_wide_signal(self):
        history = [-20, -18, -22, -19, -21] * 4
        sig = basis_monitor("EUR/USD", "5Y", -40.0, history, threshold=2.0)
        assert sig.signal == "tight"  # more negative = tighter

    def test_fair(self):
        history = [-20, -18, -22, -19, -21] * 4
        sig = basis_monitor("EUR/USD", "5Y", -20.0, history)
        assert sig.signal == "fair"

    def test_no_history(self):
        sig = basis_monitor("EUR/USD", "5Y", -20.0, [])
        assert sig.z_score is None


class TestBasisTermStructure:
    def test_build(self):
        ts = basis_term_structure("EUR/USD", [
            ("1Y", -10.0), ("3Y", -15.0), ("5Y", -20.0), ("10Y", -25.0),
        ])
        assert len(ts) == 4
        assert ts[0].tenor == "1Y"
        assert ts[0].basis_bps == -10.0


class TestCrossMarketBasisRV:
    def test_ranking(self):
        pairs = [
            ("EUR/USD", -20.0, [-15, -18, -16, -14, -17] * 4),
            ("USD/JPY", -40.0, [-15, -18, -16, -14, -17] * 4),
            ("GBP/USD", -10.0, [-15, -18, -16, -14, -17] * 4),
        ]
        ranked = cross_market_basis_rv(pairs)
        assert len(ranked) == 3
        assert ranked[0].rank == 1

    def test_empty(self):
        assert cross_market_basis_rv([]) == []


class TestBasisCarry:
    def test_positive_basis(self):
        carry = basis_carry(10_000_000, -20.0, 90)
        # 10M × (-20/10000) × 90/365
        assert carry == pytest.approx(10_000_000 * (-0.002) * 90 / 365)

    def test_zero_days(self):
        assert basis_carry(10_000_000, -20.0, 0) == pytest.approx(0.0)


class TestBasisCurveTrade:
    def test_curve_spread(self):
        trade = BasisCurveTrade("EUR/USD", "1Y", "5Y", -10.0, -25.0, 10_000_000)
        assert trade.curve_spread_bps == pytest.approx(-15.0)

    def test_pv_change_steepener_profits(self):
        trade = BasisCurveTrade("EUR/USD", "1Y", "5Y", -10.0, -25.0,
                                10_000_000, direction=1)
        # Curve steepens: long−short widens from -15 to -20
        pv = trade.pv_change(-8.0, -28.0)
        # new_spread = -28 − (-8) = -20, old = -15, diff = -5
        # pv = 1 × 10M × (-5/10000) = -5000
        assert pv == pytest.approx(-5_000)

    def test_flattener_opposite(self):
        steep = BasisCurveTrade("EUR/USD", "1Y", "5Y", -10.0, -25.0,
                                10_000_000, direction=1)
        flat = BasisCurveTrade("EUR/USD", "1Y", "5Y", -10.0, -25.0,
                               10_000_000, direction=-1)
        assert steep.pv_change(-8.0, -28.0) == pytest.approx(
            -flat.pv_change(-8.0, -28.0)
        )
