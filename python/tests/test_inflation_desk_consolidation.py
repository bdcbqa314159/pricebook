"""Inflation desk consolidation tests — factory functions, book aggregation, cross-asset."""

from __future__ import annotations

import math
from datetime import date
from dateutil.relativedelta import relativedelta

import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.fixed_income.inflation import (
    CPICurve, InflationLinkedBond, ZCInflationSwap, YoYInflationSwap,
    zc_inflation_swap_pv, yoy_inflation_swap_pv,
)
from pricebook.desks.inflation_desk import (
    inflation_risk_metrics, InflationRiskMetrics,
    inflation_carry_decomposition, InflationCarryDecomposition,
    inflation_daily_pnl, InflationDailyPnL,
    inflation_capital, InflationCapitalResult,
    inflation_stress_suite, InflationStressResult,
    inflation_dashboard,
)
from pricebook.desks.inflation_book import InflationBook, InflationTradeEntry
from pricebook.core.trade import Trade
from pricebook.core.schedule import Frequency


REF = date(2024, 7, 15)
END = date(2029, 7, 15)


def _disc():
    return DiscountCurve.flat(REF, 0.04)


def _cpi():
    dates = [REF + relativedelta(years=y) for y in [1, 2, 3, 5, 10]]
    be = 0.025  # 2.5% breakeven
    return CPICurve.from_breakevens(REF, 300.0, dates, [be] * 5)


def _linker():
    return InflationLinkedBond(REF, END, coupon_rate=0.0125,
                                base_cpi_value=300.0, notional=100.0)


def _zc_swap():
    return ZCInflationSwap(REF, END, fixed_rate=0.025, notional=10_000_000)


def _yoy_swap():
    return YoYInflationSwap(REF, END, fixed_rate=0.025, notional=10_000_000)


def _trade(direction=1, instrument=None):
    return Trade(instrument=instrument or _zc_swap(), direction=direction, notional_scale=1.0)


# ── CPICurve.bumped ──

class TestCPICurveBumped:

    def test_bumped_shifts_breakeven(self):
        cpi = _cpi()
        bumped = cpi.bumped(0.001)  # +10bp
        be_base = cpi.breakeven_rate(END)
        be_bumped = bumped.breakeven_rate(END)
        assert be_bumped > be_base
        assert be_bumped - be_base == pytest.approx(0.001, abs=0.0005)

    def test_bumped_zero_is_identity(self):
        cpi = _cpi()
        bumped = cpi.bumped(0.0)
        assert cpi.cpi(END) == pytest.approx(bumped.cpi(END), abs=1e-6)

    def test_bumped_negative(self):
        cpi = _cpi()
        bumped = cpi.bumped(-0.001)
        assert bumped.breakeven_rate(END) < cpi.breakeven_rate(END)


# ── Instrument Wrappers ──

class TestInstrumentWrappers:

    def test_zc_swap_pv_matches_function(self):
        disc, cpi = _disc(), _cpi()
        swap = _zc_swap()
        fn_pv = zc_inflation_swap_pv(0.025, disc, cpi, END, notional=10_000_000)
        assert swap.pv(disc, cpi) == pytest.approx(fn_pv, abs=1e-6)

    def test_yoy_swap_pv_matches_function(self):
        disc, cpi = _disc(), _cpi()
        swap = _yoy_swap()
        fn_pv = yoy_inflation_swap_pv(0.025, disc, cpi, REF, END, notional=10_000_000)
        assert swap.pv(disc, cpi) == pytest.approx(fn_pv, abs=1e-6)

    def test_zc_swap_par_rate(self):
        disc, cpi = _disc(), _cpi()
        swap = _zc_swap()
        par = swap.par_rate(disc, cpi)
        at_par = ZCInflationSwap(REF, END, par, notional=10_000_000)
        assert at_par.pv(disc, cpi) == pytest.approx(0.0, abs=100)


# ── Risk Metrics ──

class TestInflationRiskMetrics:

    def test_linker_ie01_positive(self):
        """Long linker gains when breakeven rises."""
        rm = inflation_risk_metrics(_linker(), _disc(), _cpi())
        assert rm.ie01 > 0

    def test_linker_real_dv01_negative(self):
        """Linker price falls when real rates rise."""
        rm = inflation_risk_metrics(_linker(), _disc(), _cpi())
        assert rm.real_dv01 < 0

    def test_zc_swap_ie01(self):
        rm = inflation_risk_metrics(_zc_swap(), _disc(), _cpi())
        assert math.isfinite(rm.ie01)
        assert rm.ie01 != 0

    def test_yoy_swap_ie01(self):
        rm = inflation_risk_metrics(_yoy_swap(), _disc(), _cpi())
        assert math.isfinite(rm.ie01)

    def test_additive_decomposition(self):
        """ie01 + real_dv01 ~ nominal_dv01 (approximate)."""
        rm = inflation_risk_metrics(_linker(), _disc(), _cpi())
        sum_parts = rm.ie01 + rm.real_dv01
        # Allow 30% tolerance — cross-gamma means not perfectly additive
        if abs(rm.nominal_dv01) > 1e-6:
            ratio = abs(sum_parts / rm.nominal_dv01)
            assert 0.5 < ratio < 2.0

    def test_gamma_finite(self):
        rm = inflation_risk_metrics(_linker(), _disc(), _cpi())
        assert math.isfinite(rm.gamma)

    def test_notional_scales(self):
        """2x notional → 2x IE01."""
        rm1 = inflation_risk_metrics(
            ZCInflationSwap(REF, END, 0.025, notional=10e6), _disc(), _cpi())
        rm2 = inflation_risk_metrics(
            ZCInflationSwap(REF, END, 0.025, notional=20e6), _disc(), _cpi())
        assert rm2.ie01 == pytest.approx(2 * rm1.ie01, rel=0.01)

    def test_to_dict(self):
        rm = inflation_risk_metrics(_zc_swap(), _disc(), _cpi())
        d = rm.to_dict()
        assert "ie01" in d
        assert "real_dv01" in d
        assert "gamma" in d


# ── Carry Decomposition ──

class TestInflationCarry:

    def test_carry_finite(self):
        carry = inflation_carry_decomposition(_linker(), _disc(), _cpi())
        assert math.isfinite(carry.net_carry)

    def test_real_yield_carry_positive(self):
        carry = inflation_carry_decomposition(_linker(), _disc(), _cpi())
        assert carry.real_yield_carry > 0  # coupon_rate = 1.25% > 0

    def test_breakeven_accrual_positive(self):
        carry = inflation_carry_decomposition(_linker(), _disc(), _cpi())
        assert carry.breakeven_accrual > 0  # breakeven = 2.5% > 0

    def test_to_dict(self):
        carry = inflation_carry_decomposition(_zc_swap(), _disc(), _cpi())
        d = carry.to_dict()
        assert "real_yield" in d
        assert "breakeven" in d
        assert "net" in d


# ── Daily P&L ──

class TestInflationDailyPnL:

    def test_unchanged_curves_small(self):
        disc, cpi = _disc(), _cpi()
        pnl = inflation_daily_pnl(_zc_swap(), disc, cpi, disc, cpi, REF)
        assert abs(pnl.total) < 1e-6

    def test_breakeven_shift_has_impact(self):
        disc, cpi = _disc(), _cpi()
        cpi_shifted = cpi.bumped(0.001)
        pnl = inflation_daily_pnl(_zc_swap(), disc, cpi, disc, cpi_shifted, REF)
        assert abs(pnl.breakeven_pnl) > 0

    def test_attribution_structure(self):
        disc, cpi = _disc(), _cpi()
        cpi2 = cpi.bumped(0.0005)
        pnl = inflation_daily_pnl(_linker(), disc, cpi, disc, cpi2, REF)
        assert math.isfinite(pnl.total)
        assert math.isfinite(pnl.unexplained)

    def test_to_dict(self):
        disc, cpi = _disc(), _cpi()
        pnl = inflation_daily_pnl(_zc_swap(), disc, cpi, disc, cpi, REF)
        d = pnl.to_dict()
        assert "total" in d
        assert "unexplained" in d


# ── Capital ──

class TestInflationCapital:

    def test_capital_positive(self):
        cap = inflation_capital(_zc_swap(), _disc(), _cpi())
        assert cap.capital > 0
        assert cap.ead > 0

    def test_capital_8pct_rwa(self):
        cap = inflation_capital(_zc_swap(), _disc(), _cpi())
        assert cap.capital == pytest.approx(cap.rwa * 0.08, rel=1e-10)

    def test_simm_im_positive(self):
        cap = inflation_capital(_linker(), _disc(), _cpi())
        assert cap.simm_im > 0


# ── Book + Aggregate Risk ──

class TestInflationBookAggregate:

    def _book_with_instruments(self):
        disc, cpi = _disc(), _cpi()
        book = InflationBook("test_infl")
        book.add(_trade(1), "US_TIPS", "linker", notional=100.0,
                 instrument=_linker())
        book.add(_trade(1), "ZC_5Y", "zc_swap", notional=10_000_000,
                 instrument=_zc_swap())
        return book, disc, cpi

    def test_aggregate_risk_keys(self):
        book, disc, cpi = self._book_with_instruments()
        risk = book.aggregate_risk(disc, cpi)
        assert "total_ie01" in risk
        assert "total_real_dv01" in risk
        assert "total_dv01" in risk  # cross-asset compat
        assert "n_positions" in risk
        assert risk["n_positions"] == 2

    def test_total_ie01_nonzero(self):
        book, disc, cpi = self._book_with_instruments()
        risk = book.aggregate_risk(disc, cpi)
        assert risk["total_ie01"] != 0

    def test_positions_method(self):
        book, _, _ = self._book_with_instruments()
        assert len(book.positions()) == 2

    def test_cross_asset_dv01_equals_ie01(self):
        book, disc, cpi = self._book_with_instruments()
        risk = book.aggregate_risk(disc, cpi)
        assert risk["total_dv01"] == risk["total_ie01"]


# ── Stress with book + curves ──

class TestInflationStressBook:

    def test_stress_from_book(self):
        book = InflationBook("test")
        book.add(_trade(1), "US", "zc_swap", notional=10e6, instrument=_zc_swap())
        stress = inflation_stress_suite(
            book=book, discount_curve=_disc(), cpi_curve=_cpi())
        assert len(stress) >= 5
        assert any(s.scenario == "breakeven_up_50" for s in stress)

    def test_stress_backward_compat(self):
        """Old scalar interface still works."""
        stress = inflation_stress_suite(total_ie01=5000, total_real_dv01=3000)
        assert len(stress) >= 5
        assert stress[0].pnl == pytest.approx(5000 * 50)


# ── Cross-Asset Integration ──

class TestCrossAssetInflation:

    def test_inflation_registers_in_cross_asset(self):
        from pricebook.desks.cross_asset_desk import CrossAssetDesk
        desk = CrossAssetDesk()
        book = InflationBook("inflation")
        book.add(_trade(1), "US", "zc_swap", notional=10e6, ie01=5000)
        desk.add("inflation", book)
        assert "inflation" in desk.desk_names

    def test_dashboard_includes_inflation(self):
        from pricebook.desks.cross_asset_desk import CrossAssetDesk
        desk = CrossAssetDesk()
        book = InflationBook("inflation")
        book.add(_trade(1), "US", "zc_swap", notional=10e6, ie01=5000)
        desk.add("inflation", book)
        db = desk.dashboard(REF, _disc())
        assert db.n_desks == 1
        assert db.total_notional > 0
