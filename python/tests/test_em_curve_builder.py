"""Tests for EM local currency curve builders."""

import math
import pytest
from datetime import date

from pricebook.core.discount_curve import DiscountCurve
from pricebook.curves.em_curve_builder import (
    build_em_curve, build_cdi_curve, build_tiie_curve, build_shibor_curve,
    get_em_curve_conventions, list_em_curve_currencies, EMCurveConventions,
)


REF = date(2024, 1, 15)


def _make_deposits(ref, rates_by_months):
    """Helper: create deposit tuples."""
    return [(date(ref.year + m // 12, ref.month + m % 12, ref.day), r)
            for m, r in rates_by_months]


def _make_swaps(ref, rates_by_years):
    """Helper: create swap tuples."""
    return [(date(ref.year + y, ref.month, ref.day), r) for y, r in rates_by_years]


class TestConventions:
    def test_brl(self):
        c = get_em_curve_conventions("BRL")
        assert c.deposit_day_count.value == "BUS/252"

    def test_mxn(self):
        c = get_em_curve_conventions("MXN")
        assert c.deposit_day_count.value == "ACT/360"

    def test_cny(self):
        c = get_em_curve_conventions("CNY")
        assert c.deposit_day_count.value == "ACT/365F"

    def test_list_count(self):
        currencies = list_em_curve_currencies()
        assert len(currencies) == 16

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="No EM curve"):
            get_em_curve_conventions("USD")

    def test_to_dict(self):
        d = get_em_curve_conventions("BRL").to_dict()
        assert d["currency"] == "BRL"


class TestBuildEMCurve:
    def test_zar_curve(self):
        deposits = _make_deposits(REF, [(3, 0.08)])
        swaps = _make_swaps(REF, [(1, 0.085), (2, 0.09), (5, 0.095)])
        curve = build_em_curve("ZAR", REF, deposits, swaps)
        assert curve.df(date(2029, 1, 15)) < 1.0
        assert curve.df(date(2029, 1, 15)) > 0.5

    def test_inr_curve(self):
        deposits = _make_deposits(REF, [(3, 0.065)])
        swaps = _make_swaps(REF, [(1, 0.07), (5, 0.075)])
        curve = build_em_curve("INR", REF, deposits, swaps)
        assert curve.df(date(2029, 1, 15)) > 0

    def test_all_currencies_build(self):
        """Every EM currency should be able to build a curve."""
        deposits = _make_deposits(REF, [(3, 0.05)])
        swaps = _make_swaps(REF, [(1, 0.05), (2, 0.055)])
        for ccy in list_em_curve_currencies():
            if ccy == "BRL":
                continue  # BRL needs DI futures, not generic deposits
            curve = build_em_curve(ccy, REF, deposits, swaps)
            assert curve.df(date(2026, 1, 15)) > 0


class TestCDICurve:
    def test_basic_cdi(self):
        """Build CDI curve from DI futures."""
        di_futures = [
            (date(2024, 7, 1), 0.1175),
            (date(2025, 1, 2), 0.1150),
            (date(2026, 1, 2), 0.1100),
        ]
        curve = build_cdi_curve(REF, di_futures)
        # 6-month rate ~11.75%: df ≈ 1/(1.1175)^(~126/252) ≈ 0.946
        df_6m = curve.df(date(2024, 7, 1))
        assert 0.90 < df_6m < 0.98

    def test_cdi_df_formula(self):
        """Verify df = 1 / (1+r)^(bd/252)."""
        di_futures = [(date(2025, 1, 15), 0.10)]
        curve = build_cdi_curve(REF, di_futures)
        from pricebook.core.day_count import business_days_between
        from pricebook.core.calendar import SaoPauloCalendar
        cal = SaoPauloCalendar()
        bd = business_days_between(REF, date(2025, 1, 15), cal)
        expected_df = 1.0 / (1.10 ** (bd / 252.0))
        actual_df = curve.df(date(2025, 1, 15))
        assert abs(actual_df - expected_df) < 0.001

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            build_cdi_curve(REF, [])


class TestTIIECurve:
    def test_basic_tiie(self):
        deposits = [(date(2024, 2, 15), 0.115)]
        swaps = [(date(2025, 1, 15), 0.112), (date(2026, 1, 15), 0.108)]
        curve = build_tiie_curve(REF, deposits, swaps)
        assert curve.df(date(2026, 1, 15)) > 0


class TestSHIBORCurve:
    def test_basic_shibor(self):
        deposits = [(date(2024, 4, 15), 0.025)]
        swaps = [(date(2025, 1, 15), 0.026), (date(2027, 1, 15), 0.028)]
        curve = build_shibor_curve(REF, deposits, swaps)
        assert curve.df(date(2027, 1, 15)) > 0
