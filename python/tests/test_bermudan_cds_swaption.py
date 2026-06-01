"""Tests for Bermudan CDS swaption."""

import pytest
import math
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.credit.bermudan_cds_swaption import (
    bermudan_cds_swaption_price, BermudanCDSSwaptionResult,
)

REF = date(2024, 11, 4)


def _make_curves(hazard=0.02, rate=0.04):
    from pricebook.core.discount_curve import DiscountCurve
    from pricebook.core.survival_curve import SurvivalCurve
    from pricebook.core.interpolation import InterpolationMethod
    dates = [REF + relativedelta(years=y) for y in range(1, 15)]
    dfs = [math.exp(-rate * y) for y in range(1, 15)]
    survs = [math.exp(-hazard * y) for y in range(1, 15)]
    dc = DiscountCurve(REF, dates, dfs, interpolation=InterpolationMethod.LOG_LINEAR)
    sc = SurvivalCurve(REF, dates, survs)
    return dc, sc


class TestBermudanCDSSwaption:
    def test_positive_price(self):
        dc, sc = _make_curves(hazard=0.03)  # wider credit → more ITM
        exercise = [REF + relativedelta(years=y) for y in [1, 2, 3]]
        maturity = REF + relativedelta(years=5)
        r = bermudan_cds_swaption_price(REF, exercise, maturity, 0.005, dc, sc)  # low strike = ITM payer
        assert r.price > 0

    def test_bermudan_ge_european(self):
        """Bermudan ≥ European (more exercise opportunities)."""
        dc, sc = _make_curves(hazard=0.03)
        exercise = [REF + relativedelta(years=y) for y in [1, 2, 3, 4]]
        maturity = REF + relativedelta(years=5)
        r = bermudan_cds_swaption_price(REF, exercise, maturity, 0.005, dc, sc)
        assert r.price >= r.european_price - 0.01
        assert r.early_exercise_premium >= 0

    def test_single_exercise_equals_european(self):
        """Single exercise date → Bermudan = European."""
        dc, sc = _make_curves()
        exercise = [REF + relativedelta(years=1)]
        maturity = REF + relativedelta(years=5)
        r = bermudan_cds_swaption_price(REF, exercise, maturity, 0.015, dc, sc)
        assert r.price == pytest.approx(r.european_price, rel=0.05)

    def test_payer_vs_receiver(self):
        """Payer and receiver swaptions should both be non-negative."""
        dc, sc = _make_curves(hazard=0.03)
        exercise = [REF + relativedelta(years=y) for y in [1, 2, 3]]
        maturity = REF + relativedelta(years=5)
        payer = bermudan_cds_swaption_price(REF, exercise, maturity, 0.005, dc, sc, is_payer=True)
        receiver = bermudan_cds_swaption_price(REF, exercise, maturity, 0.05, dc, sc, is_payer=False)
        assert payer.price > 0
        assert receiver.price >= 0

    def test_itm_higher_than_otm(self):
        """Lower strike (more ITM for payer) → higher price."""
        dc, sc = _make_curves(hazard=0.03)  # wider credit
        exercise = [REF + relativedelta(years=1)]
        maturity = REF + relativedelta(years=5)
        itm = bermudan_cds_swaption_price(REF, exercise, maturity, 0.01, dc, sc)
        otm = bermudan_cds_swaption_price(REF, exercise, maturity, 0.05, dc, sc)
        assert itm.price > otm.price

    def test_exercise_probability(self):
        dc, sc = _make_curves()
        exercise = [REF + relativedelta(years=y) for y in [1, 2, 3]]
        maturity = REF + relativedelta(years=5)
        r = bermudan_cds_swaption_price(REF, exercise, maturity, 0.015, dc, sc)
        assert 0 <= r.exercise_probability <= 1

    def test_n_exercise_dates(self):
        dc, sc = _make_curves()
        exercise = [REF + relativedelta(years=y) for y in [1, 2, 3, 4]]
        maturity = REF + relativedelta(years=5)
        r = bermudan_cds_swaption_price(REF, exercise, maturity, 0.015, dc, sc)
        assert r.n_exercise_dates == 4

    def test_to_dict(self):
        dc, sc = _make_curves()
        exercise = [REF + relativedelta(years=1)]
        maturity = REF + relativedelta(years=5)
        r = bermudan_cds_swaption_price(REF, exercise, maturity, 0.015, dc, sc)
        d = r.to_dict()
        assert "early_exercise_premium" in d
