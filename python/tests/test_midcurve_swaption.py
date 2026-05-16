"""Tests for mid-curve swaption."""
from __future__ import annotations
from datetime import date
import pytest
from dateutil.relativedelta import relativedelta
from pricebook.curves.bootstrap import bootstrap
from pricebook.midcurve_swaption import midcurve_swaption


REF = date(2024, 7, 15)

def _ois():
    deposits = [(REF + relativedelta(months=6), 0.043)]
    swaps = [(REF + relativedelta(years=y), r) for y, r in
             [(1, 0.041), (2, 0.039), (5, 0.038), (10, 0.036), (30, 0.035)]]
    return bootstrap(REF, deposits, swaps)


class TestMidCurveSwaption:

    def test_price_positive(self):
        ois = _ois()
        r = midcurve_swaption(REF, REF + relativedelta(years=1),
                               REF + relativedelta(years=2),
                               REF + relativedelta(years=7),
                               strike=0.04, vol=0.50, curve=ois)
        assert r.price > 0

    def test_gap_positive(self):
        ois = _ois()
        r = midcurve_swaption(REF, REF + relativedelta(years=1),
                               REF + relativedelta(years=3),
                               REF + relativedelta(years=8),
                               strike=0.04, vol=0.50, curve=ois)
        assert r.gap > 0  # 2 year gap between expiry and swap start

    def test_standard_swaption_zero_gap(self):
        ois = _ois()
        # When swap_start = option_expiry, this is a standard swaption
        r = midcurve_swaption(REF, REF + relativedelta(years=1),
                               REF + relativedelta(years=1),
                               REF + relativedelta(years=6),
                               strike=0.04, vol=0.50, curve=ois)
        assert r.gap == pytest.approx(0, abs=0.01)
        assert r.price > 0

    def test_higher_vol_higher_price(self):
        ois = _ois()
        low = midcurve_swaption(REF, REF + relativedelta(years=1),
                                 REF + relativedelta(years=2),
                                 REF + relativedelta(years=7),
                                 strike=0.04, vol=0.30, curve=ois)
        high = midcurve_swaption(REF, REF + relativedelta(years=1),
                                  REF + relativedelta(years=2),
                                  REF + relativedelta(years=7),
                                  strike=0.04, vol=0.60, curve=ois)
        assert high.price > low.price

    def test_receiver_positive(self):
        ois = _ois()
        r = midcurve_swaption(REF, REF + relativedelta(years=1),
                               REF + relativedelta(years=2),
                               REF + relativedelta(years=7),
                               strike=0.04, vol=0.50, curve=ois,
                               is_payer=False)
        assert r.price > 0

    def test_delta_positive_payer(self):
        ois = _ois()
        r = midcurve_swaption(REF, REF + relativedelta(years=1),
                               REF + relativedelta(years=2),
                               REF + relativedelta(years=7),
                               strike=0.04, vol=0.50, curve=ois)
        assert r.delta > 0  # payer benefits from higher rates

    def test_vega_positive(self):
        ois = _ois()
        r = midcurve_swaption(REF, REF + relativedelta(years=1),
                               REF + relativedelta(years=2),
                               REF + relativedelta(years=7),
                               strike=0.04, vol=0.50, curve=ois)
        assert r.vega > 0

    def test_invalid_dates_raises(self):
        ois = _ois()
        with pytest.raises(ValueError):
            midcurve_swaption(REF, REF + relativedelta(years=2),
                               REF + relativedelta(years=1),  # swap_start < expiry
                               REF + relativedelta(years=6),
                               strike=0.04, vol=0.50, curve=ois)
