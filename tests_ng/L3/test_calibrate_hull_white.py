"""Calibration front oracle — HW vol fit to a ZCB-option / caplet (L3).

The first tenant of the unified calibration front: `market -> calibrate -> model`.
Two facts pin the fit: (1) round-trip — a quote priced under a known sigma
recovers that sigma; (2) the calibrated model reprices the quote to tolerance.
The fit uses only the HW analytic `zero_bond_option` (no L4 engine), keeping
calibration inside its L0/L1/L3 dependency budget.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.calibration.hull_white import ZCBOptionQuote, calibrate_hull_white
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.models.hull_white import HullWhite

D0 = date(2026, 1, 15)
ACT365 = DC.ACT_365_FIXED
EXPIRY = date(2028, 1, 15)
BOND_MAT = date(2031, 1, 15)
STRIKE = 0.90


def _snapshot(rate=0.03):
    return MarketSnapshot(valuation_date=D0, discount_curve=FlatDiscountCurve(rate, D0, ACT365))


def _quote(snap, sigma, a, is_call=False):
    # target = price of the ZCB option under a known (a, sigma): a self-made market quote
    price = HullWhite(a, sigma, snap).zero_bond_option(EXPIRY, BOND_MAT, STRIKE, is_call)
    return ZCBOptionQuote(EXPIRY, BOND_MAT, STRIKE, is_call, price)


def test_calibrate_recovers_known_sigma():
    snap = _snapshot()
    a, true_sigma = 0.05, 0.012
    model = calibrate_hull_white(snap, _quote(snap, true_sigma, a), a)
    assert model.sigma == pytest.approx(true_sigma, abs=1e-9)
    assert model.a == a
    assert model.market is snap                              # carries the calibrating snapshot (A1)


def test_calibrated_model_reprices_the_quote():
    snap = _snapshot(0.025)
    a = 0.03
    quote = _quote(snap, 0.02, a)
    model = calibrate_hull_white(snap, quote, a)
    reprice = model.zero_bond_option(EXPIRY, BOND_MAT, STRIKE, quote.is_call)
    assert reprice == pytest.approx(quote.price, abs=1e-10)


def test_unreachable_quote_raises():
    snap = _snapshot()
    intrinsic = HullWhite(0.05, 1e-12, snap).zero_bond_option(EXPIRY, BOND_MAT, STRIKE, False)
    below_intrinsic = ZCBOptionQuote(EXPIRY, BOND_MAT, STRIKE, False, intrinsic - 1e-4)
    with pytest.raises(ValueError):
        calibrate_hull_white(snap, below_intrinsic, 0.05)
