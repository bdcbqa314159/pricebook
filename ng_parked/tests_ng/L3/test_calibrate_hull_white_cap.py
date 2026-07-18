"""Calibration front — joint HW (a, sigma) fit to a cap strip (L3).

The front's first *multi-instrument* fit: a cap is a strip of caplets (each a
European option on a zero-coupon bond), and a single caplet cannot separate mean
reversion `a` from vol `sigma` — a strip spanning expiries can. Least-squares over
the strip recovers both. Oracles: (1) round-trip — a strip priced under a known
(a*, sigma*) recovers them; (2) the fitted model reprices the whole strip (SSE ~ 0).
"""

from datetime import date

import pytest

from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.calibration.hull_white import (
    ZCBOptionQuote,
    calibrate_hull_white_cap,
)
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.models.hull_white import HullWhite

D0 = date(2026, 1, 15)
ACT365 = DC.ACT_365_FIXED
# caplets: reset at expiry Ti, pay at Ti+1 -> option on the ZCB P(Ti, Ti+1)
_CAPLETS = [
    (date(2027, 1, 15), date(2028, 1, 15)),
    (date(2028, 1, 15), date(2029, 1, 15)),
    (date(2029, 1, 15), date(2030, 1, 15)),
    (date(2030, 1, 15), date(2031, 1, 15)),
]
STRIKE = 0.95


def _snapshot(rate=0.03):
    return MarketSnapshot(valuation_date=D0, discount_curve=FlatDiscountCurve(rate, D0, ACT365))


def _strip(snap, a, sigma):
    model = HullWhite(a, sigma, snap)
    return [
        ZCBOptionQuote(
            exp, mat, STRIKE, is_call=False,
            price=model.zero_bond_option(exp, mat, STRIKE, is_call=False),
        )
        for exp, mat in _CAPLETS
    ]


def test_cap_strip_recovers_a_and_sigma():
    snap = _snapshot()
    a_true, sigma_true = 0.08, 0.015
    model = calibrate_hull_white_cap(snap, _strip(snap, a_true, sigma_true))
    assert model.a == pytest.approx(a_true, abs=1e-4)
    assert model.sigma == pytest.approx(sigma_true, abs=1e-5)
    assert model.market is snap


def test_fitted_model_reprices_the_strip():
    snap = _snapshot(0.025)
    strip = _strip(snap, 0.05, 0.02)
    model = calibrate_hull_white_cap(snap, strip)
    sse = sum(
        (model.zero_bond_option(q.expiry, q.bond_maturity, q.strike, q.is_call) - q.price) ** 2
        for q in strip
    )
    assert sse == pytest.approx(0.0, abs=1e-14)
