"""Tests for American, Basket — Layer E exotic instruments."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta

import pytest

from pricebook.american_option import AmericanOption, AmericanResult
from pricebook.basket_option import BasketOption
from pricebook.black76 import OptionType
from pricebook.discount_curve import DiscountCurve
from pricebook.serialisable import from_dict

REF = date(2026, 4, 28)
MAT = REF + timedelta(days=365)


def _curve():
    return DiscountCurve.flat(REF, 0.03)


# ---- American ----

class TestAmericanOption:

    def test_put_pde(self):
        opt = AmericanOption(strike=100, maturity=MAT, option_type=OptionType.PUT)
        r = opt.price(spot=100, curve=_curve(), vol=0.20, method="pde")
        assert math.isfinite(r.price)
        assert r.price > 0

    def test_put_lsm(self):
        opt = AmericanOption(strike=100, maturity=MAT, option_type=OptionType.PUT)
        r = opt.price(spot=100, curve=_curve(), vol=0.20, method="lsm",
                      n_paths=50_000, n_steps=50)
        assert math.isfinite(r.price)

    def test_american_geq_european(self):
        """American put >= European put (early exercise premium >= 0)."""
        opt = AmericanOption(strike=100, maturity=MAT, option_type=OptionType.PUT)
        r = opt.price(spot=100, curve=_curve(), vol=0.20)
        assert r.price >= r.european_price - 0.01  # small tolerance
        assert r.early_exercise_premium >= -0.01

    def test_call_no_div_equals_european(self):
        """American call without dividends ≈ European call."""
        opt = AmericanOption(strike=100, maturity=MAT, option_type=OptionType.CALL)
        r = opt.price(spot=100, curve=_curve(), vol=0.20)
        assert r.early_exercise_premium < 0.5  # should be near zero

    def test_greeks(self):
        opt = AmericanOption(strike=100, maturity=MAT)
        g = opt.greeks(spot=100, curve=_curve(), vol=0.20)
        assert math.isfinite(g["delta"])
        assert g["delta"] < 0  # put delta negative

    def test_notional(self):
        opt = AmericanOption(strike=100, maturity=MAT, notional=1_000_000)
        r = opt.price(spot=100, curve=_curve(), vol=0.20)
        assert r.price > 1000  # reasonable for 1M notional


class TestAmericanSerialisation:

    def test_round_trip(self):
        opt = AmericanOption(strike=105, maturity=MAT, option_type=OptionType.PUT,
                             notional=500_000)
        d = opt.to_dict()
        assert d["type"] == "american"
        opt2 = from_dict(d)
        assert opt2.strike == 105
        assert opt2.option_type == OptionType.PUT

    def test_pv_after_round_trip(self):
        opt = AmericanOption(strike=100, maturity=MAT)
        p1 = opt.price(spot=100, curve=_curve(), vol=0.20).price
        opt2 = from_dict(opt.to_dict())
        p2 = opt2.price(spot=100, curve=_curve(), vol=0.20).price
        assert p1 == pytest.approx(p2, abs=1e-8)


# ---- Basket ----

class TestBasketOption:

    def test_basket_call(self):
        opt = BasketOption(strike=100, maturity=MAT, n_assets=2)
        r = opt.price_mc(spots=[100, 100], vols=[0.20, 0.25],
                          corr_matrix=[[1, 0.5], [0.5, 1]], curve=_curve(),
                          n_paths=50_000)
        assert math.isfinite(r.price)
        assert r.price > 0

    def test_best_of(self):
        opt = BasketOption(strike=1.0, maturity=MAT, payoff_type="best_of", n_assets=2)
        r = opt.price_mc(spots=[100, 100], vols=[0.20, 0.25],
                          corr_matrix=[[1, 0.5], [0.5, 1]], curve=_curve(),
                          n_paths=50_000)
        assert r.price > 0

    def test_worst_of(self):
        opt = BasketOption(strike=1.0, maturity=MAT, payoff_type="worst_of", n_assets=2)
        r = opt.price_mc(spots=[100, 100], vols=[0.20, 0.25],
                          corr_matrix=[[1, 0.5], [0.5, 1]], curve=_curve(),
                          n_paths=50_000)
        assert r.price >= 0

    def test_best_of_gt_worst_of(self):
        """Best-of call > worst-of call."""
        best = BasketOption(strike=1.0, maturity=MAT, payoff_type="best_of", n_assets=2)
        worst = BasketOption(strike=1.0, maturity=MAT, payoff_type="worst_of", n_assets=2)
        corr = [[1, 0.5], [0.5, 1]]
        rb = best.price_mc([100, 100], [0.20, 0.20], corr, _curve(), n_paths=50_000)
        rw = worst.price_mc([100, 100], [0.20, 0.20], corr, _curve(), n_paths=50_000)
        assert rb.price > rw.price

    def test_three_assets(self):
        opt = BasketOption(strike=100, maturity=MAT, n_assets=3,
                            weights=[0.4, 0.3, 0.3])
        corr = [[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]]
        r = opt.price_mc([100, 80, 120], [0.20, 0.25, 0.15], corr, _curve(),
                          n_paths=50_000)
        assert math.isfinite(r.price)
        assert r.n_assets == 3


class TestBasketSerialisation:

    def test_round_trip(self):
        opt = BasketOption(strike=100, maturity=MAT, weights=[0.5, 0.5],
                            payoff_type="best_of", notional=1_000_000)
        d = opt.to_dict()
        assert d["type"] == "basket_option"
        opt2 = from_dict(d)
        assert opt2.payoff_type == "best_of"
        assert opt2.weights == [0.5, 0.5]

    def test_json(self):
        opt = BasketOption(strike=100, maturity=MAT)
        s = json.dumps(opt.to_dict())
        opt2 = from_dict(json.loads(s))
        assert opt2.strike == 100
