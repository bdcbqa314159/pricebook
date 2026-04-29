"""Tests for BarrierOption: PDE, MC, parity, serialisation."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta

import pytest

from pricebook.barrier_option import BarrierOption, BarrierResult
from pricebook.black76 import OptionType
from pricebook.discount_curve import DiscountCurve
from pricebook.serialisable import from_dict

REF = date(2026, 4, 28)
MAT = REF + timedelta(days=365)


def _curve():
    return DiscountCurve.flat(REF, 0.03)


class TestBarrierPDE:

    def test_up_out_call(self):
        opt = BarrierOption(strike=100, barrier=120, barrier_type="up_out", maturity=MAT)
        r = opt.price(spot=100, curve=_curve(), vol=0.20, method="pde")
        assert math.isfinite(r.price)
        assert r.price > 0
        assert r.price < r.vanilla_price  # knocked out → cheaper

    def test_down_out_put(self):
        opt = BarrierOption(strike=100, barrier=80, barrier_type="down_out", maturity=MAT,
                            option_type=OptionType.PUT)
        r = opt.price(spot=100, curve=_curve(), vol=0.20, method="pde")
        assert r.price < r.vanilla_price

    def test_up_in_call(self):
        opt = BarrierOption(strike=100, barrier=120, barrier_type="up_in", maturity=MAT)
        r = opt.price(spot=100, curve=_curve(), vol=0.20, method="pde")
        assert math.isfinite(r.price)

    def test_down_in_put(self):
        opt = BarrierOption(strike=100, barrier=80, barrier_type="down_in", maturity=MAT,
                            option_type=OptionType.PUT)
        r = opt.price(spot=100, curve=_curve(), vol=0.20, method="pde")
        assert math.isfinite(r.price)


class TestBarrierMC:

    def test_up_out_mc(self):
        opt = BarrierOption(strike=100, barrier=120, barrier_type="up_out", maturity=MAT)
        r = opt.price(spot=100, curve=_curve(), vol=0.20, method="mc", n_paths=50_000)
        assert r.method == "mc"
        assert r.price < r.vanilla_price

    def test_barrier_hit_prob(self):
        opt = BarrierOption(strike=100, barrier=110, barrier_type="up_out", maturity=MAT)
        r = opt.price(spot=100, curve=_curve(), vol=0.30, method="mc", n_paths=50_000)
        assert r.barrier_hit_prob > 0  # should hit sometimes with 30% vol


class TestBarrierParity:

    def test_in_out_parity(self):
        """knock_in + knock_out ≈ vanilla."""
        ko = BarrierOption(strike=100, barrier=120, barrier_type="up_out", maturity=MAT)
        ki = BarrierOption(strike=100, barrier=120, barrier_type="up_in", maturity=MAT)
        rko = ko.price(spot=100, curve=_curve(), vol=0.20, method="pde")
        rki = ki.price(spot=100, curve=_curve(), vol=0.20, method="pde")
        # KI + KO ≈ vanilla
        assert rko.price + rki.price == pytest.approx(rko.vanilla_price, rel=0.05)


class TestBarrierGreeks:

    def test_greeks_finite(self):
        opt = BarrierOption(strike=100, barrier=120, barrier_type="up_out", maturity=MAT)
        g = opt.greeks(spot=100, curve=_curve(), vol=0.20)
        assert math.isfinite(g["delta"])
        assert math.isfinite(g["gamma"])
        assert math.isfinite(g["vega"])


class TestBarrierValidation:

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="barrier_type"):
            BarrierOption(strike=100, barrier=120, barrier_type="sideways", maturity=MAT)


class TestBarrierSerialisation:

    def test_round_trip(self):
        opt = BarrierOption(strike=100, barrier=120, barrier_type="up_out", maturity=MAT,
                            option_type=OptionType.PUT, notional=1_000_000)
        d = opt.to_dict()
        assert d["type"] == "barrier"
        opt2 = from_dict(d)
        assert opt2.strike == 100
        assert opt2.barrier == 120
        assert opt2.barrier_type == "up_out"
        assert opt2.option_type == OptionType.PUT

    def test_json_round_trip(self):
        opt = BarrierOption(strike=100, barrier=80, barrier_type="down_in", maturity=MAT)
        s = json.dumps(opt.to_dict())
        opt2 = from_dict(json.loads(s))
        assert opt2.barrier_type == "down_in"

    def test_pv_after_round_trip(self):
        opt = BarrierOption(strike=100, barrier=120, barrier_type="up_out", maturity=MAT)
        p1 = opt.price(spot=100, curve=_curve(), vol=0.20).price
        opt2 = from_dict(opt.to_dict())
        p2 = opt2.price(spot=100, curve=_curve(), vol=0.20).price
        assert p1 == pytest.approx(p2, abs=1e-8)
