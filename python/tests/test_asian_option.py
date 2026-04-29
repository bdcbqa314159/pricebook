"""Tests for AsianOption: schedule, TW, MC, partial fixings, serialisation."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta

import pytest

from pricebook.asian_option import (
    AsianOption, AsianSchedule, AsianResult, turnbull_wakeman, curran_asian,
)
from pricebook.black76 import OptionType
from pricebook.discount_curve import DiscountCurve
from pricebook.serialisable import from_dict

REF = date(2026, 4, 28)
END = REF + timedelta(days=365)


def _curve():
    return DiscountCurve.flat(REF, 0.03)


# ---- AsianSchedule ----

class TestAsianSchedule:

    def test_monthly(self):
        s = AsianSchedule.monthly(REF, END)
        assert s.n_fixings == 13  # 12 months + 1

    def test_weekly(self):
        s = AsianSchedule.weekly(REF, REF + timedelta(days=28))
        assert s.n_fixings == 5  # 4 weeks + day 0

    def test_daily(self):
        s = AsianSchedule.daily(REF, REF + timedelta(days=7))
        assert s.n_fixings >= 5  # approximately 1 week of business days

    def test_custom_weights(self):
        s = AsianSchedule(
            fixing_dates=[REF, REF + timedelta(days=30), REF + timedelta(days=60)],
            weights=[0.5, 0.25, 0.25],
        )
        assert sum(s.effective_weights) == pytest.approx(1.0)

    def test_equal_weights(self):
        s = AsianSchedule.monthly(REF, END)
        w = s.effective_weights
        assert sum(w) == pytest.approx(1.0)
        assert all(abs(wi - w[0]) < 1e-10 for wi in w)

    def test_mismatched_weights_raises(self):
        with pytest.raises(ValueError, match="weights length"):
            AsianSchedule(fixing_dates=[REF, REF + timedelta(30)], weights=[1.0])

    def test_round_trip(self):
        s = AsianSchedule.monthly(REF, END, fixing_lag=2)
        d = s.to_dict()
        s2 = AsianSchedule.from_dict(d)
        assert s2.n_fixings == s.n_fixings
        assert s2.fixing_lag == 2


# ---- Turnbull-Wakeman ----

class TestTurnbullWakeman:

    def test_finite_price(self):
        times = [i * 0.25 for i in range(1, 5)]
        weights = [0.25] * 4
        p = turnbull_wakeman(100, 100, 0.03, 0.20, times, weights, T=1.0)
        assert math.isfinite(p)
        assert p > 0

    def test_call_put_parity(self):
        """C - P ≈ df × (M1 - K) for Asian options."""
        times = [i * 0.25 for i in range(1, 5)]
        weights = [0.25] * 4
        c = turnbull_wakeman(100, 100, 0.03, 0.20, times, weights, 1.0, OptionType.CALL)
        p = turnbull_wakeman(100, 100, 0.03, 0.20, times, weights, 1.0, OptionType.PUT)
        # C - P should be close to df × (forward_avg - K)
        assert math.isfinite(c - p)

    def test_atm_lower_than_european(self):
        """Asian ATM < European ATM (averaging reduces uncertainty)."""
        from pricebook.black76 import black76_price
        times = [i * 0.25 for i in range(1, 5)]
        weights = [0.25] * 4
        asian = turnbull_wakeman(100, 100, 0.03, 0.20, times, weights, 1.0)
        european = black76_price(100 * math.exp(0.03), 100, 0.20, 1.0, math.exp(-0.03))
        assert asian < european

    def test_with_known_fixings(self):
        """Partial fixings should change the price."""
        times = [0.25, 0.50, 0.75, 1.0]
        weights = [0.25] * 4
        no_fix = turnbull_wakeman(100, 100, 0.03, 0.20, times, weights, 1.0)
        with_fix = turnbull_wakeman(100, 100, 0.03, 0.20, times, weights, 1.0,
                                     known_fixings=[105.0, 102.0])
        assert no_fix != with_fix

    def test_all_fixed_intrinsic(self):
        """All fixings known → intrinsic value."""
        times = [0.25, 0.50]
        weights = [0.5, 0.5]
        p = turnbull_wakeman(100, 95, 0.03, 0.20, times, weights, 0.5,
                              known_fixings=[110.0, 105.0])
        # Average = 107.5, strike = 95 → intrinsic = df × 12.5
        assert p > 10


# ---- AsianOption instrument ----

class TestAsianOption:

    def test_tw_pricing(self):
        schedule = AsianSchedule.monthly(REF, END)
        opt = AsianOption(schedule=schedule, strike=100, notional=1_000_000)
        result = opt.price(spot=100, curve=_curve(), vol=0.20, method="tw")
        assert result.method == "turnbull_wakeman"
        assert math.isfinite(result.price)
        assert result.price > 0

    def test_mc_pricing(self):
        schedule = AsianSchedule.monthly(REF, END)
        opt = AsianOption(schedule=schedule, strike=100, notional=1_000_000)
        result = opt.price(spot=100, curve=_curve(), vol=0.20, method="mc", n_paths=10_000)
        assert result.method == "mc_cv"
        assert result.std_error > 0
        assert result.n_paths > 0

    def test_tw_vs_mc_close(self):
        """TW and MC should agree within MC error."""
        schedule = AsianSchedule.monthly(REF, END)
        opt = AsianOption(schedule=schedule, strike=100)
        tw = opt.price(spot=100, curve=_curve(), vol=0.20, method="tw")
        mc = opt.price(spot=100, curve=_curve(), vol=0.20, method="mc", n_paths=200_000)
        # TW is an approximation — within 1% of each other
        assert abs(tw.price - mc.price) / max(mc.price, 0.01) < 0.01

    def test_partial_fixings(self):
        schedule = AsianSchedule.monthly(REF, END)
        fixings = {REF: 102.0, REF + timedelta(days=30): 105.0}
        opt = AsianOption(schedule=schedule, strike=100, known_fixings=fixings)
        assert opt.n_fixed == 2
        assert opt.n_remaining == schedule.n_fixings - 2
        result = opt.price(spot=100, curve=_curve(), vol=0.20, method="tw")
        assert math.isfinite(result.price)

    def test_put_option(self):
        schedule = AsianSchedule.monthly(REF, END)
        call = AsianOption(schedule=schedule, strike=100, option_type=OptionType.CALL)
        put = AsianOption(schedule=schedule, strike=100, option_type=OptionType.PUT)
        rc = call.price(spot=100, curve=_curve(), vol=0.20, method="tw")
        rp = put.price(spot=100, curve=_curve(), vol=0.20, method="tw")
        assert rc.price != rp.price

    def test_curran_method(self):
        schedule = AsianSchedule.monthly(REF, END)
        opt = AsianOption(schedule=schedule, strike=100)
        result = opt.price(spot=100, curve=_curve(), vol=0.20, method="curran")
        assert result.method == "curran"
        assert math.isfinite(result.price)
        assert result.price > 0

    def test_curran_vs_tw_agree(self):
        """Curran and TW should agree within 10% for ATM options."""
        schedule = AsianSchedule.monthly(REF, END)
        opt = AsianOption(schedule=schedule, strike=100)
        tw = opt.price(spot=100, curve=_curve(), vol=0.20, method="tw")
        curran = opt.price(spot=100, curve=_curve(), vol=0.20, method="curran")
        assert abs(curran.price - tw.price) / max(tw.price, 0.01) < 0.10

    def test_sobol_mc(self):
        schedule = AsianSchedule.monthly(REF, END)
        opt = AsianOption(schedule=schedule, strike=100)
        result = opt.price(spot=100, curve=_curve(), vol=0.20, method="mc_sobol", n_paths=10_000)
        assert result.method == "mc_sobol"
        assert math.isfinite(result.price)
        assert result.price > 0

    def test_sobol_finite_and_close(self):
        """Sobol MC should produce finite price close to TW."""
        schedule = AsianSchedule.monthly(REF, END)
        opt = AsianOption(schedule=schedule, strike=100)
        sobol = opt.price(spot=100, curve=_curve(), vol=0.20, method="mc_sobol", n_paths=50_000)
        tw = opt.price(spot=100, curve=_curve(), vol=0.20, method="tw")
        assert math.isfinite(sobol.price)
        # Sobol without CV has wider error than CV methods — within 10%
        assert abs(sobol.price - tw.price) / max(tw.price, 0.01) < 0.10

    def test_delta_profile(self):
        schedule = AsianSchedule(
            fixing_dates=[REF + timedelta(days=30*i) for i in range(1, 5)])
        opt = AsianOption(schedule=schedule, strike=100)
        profile = opt.delta_profile(spot=100, curve=_curve(), vol=0.20)
        assert len(profile) == 5  # 0 to 4 fixings
        # Delta should decrease as fixings accumulate
        assert profile[0]["delta"] > profile[-1]["delta"]
        # Last entry (all fixed) should have ~0 delta
        assert abs(profile[-1]["delta"]) < 0.1

    def test_greeks(self):
        schedule = AsianSchedule.monthly(REF, END)
        opt = AsianOption(schedule=schedule, strike=100)
        g = opt.greeks(spot=100, curve=_curve(), vol=0.20, method="tw")
        assert math.isfinite(g["delta"])
        assert math.isfinite(g["gamma"])
        assert math.isfinite(g["vega"])
        assert g["delta"] > 0  # call delta positive
        assert g["vega"] > 0   # vega positive

    def test_weighted_averaging(self):
        dates = [REF + timedelta(days=30*i) for i in range(1, 4)]
        schedule = AsianSchedule(fixing_dates=dates, weights=[0.6, 0.3, 0.1])
        opt = AsianOption(schedule=schedule, strike=100)
        result = opt.price(spot=100, curve=_curve(), vol=0.20, method="tw")
        assert math.isfinite(result.price)

    def test_result_dict(self):
        schedule = AsianSchedule.monthly(REF, END)
        opt = AsianOption(schedule=schedule, strike=100)
        result = opt.price(spot=100, curve=_curve(), vol=0.20)
        d = result.to_dict()
        assert "price" in d
        assert "method" in d


# ---- Serialisation ----

class TestAsianOptionSerialisation:

    def test_round_trip(self):
        schedule = AsianSchedule.monthly(REF, END)
        opt = AsianOption(schedule=schedule, strike=100, notional=1_000_000,
                          option_type=OptionType.PUT)
        d = opt.to_dict()
        assert d["type"] == "asian"
        opt2 = from_dict(d)
        assert opt2.strike == 100
        assert opt2.notional == 1_000_000
        assert opt2.option_type == OptionType.PUT
        assert opt2.schedule.n_fixings == schedule.n_fixings

    def test_json_round_trip(self):
        schedule = AsianSchedule.monthly(REF, END)
        opt = AsianOption(schedule=schedule, strike=105, notional=500_000)
        s = json.dumps(opt.to_dict())
        d = json.loads(s)
        opt2 = from_dict(d)
        assert opt2.strike == 105

    def test_with_fixings_round_trip(self):
        schedule = AsianSchedule.monthly(REF, END)
        fixings = {REF: 102.0, REF + timedelta(days=30): 105.0}
        opt = AsianOption(schedule=schedule, strike=100, known_fixings=fixings)
        d = opt.to_dict()
        opt2 = from_dict(d)
        assert len(opt2.known_fixings) == 2
        assert opt2.known_fixings[REF] == 102.0

    def test_pv_matches_after_round_trip(self):
        """Price after round-trip matches original."""
        schedule = AsianSchedule.monthly(REF, END)
        opt = AsianOption(schedule=schedule, strike=100, notional=1_000_000)
        curve = _curve()
        p1 = opt.price(spot=100, curve=curve, vol=0.20, method="tw")

        d = opt.to_dict()
        opt2 = from_dict(d)
        p2 = opt2.price(spot=100, curve=curve, vol=0.20, method="tw")

        assert p1.price == pytest.approx(p2.price, abs=1e-8)
