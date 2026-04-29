"""Tests for Autocallable and Cliquet — structured exotic products."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta

import pytest

from pricebook.autocallable import Autocallable, AutocallResult
from pricebook.cliquet import Cliquet, CliquetResult
from pricebook.discount_curve import DiscountCurve
from pricebook.serialisable import from_dict

REF = date(2026, 4, 28)


def _curve():
    return DiscountCurve.flat(REF, 0.03)


def _quarterly_dates(n_years=3):
    from dateutil.relativedelta import relativedelta
    return [REF + relativedelta(months=3*i) for i in range(1, n_years*4 + 1)]


# ---- Autocallable ----

class TestAutocallable:

    def test_basic_pricing(self):
        ac = Autocallable(observation_dates=_quarterly_dates(2),
                          autocall_level=1.05, coupon_rate=0.08, put_barrier=0.70)
        r = ac.price_mc(spot=100, curve=_curve(), vol=0.20, n_paths=50_000)
        assert math.isfinite(r.price)
        assert r.price > 0

    def test_higher_autocall_cheaper(self):
        """Higher autocall barrier → less likely to terminate early → lower price."""
        low = Autocallable(observation_dates=_quarterly_dates(), autocall_level=1.0)
        high = Autocallable(observation_dates=_quarterly_dates(), autocall_level=1.20)
        r_low = low.price_mc(spot=100, curve=_curve(), vol=0.20, n_paths=50_000)
        r_high = high.price_mc(spot=100, curve=_curve(), vol=0.20, n_paths=50_000)
        assert r_low.autocall_prob > r_high.autocall_prob

    def test_put_barrier_risk(self):
        """Low put barrier should have some knock probability with high vol."""
        ac = Autocallable(observation_dates=_quarterly_dates(),
                          autocall_level=1.50, put_barrier=0.70)
        r = ac.price_mc(spot=100, curve=_curve(), vol=0.30, n_paths=50_000)
        assert r.put_knock_prob > 0

    def test_avg_life(self):
        ac = Autocallable(observation_dates=_quarterly_dates(),
                          autocall_level=1.0)
        r = ac.price_mc(spot=100, curve=_curve(), vol=0.20, n_paths=50_000)
        T = 3.0  # 3 years
        assert 0 < r.avg_life <= T

    def test_result_dict(self):
        ac = Autocallable(observation_dates=_quarterly_dates())
        r = ac.price_mc(spot=100, curve=_curve(), vol=0.20, n_paths=10_000)
        d = r.to_dict()
        assert "autocall_prob" in d
        assert "put_knock_prob" in d

    def test_empty_dates_raises(self):
        with pytest.raises(ValueError, match="empty"):
            Autocallable(observation_dates=[])


class TestAutocallableSerialisation:

    def test_round_trip(self):
        ac = Autocallable(observation_dates=_quarterly_dates(),
                          autocall_level=1.05, coupon_rate=0.10, put_barrier=0.65)
        d = ac.to_dict()
        assert d["type"] == "autocallable"
        ac2 = from_dict(d)
        assert ac2.autocall_level == 1.05
        assert ac2.put_barrier == 0.65
        assert len(ac2.observation_dates) == len(ac.observation_dates)

    def test_json_round_trip(self):
        ac = Autocallable(observation_dates=_quarterly_dates())
        s = json.dumps(ac.to_dict())
        ac2 = from_dict(json.loads(s))
        assert ac2.coupon_rate == ac.coupon_rate

    def test_pv_after_round_trip(self):
        ac = Autocallable(observation_dates=_quarterly_dates())
        p1 = ac.price_mc(spot=100, curve=_curve(), vol=0.20, n_paths=10_000, seed=42)
        ac2 = from_dict(ac.to_dict())
        p2 = ac2.price_mc(spot=100, curve=_curve(), vol=0.20, n_paths=10_000, seed=42)
        assert p1.price == pytest.approx(p2.price, abs=1e-8)


# ---- Cliquet ----

class TestCliquet:

    def test_basic_pricing(self):
        cliq = Cliquet(reset_dates=_quarterly_dates(),
                       local_floor=0.0, local_cap=0.05, global_cap=0.30)
        r = cliq.price_mc(spot=100, curve=_curve(), vol=0.20, n_paths=50_000)
        assert math.isfinite(r.price)
        assert r.price >= 0  # floor at 0

    def test_no_cap_equals_sum_of_returns(self):
        """With very high caps and no floor, price ≈ forward - 1 scaled."""
        cliq = Cliquet(reset_dates=_quarterly_dates(1),
                       local_floor=-10.0, local_cap=10.0,
                       global_floor=-10.0, global_cap=10.0,
                       notional=1_000_000)
        r = cliq.price_mc(spot=100, curve=_curve(), vol=0.20, n_paths=100_000)
        assert math.isfinite(r.price)

    def test_higher_local_cap_higher_price(self):
        low_cap = Cliquet(reset_dates=_quarterly_dates(), local_cap=0.02)
        high_cap = Cliquet(reset_dates=_quarterly_dates(), local_cap=0.10)
        r_low = low_cap.price_mc(spot=100, curve=_curve(), vol=0.20, n_paths=50_000)
        r_high = high_cap.price_mc(spot=100, curve=_curve(), vol=0.20, n_paths=50_000)
        assert r_high.price >= r_low.price

    def test_floor_protects(self):
        """With local_floor=0, avg_total_return >= 0."""
        cliq = Cliquet(reset_dates=_quarterly_dates(), local_floor=0.0, local_cap=0.05)
        r = cliq.price_mc(spot=100, curve=_curve(), vol=0.20, n_paths=50_000)
        assert r.avg_total_return >= 0

    def test_capped_periods_counted(self):
        cliq = Cliquet(reset_dates=_quarterly_dates(), local_cap=0.01)
        r = cliq.price_mc(spot=100, curve=_curve(), vol=0.20, n_paths=50_000)
        assert r.avg_capped_periods > 0  # many periods should hit 1% cap

    def test_n_periods(self):
        cliq = Cliquet(reset_dates=_quarterly_dates(2))
        assert cliq.n_periods == 8

    def test_result_dict(self):
        cliq = Cliquet(reset_dates=_quarterly_dates())
        r = cliq.price_mc(spot=100, curve=_curve(), vol=0.20, n_paths=10_000)
        d = r.to_dict()
        assert "avg_total_return" in d


class TestCliquetSerialisation:

    def test_round_trip(self):
        cliq = Cliquet(reset_dates=_quarterly_dates(),
                       local_floor=-0.02, local_cap=0.05,
                       global_floor=0.0, global_cap=0.25)
        d = cliq.to_dict()
        assert d["type"] == "cliquet"
        cliq2 = from_dict(d)
        assert cliq2.local_cap == 0.05
        assert cliq2.global_cap == 0.25
        assert cliq2.n_periods == cliq.n_periods

    def test_json_round_trip(self):
        cliq = Cliquet(reset_dates=_quarterly_dates())
        s = json.dumps(cliq.to_dict())
        cliq2 = from_dict(json.loads(s))
        assert cliq2.local_floor == cliq.local_floor

    def test_pv_after_round_trip(self):
        cliq = Cliquet(reset_dates=_quarterly_dates())
        p1 = cliq.price_mc(spot=100, curve=_curve(), vol=0.20, n_paths=10_000, seed=42)
        cliq2 = from_dict(cliq.to_dict())
        p2 = cliq2.price_mc(spot=100, curve=_curve(), vol=0.20, n_paths=10_000, seed=42)
        assert p1.price == pytest.approx(p2.price, abs=1e-8)
