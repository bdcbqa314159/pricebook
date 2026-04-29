"""Tests for TARF — Target Redemption Forward."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta

import pytest

from pricebook.tarf import TARF, TARFResult
from pricebook.discount_curve import DiscountCurve
from pricebook.serialisable import from_dict

REF = date(2026, 4, 28)


def _curve():
    return DiscountCurve.flat(REF, 0.03)


def _monthly_dates(n_months=12):
    from dateutil.relativedelta import relativedelta
    return [REF + relativedelta(months=i) for i in range(1, n_months + 1)]


# ---- Pricing ----

class TestTARFPricing:

    def test_basic(self):
        tarf = TARF(strike=1.10, target=0.05, leverage=2.0,
                     fixing_dates=_monthly_dates())
        r = tarf.price_mc(spot=1.10, curve=_curve(), vol=0.08, n_paths=50_000)
        assert math.isfinite(r.price)
        assert r.n_paths == 50_000

    def test_target_hit_prob(self):
        """With favorable drift, target should be hit sometimes."""
        tarf = TARF(strike=1.10, target=0.05, fixing_dates=_monthly_dates())
        r = tarf.price_mc(spot=1.12, curve=_curve(), vol=0.08, n_paths=50_000)
        assert r.target_hit_prob > 0

    def test_higher_target_less_likely(self):
        """Higher target → less likely to hit early."""
        low = TARF(strike=1.10, target=0.02, fixing_dates=_monthly_dates())
        high = TARF(strike=1.10, target=0.20, fixing_dates=_monthly_dates())
        r_low = low.price_mc(spot=1.12, curve=_curve(), vol=0.08, n_paths=50_000)
        r_high = high.price_mc(spot=1.12, curve=_curve(), vol=0.08, n_paths=50_000)
        assert r_low.target_hit_prob >= r_high.target_hit_prob

    def test_avg_life_shorter_with_low_target(self):
        """Low target → earlier termination → shorter avg life."""
        low = TARF(strike=1.10, target=0.02, fixing_dates=_monthly_dates())
        high = TARF(strike=1.10, target=0.50, fixing_dates=_monthly_dates())
        r_low = low.price_mc(spot=1.12, curve=_curve(), vol=0.08, n_paths=50_000)
        r_high = high.price_mc(spot=1.12, curve=_curve(), vol=0.08, n_paths=50_000)
        assert r_low.avg_life <= r_high.avg_life + 0.01

    def test_higher_leverage_worse_for_buyer(self):
        """Higher leverage → more loss when below strike → lower PV for buyer."""
        low_lev = TARF(strike=1.10, target=0.05, leverage=1.0,
                        fixing_dates=_monthly_dates())
        high_lev = TARF(strike=1.10, target=0.05, leverage=3.0,
                         fixing_dates=_monthly_dates())
        r_low = low_lev.price_mc(spot=1.10, curve=_curve(), vol=0.10, n_paths=50_000)
        r_high = high_lev.price_mc(spot=1.10, curve=_curve(), vol=0.10, n_paths=50_000)
        assert r_low.price > r_high.price

    def test_result_dict(self):
        tarf = TARF(strike=1.10, target=0.05, fixing_dates=_monthly_dates())
        r = tarf.price_mc(spot=1.10, curve=_curve(), vol=0.08, n_paths=10_000)
        d = r.to_dict()
        assert "target_hit_prob" in d
        assert "avg_life" in d
        assert "avg_accumulated" in d


# ---- Validation ----

class TestTARFValidation:

    def test_empty_dates_raises(self):
        with pytest.raises(ValueError, match="empty"):
            TARF(strike=1.10, target=0.05, fixing_dates=[])

    def test_none_dates_raises(self):
        with pytest.raises(ValueError, match="empty"):
            TARF(strike=1.10, target=0.05, fixing_dates=None)


# ---- Serialisation ----

class TestTARFSerialisation:

    def test_round_trip(self):
        tarf = TARF(strike=1.10, target=0.08, leverage=2.5,
                     fixing_dates=_monthly_dates(), notional=5_000_000)
        d = tarf.to_dict()
        assert d["type"] == "tarf"
        tarf2 = from_dict(d)
        assert tarf2.strike == 1.10
        assert tarf2.target == 0.08
        assert tarf2.leverage == 2.5
        assert len(tarf2.fixing_dates) == 12

    def test_json_round_trip(self):
        tarf = TARF(strike=1.10, target=0.05, fixing_dates=_monthly_dates())
        s = json.dumps(tarf.to_dict())
        tarf2 = from_dict(json.loads(s))
        assert tarf2.strike == tarf.strike

    def test_pv_after_round_trip(self):
        tarf = TARF(strike=1.10, target=0.05, fixing_dates=_monthly_dates())
        p1 = tarf.price_mc(spot=1.10, curve=_curve(), vol=0.08,
                            n_paths=10_000, seed=42)
        tarf2 = from_dict(tarf.to_dict())
        p2 = tarf2.price_mc(spot=1.10, curve=_curve(), vol=0.08,
                             n_paths=10_000, seed=42)
        assert p1.price == pytest.approx(p2.price, abs=1e-8)

    def test_pivot(self):
        tarf = TARF(strike=1.10, target=0.05, pivot=1.12,
                     fixing_dates=_monthly_dates())
        d = tarf.to_dict()
        tarf2 = from_dict(d)
        assert tarf2.pivot == 1.12
