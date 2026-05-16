"""Tests for benchmark bonds: universe, curve fitting, strategies, rankings."""
from __future__ import annotations
from datetime import date
import math
import pytest
import numpy as np
from dateutil.relativedelta import relativedelta
from pricebook.curves.bootstrap import bootstrap
from pricebook.fixed_income.benchmark_bonds import (
    create_ust_universe, create_bund_universe, create_universe,
    fitted_curve_nss, duration_neutral_spread, butterfly_trade,
    barbell_vs_bullet, roll_down_ranking, carry_ranking, rv_scorecard,
    UST, BUND, GILT, JGB, CONVENTIONS,
)

REF = date(2024, 7, 15)
UST_YIELDS = {2: 0.0455, 3: 0.0435, 5: 0.0410, 7: 0.0400, 10: 0.0390, 20: 0.0405, 30: 0.0415}

def _curve():
    deposits = [(REF + relativedelta(months=6), 0.0515)]
    swaps = [(REF + relativedelta(years=y), r) for y, r in
             [(1, 0.0490), (2, 0.0455), (5, 0.0410), (10, 0.0390), (30, 0.0415)]]
    return bootstrap(REF, deposits, swaps)


class TestBenchmarkConventions:
    def test_ust_convention(self):
        assert UST.settlement_days == 1
        assert UST.quoting == "32nds"
        assert 10 in UST.standard_tenors

    def test_bund_convention(self):
        assert BUND.settlement_days == 2
        assert BUND.frequency == Frequency.ANNUAL

    def test_all_markets_exist(self):
        assert len(CONVENTIONS) == 6

from pricebook.schedule import Frequency

class TestBenchmarkUniverse:
    def test_create_ust(self):
        u = create_ust_universe(REF, UST_YIELDS)
        assert len(u.bonds) == 7
        assert u.market == "UST"

    def test_yields_dict(self):
        u = create_ust_universe(REF, UST_YIELDS)
        y = u.yields()
        assert y[10] == 0.0390

    def test_bond_at(self):
        u = create_ust_universe(REF, UST_YIELDS)
        b = u.bond_at(10)
        assert b is not None
        assert b.tenor == 10

    def test_bund_universe(self):
        bund_yields = {2: 0.028, 5: 0.024, 10: 0.025, 30: 0.027}
        u = create_bund_universe(REF, bund_yields)
        assert u.convention.country == "DE"
        assert len(u.bonds) == 4

    def test_price_from_yield(self):
        u = create_ust_universe(REF, UST_YIELDS)
        b10 = u.bond_at(10)
        assert b10.market_price is not None
        assert 80 < b10.market_price < 120


class TestNSSFitting:
    def test_fit_converges(self):
        u = create_ust_universe(REF, UST_YIELDS)
        nss = fitted_curve_nss(u)
        assert nss.beta0 > 0
        # Fitted yields should be close to market
        for b in u.bonds:
            fitted = nss.yield_at(b.tenor)
            assert abs(fitted - b.market_yield) < 0.005  # within 50bp

    def test_yield_at_interpolation(self):
        u = create_ust_universe(REF, UST_YIELDS)
        nss = fitted_curve_nss(u)
        y15 = nss.yield_at(15.0)
        assert 0.03 < y15 < 0.05


class TestTradingStrategies:
    def test_spread_trade(self):
        u = create_ust_universe(REF, UST_YIELDS)
        curve = _curve()
        r = duration_neutral_spread(u, 2, 10, curve)
        assert r.notional_ratio > 0
        assert r.spread_bps != 0

    def test_butterfly(self):
        u = create_ust_universe(REF, UST_YIELDS)
        curve = _curve()
        r = butterfly_trade(u, 2, 5, 10, curve)
        assert r.belly_weight == 1.0
        assert r.wing_weights[0] > 0

    def test_barbell_vs_bullet(self):
        u = create_ust_universe(REF, UST_YIELDS)
        curve = _curve()
        r = barbell_vs_bullet(u, 2, 5, 10, curve)
        assert r.barbell_convexity > r.bullet_convexity  # barbell has more convexity


class TestRankings:
    def test_carry_ranking(self):
        u = create_ust_universe(REF, UST_YIELDS)
        curve = _curve()
        ranks = carry_ranking(u, curve, repo_rate=0.0520)
        assert len(ranks) > 0
        # Sorted by score descending
        for i in range(len(ranks) - 1):
            assert ranks[i].score >= ranks[i+1].score

    def test_rv_scorecard(self):
        u = create_ust_universe(REF, UST_YIELDS)
        curve = _curve()
        nss = fitted_curve_nss(u)
        scores = rv_scorecard(u, nss, curve, repo_rate=0.0520)
        assert len(scores) > 0
        for s in scores:
            assert "rich_cheap_bps" in s.details
