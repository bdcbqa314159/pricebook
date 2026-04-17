"""Tests for advanced repo: curve, specials, financing optimisation."""

import math

import numpy as np
import pytest

from pricebook.repo_advanced import (
    FinancingPlan,
    HaircutCurve,
    RepoCounterparty,
    RepoCurve,
    RepoOISSpread,
    SpecialBond,
    build_repo_curve,
    identify_specials,
    optimise_financing,
    repo_haircut_curve,
    repo_spread_to_ois,
)


# ---- Repo curve ----

class TestRepoCurve:
    def test_build(self):
        curve = build_repo_curve([1, 7, 30, 90], [5.00, 5.05, 5.10, 5.15])
        assert isinstance(curve, RepoCurve)
        assert curve.collateral_type == "GC"

    def test_rate_interpolation(self):
        curve = build_repo_curve([1, 30], [5.00, 5.10])
        # Interpolate at 15 days
        assert curve.rate_at(15) == pytest.approx(5.05, abs=0.01)

    def test_rate_extrapolation(self):
        curve = build_repo_curve([1, 30], [5.00, 5.10])
        assert curve.rate_at(100) == 5.10
        assert curve.rate_at(0) == 5.00

    def test_financing_cost(self):
        curve = build_repo_curve([30], [5.00])
        # 1M notional, 30 days at 5% → 1M × 0.05 × 30/360 = 4166.67
        cost = curve.financing_cost(1_000_000, 30)
        assert cost == pytest.approx(1_000_000 * 0.05 * 30 / 360)


# ---- Repo-OIS spread ----

class TestRepoOISSpread:
    def test_normal_regime(self):
        curve = build_repo_curve([1], [5.02])
        result = repo_spread_to_ois(curve, ois_rate_pct=5.00, tenor_days=1)
        assert isinstance(result, RepoOISSpread)
        assert result.spread_bps == pytest.approx(2)
        assert result.regime == "normal"

    def test_tightening_regime(self):
        curve = build_repo_curve([1], [5.15])
        result = repo_spread_to_ois(curve, 5.00, 1)
        assert result.spread_bps == pytest.approx(15)
        assert result.regime == "tightening"

    def test_stressed_regime(self):
        curve = build_repo_curve([1], [5.30])
        result = repo_spread_to_ois(curve, 5.00, 1)
        assert result.spread_bps == pytest.approx(30)
        assert result.regime == "stressed"


# ---- Specials ----

class TestIdentifySpecials:
    def test_basic(self):
        bonds = {
            "10Y_ontr": 4.50,   # 50 bps special
            "5Y_new": 4.95,     # 5 bps (not special)
            "10Y_ofr": 5.00,    # flat (not special)
            "2Y_squeeze": 3.80,  # 120 bps ultra-special
        }
        specials = identify_specials(gc_rate_pct=5.00, bond_repo_rates=bonds)
        assert len(specials) == 2   # 10Y_ontr and 2Y_squeeze
        # Sorted by specialness desc
        assert specials[0].bond_id == "2Y_squeeze"
        assert specials[0].is_ultra_special

    def test_no_specials(self):
        bonds = {"a": 4.99, "b": 5.00, "c": 5.01}
        specials = identify_specials(5.00, bonds)
        assert len(specials) == 0

    def test_custom_threshold(self):
        bonds = {"a": 4.97, "b": 4.93}
        specials = identify_specials(5.00, bonds, specialness_threshold_bps=5)
        assert len(specials) == 1   # only b has 7bp > 5bp
        assert specials[0].bond_id == "b"


# ---- Counterparty financing ----

class TestOptimiseFinancing:
    def test_basic(self):
        cps = [
            RepoCounterparty("bank_a", 5.00, max_capacity=500_000,
                              haircut_pct=2, credit_quality="AA"),
            RepoCounterparty("bank_b", 4.95, max_capacity=300_000,
                              haircut_pct=2, credit_quality="AA"),
            RepoCounterparty("bank_c", 5.10, max_capacity=1_000_000,
                              haircut_pct=2, credit_quality="A"),
        ]
        result = optimise_financing(1_000_000, cps, days=1)
        assert isinstance(result, FinancingPlan)
        # bank_b cheapest → fully used first
        assert result.counterparty_allocations["bank_b"] == 300_000
        assert result.counterparty_allocations["bank_a"] == 500_000
        assert result.counterparty_allocations["bank_c"] == 200_000

    def test_unmet_demand(self):
        cps = [
            RepoCounterparty("a", 5.00, 100_000, 2, "AA"),
        ]
        result = optimise_financing(1_000_000, cps, 1)
        assert result.total_notional == 100_000
        assert result.unmet_demand == 900_000

    def test_cheapest_first(self):
        """Ensure cheapest counterparty is always used first."""
        cps = [
            RepoCounterparty("expensive", 6.0, 500_000, 2, "AA"),
            RepoCounterparty("cheap", 4.5, 500_000, 2, "AA"),
        ]
        result = optimise_financing(500_000, cps, 1)
        assert result.counterparty_allocations["cheap"] == 500_000
        assert "expensive" not in result.counterparty_allocations or result.counterparty_allocations["expensive"] == 0

    def test_weighted_rate(self):
        cps = [
            RepoCounterparty("a", 5.00, 500_000, 2, "AA"),
            RepoCounterparty("b", 6.00, 500_000, 2, "AA"),
        ]
        result = optimise_financing(1_000_000, cps, 360)  # 1 year
        # Weighted rate: 50% at 5%, 50% at 6% → 5.5%
        assert result.avg_weighted_rate_pct == pytest.approx(5.5, abs=0.05)


# ---- Haircut curve ----

class TestHaircutCurve:
    def test_treasury(self):
        curve = repo_haircut_curve("treasury")
        assert isinstance(curve, HaircutCurve)
        assert curve.asset_type == "treasury"
        assert curve.haircut_at(30) < 5

    def test_hy_corp_higher_than_treasury(self):
        tsy = repo_haircut_curve("treasury")
        hy = repo_haircut_curve("hy_corp")
        assert hy.haircut_at(30) > tsy.haircut_at(30)

    def test_equity_highest(self):
        """Equities have the highest haircuts."""
        equity = repo_haircut_curve("equity")
        tsy = repo_haircut_curve("treasury")
        hy = repo_haircut_curve("hy_corp")
        assert equity.haircut_at(30) > hy.haircut_at(30) > tsy.haircut_at(30)

    def test_haircut_interpolation(self):
        curve = repo_haircut_curve("treasury")
        # Should interpolate between tenor points
        h_15 = curve.haircut_at(15)
        h_1 = curve.haircut_at(1)
        h_30 = curve.haircut_at(30)
        assert h_1 <= h_15 <= h_30
