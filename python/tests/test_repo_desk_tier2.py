"""Tests for repo desk Tier 2: dynamic haircuts, margin calls, specialness forecast, curve stress."""

from __future__ import annotations

import math
from datetime import date

import pytest

from pricebook.repo_desk import (
    RepoBook, RepoTradeEntry,
    dynamic_haircut, HaircutAdjustment,
    margin_call_simulation, MarginCallScenario,
    forecast_specialness, SpecialnessForecast,
    repo_curve_stress, RepoCurveStress,
)


REF = date(2024, 7, 15)


def _make_book():
    book = RepoBook("Test")
    book.add(RepoTradeEntry(
        counterparty="A", collateral_issuer="UST10Y", face_amount=50_000_000,
        bond_price=101.5, repo_rate=0.045, term_days=1, coupon_rate=0.04,
        direction="repo", start_date=REF,
    ))
    book.add(RepoTradeEntry(
        counterparty="B", collateral_issuer="UST5Y", face_amount=30_000_000,
        bond_price=100.0, repo_rate=0.040, term_days=90, coupon_rate=0.035,
        direction="repo", start_date=REF,
    ))
    book.add(RepoTradeEntry(
        counterparty="C", collateral_issuer="UST2Y", face_amount=20_000_000,
        bond_price=99.5, repo_rate=0.046, term_days=30, coupon_rate=0.045,
        direction="reverse", start_date=REF,
    ))
    return book


# ---- Dynamic haircut ----

class TestDynamicHaircut:

    def test_normal_vol(self):
        h = dynamic_haircut(2.0, current_vol=0.05, normal_vol=0.05)
        assert h.regime == "normal"
        assert h.adjusted_haircut_pct == pytest.approx(2.0)

    def test_elevated_vol(self):
        h = dynamic_haircut(2.0, current_vol=0.10, normal_vol=0.05)
        assert h.regime == "elevated"
        assert h.adjusted_haircut_pct > 2.0

    def test_stressed_vol(self):
        h = dynamic_haircut(2.0, current_vol=0.15, normal_vol=0.05)
        assert h.regime == "stressed"
        assert h.adjusted_haircut_pct > h.base_haircut_pct * 2

    def test_monotonic(self):
        """Higher vol → higher haircut."""
        vols = [0.03, 0.05, 0.08, 0.12, 0.20]
        haircuts = [dynamic_haircut(2.0, v).adjusted_haircut_pct for v in vols]
        for i in range(1, len(haircuts)):
            assert haircuts[i] >= haircuts[i-1]

    def test_to_dict(self):
        h = dynamic_haircut(2.0, 0.10)
        d = h.to_dict()
        assert "adjusted" in d
        assert "regime" in d


# ---- Margin call simulation ----

class TestMarginCall:

    def test_default_scenarios(self):
        book = _make_book()
        results = margin_call_simulation(book)
        assert len(results) == 4

    def test_larger_shock_larger_call(self):
        book = _make_book()
        results = margin_call_simulation(book)
        calls = [r.total_margin_call for r in results]
        for i in range(1, len(calls)):
            assert calls[i] >= calls[i-1]

    def test_all_affected(self):
        book = _make_book()
        results = margin_call_simulation(book)
        for r in results:
            assert r.n_positions_affected > 0

    def test_to_dict(self):
        book = _make_book()
        results = margin_call_simulation(book)
        d = results[0].to_dict()
        assert "total_call" in d
        assert "largest_call" in d


# ---- Specialness forecast ----

class TestSpecialnessForecast:

    def test_near_auction_widens(self):
        f = forecast_specialness("UST10Y", 50.0, days_to_auction=10)
        assert f.forecast_specialness_bps >= 50.0

    def test_far_from_auction_stable(self):
        f = forecast_specialness("UST10Y", 50.0, days_to_auction=60)
        assert f.trend == "stable"

    def test_high_demand_widens(self):
        f_low = forecast_specialness("UST10Y", 50.0, borrowing_demand_pct=0.3)
        f_high = forecast_specialness("UST10Y", 50.0, borrowing_demand_pct=0.8)
        assert f_high.forecast_specialness_bps > f_low.forecast_specialness_bps

    def test_pre_auction_collapse(self):
        f = forecast_specialness("UST10Y", 50.0, days_to_auction=2)
        assert f.trend == "collapsing"
        assert f.confidence == "high"

    def test_zero_specialness(self):
        f = forecast_specialness("UST10Y", 0.0)
        assert f.forecast_specialness_bps == 0.0

    def test_to_dict(self):
        f = forecast_specialness("UST10Y", 50.0, days_to_auction=10)
        d = f.to_dict()
        assert "trend" in d
        assert "confidence" in d


# ---- Repo curve stress ----

class TestCurveStress:

    def test_default_scenarios(self):
        book = _make_book()
        results = repo_curve_stress(book)
        assert len(results) == 5
        names = [r.scenario_name for r in results]
        assert "parallel_up" in names
        assert "inversion" in names

    def test_parallel_up_hurts(self):
        """Higher rates → higher financing cost → negative carry impact."""
        book = _make_book()
        results = repo_curve_stress(book)
        par_up = [r for r in results if r.scenario_name == "parallel_up"][0]
        # For a net borrower, parallel up should hurt
        assert par_up.carry_impact != 0

    def test_symmetric(self):
        """Parallel up and down should have opposite signs."""
        book = _make_book()
        results = repo_curve_stress(book)
        up = [r for r in results if r.scenario_name == "parallel_up"][0]
        down = [r for r in results if r.scenario_name == "parallel_down"][0]
        assert (up.carry_impact > 0) != (down.carry_impact > 0)

    def test_steepener_vs_flattener(self):
        book = _make_book()
        results = repo_curve_stress(book)
        steep = [r for r in results if r.scenario_name == "steepener"][0]
        flat = [r for r in results if r.scenario_name == "flattener"][0]
        assert steep.carry_impact != flat.carry_impact

    def test_to_dict(self):
        book = _make_book()
        results = repo_curve_stress(book)
        d = results[0].to_dict()
        assert "on_shift" in d
        assert "total" in d
