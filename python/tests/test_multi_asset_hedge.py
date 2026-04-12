"""Tests for multi-asset hedging."""

import pytest

from pricebook.multi_asset_hedge import (
    HedgeAllocation,
    HedgeInstrument,
    HedgeRecommendation,
    HedgeResult,
    HedgeTarget,
    WhatIfResult,
    hedge_recommendation,
    hedge_residual,
    optimal_hedge,
    what_if_analysis,
)


def _targets():
    return [
        HedgeTarget("delta", 1000.0),
        HedgeTarget("gamma", 50.0),
        HedgeTarget("vega", 5000.0),
    ]


def _instruments():
    return [
        HedgeInstrument("stock", {"delta": 1.0, "gamma": 0.0, "vega": 0.0}),
        HedgeInstrument("atm_call", {"delta": 0.5, "gamma": 0.05, "vega": 120.0}),
        HedgeInstrument("otm_put", {"delta": -0.3, "gamma": 0.03, "vega": 80.0}),
    ]


# ---- Step 1: generalised hedge optimiser ----

class TestOptimalHedge:
    def test_hedged_near_zero(self):
        """Step 1 test: hedged book has near-zero for all targeted Greeks."""
        result = optimal_hedge(_targets(), _instruments())
        for greek, residual in result.residuals.items():
            assert abs(residual) < 1.0, f"{greek} residual {residual} too large"

    def test_two_targets_two_instruments(self):
        targets = [HedgeTarget("delta", 1000), HedgeTarget("vega", 5000)]
        instruments = [
            HedgeInstrument("stock", {"delta": 1.0, "vega": 0.0}),
            HedgeInstrument("option", {"delta": 0.5, "vega": 100.0}),
        ]
        result = optimal_hedge(targets, instruments)
        assert abs(result.residuals["delta"]) < 1e-6
        assert abs(result.residuals["vega"]) < 1e-6

    def test_overdetermined_reduces_risk(self):
        """More targets than instruments → can't zero all, but reduces."""
        targets = [
            HedgeTarget("delta", 1000), HedgeTarget("gamma", 50),
            HedgeTarget("vega", 5000), HedgeTarget("vanna", 200),
            HedgeTarget("volga", 100),
        ]
        instruments = [
            HedgeInstrument("option", {"delta": 0.5, "gamma": 0.05,
                                        "vega": 100, "vanna": 5, "volga": 3}),
        ]
        result = optimal_hedge(targets, instruments)
        assert result.max_residual < max(abs(t.exposure) for t in targets)

    def test_empty_instruments(self):
        result = optimal_hedge(_targets(), [])
        assert result.residuals["delta"] == pytest.approx(1000.0)

    def test_empty_targets(self):
        result = optimal_hedge([], _instruments())
        assert result.allocations == []

    def test_cost_penalty(self):
        targets = [HedgeTarget("delta", 1000)]
        inst_cheap = HedgeInstrument("cheap", {"delta": 1.0}, cost_per_unit=0.01)
        inst_expensive = HedgeInstrument("pricey", {"delta": 1.0}, cost_per_unit=10.0)
        result = optimal_hedge(targets, [inst_cheap, inst_expensive], cost_penalty=1.0)
        # With cost penalty, should prefer the cheap instrument
        cheap_alloc = next(a for a in result.allocations if a.instrument.name == "cheap")
        pricey_alloc = next(a for a in result.allocations if a.instrument.name == "pricey")
        assert abs(cheap_alloc.quantity) > abs(pricey_alloc.quantity)


# ---- Hedge residual ----

class TestHedgeResidual:
    def test_no_allocations(self):
        residuals = hedge_residual(_targets(), [])
        assert residuals["delta"] == pytest.approx(1000.0)

    def test_perfect_hedge(self):
        targets = [HedgeTarget("delta", 1000)]
        allocs = [HedgeAllocation(HedgeInstrument("s", {"delta": 1.0}), -1000)]
        residuals = hedge_residual(targets, allocs)
        assert residuals["delta"] == pytest.approx(0.0)


# ---- Step 2: what-if + recommendation ----

class TestWhatIfAnalysis:
    def test_adding_reduces_residual(self):
        targets = [HedgeTarget("delta", 1000)]
        current = []
        candidate = HedgeInstrument("stock", {"delta": 1.0})
        result = what_if_analysis(targets, current, candidate, -500, "add")
        assert result.improvement > 0  # adding a hedge reduces risk

    def test_removing_increases_residual(self):
        targets = [HedgeTarget("delta", 1000)]
        alloc = HedgeAllocation(HedgeInstrument("stock", {"delta": 1.0}), -1000)
        candidate = alloc.instrument
        result = what_if_analysis(targets, [alloc], candidate, 0, "remove")
        assert result.improvement < 0  # removing hedge worsens risk


class TestHedgeRecommendation:
    def test_residual_decreases(self):
        """Step 2 test: residual risk decreases after applying recommendation."""
        targets = _targets()
        instruments = _instruments()
        rec = hedge_recommendation(targets, instruments)
        assert rec.risk_reduction_pct > 0
        assert rec.hedge.max_residual < max(abs(t.exposure) for t in targets)

    def test_reports_largest_residual(self):
        targets = _targets()
        rec = hedge_recommendation(targets, _instruments())
        assert rec.largest_residual_greek in {"delta", "gamma", "vega"}

    def test_empty(self):
        rec = hedge_recommendation([], [])
        assert rec.risk_reduction_pct == 100.0
