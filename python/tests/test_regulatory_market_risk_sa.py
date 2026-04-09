"""Tests for FRTB Standardised Approach."""

import pytest

from pricebook.regulatory.market_risk_sa import (
    aggregate_within_bucket, aggregate_across_buckets,
    calculate_delta_capital, calculate_vega_capital, calculate_curvature_capital,
    calculate_sbm_capital, calculate_drc_charge, calculate_rrao, calculate_frtb_sa,
    GIRR_RISK_WEIGHTS, DRC_RISK_WEIGHTS, DRC_LGD,
)


# ---- Aggregation ----

class TestAggregation:
    def test_single_sensitivity(self):
        assert aggregate_within_bucket([100]) == 100

    def test_perfect_correlation(self):
        k = aggregate_within_bucket([100, 200], 1.0)
        assert k == pytest.approx(300)

    def test_zero_correlation(self):
        import math
        k = aggregate_within_bucket([100, 200], 0.0)
        assert k == pytest.approx(math.sqrt(100**2 + 200**2))

    def test_diversification(self):
        """Partial correlation → K < sum of absolute."""
        k = aggregate_within_bucket([100, -50], 0.5)
        assert k < 150

    def test_across_buckets(self):
        caps = {"A": 100, "B": 200}
        nets = {"A": 80, "B": 150}
        k = aggregate_across_buckets(caps, nets, 0.25)
        assert k > 0


# ---- Delta capital ----

class TestDeltaCapital:
    def test_single_position(self):
        pos = [{"bucket": "default", "sensitivity": 1_000_000, "risk_weight": 20}]
        r = calculate_delta_capital(pos, "EQ")
        assert r["capital"] > 0

    def test_offsetting_reduces_girr(self):
        """Perfect offset in GIRR (high intra-corr 0.99) → near-zero capital."""
        long_only = [{"bucket": "USD", "sensitivity": 1_000_000, "risk_weight": 11}]
        hedged = [
            {"bucket": "USD", "sensitivity": 1_000_000, "risk_weight": 11},
            {"bucket": "USD", "sensitivity": -1_000_000, "risk_weight": 11},
        ]
        r1 = calculate_delta_capital(long_only, "GIRR")
        r2 = calculate_delta_capital(hedged, "GIRR")
        assert r2["capital"] < r1["capital"]

    def test_all_risk_classes(self):
        pos = [{"bucket": "default", "sensitivity": 500_000, "risk_weight": 10}]
        for rc in ["GIRR", "CSR", "EQ", "FX", "COM"]:
            r = calculate_delta_capital(pos, rc)
            assert r["capital"] > 0
            assert r["risk_class"] == rc

    def test_multiple_buckets(self):
        pos = [
            {"bucket": "A", "sensitivity": 1_000_000, "risk_weight": 20},
            {"bucket": "B", "sensitivity": 500_000, "risk_weight": 30},
        ]
        r = calculate_delta_capital(pos, "EQ")
        assert len(r["bucket_capitals"]) == 2


# ---- Vega capital ----

class TestVegaCapital:
    def test_positive(self):
        pos = [{"bucket": "default", "vega": 100_000, "vega_risk_weight": 50}]
        r = calculate_vega_capital(pos, "EQ")
        assert r["capital"] > 0


# ---- Curvature capital ----

class TestCurvatureCapital:
    def test_positive(self):
        pos = [{"bucket": "default", "cvr_up": 50_000, "cvr_down": 30_000}]
        r = calculate_curvature_capital(pos, "EQ")
        assert r["capital"] == 50_000

    def test_zero_when_both_negative(self):
        pos = [{"bucket": "default", "cvr_up": -10_000, "cvr_down": -20_000}]
        r = calculate_curvature_capital(pos, "EQ")
        assert r["capital"] == 0


# ---- SbM total ----

class TestSbM:
    def test_total_is_sum(self):
        delta = [{"bucket": "A", "sensitivity": 1_000_000, "risk_weight": 20}]
        vega = [{"bucket": "A", "vega": 50_000, "vega_risk_weight": 50}]
        r = calculate_sbm_capital(delta, vega, risk_class="EQ")
        assert r["total_capital"] == pytest.approx(
            r["delta_capital"] + r["vega_capital"] + r["curvature_capital"]
        )


# ---- DRC ----

class TestDRC:
    def test_single_obligor(self):
        pos = [{"obligor": "A", "notional": 10_000_000, "rating": "BBB", "seniority": "senior", "is_long": True}]
        r = calculate_drc_charge(pos)
        # JTD = 10M × 0.75 = 7.5M, RW = 2/100 = 0.02, DRC = 7.5M × 0.02 = 150K
        assert r["total_drc"] == pytest.approx(150_000)

    def test_netting_within_obligor(self):
        pos = [
            {"obligor": "A", "notional": 10_000_000, "rating": "BBB", "seniority": "senior", "is_long": True},
            {"obligor": "A", "notional": 10_000_000, "rating": "BBB", "seniority": "senior", "is_long": False},
        ]
        r = calculate_drc_charge(pos)
        assert r["total_drc"] == pytest.approx(0.0)

    def test_higher_rating_lower_drc(self):
        pos_a = [{"obligor": "X", "notional": 10_000_000, "rating": "A", "seniority": "senior", "is_long": True}]
        pos_bb = [{"obligor": "Y", "notional": 10_000_000, "rating": "BB", "seniority": "senior", "is_long": True}]
        r_a = calculate_drc_charge(pos_a)
        r_bb = calculate_drc_charge(pos_bb)
        assert r_a["total_drc"] < r_bb["total_drc"]

    def test_rwa_is_12_5x(self):
        pos = [{"obligor": "A", "notional": 10_000_000, "rating": "A", "seniority": "senior", "is_long": True}]
        r = calculate_drc_charge(pos)
        assert r["rwa"] == pytest.approx(r["total_drc"] * 12.5)


# ---- RRAO ----

class TestRRAO:
    def test_exotic(self):
        pos = [{"notional": 10_000_000, "is_exotic": True}]
        r = calculate_rrao(pos)
        assert r["total_rrao"] == pytest.approx(100_000)  # 1%

    def test_other(self):
        pos = [{"notional": 10_000_000, "has_other_residual_risk": True}]
        r = calculate_rrao(pos)
        assert r["total_rrao"] == pytest.approx(10_000)  # 0.1%

    def test_empty(self):
        r = calculate_rrao([])
        assert r["total_rrao"] == 0


# ---- Full FRTB-SA ----

class TestFRTBSA:
    def test_total_is_sum(self):
        delta = {"EQ": [{"bucket": "A", "sensitivity": 1_000_000, "risk_weight": 20}]}
        drc = [{"obligor": "X", "notional": 5_000_000, "rating": "BBB", "seniority": "senior", "is_long": True}]
        rrao = [{"notional": 10_000_000, "is_exotic": True}]
        r = calculate_frtb_sa(delta, drc_positions=drc, rrao_positions=rrao)
        assert r["total_capital"] == pytest.approx(
            r["sbm_capital"] + r["drc_capital"] + r["rrao_capital"]
        )

    def test_multi_risk_class(self):
        delta = {
            "EQ": [{"bucket": "A", "sensitivity": 1_000_000, "risk_weight": 20}],
            "GIRR": [{"bucket": "1Y", "sensitivity": 500_000, "risk_weight": 16}],
        }
        r = calculate_frtb_sa(delta)
        assert len(r["sbm_by_risk_class"]) == 2
        assert r["total_capital"] > 0

    def test_rwa_is_12_5x(self):
        delta = {"EQ": [{"bucket": "A", "sensitivity": 1_000_000, "risk_weight": 20}]}
        r = calculate_frtb_sa(delta)
        assert r["total_rwa"] == pytest.approx(r["total_capital"] * 12.5)
