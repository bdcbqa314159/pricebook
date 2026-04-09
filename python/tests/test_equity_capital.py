"""Tests for equity FRTB SA capital wiring."""

import math

import pytest

from pricebook.equity_capital import (
    EquityCapitalReport,
    EquityClassification,
    EquityRiskInputs,
    equity_frtb_capital,
    equity_to_frtb_positions,
    map_to_frtb_bucket,
)
from pricebook.regulatory.market_risk_sa import (
    EQ_RISK_WEIGHTS,
    calculate_frtb_sa,
)


# ---- Step 1: bucket mapping + wiring ----

class TestBucketMapping:
    def test_large_developed(self):
        b = map_to_frtb_bucket(market_cap=200_000_000_000, region="US")
        assert b == "large_cap_developed"

    def test_large_emerging(self):
        b = map_to_frtb_bucket(market_cap=50_000_000_000, region="brazil")
        assert b == "large_cap_emerging"

    def test_small_developed(self):
        b = map_to_frtb_bucket(market_cap=500_000_000, region="europe")
        assert b == "small_cap_developed"

    def test_small_emerging(self):
        b = map_to_frtb_bucket(market_cap=100_000_000, region="india")
        assert b == "small_cap_emerging"

    def test_threshold_boundary(self):
        # 2bn exactly is "large"
        assert map_to_frtb_bucket(2_000_000_000, "US") == "large_cap_developed"
        # Just below
        assert map_to_frtb_bucket(1_999_999_999, "US") == "small_cap_developed"

    def test_case_insensitive_region(self):
        assert map_to_frtb_bucket(10e9, "DEVELOPED") == "large_cap_developed"
        assert map_to_frtb_bucket(10e9, "Europe") == "large_cap_developed"


class TestEquityToFrtbPositions:
    def test_single_name_structure(self):
        inputs = [EquityRiskInputs(
            ticker="AAPL", delta=1_000_000, vega=10_000,
            cvr_up=5_000, cvr_down=3_000, notional=10_000_000,
        )]
        classifications = {"AAPL": EquityClassification(
            ticker="AAPL", market_cap=3_000_000_000_000, region="US",
            rating="AA",
        )}
        result = equity_to_frtb_positions(inputs, classifications)
        assert "EQ" in result["delta_positions"]
        assert len(result["delta_positions"]["EQ"]) == 1
        assert result["delta_positions"]["EQ"][0]["bucket"] == "large_cap_developed"
        assert result["delta_positions"]["EQ"][0]["risk_weight"] == 20  # large_cap_developed
        assert len(result["vega_positions"]["EQ"]) == 1
        assert len(result["curvature_positions"]["EQ"]) == 1
        assert len(result["drc_positions"]) == 1
        assert result["drc_positions"][0]["seniority"] == "equity"

    def test_zero_vega_and_curvature_omitted(self):
        inputs = [EquityRiskInputs(
            ticker="AAPL", delta=1_000_000, notional=10_000_000,
        )]
        classifications = {"AAPL": EquityClassification(
            "AAPL", 3e12, "US",
        )}
        result = equity_to_frtb_positions(inputs, classifications)
        assert len(result["vega_positions"]["EQ"]) == 0
        assert len(result["curvature_positions"]["EQ"]) == 0

    def test_zero_notional_no_drc(self):
        inputs = [EquityRiskInputs(
            ticker="AAPL", delta=1_000_000, notional=0.0,
        )]
        classifications = {"AAPL": EquityClassification("AAPL", 3e12, "US")}
        result = equity_to_frtb_positions(inputs, classifications)
        assert len(result["drc_positions"]) == 0

    def test_exotic_creates_rrao(self):
        inputs = [EquityRiskInputs(ticker="X", delta=100_000, notional=1_000_000)]
        cls = {"X": EquityClassification("X", 3e12, "US", is_exotic=True)}
        result = equity_to_frtb_positions(inputs, cls)
        assert len(result["rrao_positions"]) == 1
        assert result["rrao_positions"][0]["is_exotic"] is True

    def test_missing_classification_falls_back_to_other(self):
        inputs = [EquityRiskInputs("UNKNOWN", delta=1_000_000)]
        result = equity_to_frtb_positions(inputs, {})
        assert result["delta_positions"]["EQ"][0]["bucket"] == "other"


class TestManualCalculation:
    """Slice 140 step 1 test: FRTB SA capital matches manual calculation."""

    def test_single_name_delta_only(self):
        # 1M delta, large-cap developed → RW = 20%
        inputs = [EquityRiskInputs(
            ticker="AAPL", delta=1_000_000, notional=0.0,
        )]
        cls = {"AAPL": EquityClassification("AAPL", 3e12, "US", rating="AA")}
        report = equity_frtb_capital(inputs, cls)

        # Manual: weighted = 1M × 20/100 = 200K
        # Single bucket → K_b = 200K, total = 200K
        # No DRC (notional = 0), no vega, no curvature, no RRAO
        assert report.delta_capital == pytest.approx(200_000.0)
        assert report.vega_capital == 0.0
        assert report.curvature_capital == 0.0
        assert report.drc_capital == 0.0
        assert report.rrao_capital == 0.0
        assert report.total_capital == pytest.approx(200_000.0)

    def test_single_name_with_drc(self):
        # 1M delta + 10M notional, BBB rating → DRC = 10M × 1.0 × 0.02 = 200K
        inputs = [EquityRiskInputs(
            ticker="X", delta=1_000_000, notional=10_000_000,
        )]
        cls = {"X": EquityClassification("X", 3e12, "US", rating="BBB")}
        report = equity_frtb_capital(inputs, cls)

        # Delta: 1M × 0.20 = 200K
        # DRC: 10M × LGD(equity)=1.0 × RW(BBB)=2/100 = 200K
        assert report.delta_capital == pytest.approx(200_000.0)
        assert report.drc_capital == pytest.approx(200_000.0)
        assert report.total_capital == pytest.approx(400_000.0)

    def test_multi_name_intra_bucket_correlation(self):
        # Two names in same bucket, both long → some diversification benefit
        inputs = [
            EquityRiskInputs("A", delta=1_000_000),
            EquityRiskInputs("B", delta=1_000_000),
        ]
        cls = {
            "A": EquityClassification("A", 3e12, "US"),
            "B": EquityClassification("B", 3e12, "US"),
        }
        report = equity_frtb_capital(inputs, cls)
        # Each weighted = 200K
        # K_b = sqrt(200K² + 200K² + 0.20 × ((400K)² - (200K² + 200K²)))
        #     = sqrt(2·40e9 + 0.20 × (16e10 - 8e10))
        #     = sqrt(8e10 + 0.20 × 8e10)
        #     = sqrt(9.6e10)
        expected = math.sqrt(9.6e10)
        assert report.delta_capital == pytest.approx(expected)


# ---- Step 2: capital report aggregation ----

class TestEquityCapitalReport:
    def _book(self):
        inputs = [
            EquityRiskInputs("AAPL", delta=2_000_000, vega=10_000,
                             notional=20_000_000),
            EquityRiskInputs("MSFT", delta=1_500_000, vega=8_000,
                             notional=15_000_000),
            EquityRiskInputs("EM_NAME", delta=500_000, vega=2_000,
                             notional=5_000_000),
        ]
        classifications = {
            "AAPL": EquityClassification("AAPL", 3e12, "US", rating="AA"),
            "MSFT": EquityClassification("MSFT", 3e12, "US", rating="AAA"),
            "EM_NAME": EquityClassification(
                "EM_NAME", 5e9, "brazil", rating="BB",
            ),
        }
        return inputs, classifications

    def test_report_n_names(self):
        inputs, cls = self._book()
        report = equity_frtb_capital(inputs, cls)
        assert report.n_names == 3

    def test_total_notional(self):
        inputs, cls = self._book()
        report = equity_frtb_capital(inputs, cls)
        assert report.total_notional == pytest.approx(40_000_000)

    def test_capital_efficiency(self):
        inputs, cls = self._book()
        report = equity_frtb_capital(inputs, cls)
        assert report.capital_efficiency == pytest.approx(
            report.total_capital / report.total_notional
        )

    def test_total_rwa(self):
        inputs, cls = self._book()
        report = equity_frtb_capital(inputs, cls)
        assert report.total_rwa == pytest.approx(report.total_capital * 12.5)

    def test_zero_notional_zero_efficiency(self):
        inputs = [EquityRiskInputs("X", delta=100_000)]
        cls = {"X": EquityClassification("X", 3e12, "US")}
        report = equity_frtb_capital(inputs, cls)
        assert report.total_notional == 0.0
        assert report.capital_efficiency == 0.0

    def test_sbm_components_sum(self):
        inputs, cls = self._book()
        report = equity_frtb_capital(inputs, cls)
        # SbM total = delta + vega + curvature
        assert report.sbm_capital == pytest.approx(report.sbm_components_sum)

    def test_total_matches_calculate_frtb_sa(self):
        """Slice 140 step 2 test: capital sums match calculate_frtb_sa."""
        inputs, cls = self._book()
        positions = equity_to_frtb_positions(inputs, cls)
        direct = calculate_frtb_sa(**positions)
        report = equity_frtb_capital(inputs, cls)
        assert report.total_capital == pytest.approx(direct["total_capital"])
        assert report.total_rwa == pytest.approx(direct["total_rwa"])

    def test_components_match_calculate_frtb_sa(self):
        inputs, cls = self._book()
        positions = equity_to_frtb_positions(inputs, cls)
        direct = calculate_frtb_sa(**positions)
        report = equity_frtb_capital(inputs, cls)
        eq = direct["sbm_by_risk_class"]["EQ"]
        assert report.delta_capital == pytest.approx(eq["delta_capital"])
        assert report.vega_capital == pytest.approx(eq["vega_capital"])
        assert report.curvature_capital == pytest.approx(eq["curvature_capital"])
        assert report.drc_capital == pytest.approx(direct["drc_capital"])
        assert report.rrao_capital == pytest.approx(direct["rrao_capital"])

    def test_bucket_capitals_populated(self):
        inputs, cls = self._book()
        report = equity_frtb_capital(inputs, cls)
        # Two distinct buckets used: large_cap_developed (AAPL, MSFT)
        # and large_cap_emerging (EM_NAME, $5bn)
        assert "large_cap_developed" in report.bucket_capitals
        assert "large_cap_emerging" in report.bucket_capitals
