"""Tests for commodity FRTB SA capital wiring."""

import math

import pytest

from pricebook.commodity_capital import (
    CommodityCapitalReport,
    CommodityClassification,
    CommodityRiskInputs,
    commodity_frtb_capital,
    commodity_to_frtb_positions,
    map_to_com_bucket,
)
from pricebook.regulatory.market_risk_sa import (
    COM_RISK_WEIGHTS,
    calculate_frtb_sa,
)


# ---- Step 1: bucket mapping + wiring ----

class TestBucketMapping:
    def test_crude_is_energy_liquid(self):
        assert map_to_com_bucket("crude") == "energy_liquid"
        assert map_to_com_bucket("WTI") == "energy_liquid"
        assert map_to_com_bucket("Brent") == "energy_liquid"

    def test_natgas_is_energy_solid(self):
        assert map_to_com_bucket("natgas") == "energy_solid"
        assert map_to_com_bucket("coal") == "energy_solid"

    def test_power_is_electricity(self):
        assert map_to_com_bucket("power") == "energy_electricity"

    def test_gold_is_precious(self):
        assert map_to_com_bucket("gold") == "metals_precious"
        assert map_to_com_bucket("Silver") == "metals_precious"

    def test_copper_is_non_precious(self):
        assert map_to_com_bucket("copper") == "metals_non_precious"

    def test_wheat_is_grains(self):
        assert map_to_com_bucket("wheat") == "agriculture_grains"
        assert map_to_com_bucket("corn") == "agriculture_grains"

    def test_coffee_is_softs(self):
        assert map_to_com_bucket("coffee") == "agriculture_softs"

    def test_unknown_is_other(self):
        assert map_to_com_bucket("unobtainium") == "other"

    def test_case_insensitive(self):
        assert map_to_com_bucket("GOLD") == "metals_precious"
        assert map_to_com_bucket("  WTI  ") == "energy_liquid"


class TestCommodityToFrtbPositions:
    def test_single_commodity(self):
        inputs = [CommodityRiskInputs(
            commodity="WTI", delta=500_000, vega=5_000,
            cvr_up=2_000, cvr_down=1_000, notional=5_000_000,
        )]
        result = commodity_to_frtb_positions(inputs)
        assert "COM" in result["delta_positions"]
        assert len(result["delta_positions"]["COM"]) == 1
        assert result["delta_positions"]["COM"][0]["bucket"] == "energy_liquid"
        assert result["delta_positions"]["COM"][0]["risk_weight"] == 25
        assert len(result["vega_positions"]["COM"]) == 1
        assert len(result["curvature_positions"]["COM"]) == 1
        assert len(result["drc_positions"]) == 0  # no DRC for commodities

    def test_classification_overrides_auto_map(self):
        inputs = [CommodityRiskInputs("CUSTOM", delta=100_000)]
        cls = {"CUSTOM": CommodityClassification("CUSTOM", "freight")}
        result = commodity_to_frtb_positions(inputs, cls)
        assert result["delta_positions"]["COM"][0]["bucket"] == "freight"
        assert result["delta_positions"]["COM"][0]["risk_weight"] == 80

    def test_exotic_creates_rrao(self):
        inputs = [CommodityRiskInputs("X", delta=100_000, notional=1_000_000)]
        cls = {"X": CommodityClassification("X", "other", is_exotic=True)}
        result = commodity_to_frtb_positions(inputs, cls)
        assert len(result["rrao_positions"]) == 1


class TestManualCalculation:
    """Step 1 test: FRTB SA capital matches manual calculation."""

    def test_single_commodity_delta_only(self):
        # 500K delta, energy_liquid → RW = 25%
        inputs = [CommodityRiskInputs(commodity="WTI", delta=500_000)]
        report = commodity_frtb_capital(inputs)

        # Manual: weighted = 500K × 25/100 = 125K
        # Single bucket → K_b = 125K, total = 125K
        assert report.delta_capital == pytest.approx(125_000)
        assert report.vega_capital == 0.0
        assert report.curvature_capital == 0.0
        assert report.rrao_capital == 0.0
        assert report.total_capital == pytest.approx(125_000)

    def test_two_commodities_same_bucket(self):
        # Two in energy_liquid → intra-bucket correlation 0.55
        inputs = [
            CommodityRiskInputs("WTI", delta=500_000),
            CommodityRiskInputs("Brent", delta=500_000),
        ]
        report = commodity_frtb_capital(inputs)
        # Each weighted = 125K
        # K_b = sqrt(125K² + 125K² + 0.55 × ((250K)² − 2×125K²))
        #     = sqrt(2×15.625e9 + 0.55 × (62.5e9 − 31.25e9))
        #     = sqrt(31.25e9 + 0.55 × 31.25e9)
        #     = sqrt(31.25e9 × 1.55)
        expected = math.sqrt(31.25e9 * 1.55)
        assert report.delta_capital == pytest.approx(expected)

    def test_two_commodities_different_buckets(self):
        # energy_liquid (RW=25) and metals_precious (RW=20)
        inputs = [
            CommodityRiskInputs("WTI", delta=500_000),
            CommodityRiskInputs("gold", delta=500_000),
        ]
        report = commodity_frtb_capital(inputs)
        # Bucket A (energy_liquid): K_a = 500K × 0.25 = 125K
        # Bucket B (metals_precious): K_b = 500K × 0.20 = 100K
        # Cross: inter_corr = 0.20, S_a = 125K, S_b = 100K
        # Total = sqrt(125K² + 100K² + 2 × 0.20 × 125K × 100K)
        expected = math.sqrt(125_000**2 + 100_000**2 + 2 * 0.20 * 125_000 * 100_000)
        assert report.delta_capital == pytest.approx(expected)


# ---- Step 2: capital report ----

class TestCommodityCapitalReport:
    def _book(self):
        return [
            CommodityRiskInputs("WTI", delta=1_000_000, vega=10_000,
                                notional=20_000_000),
            CommodityRiskInputs("gold", delta=500_000, vega=5_000,
                                notional=10_000_000),
            CommodityRiskInputs("wheat", delta=200_000,
                                notional=5_000_000),
        ]

    def test_n_commodities(self):
        report = commodity_frtb_capital(self._book())
        assert report.n_commodities == 3

    def test_total_notional(self):
        report = commodity_frtb_capital(self._book())
        assert report.total_notional == pytest.approx(35_000_000)

    def test_capital_efficiency(self):
        report = commodity_frtb_capital(self._book())
        assert report.capital_efficiency == pytest.approx(
            report.total_capital / report.total_notional
        )

    def test_total_rwa(self):
        report = commodity_frtb_capital(self._book())
        assert report.total_rwa == pytest.approx(report.total_capital * 12.5)

    def test_sbm_components_sum(self):
        report = commodity_frtb_capital(self._book())
        assert report.sbm_capital == pytest.approx(report.sbm_components_sum)

    def test_total_matches_calculate_frtb_sa(self):
        inputs = self._book()
        positions = commodity_to_frtb_positions(inputs)
        direct = calculate_frtb_sa(**positions)
        report = commodity_frtb_capital(inputs)
        assert report.total_capital == pytest.approx(direct["total_capital"])
        assert report.total_rwa == pytest.approx(direct["total_rwa"])

    def test_bucket_capitals_populated(self):
        report = commodity_frtb_capital(self._book())
        assert "energy_liquid" in report.bucket_capitals
        assert "metals_precious" in report.bucket_capitals
        assert "agriculture_grains" in report.bucket_capitals
