"""Tests for FRTB-IMA desk bridge."""

import pytest

from pricebook.regulatory.ima_bridge import (
    DeskRiskExtract, IMABridgeResult, RISK_CLASS_MAP,
    extract_risk_factors_from_desk, extract_drc_positions_from_desk,
    extract_from_risk_metrics, aggregate_desk_ima,
)
from pricebook.regulatory.market_risk_ima import ESRiskFactor, DRCPosition, DeskPLA


class TestRiskClassMap:
    def test_swap_is_ir(self):
        assert RISK_CLASS_MAP["swap"] == ("IR", "major")

    def test_equity_is_eq(self):
        assert RISK_CLASS_MAP["equity"] == ("EQ", "large_cap")

    def test_cds_is_cr(self):
        assert RISK_CLASS_MAP["cds"] == ("CR", "HY")


class TestExtractRiskFactors:
    def test_dv01_produces_es(self):
        extract = DeskRiskExtract("rates_1", "swap", "IR", "major", dv01=50_000)
        factors = extract_risk_factors_from_desk(extract, vol_of_returns=0.01)
        assert len(factors) >= 1
        assert all(isinstance(f, ESRiskFactor) for f in factors)
        assert factors[0].es_10day > 0
        assert factors[0].risk_class == "IR"

    def test_vega_produces_separate_factor(self):
        extract = DeskRiskExtract("eq_1", "equity", "EQ", "large_cap", delta=100_000, vega=50_000)
        factors = extract_risk_factors_from_desk(extract)
        assert len(factors) == 2  # delta + vega

    def test_cs01_produces_cr_factor(self):
        extract = DeskRiskExtract("cds_1", "cds", "CR", "HY", cs01=30_000, notional=10e6)
        factors = extract_risk_factors_from_desk(extract)
        cr_factors = [f for f in factors if f.risk_class == "CR"]
        assert len(cr_factors) >= 1

    def test_zero_sensitivity_no_factors(self):
        extract = DeskRiskExtract("empty", "swap", "IR", "major")
        factors = extract_risk_factors_from_desk(extract)
        assert len(factors) == 0

    def test_stressed_es_higher(self):
        extract = DeskRiskExtract("rates_1", "swap", "IR", "major", dv01=50_000)
        factors = extract_risk_factors_from_desk(extract, vol_of_returns=0.01)
        assert factors[0].stressed_es_10day > factors[0].es_10day


class TestExtractDRC:
    def test_credit_desk_produces_drc(self):
        extract = DeskRiskExtract("bond_1", "bond", "CR", "IG_corporate",
                                   notional=10_000_000, obligor="Apple")
        drc = extract_drc_positions_from_desk(extract)
        assert drc is not None
        assert isinstance(drc, DRCPosition)
        assert drc.obligor == "Apple"

    def test_non_credit_no_drc(self):
        extract = DeskRiskExtract("eq_1", "equity", "EQ", "large_cap", notional=10e6)
        drc = extract_drc_positions_from_desk(extract)
        assert drc is None

    def test_zero_notional_no_drc(self):
        extract = DeskRiskExtract("cds_1", "cds", "CR", "HY", notional=0)
        drc = extract_drc_positions_from_desk(extract)
        assert drc is None


class TestExtractFromMetrics:
    def test_swap_metrics(self):
        metrics = {"dv01": 50_000, "notional": 100_000_000, "pv": 500_000}
        extract = extract_from_risk_metrics("swap_desk", "swap", metrics)
        assert extract.risk_class == "IR"
        assert extract.dv01 == 50_000
        assert extract.notional == 100_000_000

    def test_equity_metrics(self):
        metrics = {"delta": 200_000, "gamma": 5_000, "vega": 80_000, "notional": 50_000_000}
        extract = extract_from_risk_metrics("eq_desk", "equity", metrics)
        assert extract.risk_class == "EQ"
        assert extract.delta == 200_000
        assert extract.vega == 80_000


class TestAggregation:
    def test_multi_desk(self):
        extracts = [
            DeskRiskExtract("rates", "swap", "IR", "major", dv01=50_000),
            DeskRiskExtract("credit", "cds", "CR", "HY", cs01=30_000, notional=10e6, obligor="IBM"),
            DeskRiskExtract("equity", "equity", "EQ", "large_cap", delta=100_000, vega=50_000),
        ]
        result = aggregate_desk_ima(extracts)
        assert isinstance(result, IMABridgeResult)
        assert result.n_risk_factors > 0
        assert result.n_drc_positions >= 1  # credit desk has DRC
        assert "total_ima_capital" in result.ima_capital or "imcc" in result.ima_capital

    def test_with_pla(self):
        extracts = [DeskRiskExtract("rates", "swap", "IR", "major", dv01=50_000)]
        pla = [DeskPLA("rates", spearman_correlation=0.85, kl_divergence=0.05)]
        result = aggregate_desk_ima(extracts, desk_pla=pla)
        assert result.pla_results is not None

    def test_to_dict(self):
        extracts = [DeskRiskExtract("rates", "swap", "IR", "major", dv01=50_000)]
        d = aggregate_desk_ima(extracts).to_dict()
        assert "ima_capital" in d
        assert "n_risk_factors" in d
