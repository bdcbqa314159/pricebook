"""Tests for FX FRTB SA capital."""

import pytest

from pricebook.fx_capital import (
    FXCapitalReport,
    FXRiskInputs,
    fx_frtb_capital,
    fx_to_frtb_positions,
    is_liquid_pair,
)
from pricebook.regulatory.market_risk_sa import calculate_frtb_sa


class TestLiquidPairs:
    def test_eur_usd_liquid(self):
        assert is_liquid_pair("EUR/USD") is True

    def test_usd_jpy_liquid(self):
        assert is_liquid_pair("USD/JPY") is True

    def test_inverse_liquid(self):
        assert is_liquid_pair("JPY/USD") is True

    def test_exotic_not_liquid(self):
        assert is_liquid_pair("USD/BRL") is False

    def test_case_insensitive(self):
        assert is_liquid_pair("eur/usd") is True


class TestFXToFrtbPositions:
    def test_liquid_pair_rw(self):
        inputs = [FXRiskInputs("EUR/USD", delta=1_000_000)]
        result = fx_to_frtb_positions(inputs)
        assert result["delta_positions"]["FX"][0]["risk_weight"] == 11.25

    def test_illiquid_pair_rw(self):
        inputs = [FXRiskInputs("USD/BRL", delta=1_000_000)]
        result = fx_to_frtb_positions(inputs)
        assert result["delta_positions"]["FX"][0]["risk_weight"] == 15.0


class TestManualCalculation:
    def test_single_liquid_pair(self):
        inputs = [FXRiskInputs("EUR/USD", delta=1_000_000, notional=10_000_000)]
        report = fx_frtb_capital(inputs)
        # 1M × 11.25% = 112,500
        assert report.delta_capital == pytest.approx(112_500)

    def test_single_illiquid_pair(self):
        inputs = [FXRiskInputs("USD/BRL", delta=1_000_000, notional=10_000_000)]
        report = fx_frtb_capital(inputs)
        # 1M × 15% = 150,000
        assert report.delta_capital == pytest.approx(150_000)


class TestFXCapitalReport:
    def _inputs(self):
        return [
            FXRiskInputs("EUR/USD", delta=2_000_000, notional=20_000_000),
            FXRiskInputs("USD/BRL", delta=500_000, notional=5_000_000),
        ]

    def test_n_pairs(self):
        assert fx_frtb_capital(self._inputs()).n_pairs == 2

    def test_total_notional(self):
        assert fx_frtb_capital(self._inputs()).total_notional == pytest.approx(25_000_000)

    def test_capital_efficiency(self):
        report = fx_frtb_capital(self._inputs())
        assert report.capital_efficiency == pytest.approx(
            report.total_capital / report.total_notional
        )

    def test_total_rwa(self):
        report = fx_frtb_capital(self._inputs())
        assert report.total_rwa == pytest.approx(report.total_capital * 12.5)

    def test_matches_calculate_frtb_sa(self):
        inputs = self._inputs()
        positions = fx_to_frtb_positions(inputs)
        direct = calculate_frtb_sa(**positions)
        report = fx_frtb_capital(inputs)
        assert report.total_capital == pytest.approx(direct["total_capital"])
