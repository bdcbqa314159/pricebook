"""Tests for bond FRTB SA capital wiring."""

import pytest

from pricebook.bond_capital import (
    BondCapitalReport,
    BondRiskInputs,
    bond_frtb_capital,
    bond_to_frtb_positions,
)
from pricebook.regulatory.market_risk_sa import calculate_frtb_sa


# ---- Step 1: wiring + manual calc ----

class TestBondToFrtbPositions:
    def test_girr_positions(self):
        inputs = [BondRiskInputs(
            issuer="UST", currency="USD", ir_sensitivity=1_000,
            notional=10_000_000,
        )]
        result = bond_to_frtb_positions(inputs)
        assert "GIRR" in result["delta_positions"]
        assert len(result["delta_positions"]["GIRR"]) == 1

    def test_csr_positions(self):
        inputs = [BondRiskInputs(
            issuer="AAPL", sector="corporate_IG", cs_sensitivity=500,
            notional=5_000_000,
        )]
        result = bond_to_frtb_positions(inputs)
        assert "CSR" in result["delta_positions"]
        assert len(result["delta_positions"]["CSR"]) == 1

    def test_drc_positions(self):
        inputs = [BondRiskInputs(
            issuer="AAPL", rating="A", seniority="senior",
            notional=10_000_000, is_long=True,
        )]
        result = bond_to_frtb_positions(inputs)
        assert len(result["drc_positions"]) == 1
        assert result["drc_positions"][0]["seniority"] == "senior"

    def test_short_position_drc(self):
        inputs = [BondRiskInputs(
            issuer="AAPL", notional=10_000_000, is_long=False,
        )]
        result = bond_to_frtb_positions(inputs)
        assert result["drc_positions"][0]["is_long"] is False

    def test_zero_sensitivity_excluded(self):
        inputs = [BondRiskInputs(
            issuer="X", ir_sensitivity=0, cs_sensitivity=0,
            notional=10_000_000,
        )]
        result = bond_to_frtb_positions(inputs)
        assert "GIRR" not in result["delta_positions"]
        assert "CSR" not in result["delta_positions"]


class TestManualCalculation:
    """Step 1 test: capital matches manual calculation."""

    def test_girr_only(self):
        inputs = [BondRiskInputs(
            issuer="UST", currency="USD",
            ir_sensitivity=1_000_000, notional=0,
        )]
        report = bond_frtb_capital(inputs)
        # GIRR: sensitivity=1M, RW=11% → weighted=110K
        # Single bucket → K=110K
        assert report.girr_capital == pytest.approx(110_000)
        assert report.csr_capital == 0.0
        assert report.drc_capital == 0.0

    def test_drc_senior_bbb(self):
        inputs = [BondRiskInputs(
            issuer="ACME", rating="BBB", seniority="senior",
            notional=10_000_000, is_long=True,
        )]
        report = bond_frtb_capital(inputs)
        # DRC: 10M × LGD(senior)=0.75 × RW(BBB)=2% = 150K
        assert report.drc_capital == pytest.approx(150_000)

    def test_combined_girr_csr_drc(self):
        inputs = [BondRiskInputs(
            issuer="AAPL", currency="USD", sector="corporate_IG",
            rating="A", seniority="senior",
            ir_sensitivity=500_000, cs_sensitivity=300_000,
            notional=10_000_000,
        )]
        report = bond_frtb_capital(inputs)
        assert report.girr_capital > 0
        assert report.csr_capital > 0
        assert report.drc_capital > 0
        assert report.total_capital == pytest.approx(
            report.girr_capital + report.csr_capital + report.drc_capital
        )


# ---- Step 2: capital report ----

class TestBondCapitalReport:
    def _inputs(self):
        return [
            BondRiskInputs("UST", "USD", "sovereign_IG", "AAA", "senior",
                           ir_sensitivity=1_000_000, notional=50_000_000),
            BondRiskInputs("AAPL", "USD", "corporate_IG", "AA", "senior",
                           ir_sensitivity=500_000, cs_sensitivity=300_000,
                           notional=20_000_000),
        ]

    def test_n_bonds(self):
        report = bond_frtb_capital(self._inputs())
        assert report.n_bonds == 2

    def test_total_notional(self):
        report = bond_frtb_capital(self._inputs())
        assert report.total_notional == pytest.approx(70_000_000)

    def test_capital_efficiency(self):
        report = bond_frtb_capital(self._inputs())
        assert report.capital_efficiency == pytest.approx(
            report.total_capital / report.total_notional
        )

    def test_total_rwa(self):
        report = bond_frtb_capital(self._inputs())
        assert report.total_rwa == pytest.approx(report.total_capital * 12.5)

    def test_sbm_property(self):
        report = bond_frtb_capital(self._inputs())
        assert report.sbm_capital == pytest.approx(
            report.girr_capital + report.csr_capital
        )

    def test_matches_calculate_frtb_sa(self):
        """Step 2 test: sums match calculate_frtb_sa."""
        inputs = self._inputs()
        positions = bond_to_frtb_positions(inputs)
        direct = calculate_frtb_sa(**positions)
        report = bond_frtb_capital(inputs)
        assert report.total_capital == pytest.approx(direct["total_capital"])
        assert report.total_rwa == pytest.approx(direct["total_rwa"])
