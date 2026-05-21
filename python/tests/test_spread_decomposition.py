"""Tests for spread decomposition."""

import pytest

from pricebook.credit.spread_decomposition import (
    decompose_spread, SpreadComponents, cds_bond_basis, decompose_portfolio,
)


class TestDecomposeSpread:
    def test_basic(self):
        d = decompose_spread(250, cds_spread_bp=180, bid_ask_bp=8)
        assert d.total_spread_bp == 250
        assert d.credit_bp == 180
        assert d.liquidity_bp > 0

    def test_components_sum(self):
        d = decompose_spread(300, cds_spread_bp=200, bid_ask_bp=10)
        total = d.credit_bp + d.liquidity_bp + d.tax_bp + d.optionality_bp + d.residual_bp
        assert abs(total - d.total_spread_bp) < 0.01

    def test_no_cds_estimate(self):
        """Without CDS, credit ≈ 70% of spread."""
        d = decompose_spread(200)
        assert abs(d.credit_bp - 140) < 1

    def test_with_oas(self):
        """OAS provided → optionality = spread - OAS."""
        d = decompose_spread(250, cds_spread_bp=180, oas_bp=220)
        assert d.optionality_bp == 30

    def test_credit_share(self):
        d = decompose_spread(200, cds_spread_bp=140)
        assert abs(d.credit_share - 0.70) < 0.05

    def test_non_credit(self):
        d = decompose_spread(250, cds_spread_bp=180)
        assert d.non_credit_bp == d.total_spread_bp - d.credit_bp

    def test_to_dict(self):
        d = decompose_spread(200, cds_spread_bp=150).to_dict()
        assert "credit_share" in d
        assert "non_credit_bp" in d


class TestCDSBondBasis:
    def test_positive_basis(self):
        assert cds_bond_basis(250, 200) == 50

    def test_negative_basis(self):
        assert cds_bond_basis(180, 200) == -20

    def test_zero(self):
        assert cds_bond_basis(100, 100) == 0


class TestPortfolio:
    def test_basic(self):
        bonds = [
            {"bond_spread_bp": 200, "cds_spread_bp": 150, "weight": 1.0},
            {"bond_spread_bp": 300, "cds_spread_bp": 250, "weight": 1.0},
        ]
        d = decompose_portfolio(bonds)
        assert d["n_bonds"] == 2
        assert d["total_spread_bp"] == 250  # average

    def test_weighted(self):
        bonds = [
            {"bond_spread_bp": 100, "weight": 3.0},
            {"bond_spread_bp": 300, "weight": 1.0},
        ]
        d = decompose_portfolio(bonds)
        assert d["total_spread_bp"] == 150  # 3*100+1*300 / 4

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            decompose_portfolio([])
