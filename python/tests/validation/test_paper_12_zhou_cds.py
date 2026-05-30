"""Paper 12: Zhou (2008) — CDS-Bond Basis (rewired through pricebook).

Uses: bond_implied_cds_spread(), compute_basis(), RiskyBond, CDS.
Canonical case: 10Y 7% semi-annual, R=40%, 7 D-levels.
"""

import pytest
import math

from pricebook.credit.cds_bond_basis import bond_implied_cds_spread, compute_basis


COUPON = 0.07
R = 0.40
T = 10
FLAT_RATE = 0.047


class TestBondImpliedCDS:
    """Use pricebook's bond_implied_cds_spread (Zhou eq. 8)."""

    @pytest.mark.parametrize("D_pct", [0, 5, 10, 15, 20])
    def test_implied_spread_positive(self, D_pct):
        price = 100 - D_pct
        result = bond_implied_cds_spread(COUPON, price, T, FLAT_RATE, R)
        assert result["cds_spread"] >= 0

    @pytest.mark.parametrize("D_pct", [0, 5, 10, 15, 20])
    def test_risky_price_matches_market(self, D_pct):
        price = 100 - D_pct
        result = bond_implied_cds_spread(COUPON, price, T, FLAT_RATE, R)
        assert abs(result["risky_price"] - price) < 0.01

    def test_spread_increases_with_discount(self):
        spreads = [bond_implied_cds_spread(COUPON, 100 - D, T, FLAT_RATE, R)["cds_spread"]
                   for D in [0, 5, 10, 15, 20]]
        for i in range(1, len(spreads)):
            assert spreads[i] >= spreads[i-1] - 0.0001

    def test_hazard_increases_with_discount(self):
        hazards = [bond_implied_cds_spread(COUPON, 100 - D, T, FLAT_RATE, R)["hazard_rate"]
                   for D in [0, 5, 10, 15, 20]]
        for i in range(1, len(hazards)):
            assert hazards[i] >= hazards[i-1] - 0.0001


class TestZhouTable1:
    def test_d0(self):
        r = bond_implied_cds_spread(COUPON, 100.0, T, FLAT_RATE, R)
        assert 0.01 < r["cds_spread"] < 0.04

    def test_d10(self):
        r = bond_implied_cds_spread(COUPON, 90.0, T, FLAT_RATE, R)
        assert 0.02 < r["cds_spread"] < 0.06

    def test_d20(self):
        r = bond_implied_cds_spread(COUPON, 80.0, T, FLAT_RATE, R)
        assert 0.03 < r["cds_spread"] < 0.10


class TestBasisDecomposition:
    def test_basis_components(self):
        result = compute_basis(cds_spread_bp=300, bond_spread_bp=250, repo_spread_bp=10, funding_spread_bp=20)
        assert result.basis_bp == 50
        assert hasattr(result, 'funding_component_bp')

    def test_positive_basis(self):
        r = compute_basis(cds_spread_bp=200, bond_spread_bp=180)
        assert r.basis_bp > 0

    def test_negative_basis(self):
        r = compute_basis(cds_spread_bp=150, bond_spread_bp=200)
        assert r.basis_bp < 0
        assert "NEGATIVE" in r.signal.upper()

    def test_basis_widens_with_discount(self):
        bases = []
        for D in [0, 5, 10, 15, 20]:
            r = bond_implied_cds_spread(COUPON, 100 - D, T, FLAT_RATE, R)
            annuity = sum(0.5 * math.exp(-FLAT_RATE * 0.5 * i) for i in range(1, 21))
            asw = (COUPON - FLAT_RATE) - D / 100 / annuity
            bases.append(r["cds_spread"] - max(asw, 0))
        assert bases[-1] > bases[0]
