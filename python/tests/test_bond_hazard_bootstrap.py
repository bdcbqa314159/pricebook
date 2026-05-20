"""Tests for hazard rate bootstrap from bond prices.

Covers: sequential, global, edge cases, convergence, round-trip.
"""
import math
import pytest
import numpy as np
from datetime import date

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve
from pricebook.credit.bond_hazard_bootstrap import (
    BondInput, HazardBootstrapResult,
    bootstrap_hazard_from_bonds, implied_hazard_from_spread,
    minimum_bonds_for_calibration, _price_risky_bond,
)


@pytest.fixture
def ref_date():
    return date(2024, 1, 1)


@pytest.fixture
def flat_curve(ref_date):
    return DiscountCurve.flat(ref_date, 0.04)


def _make_bond_at_spread(ref_date, maturity_years, coupon, spread_bp, recovery, flat_rate):
    """Helper: create a BondInput with price consistent with a given spread."""
    mat = date(ref_date.year + maturity_years, ref_date.month, ref_date.day)
    dc = DiscountCurve.flat(ref_date, flat_rate)
    hazard = spread_bp / 10_000 / (1 - recovery)
    sc = SurvivalCurve.flat(ref_date, hazard, tenors=list(range(1, maturity_years + 2)))
    price = _price_risky_bond(ref_date, mat, coupon, 2, recovery, dc, sc)
    return BondInput(maturity=mat, coupon=coupon, market_price=price,
                     frequency=2, recovery=recovery)


# ═══════════════════════════════════════════════════════════════
# Sequential Bootstrap
# ═══════════════════════════════════════════════════════════════

class TestSequential:
    def test_single_bond(self, ref_date, flat_curve):
        """1 bond → flat hazard rate."""
        bond = _make_bond_at_spread(ref_date, 5, 0.05, 200, 0.40, 0.04)
        result = bootstrap_hazard_from_bonds(ref_date, [bond], flat_curve, method="sequential")
        assert result.converged
        assert result.n_bonds == 1
        assert len(result.pillar_hazards) == 1
        assert result.pillar_hazards[0] > 0
        assert result.rmse_bp < 1.0

    def test_three_bonds(self, ref_date, flat_curve):
        """3 bonds at different maturities → 3-segment hazard curve."""
        bonds = [
            _make_bond_at_spread(ref_date, 2, 0.04, 100, 0.40, 0.04),
            _make_bond_at_spread(ref_date, 5, 0.05, 150, 0.40, 0.04),
            _make_bond_at_spread(ref_date, 10, 0.06, 200, 0.40, 0.04),
        ]
        result = bootstrap_hazard_from_bonds(ref_date, bonds, flat_curve, method="sequential")
        assert result.converged
        assert len(result.pillar_hazards) == 3
        assert result.rmse_bp < 1.0
        # Increasing spread → increasing hazard
        assert result.pillar_hazards[2] > result.pillar_hazards[0]

    def test_round_trip(self, ref_date, flat_curve):
        """Bootstrapped curve should reprice all input bonds."""
        bonds = [
            _make_bond_at_spread(ref_date, 3, 0.045, 120, 0.40, 0.04),
            _make_bond_at_spread(ref_date, 7, 0.055, 180, 0.40, 0.04),
        ]
        result = bootstrap_hazard_from_bonds(ref_date, bonds, flat_curve, method="sequential")
        for i, bond in enumerate(sorted(bonds, key=lambda b: b.maturity)):
            assert abs(result.fitted_prices[i] - bond.market_price) < 0.01, \
                f"Bond {i}: fitted={result.fitted_prices[i]:.4f}, market={bond.market_price:.4f}"


# ═══════════════════════════════════════════════════════════════
# Global Fit
# ═══════════════════════════════════════════════════════════════

class TestGlobal:
    def test_overdetermined(self, ref_date, flat_curve):
        """5 bonds → 3 pillars (over-determined, least-squares)."""
        bonds = [
            _make_bond_at_spread(ref_date, 1, 0.03, 80, 0.40, 0.04),
            _make_bond_at_spread(ref_date, 2, 0.04, 100, 0.40, 0.04),
            _make_bond_at_spread(ref_date, 3, 0.045, 120, 0.40, 0.04),
            _make_bond_at_spread(ref_date, 5, 0.05, 150, 0.40, 0.04),
            _make_bond_at_spread(ref_date, 7, 0.055, 180, 0.40, 0.04),
        ]
        result = bootstrap_hazard_from_bonds(ref_date, bonds, flat_curve,
                                              method="global", n_pillars=3)
        assert result.method == "global_ls"
        assert len(result.pillar_hazards) == 3
        assert result.rmse_bp < 50  # not exact, but reasonable fit

    def test_same_maturity(self, ref_date, flat_curve):
        """Bonds with same maturity → global handles it."""
        mat = date(2029, 1, 1)
        bonds = [
            BondInput(mat, 0.04, 95.0, recovery=0.40),
            BondInput(mat, 0.06, 98.0, recovery=0.40),
        ]
        result = bootstrap_hazard_from_bonds(ref_date, bonds, flat_curve, method="global")
        assert result.n_bonds == 2
        assert len(result.pillar_hazards) >= 1

    def test_same_maturity_sequential_fails(self, ref_date, flat_curve):
        """Sequential should reject duplicate maturities."""
        mat = date(2029, 1, 1)
        bonds = [
            BondInput(mat, 0.04, 95.0, recovery=0.40),
            BondInput(mat, 0.06, 98.0, recovery=0.40),
        ]
        with pytest.raises(ValueError, match="distinct maturities"):
            bootstrap_hazard_from_bonds(ref_date, bonds, flat_curve, method="sequential")


# ═══════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_par_bond(self, ref_date, flat_curve):
        """Bond trading at par → low hazard (spread ≈ 0)."""
        mat = date(2029, 1, 1)
        bond = BondInput(mat, 0.04, 100.0, recovery=0.40)
        result = bootstrap_hazard_from_bonds(ref_date, [bond], flat_curve)
        assert result.pillar_hazards[0] < 0.02  # near-zero hazard

    def test_deep_discount(self, ref_date, flat_curve):
        """Distressed bond at 60 → high hazard."""
        mat = date(2029, 1, 1)
        bond = BondInput(mat, 0.05, 60.0, recovery=0.40)
        result = bootstrap_hazard_from_bonds(ref_date, [bond], flat_curve)
        assert result.pillar_hazards[0] > 0.05  # high hazard

    def test_recovery_sensitivity(self, ref_date, flat_curve):
        """Different recovery → different hazard for same price."""
        mat = date(2029, 1, 1)
        bond_low_r = BondInput(mat, 0.05, 90.0, recovery=0.20)
        bond_high_r = BondInput(mat, 0.05, 90.0, recovery=0.60)
        r1 = bootstrap_hazard_from_bonds(ref_date, [bond_low_r], flat_curve)
        r2 = bootstrap_hazard_from_bonds(ref_date, [bond_high_r], flat_curve)
        # Higher recovery → higher hazard needed to explain same price discount
        assert r2.pillar_hazards[0] > r1.pillar_hazards[0]

    def test_empty_raises(self, ref_date, flat_curve):
        with pytest.raises(ValueError):
            bootstrap_hazard_from_bonds(ref_date, [], flat_curve)

    def test_auto_method_selection(self, ref_date, flat_curve):
        """Auto selects sequential for small distinct set, global for large."""
        bonds_3 = [_make_bond_at_spread(ref_date, y, 0.05, 150, 0.40, 0.04)
                    for y in [2, 5, 10]]
        r = bootstrap_hazard_from_bonds(ref_date, bonds_3, flat_curve, method="auto")
        assert r.method == "sequential"

        bonds_10 = [_make_bond_at_spread(ref_date, y, 0.05, 150, 0.40, 0.04)
                     for y in range(1, 11)]
        r2 = bootstrap_hazard_from_bonds(ref_date, bonds_10, flat_curve, method="auto")
        assert r2.method == "global_ls"

    def test_to_dict(self, ref_date, flat_curve):
        bond = _make_bond_at_spread(ref_date, 5, 0.05, 150, 0.40, 0.04)
        d = bootstrap_hazard_from_bonds(ref_date, [bond], flat_curve).to_dict()
        assert "pillar_hazards" in d
        assert "rmse_bp" in d


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

class TestHelpers:
    def test_implied_hazard(self):
        h = implied_hazard_from_spread(200, 0.40)
        assert abs(h - 0.02/0.60) < 1e-10

    def test_minimum_bonds(self):
        guide = minimum_bonds_for_calibration([2, 5, 10])
        assert guide["minimum"] == 1
        assert guide["recommended"] == 3

    def test_risky_bond_pricing(self, ref_date, flat_curve):
        """Risky bond with zero hazard should equal risk-free bond."""
        sc = SurvivalCurve.flat(ref_date, 0.0, tenors=list(range(1, 12)))
        mat = date(2034, 1, 1)
        risky = _price_risky_bond(ref_date, mat, 0.04, 2, 0.40, flat_curve, sc)
        # With zero hazard, should be close to risk-free par bond
        assert abs(risky - 100.0) < 1.0  # near par for coupon ≈ rate


# ═══════════════════════════════════════════════════════════════
# Stress / Convergence
# ═══════════════════════════════════════════════════════════════

class TestStress:
    def test_inverted_spread_curve(self, ref_date, flat_curve):
        """Short-term spread > long-term (inverted credit curve)."""
        bonds = [
            _make_bond_at_spread(ref_date, 2, 0.05, 300, 0.40, 0.04),
            _make_bond_at_spread(ref_date, 5, 0.05, 200, 0.40, 0.04),
            _make_bond_at_spread(ref_date, 10, 0.05, 150, 0.40, 0.04),
        ]
        result = bootstrap_hazard_from_bonds(ref_date, bonds, flat_curve)
        assert result.converged
        # Front hazard > back hazard
        assert result.pillar_hazards[0] > result.pillar_hazards[2]

    def test_noisy_prices(self, ref_date, flat_curve):
        """Bonds with noisy prices → global fit handles gracefully."""
        rng = np.random.default_rng(42)
        bonds = []
        for y in [1, 2, 3, 5, 7, 10]:
            clean = _make_bond_at_spread(ref_date, y, 0.05, 150, 0.40, 0.04)
            noisy_price = clean.market_price + rng.normal(0, 0.5)  # ±50bp noise
            bonds.append(BondInput(clean.maturity, clean.coupon, noisy_price, recovery=0.40))
        result = bootstrap_hazard_from_bonds(ref_date, bonds, flat_curve, method="global")
        assert result.rmse_bp < 100  # should still fit reasonably
