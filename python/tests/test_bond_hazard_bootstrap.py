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
    minimum_bonds_for_calibration, _price_risky_bond, _price_risky_bond_rmv,
    _price_bond, RECOVERY_PAR, RECOVERY_MARKET_VALUE,
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


# ═══════════════════════════════════════════════════════════════
# Recovery of Market Value (Duffie-Singleton)
# ═══════════════════════════════════════════════════════════════

class TestRecoveryOfMarketValue:
    def test_rmv_zero_hazard_equals_riskfree(self, ref_date, flat_curve):
        """With zero hazard, RMV price should equal risk-free price (same as RP)."""
        sc = SurvivalCurve.flat(ref_date, 0.0, tenors=list(range(1, 12)))
        mat = date(2034, 1, 1)
        price_rp = _price_risky_bond(ref_date, mat, 0.04, 2, 0.40, flat_curve, sc)
        price_rmv = _price_risky_bond_rmv(ref_date, mat, 0.04, 2, 0.40, flat_curve, sc)
        assert abs(price_rp - price_rmv) < 0.01

    def test_rmv_different_from_par(self, ref_date, flat_curve):
        """For non-zero hazard, RMV and RP give different prices."""
        sc = SurvivalCurve.flat(ref_date, 0.03, tenors=list(range(1, 12)))
        mat = date(2034, 1, 1)
        price_rp = _price_risky_bond(ref_date, mat, 0.05, 2, 0.40, flat_curve, sc)
        price_rmv = _price_risky_bond_rmv(ref_date, mat, 0.05, 2, 0.40, flat_curve, sc)
        assert abs(price_rp - price_rmv) > 0.1  # meaningfully different

    def test_rmv_lower_price_than_par(self, ref_date, flat_curve):
        """RMV typically gives lower price than RP for same hazard.

        Under RP, recovery = R × 100 (face) regardless of bond value.
        Under RMV, recovery = R × V(t⁻) which < 100 for risky bonds.
        So RMV pays less on default → lower bond price.
        """
        sc = SurvivalCurve.flat(ref_date, 0.05, tenors=list(range(1, 12)))
        mat = date(2029, 1, 1)
        price_rp = _price_risky_bond(ref_date, mat, 0.05, 2, 0.40, flat_curve, sc)
        price_rmv = _price_risky_bond_rmv(ref_date, mat, 0.05, 2, 0.40, flat_curve, sc)
        assert price_rmv < price_rp

    def test_rmv_bootstrap_produces_lower_hazard(self, ref_date, flat_curve):
        """Same market prices → RMV bootstrap gives lower hazard than RP.

        RMV gives lower prices than RP for the same hazard (less recovery on
        default). So to match the same market price, RMV needs less hazard.
        """
        bond = _make_bond_at_spread(ref_date, 5, 0.05, 300, 0.40, 0.04)
        rp = bootstrap_hazard_from_bonds(
            ref_date, [bond], flat_curve, recovery_mode=RECOVERY_PAR,
        )
        rmv = bootstrap_hazard_from_bonds(
            ref_date, [bond], flat_curve, recovery_mode=RECOVERY_MARKET_VALUE,
        )
        assert rmv.pillar_hazards[0] < rp.pillar_hazards[0]

    def test_rmv_sequential_round_trip(self, ref_date, flat_curve):
        """RMV bootstrap should reprice input bonds."""
        bonds = [
            _make_bond_at_spread(ref_date, 3, 0.045, 200, 0.40, 0.04),
            _make_bond_at_spread(ref_date, 7, 0.055, 300, 0.40, 0.04),
        ]
        result = bootstrap_hazard_from_bonds(
            ref_date, bonds, flat_curve,
            method="sequential", recovery_mode=RECOVERY_MARKET_VALUE,
        )
        assert result.converged
        assert result.rmse_bp < 5.0  # should reprice well

    def test_rmv_global_fit(self, ref_date, flat_curve):
        """RMV works with global method too."""
        bonds = [
            _make_bond_at_spread(ref_date, y, 0.05, 200, 0.40, 0.04)
            for y in [2, 5, 7, 10]
        ]
        result = bootstrap_hazard_from_bonds(
            ref_date, bonds, flat_curve,
            method="global", recovery_mode=RECOVERY_MARKET_VALUE,
        )
        assert result.n_bonds == 4
        assert all(h >= 0 for h in result.pillar_hazards)

    def test_dispatcher_consistency(self, ref_date, flat_curve):
        """_price_bond dispatches correctly to both modes."""
        sc = SurvivalCurve.flat(ref_date, 0.02, tenors=list(range(1, 8)))
        mat = date(2029, 1, 1)
        p_par = _price_bond(ref_date, mat, 0.05, 2, 0.40, flat_curve, sc, RECOVERY_PAR)
        p_rmv = _price_bond(ref_date, mat, 0.05, 2, 0.40, flat_curve, sc, RECOVERY_MARKET_VALUE)
        p_direct_par = _price_risky_bond(ref_date, mat, 0.05, 2, 0.40, flat_curve, sc)
        p_direct_rmv = _price_risky_bond_rmv(ref_date, mat, 0.05, 2, 0.40, flat_curve, sc)
        assert abs(p_par - p_direct_par) < 1e-10
        assert abs(p_rmv - p_direct_rmv) < 1e-10

    def test_invalid_recovery_mode(self, ref_date, flat_curve):
        bond = _make_bond_at_spread(ref_date, 5, 0.05, 200, 0.40, 0.04)
        with pytest.raises(ValueError, match="recovery_mode"):
            bootstrap_hazard_from_bonds(
                ref_date, [bond], flat_curve, recovery_mode="invalid",
            )


# ═══════════════════════════════════════════════════════════════
# Liquidity Premium Separation
# ═══════════════════════════════════════════════════════════════

class TestLiquidityPremium:
    def test_liquidity_spread_lowers_hazard(self, ref_date, flat_curve):
        """Attributing part of the spread to liquidity → lower credit hazard.

        If bond spread = 200bp and we say 50bp is liquidity, the credit-only
        spread is 150bp, so the hazard rate should be lower.
        """
        bond_no_liq = BondInput(
            date(2029, 1, 1), 0.05, 90.0, recovery=0.40, liquidity_spread_bp=0.0,
        )
        bond_with_liq = BondInput(
            date(2029, 1, 1), 0.05, 90.0, recovery=0.40, liquidity_spread_bp=50.0,
        )
        r_no = bootstrap_hazard_from_bonds(ref_date, [bond_no_liq], flat_curve)
        r_liq = bootstrap_hazard_from_bonds(ref_date, [bond_with_liq], flat_curve)
        assert r_liq.pillar_hazards[0] < r_no.pillar_hazards[0]

    def test_zero_liquidity_unchanged(self, ref_date, flat_curve):
        """Zero liquidity spread should give same result as no liquidity spread."""
        bond = _make_bond_at_spread(ref_date, 5, 0.05, 200, 0.40, 0.04)
        bond_zero = BondInput(
            bond.maturity, bond.coupon, bond.market_price,
            recovery=bond.recovery, liquidity_spread_bp=0.0,
        )
        r1 = bootstrap_hazard_from_bonds(ref_date, [bond], flat_curve)
        r2 = bootstrap_hazard_from_bonds(ref_date, [bond_zero], flat_curve)
        assert abs(r1.pillar_hazards[0] - r2.pillar_hazards[0]) < 1e-10

    def test_liquidity_spread_global(self, ref_date, flat_curve):
        """Liquidity spread works with global method."""
        bonds = [
            BondInput(date(2026, 1, 1), 0.04, 96.0, recovery=0.40, liquidity_spread_bp=30.0),
            BondInput(date(2029, 1, 1), 0.05, 90.0, recovery=0.40, liquidity_spread_bp=50.0),
            BondInput(date(2034, 1, 1), 0.06, 85.0, recovery=0.40, liquidity_spread_bp=80.0),
        ]
        result = bootstrap_hazard_from_bonds(
            ref_date, bonds, flat_curve, method="global",
        )
        assert result.n_bonds == 3
        assert all(h >= 0 for h in result.pillar_hazards)

    def test_per_bond_liquidity(self, ref_date, flat_curve):
        """Different liquidity spreads per bond (illiquid long-end)."""
        bonds = [
            _make_bond_at_spread(ref_date, 2, 0.04, 150, 0.40, 0.04),
            _make_bond_at_spread(ref_date, 5, 0.05, 200, 0.40, 0.04),
            _make_bond_at_spread(ref_date, 10, 0.06, 250, 0.40, 0.04),
        ]
        # Apply increasing liquidity to long-end
        bonds[0] = BondInput(bonds[0].maturity, bonds[0].coupon, bonds[0].market_price,
                             recovery=0.40, liquidity_spread_bp=10.0)
        bonds[1] = BondInput(bonds[1].maturity, bonds[1].coupon, bonds[1].market_price,
                             recovery=0.40, liquidity_spread_bp=30.0)
        bonds[2] = BondInput(bonds[2].maturity, bonds[2].coupon, bonds[2].market_price,
                             recovery=0.40, liquidity_spread_bp=60.0)
        result = bootstrap_hazard_from_bonds(ref_date, bonds, flat_curve)
        assert result.converged
        assert len(result.pillar_hazards) == 3

    def test_liquidity_with_rmv(self, ref_date, flat_curve):
        """Liquidity spread + RMV recovery mode together."""
        bond = BondInput(
            date(2029, 1, 1), 0.05, 88.0, recovery=0.40, liquidity_spread_bp=40.0,
        )
        result = bootstrap_hazard_from_bonds(
            ref_date, [bond], flat_curve, recovery_mode=RECOVERY_MARKET_VALUE,
        )
        assert result.converged
        assert result.pillar_hazards[0] > 0

    def test_high_liquidity_near_full_spread(self, ref_date, flat_curve):
        """If liquidity ≈ total spread, hazard should be near zero."""
        # Bond at ~200bp spread, attribute 180bp to liquidity
        bond = _make_bond_at_spread(ref_date, 5, 0.05, 200, 0.40, 0.04)
        bond = BondInput(bond.maturity, bond.coupon, bond.market_price,
                         recovery=0.40, liquidity_spread_bp=180.0)
        result = bootstrap_hazard_from_bonds(ref_date, [bond], flat_curve)
        # With most spread attributed to liquidity, credit hazard should be small
        assert result.pillar_hazards[0] < 0.01
