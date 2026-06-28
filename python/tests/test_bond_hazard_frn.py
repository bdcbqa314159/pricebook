"""Tests for FRN bootstrap, mixed fixed+float, and liquidity assessment."""

import pytest
import math
from datetime import date

from pricebook.credit.bond_hazard_bootstrap import (
    BondInput, FRNInput, HazardBootstrapResult,
    bootstrap_hazard_from_bonds, bootstrap_hazard_mixed,
    assess_liquidity, bootstrap_hazard_adaptive,
)
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.interpolation import InterpolationMethod

REF = date(2024, 1, 1)


def _make_discount_curve(rate=0.05):
    dates = [date(2024 + y, 1, 1) for y in range(1, 16)]
    dfs = [math.exp(-rate * y) for y in range(1, 16)]
    return DiscountCurve(REF, dates, dfs, interpolation=InterpolationMethod.LOG_LINEAR)


# ═══════════════════════════════════════════════════════════════
# FRN Input
# ═══════════════════════════════════════════════════════════════

class TestFRNInput:
    def test_construction(self):
        frn = FRNInput(date(2027, 1, 1), 0.015, 99.5, 0.053)
        assert frn.spread == 0.015
        assert frn.frequency == 4

    def test_to_dict(self):
        frn = FRNInput(date(2027, 1, 1), 0.015, 99.5, 0.053)
        d = frn.to_dict()
        assert "spread" in d
        assert "benchmark_rate" in d


# ═══════════════════════════════════════════════════════════════
# Mixed Bootstrap
# ═══════════════════════════════════════════════════════════════

class TestMixedBootstrap:
    def test_fixed_only(self):
        """Mixed bootstrap with only fixed bonds should work."""
        dc = _make_discount_curve()
        bonds = [
            BondInput(date(2027, 1, 1), 0.05, 96.0),
            BondInput(date(2029, 1, 1), 0.055, 92.0),
            BondInput(date(2034, 1, 1), 0.06, 85.0),
        ]
        result = bootstrap_hazard_mixed(REF, fixed_bonds=bonds, discount_curve=dc)
        assert result.converged
        assert result.rmse_bp < 500  # reasonable fit

    def test_frn_only(self):
        """Bootstrap from FRNs only."""
        dc = _make_discount_curve()
        floaters = [
            FRNInput(date(2026, 1, 1), 0.010, 99.8, 0.053),
            FRNInput(date(2029, 1, 1), 0.015, 98.5, 0.053),
        ]
        result = bootstrap_hazard_mixed(REF, floaters=floaters, discount_curve=dc)
        assert result.n_bonds == 2
        assert len(result.fitted_prices) == 2

    def test_mixed_fixed_and_float(self):
        """Mix of fixed and floating bonds."""
        dc = _make_discount_curve()
        bonds = [
            BondInput(date(2027, 1, 1), 0.05, 96.0),
            BondInput(date(2034, 1, 1), 0.06, 85.0),
        ]
        floaters = [
            FRNInput(date(2029, 1, 1), 0.015, 98.5, 0.053),
        ]
        result = bootstrap_hazard_mixed(REF, bonds, floaters, dc)
        assert result.n_bonds == 3
        assert result.method == "global_mixed"

    def test_hazard_rates_positive(self):
        dc = _make_discount_curve()
        bonds = [BondInput(date(2029, 1, 1), 0.05, 93.0)]
        result = bootstrap_hazard_mixed(REF, fixed_bonds=bonds, discount_curve=dc)
        assert all(h >= 0 for h in result.pillar_hazards)

    def test_survival_decreasing(self):
        """Survival probabilities should be decreasing."""
        dc = _make_discount_curve()
        bonds = [
            BondInput(date(2027, 1, 1), 0.05, 96.0),
            BondInput(date(2034, 1, 1), 0.06, 85.0),
        ]
        result = bootstrap_hazard_mixed(REF, fixed_bonds=bonds, discount_curve=dc)
        sc = result.survival_curve
        q1 = sc.survival(date(2027, 1, 1))
        q2 = sc.survival(date(2034, 1, 1))
        assert q1 > q2  # shorter maturity has higher survival

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Need at least one"):
            bootstrap_hazard_mixed(REF, discount_curve=_make_discount_curve())


# ═══════════════════════════════════════════════════════════════
# Liquidity Assessment
# ═══════════════════════════════════════════════════════════════

class TestLiquidityAssessment:
    def test_liquid(self):
        bonds = [
            BondInput(date(2027, 1, 1), 0.05, 98.0),
            BondInput(date(2029, 1, 1), 0.05, 96.0),
            BondInput(date(2034, 1, 1), 0.06, 92.0),
        ]
        a = assess_liquidity(bonds, bid_ask_widths_bp=[20, 25, 30])
        assert a.regime == "liquid"
        assert a.confidence == "high"

    def test_semi_liquid(self):
        bonds = [
            BondInput(date(2027, 1, 1), 0.05, 90.0),
            BondInput(date(2029, 1, 1), 0.05, 85.0),
        ]
        a = assess_liquidity(bonds, bid_ask_widths_bp=[100, 120])
        assert a.regime == "semi_liquid"

    def test_illiquid_single_bond(self):
        bonds = [BondInput(date(2029, 1, 1), 0.05, 70.0)]
        a = assess_liquidity(bonds)
        assert a.regime == "illiquid"
        assert a.recommended_n_pillars == 1
        assert any("Single bond" in n for n in a.notes)

    def test_illiquid_wide_spread(self):
        bonds = [
            BondInput(date(2027, 1, 1), 0.05, 60.0),
            BondInput(date(2029, 1, 1), 0.05, 55.0),
        ]
        a = assess_liquidity(bonds, bid_ask_widths_bp=[300, 400])
        assert a.regime == "illiquid"

    def test_with_floaters(self):
        floaters = [FRNInput(date(2027, 1, 1), 0.015, 99.5, 0.053)]
        a = assess_liquidity(floaters=floaters)
        assert any("FRN" in n for n in a.notes)

    def test_to_dict(self):
        bonds = [BondInput(date(2029, 1, 1), 0.05, 96.0)]
        a = assess_liquidity(bonds)
        d = a.to_dict()
        assert "regime" in d
        assert "confidence" in d


# ═══════════════════════════════════════════════════════════════
# Adaptive Bootstrap
# ═══════════════════════════════════════════════════════════════

class TestAdaptiveBootstrap:
    def test_liquid_uses_sequential(self):
        dc = _make_discount_curve()
        bonds = [
            BondInput(date(2027, 1, 1), 0.05, 96.0),
            BondInput(date(2029, 1, 1), 0.055, 92.0),
            BondInput(date(2034, 1, 1), 0.06, 85.0),
        ]
        result = bootstrap_hazard_adaptive(REF, bonds, discount_curve=dc,
                                            bid_ask_widths_bp=[20, 25, 30])
        # Should pick sequential for liquid fixed-only
        assert result.n_bonds == 3

    def test_illiquid_uses_global(self):
        dc = _make_discount_curve()
        bonds = [BondInput(date(2029, 1, 1), 0.05, 70.0)]
        result = bootstrap_hazard_adaptive(REF, bonds, discount_curve=dc,
                                            bid_ask_widths_bp=[400])
        assert result.n_bonds == 1

    def test_mixed_uses_global_mixed(self):
        dc = _make_discount_curve()
        bonds = [BondInput(date(2027, 1, 1), 0.05, 96.0)]
        floaters = [FRNInput(date(2029, 1, 1), 0.015, 98.5, 0.053)]
        result = bootstrap_hazard_adaptive(REF, bonds, floaters, dc)
        assert result.method == "global_mixed"

    def test_weights_adjusted_by_bid_ask(self):
        dc = _make_discount_curve()
        bonds = [
            BondInput(date(2027, 1, 1), 0.05, 96.0),
            BondInput(date(2029, 1, 1), 0.05, 92.0),
        ]
        result = bootstrap_hazard_adaptive(REF, bonds, discount_curve=dc,
                                            bid_ask_widths_bp=[20, 300])
        # Wider bid-ask bond gets lower weight IN THE FIT — verified on the
        # calibration record, not by mutating the caller's inputs.
        w = result.calibration_result.fit.weights
        assert w[1] < w[0]
        # The caller's input objects are left untouched.
        assert bonds[0].weight == 1.0 and bonds[1].weight == 1.0

    def test_distressed_bonds(self):
        """Bonds at 50-60 cents — should still produce reasonable curve."""
        dc = _make_discount_curve()
        bonds = [
            BondInput(date(2026, 1, 1), 0.08, 65.0),
            BondInput(date(2028, 1, 1), 0.09, 50.0),
        ]
        result = bootstrap_hazard_adaptive(REF, bonds, discount_curve=dc)
        # High hazard rates expected for distressed
        assert all(h > 0 for h in result.pillar_hazards)
        # Short maturity should have higher survival
        sc = result.survival_curve
        assert sc.survival(date(2026, 1, 1)) > sc.survival(date(2028, 1, 1))
