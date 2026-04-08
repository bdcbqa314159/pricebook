"""Tests for CVA desk tools."""

import math
import pytest
import numpy as np
from datetime import date

from pricebook.cva_desk import (
    cva_cs01, cva_ir01, cva_by_trade,
    cva_hedge, incremental_cva,
)
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.xva import cva


REF = date(2024, 1, 15)
TIME_GRID = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]


def _dc(rate=0.05):
    return DiscountCurve.flat(REF, rate)


def _sc(hazard=0.02):
    return SurvivalCurve.flat(REF, hazard)


def _epe(level=100_000):
    """Constant EPE profile."""
    return np.full(len(TIME_GRID), level, dtype=float)


# ---- CVA CS01 ----

class TestCVACS01:
    def test_positive(self):
        """CVA increases when credit spreads widen → CS01 > 0."""
        dc = _dc()
        sc = _sc(0.02)
        epe = _epe()
        result = cva_cs01(epe, TIME_GRID, dc, sc)
        assert result > 0

    def test_scales_with_exposure(self):
        dc = _dc()
        sc = _sc()
        cs1 = cva_cs01(_epe(100_000), TIME_GRID, dc, sc)
        cs2 = cva_cs01(_epe(200_000), TIME_GRID, dc, sc)
        assert cs2 == pytest.approx(2 * cs1, rel=0.05)

    def test_matches_finite_difference(self):
        dc = _dc()
        sc = _sc(0.02)
        epe = _epe()
        from pricebook.credit_risk import _bump_survival_curve
        base = cva(epe, TIME_GRID, dc, sc)
        bumped_sc = _bump_survival_curve(sc, 0.0001)
        bumped = cva(epe, TIME_GRID, dc, bumped_sc)
        manual = (bumped - base) / 1.0
        computed = cva_cs01(epe, TIME_GRID, dc, sc)
        assert computed == pytest.approx(manual, rel=0.01)


# ---- CVA IR01 ----

class TestCVAIR01:
    def test_nonzero(self):
        dc = _dc()
        sc = _sc()
        epe = _epe()
        result = cva_ir01(epe, TIME_GRID, dc, sc)
        assert result != 0.0

    def test_negative(self):
        """Higher rates → lower discount factors → lower CVA → IR01 < 0."""
        dc = _dc()
        sc = _sc()
        result = cva_ir01(_epe(), TIME_GRID, dc, sc)
        assert result < 0


# ---- CVA by trade ----

class TestCVAByTrade:
    def test_decomposition(self):
        dc = _dc()
        sc = _sc()
        epes = {"t1": _epe(100_000), "t2": _epe(50_000)}
        results = cva_by_trade(epes, TIME_GRID, dc, sc)
        assert len(results) == 2
        assert "t1" in results
        assert results["t1"] > results["t2"]

    def test_single_trade(self):
        dc = _dc()
        sc = _sc()
        epe = _epe()
        results = cva_by_trade({"t1": epe}, TIME_GRID, dc, sc)
        assert results["t1"] == pytest.approx(cva(epe, TIME_GRID, dc, sc))


# ---- CVA hedge ----

class TestCVAHedge:
    def test_residual_near_zero(self):
        dc = _dc()
        sc = _sc()
        epe = _epe()
        # CDS CS01 per notional: use a reasonable value
        cds_cs01_pn = 0.0004  # 0.04bp per million notional
        result = cva_hedge(epe, TIME_GRID, dc, sc, cds_cs01_pn)
        assert abs(result.residual_cs01) < abs(result.portfolio_cs01) * 0.01

    def test_hedge_notional_nonzero(self):
        dc = _dc()
        sc = _sc()
        epe = _epe()
        result = cva_hedge(epe, TIME_GRID, dc, sc, 0.0004)
        assert result.hedge_notional != 0.0

    def test_hedge_direction(self):
        """Positive CS01 → need to sell protection (negative notional)."""
        dc = _dc()
        sc = _sc()
        epe = _epe()
        result = cva_hedge(epe, TIME_GRID, dc, sc, 0.0004)
        # CVA CS01 is positive (CVA increases with spreads)
        # To hedge, sell CDS protection (negative notional with positive CS01/not)
        assert result.portfolio_cs01 > 0
        assert result.hedge_notional < 0


# ---- Incremental CVA ----

class TestIncrementalCVA:
    def test_additive_same_sign(self):
        """Same-sign exposures: incremental ≈ standalone."""
        dc = _dc()
        sc = _sc()
        port_epe = _epe(100_000)
        new_epe = _epe(50_000)
        result = incremental_cva(port_epe, new_epe, TIME_GRID, dc, sc)
        assert result["incremental_cva"] > 0
        assert result["standalone_cva"] > 0
        assert result["combined_cva"] > result["portfolio_cva"]

    def test_netting_benefit_opposite(self):
        """Opposite-sign exposure: netting benefit > 0."""
        dc = _dc()
        sc = _sc()
        port_epe = _epe(100_000)
        # Negative EPE = exposure reduction
        new_epe = np.full(len(TIME_GRID), -50_000, dtype=float)
        result = incremental_cva(port_epe, new_epe, TIME_GRID, dc, sc)
        # Combined exposure is lower → combined CVA < portfolio CVA
        assert result["combined_cva"] < result["portfolio_cva"]
        assert result["incremental_cva"] < 0

    def test_standalone_vs_incremental(self):
        dc = _dc()
        sc = _sc()
        port_epe = _epe(100_000)
        new_epe = _epe(50_000)
        result = incremental_cva(port_epe, new_epe, TIME_GRID, dc, sc)
        # For same-sign, incremental ≈ standalone (no netting for positive + positive)
        assert result["incremental_cva"] == pytest.approx(result["standalone_cva"], rel=0.01)
