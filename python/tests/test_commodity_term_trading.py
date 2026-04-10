"""Tests for commodity term structure trading."""

import pytest
from datetime import date

from pricebook.commodity_term_trading import (
    CommodityButterfly,
    CommodityCalendarSpread,
    CommoditySteepener,
    CurveStructureSnapshot,
    curve_structure_monitor,
    dv01_neutral_quantity,
)


# Evenly-spaced deliveries: 3 months apart
D1 = date(2024, 3, 1)
D2 = date(2024, 6, 1)
D3 = date(2024, 9, 1)
D4 = date(2024, 12, 1)

# Simple curve: contango (rising with maturity)
CONTANGO = {D1: 70.0, D2: 72.0, D3: 74.0, D4: 76.0}
# Backwardation: falling
BACKWARDATED = {D1: 76.0, D2: 74.0, D3: 72.0, D4: 70.0}


# ---- Step 1: calendar spreads ----

class TestCommodityCalendarSpread:
    def test_pv_backwardated(self):
        cs = CommodityCalendarSpread("WTI", D1, D2, quantity=1_000, direction=1)
        pv = cs.pv(BACKWARDATED)
        # Long near (76) short far (74) → 1K × 2 = 2K
        assert pv == pytest.approx(2_000)

    def test_pv_contango(self):
        cs = CommodityCalendarSpread("WTI", D1, D2, quantity=1_000, direction=1)
        pv = cs.pv(CONTANGO)
        # Long near (70) short far (72) → 1K × (-2) = -2K
        assert pv == pytest.approx(-2_000)

    def test_direction_flips(self):
        long = CommodityCalendarSpread("WTI", D1, D2, 1_000, direction=1)
        short = CommodityCalendarSpread("WTI", D1, D2, 1_000, direction=-1)
        assert long.pv(CONTANGO) == pytest.approx(-short.pv(CONTANGO))

    def test_parallel_exposure_zero(self):
        """Step 1 test: calendar spread is zero parallel-DV01."""
        cs = CommodityCalendarSpread("WTI", D1, D2, quantity=10_000)
        assert cs.parallel_exposure() == 0.0
        # Also verify numerically: shift curve by +1 → same PV
        shifted = {d: f + 1.0 for d, f in CONTANGO.items()}
        assert cs.pv(shifted) == pytest.approx(cs.pv(CONTANGO))

    def test_quantity_scales_pv(self):
        cs1 = CommodityCalendarSpread("WTI", D1, D2, 1_000)
        cs3 = CommodityCalendarSpread("WTI", D1, D2, 3_000)
        assert cs3.pv(CONTANGO) == pytest.approx(3.0 * cs1.pv(CONTANGO))


class TestDV01NeutralQuantity:
    def test_equal_dv01(self):
        assert dv01_neutral_quantity(10_000, 1.0, 1.0) == pytest.approx(10_000)

    def test_different_dv01(self):
        # Near DV01 = 0.8, far DV01 = 1.2
        # far_qty = 10K × 0.8 / 1.2 ≈ 6666.7
        far = dv01_neutral_quantity(10_000, 0.8, 1.2)
        assert far == pytest.approx(10_000 * 0.8 / 1.2)

    def test_zero_far_dv01_returns_near(self):
        assert dv01_neutral_quantity(10_000, 1.0, 0.0) == pytest.approx(10_000)


# ---- Step 2: curve shape trades ----

class TestCommoditySteepener:
    def test_profits_from_steepening(self):
        steepener = CommoditySteepener("WTI", D1, D4, quantity=1_000, direction=1)
        flat = {D1: 72.0, D4: 72.0}
        steep = {D1: 70.0, D4: 74.0}
        pv_flat = steepener.pv(flat)
        pv_steep = steepener.pv(steep)
        assert pv_steep > pv_flat

    def test_flattener_opposite(self):
        steep = CommoditySteepener("WTI", D1, D4, 1_000, direction=1)
        flat = CommoditySteepener("WTI", D1, D4, 1_000, direction=-1)
        curve = {D1: 70.0, D4: 74.0}
        assert steep.pv(curve) == pytest.approx(-flat.pv(curve))

    def test_parallel_exposure_zero(self):
        s = CommoditySteepener("WTI", D1, D4, 1_000)
        assert s.parallel_exposure() == 0.0
        shifted = {d: f + 5.0 for d, f in CONTANGO.items()}
        assert s.pv(shifted) == pytest.approx(s.pv(CONTANGO))


class TestCommodityButterfly:
    def test_pv_benefits_from_belly_rise(self):
        bfly = CommodityButterfly("WTI", D1, D2, D3, quantity=1_000, direction=1)
        flat = {D1: 72.0, D2: 72.0, D3: 72.0}
        belly_up = {D1: 72.0, D2: 74.0, D3: 72.0}
        assert bfly.pv(belly_up) > bfly.pv(flat)

    def test_pv_formula(self):
        bfly = CommodityButterfly("WTI", D1, D2, D3, quantity=1_000, direction=1)
        curve = {D1: 70.0, D2: 73.0, D3: 74.0}
        # 1K × (2×73 - 70 - 74) = 1K × 2 = 2K
        assert bfly.pv(curve) == pytest.approx(2_000)

    def test_parallel_exposure_zero(self):
        """Step 2 test: butterfly has zero parallel exposure."""
        bfly = CommodityButterfly("WTI", D1, D2, D3, quantity=1_000)
        assert bfly.parallel_exposure() == 0.0
        shifted = {d: f + 10.0 for d, f in CONTANGO.items()}
        assert bfly.pv(shifted) == pytest.approx(bfly.pv(CONTANGO))

    def test_steepener_exposure_zero_even_spacing(self):
        """Step 2 test: evenly-spaced butterfly has zero steepener exposure."""
        # D1, D2, D3 are evenly spaced (3 months each)
        bfly = CommodityButterfly("WTI", D1, D2, D3, quantity=1_000)
        assert bfly.steepener_exposure() == pytest.approx(0.0, abs=1.0)

    def test_steepener_exposure_nonzero_uneven(self):
        # Near → mid: 3 months, mid → far: 6 months (uneven)
        bfly = CommodityButterfly("WTI", D1, D2, D4, quantity=1_000)
        assert bfly.steepener_exposure() != 0.0

    def test_direction_flips(self):
        long = CommodityButterfly("WTI", D1, D2, D3, 1_000, direction=1)
        short = CommodityButterfly("WTI", D1, D2, D3, 1_000, direction=-1)
        assert long.pv(CONTANGO) == pytest.approx(-short.pv(CONTANGO))


# ---- Curve structure monitor ----

class TestCurveStructureMonitor:
    def test_contango(self):
        snap = curve_structure_monitor("WTI", date(2024, 1, 15), CONTANGO)
        assert snap.structure == "contango"
        assert snap.n_deliveries == 4
        assert len(snap.spreads) == 3
        # All spreads negative (near < far)
        assert all(s < 0 for s in snap.spreads)

    def test_backwardation(self):
        snap = curve_structure_monitor("WTI", date(2024, 1, 15), BACKWARDATED)
        assert snap.structure == "backwardation"
        assert all(s > 0 for s in snap.spreads)

    def test_mixed(self):
        mixed = {D1: 72.0, D2: 74.0, D3: 73.0, D4: 75.0}
        snap = curve_structure_monitor("WTI", date(2024, 1, 15), mixed)
        assert snap.structure == "mixed"

    def test_flat(self):
        flat = {D1: 72.0, D2: 72.0, D3: 72.0}
        snap = curve_structure_monitor("WTI", date(2024, 1, 15), flat)
        assert snap.structure == "flat"

    def test_sorted_output(self):
        unsorted = {D3: 74.0, D1: 70.0, D2: 72.0}
        snap = curve_structure_monitor("WTI", date(2024, 1, 15), unsorted)
        assert snap.deliveries == [D1, D2, D3]
        assert snap.forwards == [70.0, 72.0, 74.0]

    def test_single_delivery(self):
        snap = curve_structure_monitor("WTI", date(2024, 1, 15), {D1: 72.0})
        assert snap.structure == "flat"
        assert snap.spreads == []
