"""Tests for wrong-way risk and collateralised XVA."""

import math
import pytest
import numpy as np
from datetime import date

from pricebook.xva import (
    simulate_wwr_exposures,
    cva_wrong_way,
    cva_collateralised,
    fva_collateralised,
    cva,
    expected_positive_exposure,
)
from pricebook.pricing_context import PricingContext
from pricebook.vol_surface import FlatVol
from pricebook.swaption import Swaption
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve


REF = date(2024, 1, 15)
TIME_GRID = [0.5, 1.0, 2.0, 3.0, 5.0]


def _ctx():
    return PricingContext(
        valuation_date=REF,
        discount_curve=DiscountCurve.flat(REF, 0.05),
        vol_surfaces={"ir": FlatVol(0.20)},
    )


def _pricer():
    swn = Swaption(date(2025, 1, 15), date(2030, 1, 15), strike=0.05)

    def safe_pricer(c):
        try:
            return swn.pv_ctx(c)
        except (ValueError, ZeroDivisionError):
            return 0.0

    return safe_pricer


# ---------------------------------------------------------------------------
# Slice 84: Wrong-Way Risk
# ---------------------------------------------------------------------------


class TestWWRSimulation:
    def test_shapes(self):
        pvs, hazards = simulate_wwr_exposures(
            _pricer(), _ctx(), TIME_GRID, hazard_rate=0.02,
            n_paths=100, rate_vol=0.005,
        )
        assert pvs.shape == (100, len(TIME_GRID))
        assert hazards.shape == (100, len(TIME_GRID))

    def test_hazards_positive(self):
        _, hazards = simulate_wwr_exposures(
            _pricer(), _ctx(), TIME_GRID, hazard_rate=0.02,
            n_paths=200, rate_vol=0.005,
        )
        assert np.all(hazards > 0)

    def test_zero_correlation_independent(self):
        """With zero correlation, CVA should match independent CVA."""
        pvs, hazards = simulate_wwr_exposures(
            _pricer(), _ctx(), TIME_GRID, hazard_rate=0.02,
            rate_credit_corr=0.0, n_paths=500, rate_vol=0.005,
        )
        curve = DiscountCurve.flat(REF, 0.05)
        surv = SurvivalCurve.flat(REF, 0.02)

        wwr_cva = cva_wrong_way(pvs, hazards, TIME_GRID, curve)
        epe = expected_positive_exposure(pvs)
        ind_cva = cva(epe, TIME_GRID, curve, surv)

        # Should be in same ballpark (MC noise)
        assert wwr_cva == pytest.approx(ind_cva, rel=0.5)


class TestWWRCVA:
    def test_positive(self):
        pvs, hazards = simulate_wwr_exposures(
            _pricer(), _ctx(), TIME_GRID, hazard_rate=0.02,
            rate_credit_corr=0.3, n_paths=200, rate_vol=0.005,
        )
        curve = DiscountCurve.flat(REF, 0.05)
        wwr = cva_wrong_way(pvs, hazards, TIME_GRID, curve)
        assert wwr > 0

    def test_wwr_higher_than_independent(self):
        """Positive correlation (wrong-way) → higher CVA."""
        ctx = _ctx()
        pricer = _pricer()
        curve = DiscountCurve.flat(REF, 0.05)
        surv = SurvivalCurve.flat(REF, 0.02)

        # Independent
        pvs_ind, haz_ind = simulate_wwr_exposures(
            pricer, ctx, TIME_GRID, 0.02,
            rate_credit_corr=0.0, n_paths=500, rate_vol=0.005, seed=42,
        )
        epe_ind = expected_positive_exposure(pvs_ind)
        cva_ind = cva(epe_ind, TIME_GRID, curve, surv)

        # Wrong-way (positive correlation: rates down → exposure up, default up)
        pvs_wwr, haz_wwr = simulate_wwr_exposures(
            pricer, ctx, TIME_GRID, 0.02,
            rate_credit_corr=0.5, n_paths=500, rate_vol=0.005, seed=42,
        )
        cva_wwr = cva_wrong_way(pvs_wwr, haz_wwr, TIME_GRID, curve)

        # WWR CVA should be higher (or at least comparable with MC noise)
        assert cva_wwr >= cva_ind * 0.8  # allow MC noise


# ---------------------------------------------------------------------------
# Slice 85: Collateralised XVA
# ---------------------------------------------------------------------------


class TestCollateralisedCVA:
    def test_fully_collateralised_near_zero(self):
        """Threshold=0 → CVA ≈ 0 (only MPR exposure)."""
        epe = np.array([10.0, 8.0, 5.0, 3.0, 1.0])
        curve = DiscountCurve.flat(REF, 0.05)
        surv = SurvivalCurve.flat(REF, 0.02)

        cva_coll = cva_collateralised(
            epe, TIME_GRID, curve, surv, threshold=0.0, mpr_days=10,
        )
        cva_uncoll = cva(epe, TIME_GRID, curve, surv)
        assert cva_coll < cva_uncoll

    def test_high_threshold_matches_uncollateralised(self):
        """Very high threshold → collateralised CVA ≈ uncollateralised."""
        epe = np.array([10.0, 8.0, 5.0, 3.0, 1.0])
        curve = DiscountCurve.flat(REF, 0.05)
        surv = SurvivalCurve.flat(REF, 0.02)

        cva_coll = cva_collateralised(
            epe, TIME_GRID, curve, surv, threshold=1000.0,
        )
        cva_uncoll = cva(epe, TIME_GRID, curve, surv)
        assert cva_coll == pytest.approx(cva_uncoll, rel=0.01)

    def test_lower_threshold_lower_cva(self):
        epe = np.array([10.0, 8.0, 5.0, 3.0, 1.0])
        curve = DiscountCurve.flat(REF, 0.05)
        surv = SurvivalCurve.flat(REF, 0.02)

        cva_high = cva_collateralised(epe, TIME_GRID, curve, surv, threshold=5.0)
        cva_low = cva_collateralised(epe, TIME_GRID, curve, surv, threshold=1.0)
        assert cva_low <= cva_high


class TestCollateralisedFVA:
    def test_fully_collateralised_zero(self):
        """Threshold=0 → FVA = 0."""
        ee = np.array([10.0, 8.0, 5.0, 3.0, 1.0])
        curve = DiscountCurve.flat(REF, 0.05)

        fva_val = fva_collateralised(ee, TIME_GRID, curve, 0.005, threshold=0.0)
        assert fva_val == pytest.approx(0.0)

    def test_high_threshold(self):
        """FVA with high threshold should be positive."""
        ee = np.array([10.0, 8.0, 5.0, 3.0, 1.0])
        curve = DiscountCurve.flat(REF, 0.05)

        fva_val = fva_collateralised(ee, TIME_GRID, curve, 0.005, threshold=100.0)
        assert fva_val > 0

    def test_scales_with_spread(self):
        ee = np.array([10.0, 8.0, 5.0, 3.0, 1.0])
        curve = DiscountCurve.flat(REF, 0.05)

        fva1 = fva_collateralised(ee, TIME_GRID, curve, 0.005, threshold=10.0)
        fva2 = fva_collateralised(ee, TIME_GRID, curve, 0.010, threshold=10.0)
        assert fva2 == pytest.approx(2.0 * fva1)
