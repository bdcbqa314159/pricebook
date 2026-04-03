"""Tests for CVA: exposure simulation and credit valuation adjustment."""

import pytest
import numpy as np
from datetime import date

from pricebook.xva import (
    simulate_exposures,
    expected_positive_exposure,
    expected_negative_exposure,
    cva,
)
from pricebook.pricing_context import PricingContext
from pricebook.vol_surface import FlatVol
from pricebook.swaption import Swaption
from tests.conftest import make_flat_curve, make_flat_survival


REF = date(2024, 1, 15)
TIME_GRID = [0.5, 1.0, 2.0, 3.0, 5.0]


def _make_ctx():
    curve = make_flat_curve(REF, 0.05)
    return PricingContext(
        valuation_date=REF,
        discount_curve=curve,
        vol_surfaces={"ir": FlatVol(0.20)},
    )


class TestExposureSimulation:
    def test_epe_non_negative(self):
        ctx = _make_ctx()
        swn = Swaption(date(2025, 1, 15), date(2030, 1, 15), strike=0.05)
        pvs = simulate_exposures(lambda c: swn.pv_ctx(c), ctx, TIME_GRID, n_paths=50)
        epe = expected_positive_exposure(pvs)
        assert all(e >= 0 for e in epe)

    def test_ene_non_negative(self):
        ctx = _make_ctx()
        swn = Swaption(date(2025, 1, 15), date(2030, 1, 15), strike=0.05)
        pvs = simulate_exposures(lambda c: swn.pv_ctx(c), ctx, TIME_GRID, n_paths=50)
        ene = expected_negative_exposure(pvs)
        assert all(e >= 0 for e in ene)

    def test_pvs_shape(self):
        ctx = _make_ctx()
        swn = Swaption(date(2025, 1, 15), date(2030, 1, 15), strike=0.05)
        pvs = simulate_exposures(lambda c: swn.pv_ctx(c), ctx, TIME_GRID, n_paths=30)
        assert pvs.shape == (30, len(TIME_GRID))

    def test_swaption_exposure_profile(self):
        """Swaption EPE should be positive (option always has non-negative value)."""
        ctx = _make_ctx()
        swn = Swaption(date(2025, 1, 15), date(2030, 1, 15), strike=0.05)

        def safe_pricer(c):
            try:
                return swn.pv_ctx(c)
            except (ValueError, ZeroDivisionError):
                return 0.0

        pvs = simulate_exposures(safe_pricer, ctx, TIME_GRID, n_paths=100, rate_vol=0.005)
        epe = expected_positive_exposure(pvs)
        assert max(epe) > 0


class TestCVA:
    def test_cva_positive(self):
        epe = np.array([10.0, 8.0, 5.0, 3.0, 1.0])
        curve = make_flat_curve(REF, 0.05)
        surv = make_flat_survival(REF, 0.02)
        val = cva(epe, TIME_GRID, curve, surv, recovery=0.4)
        assert val > 0

    def test_cva_zero_when_no_default(self):
        epe = np.array([10.0, 8.0, 5.0, 3.0, 1.0])
        curve = make_flat_curve(REF, 0.05)
        surv = make_flat_survival(REF, 0.0)
        val = cva(epe, TIME_GRID, curve, surv, recovery=0.4)
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_cva_zero_exposure(self):
        epe = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        curve = make_flat_curve(REF, 0.05)
        surv = make_flat_survival(REF, 0.02)
        val = cva(epe, TIME_GRID, curve, surv)
        assert val == pytest.approx(0.0)

    def test_cva_increases_with_spread(self):
        epe = np.array([10.0, 8.0, 5.0, 3.0, 1.0])
        curve = make_flat_curve(REF, 0.05)
        surv_low = make_flat_survival(REF, 0.01)
        surv_high = make_flat_survival(REF, 0.05)
        cva_low = cva(epe, TIME_GRID, curve, surv_low)
        cva_high = cva(epe, TIME_GRID, curve, surv_high)
        assert cva_high > cva_low

    def test_cva_increases_with_lgd(self):
        epe = np.array([10.0, 8.0, 5.0, 3.0, 1.0])
        curve = make_flat_curve(REF, 0.05)
        surv = make_flat_survival(REF, 0.02)
        cva_low_r = cva(epe, TIME_GRID, curve, surv, recovery=0.6)
        cva_high_r = cva(epe, TIME_GRID, curve, surv, recovery=0.2)
        assert cva_high_r > cva_low_r
