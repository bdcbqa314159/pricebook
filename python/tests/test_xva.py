"""Tests for CVA: exposure simulation and credit valuation adjustment."""

import pytest
import numpy as np
from datetime import date

from pricebook.xva import (
    simulate_exposures,
    expected_positive_exposure,
    expected_negative_exposure,
    expected_exposure,
    cva,
    dva,
    bilateral_cva,
    fva,
    mva,
    kva,
    XVAResult,
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


class TestDVA:
    def test_dva_positive(self):
        ene = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        curve = make_flat_curve(REF, 0.05)
        own_surv = make_flat_survival(REF, 0.01)
        val = dva(ene, TIME_GRID, curve, own_surv, own_recovery=0.4)
        assert val > 0

    def test_dva_zero_no_default(self):
        ene = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        curve = make_flat_curve(REF, 0.05)
        own_surv = make_flat_survival(REF, 0.0)
        val = dva(ene, TIME_GRID, curve, own_surv)
        assert val == pytest.approx(0.0, abs=1e-10)


class TestBilateralCVA:
    def test_bcva(self):
        cva_val = 10.0
        dva_val = 3.0
        assert bilateral_cva(cva_val, dva_val) == pytest.approx(7.0)

    def test_symmetric(self):
        """BCVA_A->B + BCVA_B->A = 0 (with same exposure profiles)."""
        epe = np.array([10.0, 8.0, 5.0, 3.0, 1.0])
        ene = np.array([10.0, 8.0, 5.0, 3.0, 1.0])
        curve = make_flat_curve(REF, 0.05)
        surv_a = make_flat_survival(REF, 0.02)
        surv_b = make_flat_survival(REF, 0.02)

        cva_a = cva(epe, TIME_GRID, curve, surv_b)
        dva_a = dva(ene, TIME_GRID, curve, surv_a)
        cva_b = cva(ene, TIME_GRID, curve, surv_a)
        dva_b = dva(epe, TIME_GRID, curve, surv_b)

        bcva_a = bilateral_cva(cva_a, dva_a)
        bcva_b = bilateral_cva(cva_b, dva_b)
        assert bcva_a + bcva_b == pytest.approx(0.0, abs=1e-10)


class TestFVA:
    def test_fva_positive_for_positive_exposure(self):
        ee = np.array([10.0, 8.0, 5.0, 3.0, 1.0])
        curve = make_flat_curve(REF, 0.05)
        val = fva(ee, TIME_GRID, curve, funding_spread=0.005)
        assert val > 0

    def test_fva_zero_no_spread(self):
        ee = np.array([10.0, 8.0, 5.0, 3.0, 1.0])
        curve = make_flat_curve(REF, 0.05)
        val = fva(ee, TIME_GRID, curve, funding_spread=0.0)
        assert val == pytest.approx(0.0)

    def test_fva_zero_no_exposure(self):
        ee = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        curve = make_flat_curve(REF, 0.05)
        val = fva(ee, TIME_GRID, curve, funding_spread=0.005)
        assert val == pytest.approx(0.0)

    def test_fva_scales_with_spread(self):
        ee = np.array([10.0, 8.0, 5.0, 3.0, 1.0])
        curve = make_flat_curve(REF, 0.05)
        fva1 = fva(ee, TIME_GRID, curve, funding_spread=0.005)
        fva2 = fva(ee, TIME_GRID, curve, funding_spread=0.010)
        assert fva2 == pytest.approx(2.0 * fva1)

    def test_fva_zero_fully_collateralised(self):
        """FVA = 0 when no uncollateralised exposure."""
        ee = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        curve = make_flat_curve(REF, 0.05)
        val = fva(ee, TIME_GRID, curve, funding_spread=0.005)
        assert val == pytest.approx(0.0)


class TestMVA:
    def test_mva_positive(self):
        im = np.array([100.0, 90.0, 70.0, 50.0, 20.0])
        curve = make_flat_curve(REF, 0.05)
        val = mva(im, TIME_GRID, curve, funding_spread=0.005)
        assert val > 0

    def test_mva_zero_no_im(self):
        im = np.zeros(5)
        curve = make_flat_curve(REF, 0.05)
        val = mva(im, TIME_GRID, curve, funding_spread=0.005)
        assert val == pytest.approx(0.0)

    def test_mva_scales_with_funding(self):
        im = np.array([100.0, 90.0, 70.0, 50.0, 20.0])
        curve = make_flat_curve(REF, 0.05)
        mva1 = mva(im, TIME_GRID, curve, funding_spread=0.005)
        mva2 = mva(im, TIME_GRID, curve, funding_spread=0.010)
        assert mva2 == pytest.approx(2.0 * mva1)


class TestKVA:
    def test_kva_positive(self):
        cap = np.array([50.0, 45.0, 35.0, 25.0, 10.0])
        curve = make_flat_curve(REF, 0.05)
        val = kva(cap, TIME_GRID, curve, hurdle_rate=0.10)
        assert val > 0

    def test_kva_zero_no_capital(self):
        cap = np.zeros(5)
        curve = make_flat_curve(REF, 0.05)
        val = kva(cap, TIME_GRID, curve, hurdle_rate=0.10)
        assert val == pytest.approx(0.0)

    def test_kva_increases_with_size(self):
        cap1 = np.array([50.0, 45.0, 35.0, 25.0, 10.0])
        cap2 = 2.0 * cap1
        curve = make_flat_curve(REF, 0.05)
        kva1 = kva(cap1, TIME_GRID, curve, hurdle_rate=0.10)
        kva2 = kva(cap2, TIME_GRID, curve, hurdle_rate=0.10)
        assert kva2 == pytest.approx(2.0 * kva1)


class TestXVAResult:
    def test_total(self):
        r = XVAResult(cva=10.0, dva=3.0, fva=2.0, mva=1.0, kva=0.5)
        assert r.bcva == pytest.approx(7.0)
        assert r.total == pytest.approx(10.5)  # 10 - 3 + 2 + 1 + 0.5

    def test_total_zero(self):
        r = XVAResult()
        assert r.total == pytest.approx(0.0)

    def test_collateralised_xva_small(self):
        """Fully collateralised + high credit -> XVA near 0."""
        r = XVAResult(cva=0.01, dva=0.01, fva=0.0, mva=0.0, kva=0.0)
        assert abs(r.total) < 0.1
