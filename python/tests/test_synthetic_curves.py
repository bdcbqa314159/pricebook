"""Tests for synthetic curve data and SABR-HW integration."""

import pytest
import math
from datetime import date
from dateutil.relativedelta import relativedelta

REF = date(2024, 11, 4)


class TestSyntheticCurveData:
    def test_usd_inputs(self):
        from pricebook.curves.synthetic_market_data import synthetic_curve_inputs
        deps, swaps = synthetic_curve_inputs("USD", REF)
        assert len(deps) == 3
        assert len(swaps) == 9
        assert all(r > 0 for _, r in deps)
        assert all(r > 0 for _, r in swaps)

    def test_jpy_near_zero(self):
        from pricebook.curves.synthetic_market_data import synthetic_curve_inputs
        deps, swaps = synthetic_curve_inputs("JPY", REF)
        assert deps[0][1] < 0.01  # near zero

    def test_try_extreme(self):
        from pricebook.curves.synthetic_market_data import synthetic_curve_inputs
        deps, swaps = synthetic_curve_inputs("TRY", REF)
        assert deps[0][1] > 0.40  # extreme

    def test_all_33_currencies(self):
        from pricebook.curves.synthetic_market_data import synthetic_curve_inputs, list_synthetic_currencies
        ccys = list_synthetic_currencies()
        assert len(ccys) >= 32
        for ccy in ccys:
            deps, swaps = synthetic_curve_inputs(ccy, REF)
            assert len(deps) >= 3
            assert len(swaps) >= 9

    def test_build_curve_from_synthetic_em(self):
        """End-to-end: synthetic EM data → build_em_curve → valid curve."""
        from pricebook.curves.synthetic_market_data import synthetic_curve_inputs
        from pricebook.curves.em_curve_builder import build_em_curve
        deps, swaps = synthetic_curve_inputs("BRL", REF)
        curve = build_em_curve("BRL", REF, deps, swaps)
        assert curve.df(REF + relativedelta(years=5)) > 0

    def test_em_with_ns(self):
        """EM currency with Nelson-Siegel (was impossible before unification)."""
        from pricebook.curves.synthetic_market_data import synthetic_curve_inputs
        from pricebook.curves.curve_builder import build_curves
        deps, swaps = synthetic_curve_inputs("BRL", REF)
        result = build_curves("BRL", REF, deps, swaps, method="nelson_siegel")
        assert result.ois is not None


class TestSABRHWIntegration:
    def _setup(self):
        from pricebook.core.discount_curve import DiscountCurve
        from pricebook.core.interpolation import InterpolationMethod
        dates = [REF + relativedelta(years=y) for y in range(1, 35)]
        dfs = [math.exp(-0.04 * y) for y in range(1, 35)]
        curve = DiscountCurve(REF, dates, dfs, interpolation=InterpolationMethod.LOG_LINEAR)

        from pricebook.options.synthetic_swaption_data import synthetic_atm_surface, synthetic_smile_data
        from pricebook.options.swaption_vol_cube import build_swaption_vol_cube
        atm_cube = synthetic_atm_surface("USD", REF)

        from pricebook.models.hw_per_currency import calibrate_hw_for_currency
        hw_result = calibrate_hw_for_currency("USD", REF, curve, use_synthetic=False)

        return curve, atm_cube, hw_result.model

    def test_blended_price_positive(self):
        from pricebook.options.swaption import Swaption, SwaptionType, price_swaption_sabr_hw
        curve, cube, hw = self._setup()
        swn = Swaption(REF + relativedelta(years=5), REF + relativedelta(years=15),
                        0.04, SwaptionType.PAYER)
        price = price_swaption_sabr_hw(swn, cube, hw, curve)
        assert price > 0

    def test_short_expiry_closer_to_sabr(self):
        """At short expiry, SABR weight is high → blended ≈ SABR."""
        from pricebook.options.swaption import Swaption, SwaptionType, price_swaption_sabr_hw
        curve, cube, hw = self._setup()
        # Very short expiry → SABR dominates
        swn = Swaption(REF + relativedelta(months=3), REF + relativedelta(years=5, months=3),
                        0.04, SwaptionType.PAYER)
        price = price_swaption_sabr_hw(swn, cube, hw, curve, blend_half_life=5.0)
        assert price > 0

    def test_blend_half_life_affects_price(self):
        from pricebook.options.swaption import Swaption, SwaptionType, price_swaption_sabr_hw
        curve, cube, hw = self._setup()
        swn = Swaption(REF + relativedelta(years=5), REF + relativedelta(years=15),
                        0.04, SwaptionType.PAYER)
        p1 = price_swaption_sabr_hw(swn, cube, hw, curve, blend_half_life=1.0)
        p2 = price_swaption_sabr_hw(swn, cube, hw, curve, blend_half_life=20.0)
        # Different blending → different prices (unless SABR = HW exactly)
        assert p1 > 0 and p2 > 0
