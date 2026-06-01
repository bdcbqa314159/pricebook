"""Tests for swaption conventions, synthetic data, HW per currency."""

import pytest
import math
from datetime import date
from dateutil.relativedelta import relativedelta

REF = date(2024, 11, 4)


class TestSwaptionConventions:
    def test_usd(self):
        from pricebook.options.swaption_conventions import get_swaption_convention
        c = get_swaption_convention("USD")
        assert c.quote_type.value == "black"
        assert c.smile_type.value == "shifted_sabr"

    def test_eur_normal(self):
        from pricebook.options.swaption_conventions import get_swaption_convention
        c = get_swaption_convention("EUR")
        assert c.quote_type.value == "normal"

    def test_brl_bus252(self):
        from pricebook.options.swaption_conventions import get_swaption_convention
        c = get_swaption_convention("BRL")
        assert c.fixed_day_count.value == "BUS/252"

    def test_list_currencies(self):
        from pricebook.options.swaption_conventions import list_swaption_currencies
        ccys = list_swaption_currencies()
        assert "USD" in ccys
        assert "EUR" in ccys
        assert len(ccys) >= 10

    def test_unknown_raises(self):
        from pricebook.options.swaption_conventions import get_swaption_convention
        with pytest.raises(ValueError):
            get_swaption_convention("XYZ")


class TestSyntheticSwaptionData:
    def test_usd_atm_surface(self):
        from pricebook.options.synthetic_swaption_data import synthetic_atm_surface
        cube = synthetic_atm_surface("USD", REF)
        vol = cube.vol_by_years(5.0, 10.0)
        assert 0.002 < vol < 0.010  # reasonable range

    def test_brl_higher_vol(self):
        from pricebook.options.synthetic_swaption_data import synthetic_atm_surface
        usd = synthetic_atm_surface("USD", REF)
        brl = synthetic_atm_surface("BRL", REF)
        assert brl.vol_by_years(5.0, 10.0) > usd.vol_by_years(5.0, 10.0)

    def test_jpy_lower_vol(self):
        from pricebook.options.synthetic_swaption_data import synthetic_atm_surface
        usd = synthetic_atm_surface("USD", REF)
        jpy = synthetic_atm_surface("JPY", REF)
        assert jpy.vol_by_years(5.0, 10.0) < usd.vol_by_years(5.0, 10.0)

    def test_smile_data(self):
        from pricebook.options.synthetic_swaption_data import synthetic_smile_data
        smile = synthetic_smile_data("USD", REF)
        assert len(smile) > 0
        for key, data in smile.items():
            assert "forward" in data
            assert len(data["strikes"]) == 3

    def test_hw_targets(self):
        from pricebook.options.synthetic_swaption_data import synthetic_hw_targets
        targets = synthetic_hw_targets("USD", REF)
        assert len(targets) >= 4
        assert all(v > 0 for v in targets.values())


class TestHWPerCurrency:
    def _make_curve(self, rate=0.04):
        from pricebook.core.discount_curve import DiscountCurve
        from pricebook.core.interpolation import InterpolationMethod
        dates = [REF + relativedelta(years=y) for y in range(1, 35)]
        dfs = [math.exp(-rate * y) for y in range(1, 35)]
        return DiscountCurve(REF, dates, dfs, interpolation=InterpolationMethod.LOG_LINEAR)

    def test_usd_calibrated(self):
        from pricebook.models.hw_per_currency import calibrate_hw_for_currency
        curve = self._make_curve(0.05)
        result = calibrate_hw_for_currency("USD", REF, curve, use_synthetic=True)
        assert result.a > 0
        assert result.sigma > 0
        assert result.source in ("calibrated_synthetic", "default")

    def test_jpy_low_vol(self):
        from pricebook.models.hw_per_currency import calibrate_hw_for_currency
        curve = self._make_curve(0.001)
        result = calibrate_hw_for_currency("JPY", REF, curve, use_synthetic=True)
        assert result.a > 0

    def test_default_fallback(self):
        from pricebook.models.hw_per_currency import calibrate_hw_for_currency
        curve = self._make_curve(0.04)
        result = calibrate_hw_for_currency("USD", REF, curve, use_synthetic=False)
        assert result.source == "default"
        assert result.a == pytest.approx(0.03)

    def test_em_higher_params(self):
        from pricebook.models.hw_per_currency import calibrate_hw_for_currency
        curve_usd = self._make_curve(0.05)
        curve_brl = self._make_curve(0.11)
        usd = calibrate_hw_for_currency("USD", REF, curve_usd, use_synthetic=False)
        brl = calibrate_hw_for_currency("BRL", REF, curve_brl, use_synthetic=False)
        # EM defaults have higher mean reversion + vol
        assert brl.a > usd.a
        assert brl.sigma > usd.sigma

    def test_list_currencies(self):
        from pricebook.models.hw_per_currency import list_hw_currencies
        ccys = list_hw_currencies()
        assert len(ccys) >= 30
        assert "USD" in ccys
        assert "BRL" in ccys
        assert "CNY" in ccys

    def test_to_dict(self):
        from pricebook.models.hw_per_currency import calibrate_hw_for_currency
        curve = self._make_curve()
        result = calibrate_hw_for_currency("EUR", REF, curve, use_synthetic=False)
        d = result.to_dict()
        assert d["currency"] == "EUR"
