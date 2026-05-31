"""Tests for the unified inflation unit framework (UDI/UF/UVR/CER)."""

import pytest
import math
from datetime import date
from dateutil.relativedelta import relativedelta

REF = date(2024, 11, 4)


class TestInflationUnitRegistry:
    def test_list_units(self):
        from pricebook.fixed_income.inflation_unit import list_inflation_units
        units = list_inflation_units()
        assert units == ["CER", "UDI", "UF", "UVR"]

    def test_get_udi(self):
        from pricebook.fixed_income.inflation_unit import get_inflation_unit
        u = get_inflation_unit("UDI")
        assert u.currency == "MXN"
        assert u.coupon_frequency_months == 6

    def test_get_uf(self):
        from pricebook.fixed_income.inflation_unit import get_inflation_unit
        u = get_inflation_unit("UF")
        assert u.currency == "CLP"
        assert u.typical_value > 30_000  # ~37,000 CLP

    def test_get_uvr(self):
        from pricebook.fixed_income.inflation_unit import get_inflation_unit
        u = get_inflation_unit("UVR")
        assert u.currency == "COP"

    def test_get_cer(self):
        from pricebook.fixed_income.inflation_unit import get_inflation_unit
        u = get_inflation_unit("CER")
        assert u.currency == "ARS"

    def test_unknown_raises(self):
        from pricebook.fixed_income.inflation_unit import get_inflation_unit
        with pytest.raises(ValueError, match="Unknown inflation unit"):
            get_inflation_unit("XYZ")

    def test_compare_units(self):
        from pricebook.fixed_income.inflation_unit import compare_units
        table = compare_units()
        assert len(table) == 4
        names = [r["name"] for r in table]
        assert "UDI" in names


class TestInflationUnitBond:
    """Test generic inflation unit bond pricing across all 4 units."""

    def _make_real_curve(self, real_rate: float = 0.02):
        """Build a simple real discount curve."""
        from pricebook.core.discount_curve import DiscountCurve
        from pricebook.core.interpolation import InterpolationMethod
        dates = [REF + relativedelta(years=y) for y in [1, 2, 3, 5, 10, 20, 30]]
        dfs = [math.exp(-real_rate * y) for y in [1, 2, 3, 5, 10, 20, 30]]
        return DiscountCurve(REF, dates, dfs,
                             interpolation=InterpolationMethod.LOG_LINEAR)

    def test_udi_bond(self):
        from pricebook.fixed_income.inflation_unit import InflationUnitBond
        curve = self._make_real_curve(0.04)
        bond = InflationUnitBond("UDI", REF, REF + relativedelta(years=10),
                                 real_coupon=0.04, base_unit_value=7.8)
        r = bond.price(REF, curve, current_unit_value=8.2)
        assert r.real_price > 0
        assert r.nominal_price > r.real_price  # MXN value > UDI value
        assert r.unit_name == "UDI"
        assert r.currency == "MXN"
        assert r.indexation_ratio > 1.0  # 8.2/7.8

    def test_uf_bond(self):
        from pricebook.fixed_income.inflation_unit import InflationUnitBond
        curve = self._make_real_curve(0.02)
        bond = InflationUnitBond("UF", REF, REF + relativedelta(years=5),
                                 real_coupon=0.03, face_units=1000)
        r = bond.price(REF, curve, current_unit_value=37_500.0)
        assert r.real_price > 0
        assert r.nominal_price == pytest.approx(r.real_price * 37_500.0)
        assert r.unit_name == "UF"

    def test_uvr_bond(self):
        from pricebook.fixed_income.inflation_unit import InflationUnitBond
        curve = self._make_real_curve(0.035)
        bond = InflationUnitBond("UVR", REF, REF + relativedelta(years=10),
                                 real_coupon=0.035)
        r = bond.price(REF, curve, current_unit_value=350.0)
        assert r.real_price > 0
        assert r.nominal_price == pytest.approx(r.real_price * 350.0)

    def test_cer_bond(self):
        from pricebook.fixed_income.inflation_unit import InflationUnitBond
        curve = self._make_real_curve(0.05)
        bond = InflationUnitBond("CER", REF, REF + relativedelta(years=3),
                                 real_coupon=0.05, base_unit_value=1100.0)
        r = bond.price(REF, curve, current_unit_value=1200.0)
        assert r.indexation_ratio == pytest.approx(1200.0 / 1100.0)
        assert r.currency == "ARS"

    def test_real_yield_positive(self):
        from pricebook.fixed_income.inflation_unit import InflationUnitBond
        curve = self._make_real_curve(0.03)
        bond = InflationUnitBond("UDI", REF, REF + relativedelta(years=5),
                                 real_coupon=0.02)
        r = bond.price(REF, curve, current_unit_value=8.2)
        assert r.real_yield > 0

    def test_par_bond_near_face(self):
        """Bond priced at par when coupon ≈ discount rate."""
        from pricebook.fixed_income.inflation_unit import InflationUnitBond
        curve = self._make_real_curve(0.04)
        bond = InflationUnitBond("UDI", REF, REF + relativedelta(years=10),
                                 real_coupon=0.04)
        r = bond.price(REF, curve, current_unit_value=8.2)
        # Should be approximately face (100)
        assert 95 < r.real_price < 105


class TestDualCurveBreakeven:
    def test_positive_bei(self):
        """Nominal rate > real rate → positive breakeven inflation."""
        from pricebook.fixed_income.inflation_unit import dual_curve_breakeven
        from pricebook.core.discount_curve import DiscountCurve
        from pricebook.core.interpolation import InterpolationMethod

        dates = [REF + relativedelta(years=y) for y in [1, 2, 5, 10, 30]]
        nom_dfs = [math.exp(-0.10 * y) for y in [1, 2, 5, 10, 30]]
        real_dfs = [math.exp(-0.03 * y) for y in [1, 2, 5, 10, 30]]

        nom_curve = DiscountCurve(REF, dates, nom_dfs,
                                   interpolation=InterpolationMethod.LOG_LINEAR)
        real_curve = DiscountCurve(REF, dates, real_dfs,
                                   interpolation=InterpolationMethod.LOG_LINEAR)

        bei = dual_curve_breakeven(nom_curve, real_curve, [1, 5, 10])
        for row in bei:
            assert row["bei"] > 0
            assert row["bei"] == pytest.approx(0.07, abs=0.01)

    def test_flat_curves_zero_bei(self):
        """Equal nominal and real rates → zero BEI."""
        from pricebook.fixed_income.inflation_unit import dual_curve_breakeven
        from pricebook.core.discount_curve import DiscountCurve
        from pricebook.core.interpolation import InterpolationMethod

        dates = [REF + relativedelta(years=y) for y in [1, 5, 10]]
        dfs = [math.exp(-0.05 * y) for y in [1, 5, 10]]

        curve = DiscountCurve(REF, dates, dfs,
                               interpolation=InterpolationMethod.LOG_LINEAR)
        bei = dual_curve_breakeven(curve, curve, [1, 5, 10])
        for row in bei:
            assert abs(row["bei"]) < 1e-10
