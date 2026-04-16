"""Tests for commodity basis and locational spreads."""

import math

import numpy as np
import pytest

from pricebook.commodity_basis import (
    BasisCurve,
    GasBasisResult,
    PowerLocationalBasis,
    QualityBasisResult,
    WTIBrentResult,
    basis_curve_from_futures,
    gas_basis_curve,
    power_locational_basis,
    quality_basis,
    wti_brent_basis,
)


# ---- BasisCurve ----

class TestBasisCurve:
    def test_construct(self):
        curve = basis_curve_from_futures(
            tenors=[0.25, 1.0, 2.0],
            benchmark_forwards=[80, 82, 85],
            derivative_forwards=[82, 83, 86],
        )
        assert isinstance(curve, BasisCurve)
        np.testing.assert_allclose(curve.basis_values, [2, 1, 1])

    def test_interpolation(self):
        curve = basis_curve_from_futures([0.5, 1.0, 2.0], [80, 82, 85], [82, 83, 86])
        assert curve.basis(0.75) == pytest.approx(1.5, abs=0.1)

    def test_flat_extrapolation(self):
        curve = basis_curve_from_futures([0.5, 1.0, 2.0], [80, 82, 85], [82, 83, 86])
        assert curve.basis(0.1) == 2.0   # flat below first tenor
        assert curve.basis(5.0) == 1.0   # flat above last tenor

    def test_forward_price(self):
        curve = basis_curve_from_futures([0.5, 1.0, 2.0], [80, 82, 85], [82, 83, 86])
        F_deriv = curve.forward_price(1.0, benchmark_forward=82)
        assert F_deriv == pytest.approx(83)


# ---- WTI/Brent ----

class TestWTIBrent:
    def test_basic(self):
        result = wti_brent_basis(
            tenors=[0.25, 1.0, 2.0, 5.0],
            wti_forwards=[80, 82, 84, 88],
            brent_forwards=[83, 84, 85, 89],
        )
        assert isinstance(result, WTIBrentResult)
        assert result.near_month_basis == 3.0
        assert result.long_end_basis == 1.0

    def test_positive_basis(self):
        """Brent > WTI → positive basis."""
        result = wti_brent_basis([1.0], [80], [85])
        assert result.near_month_basis > 0

    def test_backwardation_ratio(self):
        """Decreasing basis → negative ratio."""
        result = wti_brent_basis([0.25, 1.0, 5.0], [80, 82, 90],
                                   [85, 83, 91])
        assert result.backwardation_ratio < 0


# ---- Power ----

class TestPowerLocationalBasis:
    def test_basic(self):
        result = power_locational_basis(
            tenors=[0.1, 0.5, 1.0],
            hub_forwards=[50, 55, 60],
            node_forwards=[52, 57, 62],
        )
        assert isinstance(result, PowerLocationalBasis)
        np.testing.assert_allclose(result.basis_values, [2, 2, 2])
        assert result.mean_basis == 2.0

    def test_max_congestion(self):
        result = power_locational_basis([0.1, 0.5], [50, 55], [60, 53])
        # basis = [10, -2]; max abs = 10
        assert result.max_congestion == 10.0

    def test_zero_basis(self):
        result = power_locational_basis([0.1, 0.5], [50, 55], [50, 55])
        assert result.mean_basis == 0.0
        assert result.max_congestion == 0.0


# ---- Gas ----

class TestGasBasis:
    def test_basic(self):
        tenors = [i / 12 for i in range(1, 13)]
        hh = [3.0] * 12
        # CityGate: winter premium
        regional = [4.5, 4.0, 3.5, 3.2, 3.1, 3.0, 3.0, 3.1, 3.3, 3.8, 4.2, 4.8]
        result = gas_basis_curve(tenors, hh, regional, hub_name="CityGate")
        assert isinstance(result, GasBasisResult)

    def test_peak_winter(self):
        """Winter premium → peak in first or last months (Jan/Dec)."""
        tenors = [i / 12 for i in range(1, 13)]
        hh = [3.0] * 12
        # Month 12 (Dec) highest
        regional = [3.5, 3.2, 3.0, 2.9, 2.8, 2.7, 2.7, 2.8, 2.9, 3.1, 3.5, 5.0]
        result = gas_basis_curve(tenors, hh, regional)
        assert result.seasonal_peak_month == 12

    def test_annual_mean(self):
        tenors = [i / 12 for i in range(1, 13)]
        hh = [3.0] * 12
        regional = [3.5] * 12
        result = gas_basis_curve(tenors, hh, regional)
        assert result.annual_mean_basis == pytest.approx(0.5)


# ---- Quality ----

class TestQualityBasis:
    def test_basic(self):
        result = quality_basis(reference_price=80)
        assert isinstance(result, QualityBasisResult)
        assert result.reference_price == 80
        assert result.adjusted_price == 80    # no adjustments

    def test_heavy_sour_discount(self):
        """Heavy sour crude (low API, high sulphur) → discount."""
        result = quality_basis(reference_price=80,
                                api_gravity_delta=-10,     # heavier
                                sulphur_pct_delta=+1.5)     # more sulphur
        assert result.adjusted_price < 80
        assert result.total_adjustment < 0

    def test_light_sweet_premium(self):
        """Light sweet crude → premium."""
        result = quality_basis(80, api_gravity_delta=+8, sulphur_pct_delta=-0.5)
        assert result.adjusted_price > 80

    def test_heat_content_premium(self):
        result = quality_basis(80, heat_content_delta=2.0, heat_coefficient=0.5)
        assert result.adjusted_price == 81.0

    def test_breakdown_keys(self):
        result = quality_basis(80)
        assert set(result.adjustments_breakdown.keys()) == {"api", "sulphur", "heat"}
