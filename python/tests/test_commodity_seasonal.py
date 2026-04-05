"""Tests for commodity seasonality and storage."""

import math
import pytest
from datetime import date

from pricebook.commodity_seasonal import (
    SeasonalFactors,
    SeasonalForwardCurve,
    StorageCostModel,
    calendar_spread_option,
)
from pricebook.black76 import OptionType


REF = date(2024, 1, 15)


class TestSeasonalFactors:
    def test_twelve_factors(self):
        sf = SeasonalFactors.natural_gas()
        assert len(sf.factors) == 12

    def test_winter_premium(self):
        sf = SeasonalFactors.natural_gas()
        jan = sf.factor(date(2024, 1, 15))
        jul = sf.factor(date(2024, 7, 15))
        assert jan > jul  # winter > summer

    def test_flat(self):
        sf = SeasonalFactors.flat()
        assert all(f == 1.0 for f in sf.factors)

    def test_invalid_length(self):
        with pytest.raises(ValueError):
            SeasonalFactors([1.0] * 11)


class TestSeasonalForwardCurve:
    def test_winter_higher(self):
        curve = SeasonalForwardCurve(3.0, SeasonalFactors.natural_gas(), REF)
        jan_fwd = curve.forward(date(2025, 1, 15))
        jul_fwd = curve.forward(date(2025, 7, 15))
        assert jan_fwd > jul_fwd

    def test_flat_seasonal_equals_base(self):
        curve = SeasonalForwardCurve(3.0, SeasonalFactors.flat(), REF)
        assert curve.forward(date(2025, 6, 15)) == pytest.approx(3.0)

    def test_forwards_list(self):
        curve = SeasonalForwardCurve(3.0, SeasonalFactors.natural_gas(), REF)
        dates = [date(2025, m, 15) for m in range(1, 13)]
        fwds = curve.forwards(dates)
        assert len(fwds) == 12
        assert all(f > 0 for f in fwds)


class TestStorageCostModel:
    def test_net_carry(self):
        model = StorageCostModel(storage_cost_rate=0.03, convenience_yield=0.01)
        assert model.net_cost_of_carry == pytest.approx(0.02)

    def test_implied_forward(self):
        model = StorageCostModel(storage_cost_rate=0.02, convenience_yield=0.0)
        fwd = model.implied_forward(spot=100, rate=0.05, T=1.0)
        expected = 100 * math.exp(0.07)
        assert fwd == pytest.approx(expected)

    def test_convenience_yield_roundtrip(self):
        model = StorageCostModel(storage_cost_rate=0.02, convenience_yield=0.03)
        fwd = model.implied_forward(spot=100, rate=0.05, T=1.0)
        cy = model.implied_convenience_yield(100, fwd, 0.05, 1.0)
        assert cy == pytest.approx(0.03, rel=1e-10)

    def test_contango(self):
        """Positive cost of carry → contango (forward > spot)."""
        model = StorageCostModel(storage_cost_rate=0.05, convenience_yield=0.0)
        fwd = model.implied_forward(100, 0.05, 1.0)
        assert fwd > 100

    def test_backwardation(self):
        """High convenience yield → backwardation (forward < spot)."""
        model = StorageCostModel(storage_cost_rate=0.01, convenience_yield=0.10)
        fwd = model.implied_forward(100, 0.05, 1.0)
        assert fwd < 100


class TestCalendarSpreadOption:
    def test_positive(self):
        price = calendar_spread_option(
            forward_near=3.5, forward_far=3.0,
            vol_near=0.30, vol_far=0.25,
            correlation=0.8, T=0.5, df=0.975,
        )
        assert price > 0

    def test_contango_spread(self):
        """In contango, near < far → spread option on near-far is OTM."""
        price = calendar_spread_option(
            forward_near=3.0, forward_far=3.5,
            vol_near=0.30, vol_far=0.25,
            correlation=0.8, T=0.5, df=0.975,
        )
        # Should still have time value
        assert price >= 0

    def test_high_correlation_lower_price(self):
        """Higher correlation → lower spread vol → lower option price."""
        p_low = calendar_spread_option(
            3.1, 3.0, 0.30, 0.25, correlation=0.3, T=1.0, df=0.95, strike=0.1,
        )
        p_high = calendar_spread_option(
            3.1, 3.0, 0.30, 0.25, correlation=0.95, T=1.0, df=0.95, strike=0.1,
        )
        assert p_high < p_low
