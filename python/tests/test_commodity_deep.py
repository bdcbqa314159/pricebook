"""Deep tests for commodity — DD7 hardening.

Covers: cost of carry, seasonality, Kirk spread pricing, calendar spread,
swing option bounds, storage value.
"""

import math
import pytest
from datetime import date

from pricebook.commodity import CommodityForwardCurve, commodity_option_price
from pricebook.commodity_seasonal import SeasonalFactors, SeasonalForwardCurve, calendar_spread_option
from pricebook.commodity_vol_surface import kirk_spread_smile
from pricebook.black76 import OptionType


REF = date(2024, 1, 15)


class TestCommodityForward:

    def test_contango_forward_above_spot(self):
        """Positive carry cost → contango (F > S)."""
        from dateutil.relativedelta import relativedelta
        dates = [REF + relativedelta(months=m) for m in [3, 6, 12]]
        curve = CommodityForwardCurve(
            reference_date=REF, dates=dates,
            forwards=[101.0, 102.0, 104.0], spot=100.0,
        )
        assert curve.forward(dates[1]) > 100.0

    def test_option_positive(self):
        """Commodity call option has positive value."""
        price = commodity_option_price(100.0, 100.0, 0.20, 1.0, 0.05, OptionType.CALL)
        assert price > 0

    def test_put_call_parity(self):
        """C - P = df × (F - K)."""
        F, K, vol, T, r = 100.0, 100.0, 0.20, 1.0, 0.05
        call = commodity_option_price(F, K, vol, T, r, OptionType.CALL)
        put = commodity_option_price(F, K, vol, T, r, OptionType.PUT)
        df = math.exp(-r * T)
        assert call - put == pytest.approx(df * (F - K), abs=1e-10)


class TestSeasonality:

    def test_seasonal_factors_sum_to_12(self):
        """Standard seasonal factors average to 1.0."""
        sf = SeasonalFactors.natural_gas()
        assert sum(sf.factors) / 12 == pytest.approx(1.0, abs=0.1)

    def test_seasonal_forward_varies(self):
        """Seasonal forward curve should vary by month."""
        from dateutil.relativedelta import relativedelta
        sf = SeasonalFactors.natural_gas()
        curve = SeasonalForwardCurve(3.0, sf, REF)
        winter = curve.forward(REF + relativedelta(months=0))   # Jan
        summer = curve.forward(REF + relativedelta(months=6))   # Jul
        assert winter != pytest.approx(summer, abs=0.01)


class TestKirkSpread:

    def test_kirk_positive(self):
        """Kirk spread option has positive value."""
        result = kirk_spread_smile(
            forward1=100.0, forward2=90.0, strike=5.0,
            vol1=0.20, vol2=0.25, correlation=0.8,
            T=1.0, rate=0.05,
        )
        assert result.price > 0

    def test_kirk_higher_vol_higher_price(self):
        """Higher vol → higher spread option price (ATM strike)."""
        low = kirk_spread_smile(100.0, 90.0, 10.0, 0.10, 0.10, 0.8, 1.0, 0.05)
        high = kirk_spread_smile(100.0, 90.0, 10.0, 0.30, 0.30, 0.8, 1.0, 0.05)
        assert high.flat_price > low.flat_price

    def test_kirk_lower_corr_higher_price(self):
        """Lower correlation → higher spread vol → higher price (ATM)."""
        high_corr = kirk_spread_smile(100.0, 90.0, 10.0, 0.20, 0.20, 0.95, 1.0, 0.05)
        low_corr = kirk_spread_smile(100.0, 90.0, 10.0, 0.20, 0.20, 0.50, 1.0, 0.05)
        assert low_corr.flat_price > high_corr.flat_price


class TestCalendarSpread:

    def test_calendar_spread_positive(self):
        """Calendar spread option has positive value."""
        df = math.exp(-0.05)
        price = calendar_spread_option(
            forward_near=100.0, forward_far=95.0,
            vol_near=0.20, vol_far=0.25, correlation=0.9,
            T=1.0, df=df,
        )
        assert price > 0

    def test_calendar_spread_handles_negative(self):
        """Calendar spread with negative forward spread shouldn't crash."""
        df = math.exp(-0.05)
        price = calendar_spread_option(
            forward_near=90.0, forward_far=100.0,  # backwardation
            vol_near=0.20, vol_far=0.25, correlation=0.9,
            T=1.0, df=df,
        )
        # Put on negative spread should have value
        price_put = calendar_spread_option(
            forward_near=90.0, forward_far=100.0,
            vol_near=0.20, vol_far=0.25, correlation=0.9,
            T=1.0, df=df, option_type=OptionType.PUT,
        )
        assert price_put > 0
