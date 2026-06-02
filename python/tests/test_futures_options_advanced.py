"""Tests for futures options, commodity options, and spread options."""

import pytest
import math
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta

REF = date(2024, 11, 4)


# ═══════════════════════════════════════════════════════════════
# Futures Options (F1)
# ═══════════════════════════════════════════════════════════════

class TestFuturesOptions:
    def test_call_positive(self):
        from pricebook.options.futures_options import FuturesOption
        opt = FuturesOption("ES", 5000, 5000, REF + relativedelta(months=3), 0.20, "call")
        r = opt.price(REF)
        assert r.price > 0

    def test_put_positive(self):
        from pricebook.options.futures_options import FuturesOption
        opt = FuturesOption("ES", 5000, 5000, REF + relativedelta(months=3), 0.20, "put")
        r = opt.price(REF)
        assert r.price > 0

    def test_call_delta_positive(self):
        from pricebook.options.futures_options import FuturesOption
        opt = FuturesOption("ES", 5000, 5000, REF + relativedelta(months=3), 0.20, "call")
        r = opt.price(REF)
        assert 0 < r.delta < 1

    def test_put_delta_negative(self):
        from pricebook.options.futures_options import FuturesOption
        opt = FuturesOption("ES", 5000, 5000, REF + relativedelta(months=3), 0.20, "put")
        r = opt.price(REF)
        assert -1 < r.delta < 0

    def test_put_call_parity(self):
        """C − P = (F − K) × df."""
        from pricebook.options.futures_options import FuturesOption
        F, K, T_months = 5000, 4900, 3
        exp = REF + relativedelta(months=T_months)
        c = FuturesOption("ES", F, K, exp, 0.20, "call")
        p = FuturesOption("ES", F, K, exp, 0.20, "put")
        rc = c.price(REF, rate=0.04)
        rp = p.price(REF, rate=0.04)
        T = rc.expiry_years
        df = math.exp(-0.04 * T)
        assert rc.price - rp.price == pytest.approx((F - K) * df, rel=0.01)

    def test_dollar_greeks(self):
        from pricebook.options.futures_options import FuturesOption
        opt = FuturesOption("ES", 5000, 5000, REF + relativedelta(months=3), 0.20, "call", n_contracts=10)
        r = opt.price(REF)
        assert r.delta_dollars == pytest.approx(r.delta * 50.0 * 10)
        assert r.price_total == pytest.approx(r.price * 50.0 * 10)

    def test_commodity_spec(self):
        from pricebook.options.futures_options import FuturesOption
        opt = FuturesOption("CL", 75, 80, REF + relativedelta(months=2), 0.35, "call")
        r = opt.price(REF)
        assert r.price > 0
        assert r.price_total == pytest.approx(r.price * 1000.0)

    def test_moneyness(self):
        from pricebook.options.futures_options import FuturesOption
        itm = FuturesOption("ES", 5100, 5000, REF + relativedelta(months=3), 0.20, "call")
        otm = FuturesOption("ES", 4900, 5000, REF + relativedelta(months=3), 0.20, "call")
        assert itm.moneyness() > 1.0
        assert otm.moneyness() < 1.0

    def test_strip(self):
        from pricebook.options.futures_options import futures_option_strip
        prices = [5000, 5010, 5020]
        expiries = [REF + relativedelta(months=m) for m in [1, 2, 3]]
        vols = [0.22, 0.21, 0.20]
        results = futures_option_strip("ES", prices, 5000, expiries, vols, REF)
        assert len(results) == 3
        assert all(r.price > 0 for r in results)

    def test_vol_surface(self):
        from pricebook.options.futures_options import futures_option_vol_surface, interpolate_vol
        strikes = [4800, 4900, 5000, 5100, 5200]
        expiries = [0.25, 0.50, 1.0]
        vols = [
            [0.24, 0.22, 0.20, 0.21, 0.23],
            [0.23, 0.21, 0.19, 0.20, 0.22],
            [0.22, 0.20, 0.18, 0.19, 0.21],
        ]
        surface = futures_option_vol_surface(5000, strikes, expiries, vols)
        v = interpolate_vol(surface, 0.35, 4950)
        assert 0.15 < v < 0.30

    def test_to_dict(self):
        from pricebook.options.futures_options import FuturesOption
        opt = FuturesOption("ES", 5000, 5000, REF + relativedelta(months=3), 0.20)
        r = opt.price(REF)
        d = r.to_dict()
        assert "delta_dollars" in d
        assert "gamma_dollars" in d


# ═══════════════════════════════════════════════════════════════
# Commodity Options (F2)
# ═══════════════════════════════════════════════════════════════

class TestCommodityOptions:
    def test_crude_call(self):
        from pricebook.commodity.commodity_options import commodity_option_price
        r = commodity_option_price(75.0, 80.0, 0.35, REF + relativedelta(months=3), REF,
                                    "call", ticker="CL", multiplier=1000)
        assert r.price > 0
        assert r.price_per_contract == pytest.approx(r.price * 1000)

    def test_gold_put(self):
        from pricebook.commodity.commodity_options import commodity_option_price
        r = commodity_option_price(2000, 2050, 0.18, REF + relativedelta(months=6), REF,
                                    "put", ticker="GC", multiplier=100)
        assert r.price > 0

    def test_seasonal_vol_nat_gas(self):
        """Nat gas winter vol > summer vol."""
        from pricebook.commodity.commodity_options import seasonal_vol
        winter = seasonal_vol(0.40, 1, "NG")  # January
        summer = seasonal_vol(0.40, 6, "NG")  # June
        assert winter > summer

    def test_seasonal_vol_corn(self):
        """Corn summer vol > winter vol."""
        from pricebook.commodity.commodity_options import seasonal_vol
        summer = seasonal_vol(0.30, 7, "ZC")  # July
        winter = seasonal_vol(0.30, 12, "ZC")  # December
        assert summer > winter

    def test_samuelson_effect(self):
        """Front-month vol > back-month vol."""
        from pricebook.commodity.commodity_options import vol_term_structure
        front = vol_term_structure(0.30, 0.1)
        back = vol_term_structure(0.30, 1.0)
        assert front > back

    def test_option_strip(self):
        from pricebook.commodity.commodity_options import commodity_option_strip
        prices = [75.0, 74.5, 74.0]
        expiries = [REF + relativedelta(months=m) for m in [1, 2, 3]]
        months = [12, 1, 2]
        results = commodity_option_strip(prices, 75.0, 0.35, expiries, months, REF, ticker="CL")
        assert len(results) == 3
        assert all(r.price >= 0 for r in results)

    def test_implied_vol(self):
        from pricebook.commodity.commodity_options import (
            commodity_option_price, commodity_implied_vol,
        )
        r = commodity_option_price(75.0, 80.0, 0.35, REF + relativedelta(months=3), REF,
                                    apply_seasonal=False, apply_samuelson=False)
        iv = commodity_implied_vol(r.price, 75.0, 80.0, r.expiry_years)
        assert iv == pytest.approx(0.35, abs=0.01)

    def test_to_dict(self):
        from pricebook.commodity.commodity_options import commodity_option_price
        r = commodity_option_price(75.0, 75.0, 0.35, REF + relativedelta(months=3), REF)
        d = r.to_dict()
        assert "seasonal_adj" in d
        assert "commodity" in d


# ═══════════════════════════════════════════════════════════════
# Spread Options (F3)
# ═══════════════════════════════════════════════════════════════

class TestSpreadOptions:
    def test_kirk_call(self):
        from pricebook.commodity.spread_options import kirk_spread_option
        r = kirk_spread_option(80, 75, 3, 0.30, 0.25, 0.85, 0.5)
        assert r.price > 0

    def test_kirk_put(self):
        from pricebook.commodity.spread_options import kirk_spread_option
        r = kirk_spread_option(80, 75, 3, 0.30, 0.25, 0.85, 0.5, option_type="put")
        assert r.price > 0

    def test_put_call_parity_spread(self):
        """C − P = df × (F1 − F2 − K)."""
        from pricebook.commodity.spread_options import kirk_spread_option
        F1, F2, K = 80, 75, 3
        c = kirk_spread_option(F1, F2, K, 0.30, 0.25, 0.85, 0.5)
        p = kirk_spread_option(F1, F2, K, 0.30, 0.25, 0.85, 0.5, option_type="put")
        df = math.exp(-0.04 * 0.5)
        assert c.price - p.price == pytest.approx(df * (F1 - F2 - K), rel=0.02)

    def test_higher_corr_lower_spread_vol(self):
        """Higher correlation → lower spread vol → lower option price."""
        from pricebook.commodity.spread_options import kirk_spread_option
        low_corr = kirk_spread_option(80, 75, 2, 0.30, 0.25, 0.50, 0.5)
        high_corr = kirk_spread_option(80, 75, 2, 0.30, 0.25, 0.95, 0.5)
        assert high_corr.price < low_corr.price

    def test_crack_spread_option(self):
        from pricebook.commodity.spread_options import crack_spread_option
        r = crack_spread_option(2.50, 75, 0.10, 0.35, 0.30, 0.80, 0.25)
        assert r.price >= 0

    def test_calendar_spread_option(self):
        from pricebook.commodity.spread_options import calendar_spread_option
        r = calendar_spread_option(75, 74, 0.5, 0.35, 0.30, 0.95, 0.25)
        assert r.price >= 0

    def test_spread_greeks(self):
        from pricebook.commodity.spread_options import kirk_spread_option
        r = kirk_spread_option(80, 75, 2, 0.30, 0.25, 0.85, 0.5)
        assert r.delta_asset1 > 0   # long first asset
        assert r.delta_asset2 < 0   # short second asset
        assert r.vega1 > 0
        assert r.vega2 != 0  # vega2 sign depends on correlation

    def test_cross_gamma(self):
        from pricebook.commodity.spread_options import kirk_spread_option
        r = kirk_spread_option(80, 75, 2, 0.30, 0.25, 0.85, 0.5)
        assert r.cross_gamma != 0  # cross-gamma should be non-zero

    def test_to_dict(self):
        from pricebook.commodity.spread_options import kirk_spread_option
        r = kirk_spread_option(80, 75, 2, 0.30, 0.25, 0.85, 0.5)
        d = r.to_dict()
        assert "cross_gamma" in d
        assert "correlation_sensitivity" in d
