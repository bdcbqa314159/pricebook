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


# ═══════════════════════════════════════════════════════════════
# VIX/Variance Futures (F4)
# ═══════════════════════════════════════════════════════════════

class TestVarianceFutures:
    def test_vix_fair_value_contango(self):
        """VIX futures typically trade above spot (contango)."""
        from pricebook.options.variance_futures import vix_futures_fair_value
        r = vix_futures_fair_value(15, 30)
        assert r.fair_value >= r.spot_vix  # contango

    def test_vix_mean_reversion(self):
        """High VIX reverts toward long-run level."""
        from pricebook.options.variance_futures import vix_futures_fair_value
        r = vix_futures_fair_value(40, 90, long_run_vol=20)
        assert r.fair_value < 40  # reverts down

    def test_variance_swap(self):
        from pricebook.options.variance_futures import variance_swap_price
        strikes = [90, 95, 100, 105, 110]
        calls = [12.0, 8.0, 5.0, 3.0, 1.5]
        puts = [2.0, 3.5, 5.5, 8.5, 12.0]
        r = variance_swap_price(100, strikes, calls, puts, 0.25)
        assert r.fair_vol > 0

    def test_term_structure(self):
        from pricebook.options.variance_futures import vix_term_structure
        ts = vix_term_structure(15, [(30, 16), (60, 17), (90, 17.5)])
        assert len(ts) == 3
        assert all(p.contango for p in ts)

    def test_to_dict(self):
        from pricebook.options.variance_futures import vix_futures_fair_value
        r = vix_futures_fair_value(15, 30)
        d = r.to_dict()
        assert "basis" in d
        assert "roll_yield_annual" in d


# ═══════════════════════════════════════════════════════════════
# Commodity Calibration (F5)
# ═══════════════════════════════════════════════════════════════

class TestCommodityCalibration:
    def test_schwartz_calibration(self):
        from pricebook.commodity.commodity_calibration import calibrate_schwartz
        spot = 75
        # Generate synthetic futures (slight contango)
        prices = [75 * math.exp(0.02 * T) for T in [0.25, 0.5, 1, 2, 3]]
        maturities = [0.25, 0.5, 1, 2, 3]
        r = calibrate_schwartz(spot, prices, maturities)
        assert r.kappa > 0
        assert r.sigma > 0
        assert r.rmse < 5

    def test_gibson_schwartz_calibration(self):
        from pricebook.commodity.commodity_calibration import calibrate_gibson_schwartz
        spot = 75
        prices = [75 * math.exp(0.01 * T) for T in [0.25, 0.5, 1, 2, 3]]
        maturities = [0.25, 0.5, 1, 2, 3]
        r = calibrate_gibson_schwartz(spot, prices, maturities)
        assert r.sigma_s > 0
        assert r.rmse < 5

    def test_seasonal_decomposition(self):
        from pricebook.commodity.commodity_calibration import seasonal_decomposition
        # Generate seasonal data
        prices = [70 + 5 * math.sin(2 * math.pi * m / 12) + m * 0.1 for m in range(24)]
        r = seasonal_decomposition(prices, period=12)
        assert len(r.seasonal_factors) == 12
        assert r.seasonal_factors.max() > r.seasonal_factors.min()

    def test_implied_convenience_yield(self):
        from pricebook.commodity.commodity_calibration import implied_convenience_yield_term
        # Backwardation → positive convenience yield
        spot = 80
        prices = [79, 78, 77]  # declining forward curve
        maturities = [0.25, 0.5, 1.0]
        result = implied_convenience_yield_term(spot, prices, maturities)
        assert len(result) == 3
        assert all(y > 0 for _, y in result)  # positive convenience yield


# ═══════════════════════════════════════════════════════════════
# SABR Convexity (F6)
# ═══════════════════════════════════════════════════════════════

class TestFuturesConvexity:
    def test_sabr_convexity_negative(self):
        """Convexity adjustment is negative (futures rate > forward rate)."""
        from pricebook.fixed_income.futures_convexity import sabr_convexity_adjustment
        r = sabr_convexity_adjustment(0.04, 2.0)
        assert r.adjustment_bp < 0
        assert r.forward_rate < r.futures_rate

    def test_sabr_increases_with_maturity(self):
        from pricebook.fixed_income.futures_convexity import sabr_convexity_adjustment
        short = sabr_convexity_adjustment(0.04, 0.5)
        long = sabr_convexity_adjustment(0.04, 5.0)
        assert abs(long.adjustment_bp) > abs(short.adjustment_bp)

    def test_hw_convexity(self):
        from pricebook.fixed_income.futures_convexity import hw_convexity_adjustment
        r = hw_convexity_adjustment(0.04, 2.0)
        assert r.adjustment_bp < 0

    def test_compare_models(self):
        from pricebook.fixed_income.futures_convexity import compare_convexity_models
        r = compare_convexity_models(0.04, 2.0)
        assert "sabr" in r
        assert "hw" in r
        # Both should be negative
        assert r["sabr"].adjustment_bp < 0
        assert r["hw"].adjustment_bp < 0

    def test_empirical(self):
        from pricebook.fixed_income.futures_convexity import empirical_convexity
        r = empirical_convexity([0.0405, 0.0412], [0.0400, 0.0405], [1.0, 2.0])
        assert len(r) == 2
        assert all(c.adjustment_bp > 0 for c in r)  # futures > OIS


# ═══════════════════════════════════════════════════════════════
# Cost of Carry (F7)
# ═══════════════════════════════════════════════════════════════

class TestCostOfCarry:
    def test_contango(self):
        from pricebook.fixed_income.cost_of_carry import cost_of_carry
        r = cost_of_carry(100, 102, 1.0, rate=0.04)
        assert r.contango is True
        assert r.forward_premium_pct > 0

    def test_backwardation(self):
        from pricebook.fixed_income.cost_of_carry import cost_of_carry
        r = cost_of_carry(100, 97, 1.0, rate=0.04)
        assert r.contango is False
        assert r.convenience_yield > r.risk_free_rate  # backwardation implies high CY

    def test_cash_and_carry_arb(self):
        from pricebook.fixed_income.cost_of_carry import cash_and_carry_arb
        # Futures too expensive relative to cost of carry
        r = cash_and_carry_arb(100, 110, 1.0, rate=0.04, storage_cost=0.01)
        assert r.profit_per_unit > 0
        assert r.feasible is True

    def test_no_arb(self):
        from pricebook.fixed_income.cost_of_carry import cash_and_carry_arb
        fair = 100 * math.exp(0.04 * 1.0)
        r = cash_and_carry_arb(100, fair, 1.0, rate=0.04)
        assert abs(r.profit_per_unit) < 0.01

    def test_carry_roll_decomposition(self):
        from pricebook.fixed_income.cost_of_carry import carry_roll_decomposition
        r = carry_roll_decomposition(100, 101, 102, 0.25, 0.50, rate=0.04)
        assert "carry" in r
        assert "roll" in r
        assert "contango" in r


# ═══════════════════════════════════════════════════════════════
# Futures Roll (F8)
# ═══════════════════════════════════════════════════════════════

class TestFuturesRoll:
    def test_roll_schedule(self):
        from pricebook.fixed_income.futures_roll import generate_roll_schedule
        schedule = generate_roll_schedule(
            date(2024, 1, 1), date(2024, 12, 31),
            [3, 6, 9, 12],
            [75, 76, 77, 78], [76, 77, 78, 79],
        )
        assert schedule.n_rolls > 0
        assert schedule.total_roll_cost > 0

    def test_roll_slippage(self):
        from pricebook.fixed_income.futures_roll import roll_slippage
        small = roll_slippage(0.02, 10, 100)
        large = roll_slippage(0.02, 100, 100)
        assert large > small  # more contracts = more slippage

    def test_liquidity_curve(self):
        from pricebook.fixed_income.futures_roll import liquidity_curve
        r = liquidity_curve([5000, 3000, 1000], ["CLZ24", "CLF25", "CLG25"])
        assert r[0]["contract"] == "CLZ24"  # highest volume first


# ═══════════════════════════════════════════════════════════════
# Dividend Futures (F9)
# ═══════════════════════════════════════════════════════════════

class TestDividendFutures:
    def test_dividend_future(self):
        from pricebook.equity.dividend_futures import dividend_future_price
        r = dividend_future_price(100, 98, 0.04, 1.0)
        assert r.implied_dividend > 0
        assert r.dividend_yield > 0

    def test_dividend_swap(self):
        from pricebook.equity.dividend_futures import dividend_swap_fair_value
        r = dividend_swap_fair_value([2.0, 2.1, 2.2], [0.25, 0.5, 0.75])
        assert r.fixed_rate > 0

    def test_dividend_option(self):
        from pricebook.equity.dividend_futures import dividend_option_price
        r = dividend_option_price(8.0, 7.5, 0.25, 0.5)
        assert r.price > 0
        assert r.delta > 0  # call delta positive

    def test_total_return_future(self):
        from pricebook.equity.dividend_futures import total_return_future
        r = total_return_future(100, 1.0, rate=0.04, div_yield=0.02)
        assert r.tr_futures_price > r.price_futures_price


# ═══════════════════════════════════════════════════════════════
# Commodity Swaps (F10)
# ═══════════════════════════════════════════════════════════════

class TestCommoditySwaps:
    def test_commodity_swap(self):
        from pricebook.commodity.commodity_swaps import commodity_swap_price
        forwards = [75, 76, 77, 78]
        times = [0.25, 0.5, 0.75, 1.0]
        r = commodity_swap_price(forwards, times, 76.0)
        assert r.fair_fixed > 0
        assert r.n_periods == 4

    def test_swap_pv_at_fair(self):
        from pricebook.commodity.commodity_swaps import commodity_swap_price
        forwards = [75, 76, 77, 78]
        times = [0.25, 0.5, 0.75, 1.0]
        r = commodity_swap_price(forwards, times, 76.0)
        # At fair fixed, PV should be near zero
        r2 = commodity_swap_price(forwards, times, r.fair_fixed)
        assert abs(r2.pv) < 1

    def test_swaption(self):
        from pricebook.commodity.commodity_swaps import commodity_swaption_price
        forwards = [75, 76, 77, 78]
        times = [0.25, 0.5, 0.75, 1.0]
        r = commodity_swaption_price(forwards, times, 76, 0.30, 0.25)
        assert r.premium > 0
        assert r.delta > 0  # call delta

    def test_asian_swap(self):
        from pricebook.commodity.commodity_swaps import asian_commodity_swap
        forwards = [75 + i * 0.1 for i in range(20)]
        r = asian_commodity_swap(forwards, 20, 75.5)
        assert r.fair_fixed > 0
