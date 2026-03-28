"""Tests for curve bootstrapping."""

import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.bootstrap import bootstrap
from pricebook.deposit import Deposit
from pricebook.swap import InterestRateSwap, SwapDirection
from pricebook.schedule import Frequency
from pricebook.day_count import DayCountConvention


REF = date(2024, 1, 15)

# Market data: USD deposit rates
DEPOSITS = [
    (REF + relativedelta(days=1), 0.0530),    # O/N
    (REF + relativedelta(weeks=1), 0.0528),    # 1W
    (REF + relativedelta(months=1), 0.0525),   # 1M
    (REF + relativedelta(months=2), 0.0520),   # 2M
    (REF + relativedelta(months=3), 0.0515),   # 3M
    (REF + relativedelta(months=6), 0.0500),   # 6M
]

# Market data: USD swap par rates
SWAPS = [
    (REF + relativedelta(years=1), 0.0480),
    (REF + relativedelta(years=2), 0.0460),
    (REF + relativedelta(years=3), 0.0450),
    (REF + relativedelta(years=5), 0.0440),
]


class TestDepositRoundTrip:
    """Bootstrapped curve should reprice input deposits."""

    def test_deposits_reprice(self):
        curve = bootstrap(REF, DEPOSITS, [])

        for mat, rate in DEPOSITS:
            dep = Deposit(REF, mat, rate)
            df_curve = curve.df(mat)
            df_deposit = dep.discount_factor
            assert df_curve == pytest.approx(df_deposit, rel=1e-6), \
                f"Failed at {mat}: curve df={df_curve}, deposit df={df_deposit}"


class TestSwapRoundTrip:
    """Bootstrapped curve should reprice input swaps at par."""

    def test_swaps_reprice_at_par(self):
        curve = bootstrap(REF, DEPOSITS, SWAPS)

        for mat, par_rate in SWAPS:
            swap = InterestRateSwap(
                REF, mat, fixed_rate=par_rate,
                direction=SwapDirection.PAYER,
                fixed_frequency=Frequency.SEMI_ANNUAL,
                float_frequency=Frequency.QUARTERLY,
                fixed_day_count=DayCountConvention.THIRTY_360,
                float_day_count=DayCountConvention.ACT_360,
            )
            pv = swap.pv(curve)
            # PV should be close to zero (at par)
            assert abs(pv) < 100.0, \
                f"Swap {mat} not at par: PV={pv:.2f}"

    def test_swap_par_rates_recovered(self):
        curve = bootstrap(REF, DEPOSITS, SWAPS)

        for mat, par_rate in SWAPS:
            swap = InterestRateSwap(
                REF, mat, fixed_rate=0.0,
                fixed_frequency=Frequency.SEMI_ANNUAL,
                float_frequency=Frequency.QUARTERLY,
                fixed_day_count=DayCountConvention.THIRTY_360,
                float_day_count=DayCountConvention.ACT_360,
            )
            recovered_par = swap.par_rate(curve)
            assert recovered_par == pytest.approx(par_rate, abs=5e-4), \
                f"Swap {mat}: input par={par_rate:.4f}, recovered={recovered_par:.4f}"


class TestCurveShape:
    """Sanity checks on bootstrapped curve."""

    def test_discount_factors_decreasing(self):
        curve = bootstrap(REF, DEPOSITS, SWAPS)
        all_dates = [d for d, _ in DEPOSITS] + [d for d, _ in SWAPS]
        dfs = [curve.df(d) for d in all_dates]
        for i in range(1, len(dfs)):
            assert dfs[i] <= dfs[i - 1], \
                f"df not decreasing at {all_dates[i]}: {dfs[i]} > {dfs[i-1]}"

    def test_discount_factors_positive(self):
        curve = bootstrap(REF, DEPOSITS, SWAPS)
        all_dates = [d for d, _ in DEPOSITS] + [d for d, _ in SWAPS]
        for d in all_dates:
            assert curve.df(d) > 0

    def test_forward_rates_positive(self):
        curve = bootstrap(REF, DEPOSITS, SWAPS)
        all_dates = [REF] + [d for d, _ in DEPOSITS] + [d for d, _ in SWAPS]
        for i in range(1, len(all_dates)):
            fwd = curve.forward_rate(all_dates[i - 1], all_dates[i])
            assert fwd > 0, f"Negative forward rate between {all_dates[i-1]} and {all_dates[i]}"

    def test_zero_rates_positive(self):
        curve = bootstrap(REF, DEPOSITS, SWAPS)
        all_dates = [d for d, _ in DEPOSITS] + [d for d, _ in SWAPS]
        for d in all_dates:
            assert curve.zero_rate(d) > 0


class TestDepositsOnly:
    """Bootstrap with deposits only (no swaps)."""

    def test_deposits_only_builds_curve(self):
        curve = bootstrap(REF, DEPOSITS, [])
        assert curve.df(DEPOSITS[-1][0]) < 1.0
        assert curve.df(DEPOSITS[-1][0]) > 0.0

    def test_deposits_only_zero_rate_positive(self):
        curve = bootstrap(REF, DEPOSITS, [])
        for mat, _ in DEPOSITS:
            assert curve.zero_rate(mat) > 0
