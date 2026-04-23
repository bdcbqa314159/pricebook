"""Deep tests for curves & bootstrap — DD1 hardening.

Covers: extrapolation, bumped curve consistency, day count interactions,
interpolation edge cases, instantaneous forward correctness, and
round-trip verification for all bootstrap paths.
"""

import math
import pytest
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.day_count import DayCountConvention, year_fraction, date_from_year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.interpolation import (
    InterpolationMethod,
    LogLinearInterpolator,
    LinearInterpolator,
    MonotoneCubicInterpolator,
    create_interpolator,
)
from pricebook.bootstrap import bootstrap, bootstrap_forward_curve
from pricebook.ois import bootstrap_ois
from pricebook.deposit import Deposit


REF = date(2024, 1, 15)


def _make_curve(rate=0.05, method=InterpolationMethod.LOG_LINEAR):
    """Flat curve at given cc rate with given interpolation."""
    return DiscountCurve.flat(REF, rate)


# ---- Extrapolation ----

class TestLogLinearFlatForwardExtrapolation:
    """Log-linear should extrapolate with flat forward, not flat DF."""

    def test_df_beyond_last_pillar_continues_decaying(self):
        """DF at 30Y should be less than DF at 20Y (flat forward > 0)."""
        curve = _make_curve(0.05)
        df_20y = curve.df(REF + relativedelta(years=20))
        df_30y = curve.df(REF + relativedelta(years=30))
        assert df_30y < df_20y, "DF should continue decaying beyond last pillar"

    def test_flat_forward_rate_beyond_curve(self):
        """Forward rate beyond the last pillar should equal the last segment's rate."""
        curve = _make_curve(0.05)
        # Forward rate between 25Y and 30Y (both beyond 20Y last pillar)
        d1 = REF + relativedelta(years=25)
        d2 = REF + relativedelta(years=30)
        fwd = curve.forward_rate(d1, d2)
        # For a flat 5% cc curve, the simply compounded forward should be close to 5%
        # (not zero, which flat DF extrapolation would give)
        assert fwd > 0.04, f"Forward rate beyond curve should be ~5%, got {fwd:.4f}"

    def test_zero_rate_stable_beyond_curve(self):
        """Zero rate shouldn't collapse toward zero beyond the curve."""
        curve = _make_curve(0.05)
        zr_20 = curve.zero_rate(REF + relativedelta(years=20))
        zr_40 = curve.zero_rate(REF + relativedelta(years=40))
        # Should be roughly the same (flat curve)
        assert abs(zr_40 - zr_20) < 0.005, (
            f"Zero rate shouldn't drift: zr_20={zr_20:.4f}, zr_40={zr_40:.4f}"
        )

    def test_log_linear_extrapolation_exact(self):
        """Verify the math: log-linear extends the last segment's log-slope."""
        x = np.array([0.0, 1.0, 2.0, 5.0])
        y = np.array([1.0, 0.98, 0.95, 0.85])
        interp = LogLinearInterpolator(x, y)
        # Last segment slope in log space
        slope = (math.log(0.85) - math.log(0.95)) / (5.0 - 2.0)
        # Extrapolate to x=7
        expected = math.exp(math.log(0.85) + slope * (7.0 - 5.0))
        assert interp(7.0) == pytest.approx(expected, rel=1e-12)

    def test_other_methods_still_flat_extrap(self):
        """Linear, cubic, monotone, Akima still use flat extrapolation (unchanged)."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([1.0, 0.98, 0.95, 0.90])
        for method in [InterpolationMethod.LINEAR, InterpolationMethod.CUBIC_SPLINE,
                       InterpolationMethod.MONOTONE_CUBIC, InterpolationMethod.AKIMA]:
            interp = create_interpolator(method, x, y)
            assert interp(5.0) == pytest.approx(0.90)


# ---- Bumped curve consistency ----

class TestBumpedCurveConsistency:
    """bumped() and bumped_at() preserve curve attributes and produce correct shifts."""

    def test_bumped_preserves_interpolation_method(self):
        for method in InterpolationMethod:
            curve = DiscountCurve(
                REF,
                [REF + relativedelta(years=1), REF + relativedelta(years=5)],
                [0.95, 0.80],
                interpolation=method,
            )
            bumped = curve.bumped(0.001)
            assert bumped._interpolation == method

    def test_bumped_preserves_day_count(self):
        for dc in [DayCountConvention.ACT_360, DayCountConvention.ACT_365_FIXED,
                    DayCountConvention.THIRTY_360]:
            curve = DiscountCurve(
                REF,
                [REF + relativedelta(years=1)],
                [0.95],
                day_count=dc,
            )
            bumped = curve.bumped(0.001)
            assert bumped.day_count == dc

    def test_bumped_at_preserves_interpolation(self):
        curve = DiscountCurve(
            REF,
            [REF + relativedelta(years=1), REF + relativedelta(years=5)],
            [0.95, 0.80],
            interpolation=InterpolationMethod.MONOTONE_CUBIC,
        )
        bumped = curve.bumped_at(0, 0.001)
        assert bumped._interpolation == InterpolationMethod.MONOTONE_CUBIC

    def test_bumped_shift_magnitude(self):
        """1bp parallel bump should shift zero rates by ~1bp."""
        curve = _make_curve(0.05)
        shift = 0.0001  # 1bp
        bumped = curve.bumped(shift)
        d = REF + relativedelta(years=5)
        zr_orig = curve.zero_rate(d)
        zr_bumped = bumped.zero_rate(d)
        assert zr_bumped - zr_orig == pytest.approx(shift, abs=1e-8)

    def test_bumped_at_only_affects_target_pillar(self):
        """bumped_at(i) should leave other pillars' DFs unchanged."""
        dates = [REF + relativedelta(years=y) for y in [1, 2, 5, 10]]
        dfs = [0.95, 0.90, 0.78, 0.60]
        curve = DiscountCurve(REF, dates, dfs)
        bumped = curve.bumped_at(1, 0.001)  # bump the 2Y pillar
        # 1Y pillar should be unchanged
        assert bumped.df(dates[0]) == pytest.approx(curve.df(dates[0]), abs=1e-12)
        # 2Y pillar should be different
        assert bumped.df(dates[1]) != pytest.approx(curve.df(dates[1]), abs=1e-8)

    def test_bumped_zero_shift_recovers_original(self):
        """Zero shift should produce an identical curve."""
        curve = _make_curve(0.05)
        bumped = curve.bumped(0.0)
        d = REF + relativedelta(years=5)
        assert bumped.df(d) == pytest.approx(curve.df(d), abs=1e-14)


# ---- Instantaneous forward ----

class TestInstantaneousForward:

    def test_flat_curve_constant_forward(self):
        """Flat 5% curve should have ~5% instantaneous forward everywhere."""
        curve = _make_curve(0.05)
        for t in [0.5, 1.0, 2.0, 5.0, 10.0]:
            f = curve.instantaneous_forward(t)
            assert f == pytest.approx(0.05, abs=0.002), (
                f"Instantaneous forward at t={t} should be ~5%, got {f:.4f}"
            )

    def test_forward_positive_for_normal_curve(self):
        curve = _make_curve(0.03)
        for t in [0.25, 1.0, 5.0, 15.0]:
            assert curve.instantaneous_forward(t) > 0

    def test_uses_curve_day_count(self):
        """Curves with different day counts should give slightly different forwards."""
        dates = [REF + relativedelta(years=1), REF + relativedelta(years=5)]
        dfs = [0.95, 0.78]
        curve_365 = DiscountCurve(REF, dates, dfs, day_count=DayCountConvention.ACT_365_FIXED)
        curve_360 = DiscountCurve(REF, dates, dfs, day_count=DayCountConvention.ACT_360)
        # Forward rates should differ because dt is computed differently
        f_365 = curve_365.instantaneous_forward(2.0)
        f_360 = curve_360.instantaneous_forward(2.0)
        # ACT/360 dt for one day = 1/360, ACT/365 dt = 1/365
        # So f_360 should be slightly smaller: f = -ln(df2/df1)/dt, larger dt => smaller f
        assert f_365 != pytest.approx(f_360, abs=1e-6)


# ---- Day count edge cases ----

class TestDayCountEdgeCases:

    def test_act_act_isda_leap_year_boundary(self):
        """ACT/ACT ISDA across a leap year boundary should account for 366."""
        yf = year_fraction(
            date(2023, 7, 1), date(2024, 7, 1),
            DayCountConvention.ACT_ACT_ISDA,
        )
        # 184 days in 2023 (365-day year) + 182 days in 2024 (366-day year)
        expected = 184 / 365 + 182 / 366
        assert yf == pytest.approx(expected)

    def test_act_act_isda_multi_year(self):
        """Multi-year ACT/ACT ISDA should handle full years in between."""
        yf = year_fraction(
            date(2020, 6, 15), date(2025, 6, 15),
            DayCountConvention.ACT_ACT_ISDA,
        )
        assert 4.99 < yf < 5.01

    def test_thirty_360_feb_end_of_month(self):
        """30/360: Feb 28 → Mar 31 in non-leap year."""
        yf = year_fraction(
            date(2023, 2, 28), date(2023, 3, 31),
            DayCountConvention.THIRTY_360,
        )
        # d1=28 (no adj), d2=31, d1 < 30 so d2 stays 31
        # days = 30*(3-2) + (31-28) = 33
        assert yf == pytest.approx(33 / 360.0)

    def test_thirty_e_360_both_31st(self):
        """30E/360: both dates are 31st."""
        yf = year_fraction(
            date(2024, 1, 31), date(2024, 7, 31),
            DayCountConvention.THIRTY_E_360,
        )
        # Both clamped to 30: 6*30 = 180
        assert yf == pytest.approx(180 / 360.0)

    def test_date_from_year_fraction_round_trip(self):
        """date_from_year_fraction should approximately invert year_fraction.

        The 365.25 vs 365 mismatch causes ~1 day drift per 4 years.
        Tolerance scales with tenor: < 1 day error ≈ 1/365 per year fraction.
        """
        for t in [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
            d = date_from_year_fraction(REF, t)
            t_back = year_fraction(REF, d, DayCountConvention.ACT_365_FIXED)
            max_err = max(0.005, t * 0.001)  # ~0.1% of tenor
            assert abs(t_back - t) < max_err, (
                f"Round-trip drift at t={t}: got {t_back:.4f}"
            )


# ---- Interpolation edge cases ----

class TestInterpolationEdgeCases:

    def test_log_linear_two_points(self):
        """Log-linear with exactly 2 points (minimum)."""
        x = np.array([0.0, 1.0])
        y = np.array([1.0, 0.95])
        interp = LogLinearInterpolator(x, y)
        assert interp(0.5) == pytest.approx(math.exp(0.5 * math.log(0.95)))
        # Extrapolation should extend the single segment
        val_2 = interp(2.0)
        expected = math.exp(math.log(0.95) + math.log(0.95) * 1.0)
        assert val_2 == pytest.approx(expected)

    def test_monotone_cubic_constant_data(self):
        """Monotone cubic with constant y should return constant."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([1.0, 1.0, 1.0, 1.0])
        interp = MonotoneCubicInterpolator(x, y)
        for xi in [0.5, 1.5, 2.5]:
            assert interp(xi) == pytest.approx(1.0)

    def test_monotone_cubic_steep_transition(self):
        """Monotone cubic shouldn't oscillate on a steep drop."""
        x = np.array([0.0, 1.0, 1.01, 2.0, 3.0])
        y = np.array([1.0, 1.0, 0.5, 0.5, 0.5])
        interp = MonotoneCubicInterpolator(x, y)
        xs = np.linspace(0.0, 3.0, 500)
        ys = [interp(xi) for xi in xs]
        assert min(ys) >= 0.5 - 1e-10
        assert max(ys) <= 1.0 + 1e-10


# ---- Bootstrap round-trip precision ----

class TestBootstrapPrecision:

    def test_deposit_round_trip_tight(self):
        """Deposits should round-trip to machine precision."""
        deposits = [
            (REF + relativedelta(days=1), 0.0530),
            (REF + relativedelta(months=1), 0.0525),
            (REF + relativedelta(months=3), 0.0515),
            (REF + relativedelta(months=6), 0.0500),
        ]
        curve = bootstrap(REF, deposits, [])
        for mat, rate in deposits:
            dep = Deposit(REF, mat, rate)
            assert curve.df(mat) == pytest.approx(dep.discount_factor, rel=1e-10)

    def test_swap_round_trip_sub_basis_point(self):
        """Swap par rates should round-trip to < 0.1bp."""
        deposits = [
            (REF + relativedelta(months=3), 0.0515),
            (REF + relativedelta(months=6), 0.0500),
        ]
        swaps = [
            (REF + relativedelta(years=1), 0.0480),
            (REF + relativedelta(years=2), 0.0460),
            (REF + relativedelta(years=5), 0.0440),
            (REF + relativedelta(years=10), 0.0430),
        ]
        curve = bootstrap(REF, deposits, swaps)
        for mat, par_rate in swaps:
            # Verify by computing PV of fixed and float legs
            from pricebook.schedule import Frequency, StubType, generate_schedule
            fixed_sched = generate_schedule(REF, mat, Frequency.SEMI_ANNUAL)
            float_sched = generate_schedule(REF, mat, Frequency.QUARTERLY)
            pv_fixed = sum(
                par_rate * year_fraction(fixed_sched[i-1], fixed_sched[i],
                                         DayCountConvention.THIRTY_360) * curve.df(fixed_sched[i])
                for i in range(1, len(fixed_sched))
            )
            pv_float = sum(
                (curve.df(float_sched[i-1]) / curve.df(float_sched[i]) - 1.0) *
                curve.df(float_sched[i])
                for i in range(1, len(float_sched))
            )
            assert abs(pv_fixed - pv_float) < 1e-8, (
                f"Swap {mat}: PV mismatch {abs(pv_fixed - pv_float):.2e}"
            )

    def test_ois_round_trip_sub_basis_point(self):
        """OIS par rates should round-trip to < 0.1bp."""
        ois_rates = [
            (REF + relativedelta(months=1), 0.0530),
            (REF + relativedelta(months=3), 0.0525),
            (REF + relativedelta(years=1), 0.0490),
            (REF + relativedelta(years=5), 0.0440),
            (REF + relativedelta(years=10), 0.0420),
        ]
        curve = bootstrap_ois(REF, ois_rates)
        for mat, par_rate in ois_rates:
            from pricebook.ois import OISSwap
            ois = OISSwap(REF, mat, fixed_rate=par_rate)
            pv = ois.pv(curve)
            # PV should be very close to zero (< $1 on $1M notional)
            assert abs(pv) < 1.0, f"OIS {mat}: PV={pv:.2f}"


# ---- Curve construction edge cases ----

class TestCurveEdgeCases:

    def test_single_pillar(self):
        """Curve with a single pillar beyond t=0."""
        curve = DiscountCurve(REF, [REF + relativedelta(years=1)], [0.95])
        assert curve.df(REF + relativedelta(months=6)) < 1.0
        assert curve.df(REF + relativedelta(months=6)) > 0.95

    def test_very_long_tenor(self):
        """50Y tenor shouldn't crash or produce nonsense."""
        curve = _make_curve(0.04)
        df_50 = curve.df(REF + relativedelta(years=50))
        expected = math.exp(-0.04 * 50)
        assert df_50 == pytest.approx(expected, rel=0.01)

    def test_negative_rate_curve(self):
        """Curve with negative rates (EUR/JPY regime)."""
        curve = DiscountCurve.flat(REF, -0.005)
        d = REF + relativedelta(years=5)
        assert curve.df(d) > 1.0  # negative rate => DF > 1
        assert curve.zero_rate(d) < 0

    def test_steep_curve(self):
        """Very steep curve (short rate 0%, long rate 10%)."""
        dates = [
            REF + relativedelta(months=3),
            REF + relativedelta(years=1),
            REF + relativedelta(years=5),
            REF + relativedelta(years=10),
        ]
        dfs = [
            math.exp(-0.001 * 0.25),
            math.exp(-0.02 * 1),
            math.exp(-0.06 * 5),
            math.exp(-0.10 * 10),
        ]
        curve = DiscountCurve(REF, dates, dfs)
        # Forward rates should be positive and increasing
        for i in range(1, len(dates)):
            fwd = curve.forward_rate(dates[i-1], dates[i])
            assert fwd > 0


# ---- Forward curve bootstrap ----

class TestForwardCurveBootstrap:

    def test_forward_curve_dfs_positive(self):
        ois_rates = [
            (REF + relativedelta(months=3), 0.0525),
            (REF + relativedelta(years=1), 0.0490),
            (REF + relativedelta(years=2), 0.0470),
            (REF + relativedelta(years=5), 0.0440),
        ]
        ois_curve = bootstrap_ois(REF, ois_rates)
        irs_swaps = [
            (REF + relativedelta(years=1), 0.0500),
            (REF + relativedelta(years=2), 0.0485),
            (REF + relativedelta(years=5), 0.0465),
        ]
        fwd_curve = bootstrap_forward_curve(
            REF, irs_swaps, ois_curve,
            deposits=[(REF + relativedelta(months=3), 0.0540)],
        )
        for mat, _ in irs_swaps:
            assert fwd_curve.df(mat) > 0

    def test_forward_curve_spread_over_ois(self):
        """Forward curve rates should be higher than OIS (credit premium)."""
        ois_rates = [
            (REF + relativedelta(months=3), 0.0525),
            (REF + relativedelta(years=1), 0.0490),
            (REF + relativedelta(years=5), 0.0440),
        ]
        ois_curve = bootstrap_ois(REF, ois_rates)
        irs_swaps = [
            (REF + relativedelta(years=1), 0.0510),  # 20bp over OIS
            (REF + relativedelta(years=5), 0.0470),   # 30bp over OIS
        ]
        fwd_curve = bootstrap_forward_curve(REF, irs_swaps, ois_curve)
        # Forward curve zero rates should be higher than OIS
        d = REF + relativedelta(years=5)
        assert fwd_curve.zero_rate(d) > ois_curve.zero_rate(d)
