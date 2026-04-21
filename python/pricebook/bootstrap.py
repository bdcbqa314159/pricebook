"""Curve bootstrapping from deposits and swap par rates."""

from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.deposit import Deposit
from pricebook.discount_curve import DiscountCurve
from pricebook.interpolation import InterpolationMethod
from pricebook.schedule import Frequency, StubType, generate_schedule
from pricebook.solvers import brentq
from pricebook.calendar import Calendar, BusinessDayConvention


def bootstrap(
    reference_date: date,
    deposits: list[tuple[date, float]],
    swaps: list[tuple[date, float]],
    deposit_day_count: DayCountConvention = DayCountConvention.ACT_360,
    fixed_day_count: DayCountConvention = DayCountConvention.THIRTY_360,
    float_day_count: DayCountConvention = DayCountConvention.ACT_360,
    fixed_frequency: Frequency = Frequency.SEMI_ANNUAL,
    float_frequency: Frequency = Frequency.QUARTERLY,
    interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR,
    calendar: Calendar | None = None,
    convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
) -> DiscountCurve:
    """
    Bootstrap a discount curve from deposit rates and swap par rates.

    Args:
        reference_date: Curve reference date (today / spot date).
        deposits: List of (maturity_date, rate) for money market deposits.
                  Must be sorted by maturity.
        swaps: List of (maturity_date, par_rate) for vanilla IRS.
               Must be sorted by maturity, all maturities after last deposit.
        deposit_day_count: Day count for deposit year fractions.
        fixed_day_count: Day count for the fixed leg of swaps.
        float_day_count: Day count for the floating leg of swaps.
        fixed_frequency: Payment frequency of the fixed leg.
        float_frequency: Payment frequency of the floating leg.
        interpolation: Interpolation method for the resulting curve.
        calendar: Business day calendar.
        convention: Business day convention.

    Returns:
        A DiscountCurve that reprices all input instruments.
    """
    # Validate inputs are sorted
    for i in range(1, len(deposits)):
        if deposits[i][0] <= deposits[i - 1][0]:
            raise ValueError("deposits must be sorted by maturity")
    for i in range(1, len(swaps)):
        if swaps[i][0] <= swaps[i - 1][0]:
            raise ValueError("swaps must be sorted by maturity")
    if deposits and swaps and swaps[0][0] <= deposits[-1][0]:
        raise ValueError("swap maturities must come after deposit maturities")

    pillar_dates: list[date] = []
    pillar_dfs: list[float] = []

    # --- Short end: deposits give discount factors directly ---
    for mat, rate in deposits:
        dep = Deposit(reference_date, mat, rate, day_count=deposit_day_count)
        pillar_dates.append(mat)
        pillar_dfs.append(dep.discount_factor)

    # --- Long end: swap par rates, solved iteratively ---
    for mat, par_rate in swaps:
        fixed_sched = generate_schedule(
            reference_date, mat, fixed_frequency,
            calendar, convention, StubType.SHORT_FRONT, True,
        )
        float_sched = generate_schedule(
            reference_date, mat, float_frequency,
            calendar, convention, StubType.SHORT_FRONT, True,
        )

        def objective(df_guess: float, _mat=mat, _par=par_rate,
                      _fsched=fixed_sched, _flsched=float_sched) -> float:
            """Build trial curve and price the swap — PV should be 0 at par."""
            trial_dates = pillar_dates + [_mat]
            trial_dfs = pillar_dfs + [df_guess]
            trial_curve = DiscountCurve(
                reference_date, trial_dates, trial_dfs,
                day_count=DayCountConvention.ACT_365_FIXED,
                interpolation=interpolation,
            )

            # PV of fixed leg: par_rate * sum(yf_i * df_i)
            pv_fixed = 0.0
            for i in range(1, len(_fsched)):
                yf = year_fraction(_fsched[i - 1], _fsched[i], fixed_day_count)
                pv_fixed += _par * yf * trial_curve.df(_fsched[i])

            # PV of floating leg: sum(forward_rate_i * yf_i * df_i)
            # Compute forward rate using float_day_count for consistency:
            #   F = (df1/df2 - 1) / tau_float
            pv_float = 0.0
            for i in range(1, len(_flsched)):
                d1, d2 = _flsched[i - 1], _flsched[i]
                df1 = trial_curve.df(d1)
                df2 = trial_curve.df(d2)
                yf = year_fraction(d1, d2, float_day_count)
                fwd = (df1 / df2 - 1.0) / yf
                pv_float += fwd * yf * df2

            return pv_fixed - pv_float

        # Bracket: df must be between 0 and 1 (or slightly above 1 for negative rates)
        df_solved = brentq(objective, 1e-6, 3.0)  # wider bracket: handles negative rates to -3%

        pillar_dates.append(mat)
        pillar_dfs.append(df_solved)

    return DiscountCurve(
        reference_date, pillar_dates, pillar_dfs,
        day_count=DayCountConvention.ACT_365_FIXED,
        interpolation=interpolation,
    )


def bootstrap_forward_curve(
    reference_date: date,
    swaps: list[tuple[date, float]],
    discount_curve: DiscountCurve,
    deposits: list[tuple[date, float]] | None = None,
    deposit_day_count: DayCountConvention = DayCountConvention.ACT_360,
    fixed_day_count: DayCountConvention = DayCountConvention.THIRTY_360,
    float_day_count: DayCountConvention = DayCountConvention.ACT_360,
    fixed_frequency: Frequency = Frequency.SEMI_ANNUAL,
    float_frequency: Frequency = Frequency.QUARTERLY,
    interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR,
    calendar: Calendar | None = None,
    convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
) -> DiscountCurve:
    """
    Bootstrap a forward/projection curve from IRS par rates, discounting off
    an externally provided discount curve (typically OIS).

    This is the second stage of dual-curve construction:
        1. Bootstrap OIS curve from OIS rates (discount curve)
        2. Bootstrap forward curve from IRS rates using OIS for discounting

    The forward curve is used to project floating rates. The discount curve
    is used to discount all cashflows.

    Args:
        reference_date: Curve reference date.
        swaps: List of (maturity_date, par_rate) for IRS, sorted by maturity.
        discount_curve: The OIS/risk-free curve for discounting.
        deposits: Optional short-end deposit rates for the forward curve.
        deposit_day_count: Day count for deposits.
        fixed_day_count: Day count for the fixed leg.
        float_day_count: Day count for the floating leg.
        fixed_frequency: Fixed leg payment frequency.
        float_frequency: Floating leg payment frequency.
        interpolation: Interpolation method.

    Returns:
        A DiscountCurve representing the forward/projection curve.
    """
    for i in range(1, len(swaps)):
        if swaps[i][0] <= swaps[i - 1][0]:
            raise ValueError("swaps must be sorted by maturity")

    pillar_dates: list[date] = []
    pillar_dfs: list[float] = []

    # Short end: deposits imply forward curve discount factors
    if deposits:
        for i in range(1, len(deposits)):
            if deposits[i][0] <= deposits[i - 1][0]:
                raise ValueError("deposits must be sorted by maturity")
        if swaps and deposits[-1][0] >= swaps[0][0]:
            raise ValueError("deposit maturities must come before swap maturities")
        for mat, rate in deposits:
            dep = Deposit(reference_date, mat, rate, day_count=deposit_day_count)
            pillar_dates.append(mat)
            pillar_dfs.append(dep.discount_factor)

    # Long end: solve for forward curve df at each swap maturity
    for mat, par_rate in swaps:
        fixed_sched = generate_schedule(
            reference_date, mat, fixed_frequency,
            calendar, convention, StubType.SHORT_FRONT, True,
        )
        float_sched = generate_schedule(
            reference_date, mat, float_frequency,
            calendar, convention, StubType.SHORT_FRONT, True,
        )

        def objective(df_guess: float, _mat=mat, _par=par_rate,
                      _fsched=fixed_sched, _flsched=float_sched) -> float:
            # Build trial forward curve
            trial_dates = pillar_dates + [_mat]
            trial_dfs = pillar_dfs + [df_guess]
            fwd_curve = DiscountCurve(
                reference_date, trial_dates, trial_dfs,
                day_count=DayCountConvention.ACT_365_FIXED,
                interpolation=interpolation,
            )

            # PV of fixed leg: discounted off the OIS curve
            pv_fixed = 0.0
            for i in range(1, len(_fsched)):
                yf = year_fraction(_fsched[i - 1], _fsched[i], fixed_day_count)
                pv_fixed += _par * yf * discount_curve.df(_fsched[i])

            # PV of floating leg: forwards from forward curve, discounted off OIS
            pv_float = 0.0
            for i in range(1, len(_flsched)):
                d1, d2 = _flsched[i - 1], _flsched[i]
                fdf1 = fwd_curve.df(d1)
                fdf2 = fwd_curve.df(d2)
                yf = year_fraction(d1, d2, float_day_count)
                fwd = (fdf1 / fdf2 - 1.0) / yf
                pv_float += fwd * yf * discount_curve.df(d2)

            return pv_fixed - pv_float

        df_solved = brentq(objective, 1e-6, 3.0)  # wider bracket: handles negative rates to -3%
        pillar_dates.append(mat)
        pillar_dfs.append(df_solved)

    return DiscountCurve(
        reference_date, pillar_dates, pillar_dfs,
        day_count=DayCountConvention.ACT_365_FIXED,
        interpolation=interpolation,
    )
