"""Curve bootstrapping from deposits and swap par rates."""

from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.deposit import Deposit
from pricebook.discount_curve import DiscountCurve
from pricebook.interpolation import InterpolationMethod
from pricebook.schedule import Frequency, StubType, generate_schedule
from pricebook.calendar import Calendar, BusinessDayConvention


def _brentq(f, a: float, b: float, tol: float = 1e-12, maxiter: int = 100) -> float:
    """Brent's method for root finding on [a, b]. f(a) and f(b) must have opposite signs."""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError(f"f(a)={fa:.6e} and f(b)={fb:.6e} must have opposite signs")

    if abs(fa) < tol:
        return a
    if abs(fb) < tol:
        return b

    # Ensure |f(b)| <= |f(a)|
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c, fc = a, fa
    d = b - a
    mflag = True

    for _ in range(maxiter):
        if abs(fb) < tol:
            return b
        if abs(b - a) < tol:
            return b

        # Inverse quadratic interpolation or secant
        if fa != fc and fb != fc:
            s = (a * fb * fc / ((fa - fb) * (fa - fc))
                 + b * fa * fc / ((fb - fa) * (fb - fc))
                 + c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            s = b - fb * (b - a) / (fb - fa)

        # Conditions for bisection fallback
        cond1 = not ((3 * a + b) / 4 < s < b or b < s < (3 * a + b) / 4)
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2
        cond3 = not mflag and abs(s - b) >= abs(c - d) / 2
        cond4 = mflag and abs(b - c) < tol
        cond5 = not mflag and abs(c - d) < tol

        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False

        fs = f(s)
        d = c
        c, fc = b, fb

        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    return b


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
        df_solved = _brentq(objective, 0.001, 1.5)

        pillar_dates.append(mat)
        pillar_dfs.append(df_solved)

    return DiscountCurve(
        reference_date, pillar_dates, pillar_dfs,
        day_count=DayCountConvention.ACT_365_FIXED,
        interpolation=interpolation,
    )
