"""Credit default swap."""

from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.schedule import Frequency, StubType, generate_schedule
from pricebook.calendar import Calendar, BusinessDayConvention


def protection_leg_pv(
    start: date,
    end: date,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    recovery: float = 0.4,
    notional: float = 1_000_000.0,
    day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
    steps_per_year: int = 4,
) -> float:
    """
    PV of the protection leg of a CDS.

    The protection buyer receives (1 - R) * notional if default occurs.
    Discretised over small intervals:

        PV = (1 - R) * notional * sum(df(t_mid) * (Q(t_{i-1}) - Q(t_i)))

    where t_mid is the midpoint of each interval (approximation for
    the default time within the interval).

    Args:
        start: Protection start date.
        end: Protection end date.
        discount_curve: Risk-free discount curve (OIS).
        survival_curve: Credit survival curve.
        recovery: Recovery rate (fraction of notional recovered on default).
        notional: CDS notional.
        day_count: Day count for time intervals.
        steps_per_year: Discretisation granularity (4 = quarterly steps).
    """
    lgd = (1.0 - recovery) * notional

    # Generate a fine grid for numerical integration
    t_start = year_fraction(survival_curve.reference_date, start, day_count)
    t_end = year_fraction(survival_curve.reference_date, end, day_count)
    n_steps = max(1, int((t_end - t_start) * steps_per_year))
    dt = (t_end - t_start) / n_steps

    ref = survival_curve.reference_date
    pv = 0.0
    for i in range(n_steps):
        t1 = t_start + i * dt
        t2 = t_start + (i + 1) * dt
        t_mid = (t1 + t2) / 2.0

        # Convert times back to dates for curve queries
        d1 = date.fromordinal(ref.toordinal() + int(t1 * 365))
        d2 = date.fromordinal(ref.toordinal() + int(t2 * 365))
        d_mid = date.fromordinal(ref.toordinal() + int(t_mid * 365))

        q1 = survival_curve.survival(d1)
        q2 = survival_curve.survival(d2)
        df_mid = discount_curve.df(d_mid)

        pv += df_mid * (q1 - q2)

    return lgd * pv


def premium_leg_pv(
    start: date,
    end: date,
    spread: float,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    notional: float = 1_000_000.0,
    frequency: Frequency = Frequency.QUARTERLY,
    day_count: DayCountConvention = DayCountConvention.ACT_360,
    calendar: Calendar | None = None,
    convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
) -> float:
    """
    PV of the premium (fee) leg of a CDS.

    The protection buyer pays a periodic coupon contingent on survival:

        PV = notional * spread * sum(yf_i * df(t_i) * Q(t_i))

    Plus an accrued interest approximation for default mid-period:

        accrued = notional * spread * sum(yf_i/2 * df(t_mid) * (Q(t_{i-1}) - Q(t_i)))

    Args:
        start: Premium start date.
        end: Premium end date.
        spread: CDS coupon (annualised, e.g. 0.01 for 100bp).
        discount_curve: Risk-free discount curve (OIS).
        survival_curve: Credit survival curve.
        notional: CDS notional.
        frequency: Premium payment frequency (typically quarterly).
        day_count: Day count for premium accrual.
        calendar: Business day calendar.
        convention: Business day convention.
    """
    schedule = generate_schedule(
        start, end, frequency, calendar, convention,
        StubType.SHORT_FRONT, True,
    )

    pv_scheduled = 0.0
    pv_accrued = 0.0

    for i in range(1, len(schedule)):
        d1 = schedule[i - 1]
        d2 = schedule[i]
        yf = year_fraction(d1, d2, day_count)
        q1 = survival_curve.survival(d1)
        q2 = survival_curve.survival(d2)
        df2 = discount_curve.df(d2)

        # Scheduled premium: paid if survived to payment date
        pv_scheduled += yf * df2 * q2

        # Accrued on default: approximate as half-period accrual
        d_mid = date.fromordinal((d1.toordinal() + d2.toordinal()) // 2)
        df_mid = discount_curve.df(d_mid)
        pv_accrued += (yf / 2.0) * df_mid * (q1 - q2)

    return notional * spread * (pv_scheduled + pv_accrued)
