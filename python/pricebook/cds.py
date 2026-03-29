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
