"""Payment schedule generation for periodic cashflows."""

from datetime import date, timedelta
from enum import Enum
from dateutil.relativedelta import relativedelta

from pricebook.calendar import Calendar, BusinessDayConvention


class Frequency(Enum):
    WEEKLY = 0       # special: 7-day steps (not month-based)
    MONTHLY = 1
    QUARTERLY = 3
    SEMI_ANNUAL = 6
    ANNUAL = 12


class StubType(Enum):
    SHORT_FRONT = "short_front"
    LONG_FRONT = "long_front"
    SHORT_BACK = "short_back"
    LONG_BACK = "long_back"


def _add_months(d: date, months: int, eom: bool) -> date:
    """Add months to a date, respecting end-of-month rule."""
    result = d + relativedelta(months=months)
    if eom and d == _end_of_month(d):
        result = _end_of_month(result)
    return result


def _end_of_month(d: date) -> date:
    """Return the last day of the month for a given date."""
    next_month = d + relativedelta(months=1, day=1)
    return next_month - relativedelta(days=1)


def generate_schedule(
    start: date,
    end: date,
    frequency: Frequency,
    calendar: Calendar | None = None,
    convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
    stub: StubType = StubType.SHORT_FRONT,
    eom: bool = True,
) -> list[date]:
    """
    Generate a schedule of dates from start to end at the given frequency.

    Returns a list of dates including start and end. Intermediate dates
    are rolled backward from the end date to produce a front stub if
    the period doesn't divide evenly.

    Args:
        start: Effective date (first date in schedule)
        end: Termination date (last date in schedule)
        frequency: Payment frequency
        calendar: Business day calendar for adjustment (None = no adjustment)
        convention: Business day convention for adjustment
        stub: How to handle irregular first period
        eom: End-of-month rule — if start is EOM, keep rolls at EOM
    """
    if start >= end:
        raise ValueError(f"start ({start}) must be before end ({end})")

    # WEEKLY: step by 7 days (not month-based)
    if frequency == Frequency.WEEKLY:
        unadjusted = [start]
        current = start
        while True:
            current = current + timedelta(days=7)
            if current >= end:
                break
            unadjusted.append(current)
        unadjusted.append(end)
        if calendar is not None:
            return [calendar.adjust(d, convention) for d in unadjusted]
        return unadjusted

    months = frequency.value

    if stub in (StubType.SHORT_FRONT, StubType.LONG_FRONT):
        # Generate backward from end → front stub
        unadjusted = [end]
        current = end
        while True:
            current = _add_months(current, -months, eom)
            if current <= start:
                break
            unadjusted.append(current)
        unadjusted.append(start)
        unadjusted.reverse()

        # Handle long front stub: merge first two interior periods if stub is tiny
        if stub == StubType.LONG_FRONT and len(unadjusted) > 2:
            first_gap = (unadjusted[1] - unadjusted[0]).days
            regular_gap = months * 30  # approximate
            if first_gap < regular_gap * 0.5:
                unadjusted = [unadjusted[0]] + unadjusted[2:]

    else:
        # Generate forward from start → back stub
        unadjusted = [start]
        current = start
        while True:
            current = _add_months(current, months, eom)
            if current >= end:
                break
            unadjusted.append(current)
        unadjusted.append(end)

        # Handle long back stub: merge last two interior periods if stub is tiny
        if stub == StubType.LONG_BACK and len(unadjusted) > 2:
            last_gap = (unadjusted[-1] - unadjusted[-2]).days
            regular_gap = months * 30  # approximate
            if last_gap < regular_gap * 0.5:
                unadjusted = unadjusted[:-2] + [unadjusted[-1]]

    # Apply business day adjustment
    if calendar is not None:
        adjusted = []
        for i, d in enumerate(unadjusted):
            adjusted.append(calendar.adjust(d, convention))
        return adjusted

    return unadjusted
