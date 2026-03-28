"""Overnight index swap (OIS)."""

from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.fixed_leg import FixedLeg
from pricebook.schedule import Frequency, StubType, generate_schedule
from pricebook.calendar import Calendar, BusinessDayConvention
from pricebook.solvers import brentq
from pricebook.interpolation import InterpolationMethod


class OISSwap:
    """
    An overnight index swap: fixed rate vs compounded overnight rate.

    The floating leg pays the compounded overnight rate (SOFR, SONIA, ESTR)
    over each accrual period. When priced off the OIS curve itself, the
    floating PV telescopes:

        PV_float = notional * (df(start) - df(end))

    This makes OIS pricing and bootstrapping very efficient.

    USD SOFR OIS conventions:
        - Fixed: annual, ACT/360
        - Floating: annual, ACT/360, compounded in arrears
    """

    def __init__(
        self,
        start: date,
        end: date,
        fixed_rate: float,
        notional: float = 1_000_000.0,
        fixed_frequency: Frequency = Frequency.ANNUAL,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
        calendar: Calendar | None = None,
        convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
        stub: StubType = StubType.SHORT_FRONT,
        eom: bool = True,
    ):
        self.start = start
        self.end = end
        self.notional = notional
        self.day_count = day_count

        self.fixed_leg = FixedLeg(
            start, end, fixed_rate, fixed_frequency,
            notional=notional, day_count=day_count,
            calendar=calendar, convention=convention, stub=stub, eom=eom,
        )

    def pv_float(self, curve: DiscountCurve) -> float:
        """Floating leg PV via telescoping: notional * (df(start) - df(end))."""
        return self.notional * (curve.df(self.start) - curve.df(self.end))

    def pv(self, curve: DiscountCurve) -> float:
        """PV of payer OIS: receive float, pay fixed."""
        return self.pv_float(curve) - self.fixed_leg.pv(curve)

    def par_rate(self, curve: DiscountCurve) -> float:
        """The fixed rate that makes PV = 0."""
        annuity = self.fixed_leg.annuity(curve)
        return self.pv_float(curve) / (self.notional * annuity)


def bootstrap_ois(
    reference_date: date,
    ois_rates: list[tuple[date, float]],
    day_count: DayCountConvention = DayCountConvention.ACT_360,
    fixed_frequency: Frequency = Frequency.ANNUAL,
    interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR,
    calendar: Calendar | None = None,
    convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
) -> DiscountCurve:
    """
    Bootstrap an OIS (risk-free discount) curve from OIS par rates.

    Uses the telescoping property for efficient stripping:
        PV_float = df(start) - df(end)
        PV_fixed = par_rate * sum(yf_i * df_i)

    At par: df(start) - df(end) = par_rate * sum(yf_i * df_i)

    Args:
        reference_date: Curve reference date.
        ois_rates: List of (maturity_date, par_rate) sorted by maturity.
        day_count: Day count for both legs.
        fixed_frequency: Fixed leg payment frequency.
        interpolation: Interpolation method.

    Returns:
        A DiscountCurve (the OIS/risk-free curve).
    """
    for i in range(1, len(ois_rates)):
        if ois_rates[i][0] <= ois_rates[i - 1][0]:
            raise ValueError("ois_rates must be sorted by maturity")

    pillar_dates: list[date] = []
    pillar_dfs: list[float] = []

    for mat, par_rate in ois_rates:
        fixed_sched = generate_schedule(
            reference_date, mat, fixed_frequency,
            calendar, convention, StubType.SHORT_FRONT, True,
        )

        def objective(df_guess: float, _mat=mat, _par=par_rate,
                      _sched=fixed_sched) -> float:
            trial_dates = pillar_dates + [_mat]
            trial_dfs = pillar_dfs + [df_guess]
            trial_curve = DiscountCurve(
                reference_date, trial_dates, trial_dfs,
                day_count=DayCountConvention.ACT_365_FIXED,
                interpolation=interpolation,
            )

            # Telescoping: PV_float = df(ref) - df(mat)
            pv_float = trial_curve.df(reference_date) - trial_curve.df(_mat)

            # PV_fixed = par * sum(yf_i * df_i)
            pv_fixed = 0.0
            for i in range(1, len(_sched)):
                yf = year_fraction(_sched[i - 1], _sched[i], day_count)
                pv_fixed += _par * yf * trial_curve.df(_sched[i])

            return pv_fixed - pv_float

        df_solved = brentq(objective, 0.001, 1.5)
        pillar_dates.append(mat)
        pillar_dfs.append(df_solved)

    return DiscountCurve(
        reference_date, pillar_dates, pillar_dfs,
        day_count=DayCountConvention.ACT_365_FIXED,
        interpolation=interpolation,
    )
