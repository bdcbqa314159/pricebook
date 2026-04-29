"""Credit default swap."""

import math
from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.interpolation import InterpolationMethod
from pricebook.schedule import Frequency, StubType, generate_schedule
from pricebook.solvers import brentq
from pricebook.calendar import Calendar, BusinessDayConvention


def protection_leg_pv(
    start: date,
    end: date,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    recovery: float = 0.4,
    notional: float = 1_000_000.0,
    day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
    steps_per_year: int = 12,
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
    from datetime import timedelta

    lgd = (1.0 - recovery) * notional

    # Generate a fine date grid for numerical integration
    total_days = (end - start).days
    n_steps = max(1, int(year_fraction(start, end, day_count) * steps_per_year))
    step_days = total_days / n_steps

    pv = 0.0
    for i in range(n_steps):
        d1 = start + timedelta(days=int(i * step_days))
        d2 = start + timedelta(days=int((i + 1) * step_days))
        if i == n_steps - 1:
            d2 = end  # ensure exact end date
        d_mid = d1 + timedelta(days=(d2 - d1).days // 2)

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


def risky_annuity(
    start: date,
    end: date,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    frequency: Frequency = Frequency.QUARTERLY,
    day_count: DayCountConvention = DayCountConvention.ACT_360,
    calendar: Calendar | None = None,
    convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
) -> float:
    """
    Risky annuity (RPV01): PV of 1bp running spread, per unit notional.

    RPV01 = sum(yf_i * df(t_i) * Q(t_i)) + accrued-on-default terms.

    This is premium_leg_pv / (notional * spread), independent of spread.
    """
    # Use spread=1 and notional=1 to get the annuity factor
    return premium_leg_pv(
        start, end, spread=1.0,
        discount_curve=discount_curve, survival_curve=survival_curve,
        notional=1.0, frequency=frequency, day_count=day_count,
        calendar=calendar, convention=convention,
    )


class CDS:
    """
    A credit default swap.

    Protection buyer pays a periodic spread (premium leg) and receives
    (1 - recovery) * notional on default (protection leg).

    PV (protection buyer) = PV(protection) - PV(premium)

    Par spread: the coupon that makes PV = 0.

    Standard market conventions (post-Big Bang):
        - Fixed coupons: 100bp or 500bp
        - Upfront payment settles the difference from par
        - Quarterly premium, ACT/360
    """

    def __init__(
        self,
        start: date,
        end: date,
        spread: float,
        notional: float = 1_000_000.0,
        recovery: float = 0.4,
        frequency: Frequency = Frequency.QUARTERLY,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
        protection_day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
        steps_per_year: int = 4,
        calendar: Calendar | None = None,
        convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
    ):
        if notional <= 0:
            raise ValueError(f"notional must be positive, got {notional}")
        if not (0 <= recovery <= 1):
            raise ValueError(f"recovery must be in [0, 1], got {recovery}")
        if start >= end:
            raise ValueError(f"start ({start}) must be before end ({end})")

        self.start = start
        self.end = end
        self.spread = spread
        self.notional = notional
        self.recovery = recovery
        self.frequency = frequency
        self.day_count = day_count
        self.protection_day_count = protection_day_count
        self.steps_per_year = steps_per_year
        self.calendar = calendar
        self.convention = convention

    def pv_protection(
        self, discount_curve: DiscountCurve, survival_curve: SurvivalCurve,
    ) -> float:
        """PV of the protection leg."""
        return protection_leg_pv(
            self.start, self.end, discount_curve, survival_curve,
            recovery=self.recovery, notional=self.notional,
            day_count=self.protection_day_count,
            steps_per_year=self.steps_per_year,
        )

    def pv_premium(
        self, discount_curve: DiscountCurve, survival_curve: SurvivalCurve,
    ) -> float:
        """PV of the premium leg."""
        return premium_leg_pv(
            self.start, self.end, self.spread,
            discount_curve, survival_curve,
            notional=self.notional, frequency=self.frequency,
            day_count=self.day_count,
            calendar=self.calendar, convention=self.convention,
        )

    def pv(
        self, discount_curve: DiscountCurve, survival_curve: SurvivalCurve,
    ) -> float:
        """PV for the protection buyer: protection - premium."""
        return self.pv_protection(discount_curve, survival_curve) \
             - self.pv_premium(discount_curve, survival_curve)

    def pv_ctx(self, ctx, credit_curve_name: str = "default") -> float:
        """Price the CDS from a PricingContext."""
        curve = ctx.discount_curve
        # Try the specific name first, then fall back to first available
        try:
            surv = ctx.get_credit_curve(credit_curve_name)
        except KeyError:
            if ctx.credit_curves:
                surv = next(iter(ctx.credit_curves.values()))
            else:
                raise
        return self.pv(curve, surv)

    def par_spread(
        self, discount_curve: DiscountCurve, survival_curve: SurvivalCurve,
    ) -> float:
        """
        The spread that makes PV = 0.

        par_spread = PV(protection) / (notional * RPV01)
        """
        prot = self.pv_protection(discount_curve, survival_curve)
        rpv01 = risky_annuity(
            self.start, self.end, discount_curve, survival_curve,
            frequency=self.frequency, day_count=self.day_count,
            calendar=self.calendar, convention=self.convention,
        )
        if abs(rpv01) < 1e-15:
            return float('inf')
        return prot / (self.notional * rpv01)

    def upfront(
        self, discount_curve: DiscountCurve, survival_curve: SurvivalCurve,
    ) -> float:
        """
        Upfront payment (as fraction of notional).

        For standard CDS with fixed coupon, the upfront settles the
        difference between the running spread and the par spread:
            upfront ≈ (par_spread - spread) * RPV01
        """
        return self.pv(discount_curve, survival_curve) / self.notional

    def isda_upfront(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
        standard_coupon: float = 0.01,
    ) -> float:
        """ISDA standard upfront: PV at standard running coupon.

        Post-Big Bang, CDS trades with fixed 100bp (IG) or 500bp (HY)
        running coupon. The upfront settles the difference.

        Args:
            standard_coupon: 0.01 for IG (100bp), 0.05 for HY (500bp).
        """
        std = CDS(
            self.start, self.end, standard_coupon,
            notional=self.notional, recovery=self.recovery,
            frequency=self.frequency, day_count=self.day_count,
            protection_day_count=self.protection_day_count,
            steps_per_year=self.steps_per_year,
            calendar=self.calendar, convention=self.convention,
        )
        return std.pv(discount_curve, survival_curve) / self.notional

    def rpv01(
        self, discount_curve: DiscountCurve, survival_curve: SurvivalCurve,
    ) -> float:
        """Risky PV01 (risky annuity): PV of 1bp running spread per unit notional."""
        return risky_annuity(
            self.start, self.end, discount_curve, survival_curve,
            frequency=self.frequency, day_count=self.day_count,
            calendar=self.calendar, convention=self.convention,
        )

    def cs01(
        self, discount_curve: DiscountCurve, survival_curve: SurvivalCurve,
        shift: float = 0.0001,
    ) -> float:
        """CS01: PV change for a 1bp parallel shift in credit spreads.

        Bumps the survival curve's hazard rates by `shift` and reprices.
        """
        from pricebook.credit_risk import _bump_survival_curve
        pv_base = self.pv(discount_curve, survival_curve)
        pv_bumped = self.pv(discount_curve, _bump_survival_curve(survival_curve, shift))
        return pv_bumped - pv_base


def bootstrap_credit_curve(
    reference_date: date,
    cds_spreads: list[tuple[date, float]],
    discount_curve: DiscountCurve,
    recovery: float = 0.4,
    frequency: Frequency = Frequency.QUARTERLY,
    day_count: DayCountConvention = DayCountConvention.ACT_360,
    protection_day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
    steps_per_year: int = 4,
    interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR,
    calendar: Calendar | None = None,
    convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
) -> SurvivalCurve:
    """
    Bootstrap a survival (credit) curve from CDS par spreads.

    For each CDS maturity, solve for the survival probability Q(T) such
    that the CDS prices at par (PV = 0).

    Args:
        reference_date: Curve reference date.
        cds_spreads: List of (maturity_date, par_spread) sorted by maturity.
        discount_curve: Risk-free discount curve (OIS).
        recovery: Recovery rate assumption.
        frequency: CDS premium payment frequency.
        day_count: Day count for premium leg.
        protection_day_count: Day count for protection leg integration.
        steps_per_year: Discretisation for protection leg.
        interpolation: Interpolation method for survival curve.

    Returns:
        A SurvivalCurve that reprices all input CDS at par.
    """
    for i in range(1, len(cds_spreads)):
        if cds_spreads[i][0] <= cds_spreads[i - 1][0]:
            raise ValueError("cds_spreads must be sorted by maturity")

    pillar_dates: list[date] = []
    pillar_survs: list[float] = []

    for mat, par_spread in cds_spreads:

        def objective(q_guess: float, _mat=mat, _spread=par_spread) -> float:
            trial_dates = pillar_dates + [_mat]
            trial_survs = pillar_survs + [q_guess]
            trial_curve = SurvivalCurve(
                reference_date, trial_dates, trial_survs,
                day_count=DayCountConvention.ACT_365_FIXED,
                interpolation=interpolation,
            )

            cds = CDS(
                reference_date, _mat, _spread,
                notional=1.0, recovery=recovery,
                frequency=frequency, day_count=day_count,
                protection_day_count=protection_day_count,
                steps_per_year=steps_per_year,
                calendar=calendar, convention=convention,
            )
            return cds.pv(discount_curve, trial_curve)

        # Survival prob must be in (0, 1)
        q_solved = brentq(objective, 1e-6, 1.0 - 1e-10)
        pillar_dates.append(mat)
        pillar_survs.append(q_solved)

    curve = SurvivalCurve(
        reference_date, pillar_dates, pillar_survs,
        day_count=DayCountConvention.ACT_365_FIXED,
        interpolation=interpolation,
    )

    # Round-trip verification
    _verify_credit_round_trip(
        curve, reference_date, cds_spreads, discount_curve,
        recovery, frequency, day_count, protection_day_count,
        steps_per_year, calendar, convention,
    )

    return curve


def _verify_credit_round_trip(
    curve, reference_date, cds_spreads, discount_curve,
    recovery, frequency, day_count, protection_day_count,
    steps_per_year, calendar, convention,
    tol=1e-4,
):
    """Verify the bootstrapped credit curve reprices all input CDS at par."""
    import warnings

    errors = []
    for mat, input_spread in cds_spreads:
        cds = CDS(
            reference_date, mat, input_spread,
            notional=1.0, recovery=recovery,
            frequency=frequency, day_count=day_count,
            protection_day_count=protection_day_count,
            steps_per_year=steps_per_year,
            calendar=calendar, convention=convention,
        )
        model_spread = cds.par_spread(discount_curve, curve)
        err = abs(model_spread - input_spread)
        if err > tol:
            errors.append(
                f"CDS {mat}: input={input_spread:.6f}, model={model_spread:.6f}, err={err:.2e}"
            )

    if errors:
        msg = "Credit bootstrap round-trip failures:\n" + "\n".join(errors)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)

from pricebook.serialisable import serialisable as _serialisable
_serialisable("cds", ["start", "end", "spread", "notional", "recovery", "frequency", "day_count"])(CDS)
