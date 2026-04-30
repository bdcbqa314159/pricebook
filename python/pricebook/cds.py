"""Credit default swap."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

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

    def to_upfront(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
        standard_coupon: float = 0.01,
    ) -> dict[str, float]:
        """Convert to upfront quote (Big Bang convention).

        upfront = (par_spread - standard_coupon) × risky_annuity

        Args:
            standard_coupon: standard running coupon (0.01 = 100bp for IG).

        Returns:
            dict with upfront_pct, par_spread_bps, standard_coupon_bps, rpv01.
        """
        par = self.par_spread(discount_curve, survival_curve)
        rpv01 = self.rpv01(discount_curve, survival_curve)
        upfront_pct = (par - standard_coupon) * rpv01
        return {
            "upfront_pct": upfront_pct,
            "par_spread_bps": par * 10_000,
            "standard_coupon_bps": standard_coupon * 10_000,
            "rpv01": rpv01,
        }

    def carry(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
        horizon_days: int = 30,
    ) -> float:
        """Carry: premium income earned over horizon assuming no default.

        carry ≈ spread × notional × year_frac × survival_to_horizon
        """
        from pricebook.day_count import year_fraction as _yf
        horizon = self.start + timedelta(days=horizon_days)
        yf = _yf(self.start, horizon, self.day_count)
        surv = survival_curve.survival(horizon)
        return self.spread * self.notional * yf * surv

    def roll_down(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
        horizon_days: int = 30,
    ) -> float:
        """Roll-down: PV change from curve aging (time passing, spreads unchanged).

        Prices the CDS at two dates on the SAME curve shifted in time.
        """
        pv_now = self.pv(discount_curve, survival_curve)
        # Create a CDS with shorter remaining life
        horizon = self.start + timedelta(days=horizon_days)
        if horizon >= self.end:
            return -pv_now
        shorter = CDS(
            start=horizon, end=self.end, spread=self.spread,
            notional=self.notional, recovery=self.recovery,
            frequency=self.frequency, day_count=self.day_count,
        )
        pv_later = shorter.pv(discount_curve, survival_curve)
        return pv_later - pv_now

    def bucket_cs01(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
        shift_bps: float = 1.0,
    ) -> dict[str, float]:
        """Per-tenor CS01: sensitivity to each pillar of the survival curve.

        Returns dict mapping pillar date → CS01 contribution.
        """
        from pricebook.credit_risk import _bump_survival_curve_at
        pv_base = self.pv(discount_curve, survival_curve)
        shift = shift_bps / 10_000
        result = {}
        for i, d in enumerate(survival_curve._pillar_dates):
            bumped = _bump_survival_curve_at(survival_curve, i, shift)
            pv_bumped = self.pv(discount_curve, bumped)
            result[d.isoformat()] = (pv_bumped - pv_base) / shift_bps
        return result

    def rec01(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
        shift: float = 0.01,
    ) -> float:
        """Rec01: PV sensitivity to a 1% shift in recovery rate.

        Higher recovery reduces protection leg PV, so rec01 is negative
        for a protection buyer.
        """
        pv_base = self.pv(discount_curve, survival_curve)
        bumped = CDS(
            self.start, self.end, self.spread,
            notional=self.notional,
            recovery=min(self.recovery + shift, 1.0),
            frequency=self.frequency, day_count=self.day_count,
            protection_day_count=self.protection_day_count,
            steps_per_year=self.steps_per_year,
            calendar=self.calendar, convention=self.convention,
        )
        return bumped.pv(discount_curve, survival_curve) - pv_base

    def theta(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
        days: int = 1,
    ) -> float:
        """Theta: PV change for time passing (days), curves unchanged.

        Reprices a CDS with start shifted forward by `days`.
        """
        new_start = self.start + timedelta(days=days)
        if new_start >= self.end:
            return -self.pv(discount_curve, survival_curve)
        aged = CDS(
            new_start, self.end, self.spread,
            notional=self.notional, recovery=self.recovery,
            frequency=self.frequency, day_count=self.day_count,
            protection_day_count=self.protection_day_count,
            steps_per_year=self.steps_per_year,
            calendar=self.calendar, convention=self.convention,
        )
        return aged.pv(discount_curve, survival_curve) - self.pv(discount_curve, survival_curve)

    def spread_duration(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
    ) -> float:
        """Spread duration: normalised CS01.

        spread_duration = -CS01 / PV (per 1bp)
        """
        pv = self.pv(discount_curve, survival_curve)
        if abs(pv) < 1e-15:
            return 0.0
        return -self.cs01(discount_curve, survival_curve) / pv

    def spread_convexity(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
        shift: float = 0.0001,
    ) -> float:
        """Spread convexity: d²PV/ds² / PV via central difference.

        convexity = [PV(s+h) - 2×PV(s) + PV(s-h)] / (h² × PV)
        """
        from pricebook.credit_risk import _bump_survival_curve
        pv_base = self.pv(discount_curve, survival_curve)
        if abs(pv_base) < 1e-15:
            return 0.0
        pv_up = self.pv(discount_curve, _bump_survival_curve(survival_curve, shift))
        pv_down = self.pv(discount_curve, _bump_survival_curve(survival_curve, -shift))
        return (pv_up - 2 * pv_base + pv_down) / (shift ** 2 * pv_base)


# ---------------------------------------------------------------------------
# P&L attribution
# ---------------------------------------------------------------------------

@dataclass
class CDSPnLAttribution:
    """P&L decomposition for a CDS position."""
    total: float
    spread: float
    carry: float
    roll_down: float
    convexity: float
    residual: float

    def to_dict(self) -> dict:
        return {
            "total": self.total, "spread": self.spread,
            "carry": self.carry, "roll_down": self.roll_down,
            "convexity": self.convexity, "residual": self.residual,
        }


def cds_pnl_attribution(
    cds: CDS,
    disc_t0: DiscountCurve,
    surv_t0: SurvivalCurve,
    disc_t1: DiscountCurve,
    surv_t1: SurvivalCurve,
    horizon_days: int = 1,
) -> CDSPnLAttribution:
    """Decompose CDS P&L into spread, carry, roll-down, convexity, residual.

    P&L = PV(t1) - PV(t0), decomposed as:
      spread ≈ -CS01 × Δspread
      carry = spread × notional × yf × survival
      roll_down = PV(shorter CDS, t0 curves) - PV(t0)
      convexity ≈ ½ × spread_convexity × Δspread²
      residual = total - spread - carry - roll_down - convexity
    """
    pv_t0 = cds.pv(disc_t0, surv_t0)

    # Aged CDS for t1 pricing
    new_start = cds.start + timedelta(days=horizon_days)
    if new_start >= cds.end:
        return CDSPnLAttribution(
            total=-pv_t0, spread=0, carry=0,
            roll_down=-pv_t0, convexity=0, residual=0,
        )

    aged = CDS(
        new_start, cds.end, cds.spread,
        notional=cds.notional, recovery=cds.recovery,
        frequency=cds.frequency, day_count=cds.day_count,
        protection_day_count=cds.protection_day_count,
        steps_per_year=cds.steps_per_year,
    )
    pv_t1 = aged.pv(disc_t1, surv_t1)
    total = pv_t1 - pv_t0

    # Carry
    carry_pnl = cds.carry(disc_t0, surv_t0, horizon_days)

    # Roll-down
    roll = cds.roll_down(disc_t0, surv_t0, horizon_days)

    # Spread change
    par_t0 = cds.par_spread(disc_t0, surv_t0)
    par_t1 = aged.par_spread(disc_t1, surv_t1)
    delta_spread = par_t1 - par_t0
    cs01 = cds.cs01(disc_t0, surv_t0)
    spread_pnl = cs01 * delta_spread / 0.0001  # cs01 is per 1bp shift

    # Convexity
    conv = cds.spread_convexity(disc_t0, surv_t0)
    pv_for_conv = abs(pv_t0) if abs(pv_t0) > 1e-10 else cds.notional
    convexity_pnl = 0.5 * conv * pv_for_conv * delta_spread ** 2

    residual = total - spread_pnl - carry_pnl - roll - convexity_pnl

    return CDSPnLAttribution(
        total=total, spread=spread_pnl, carry=carry_pnl,
        roll_down=roll, convexity=convexity_pnl, residual=residual,
    )


# ---------------------------------------------------------------------------
# Forward CDS with curve objects
# ---------------------------------------------------------------------------

@dataclass
class ForwardCDSCurveResult:
    """Forward CDS spread computed from curve objects."""
    forward_spread: float
    risky_annuity: float
    protection_pv: float
    survival_to_start: float

    def to_dict(self) -> dict:
        return {
            "forward_spread": self.forward_spread,
            "risky_annuity": self.risky_annuity,
            "protection_pv": self.protection_pv,
            "survival_to_start": self.survival_to_start,
        }


def forward_cds_par_spread(
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    start_date: date,
    end_date: date,
    recovery: float = 0.4,
    notional: float = 1.0,
    frequency: Frequency = Frequency.QUARTERLY,
    day_count: DayCountConvention = DayCountConvention.ACT_360,
    protection_day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
    steps_per_year: int = 4,
) -> ForwardCDSCurveResult:
    """Par spread of a forward-starting CDS using curve objects.

    F(T₁,T₂) = protection_leg_pv(T₁,T₂) / risky_annuity(T₁,T₂)

    The forward CDS is simply a CDS starting at start_date with
    protection and premium legs computed on the same curves.
    """
    prot = protection_leg_pv(
        start_date, end_date, discount_curve, survival_curve,
        recovery=recovery, notional=notional,
        day_count=protection_day_count,
        steps_per_year=steps_per_year,
    )

    ann = risky_annuity(
        start_date, end_date, discount_curve, survival_curve,
        frequency=frequency, day_count=day_count,
    )

    fwd = prot / (notional * ann) if abs(ann) > 1e-15 else 0.0
    surv_start = survival_curve.survival(start_date)

    return ForwardCDSCurveResult(
        forward_spread=fwd,
        risky_annuity=ann,
        protection_pv=prot,
        survival_to_start=surv_start,
    )


def forward_risky_annuity(
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    start_date: date,
    end_date: date,
    frequency: Frequency = Frequency.QUARTERLY,
    day_count: DayCountConvention = DayCountConvention.ACT_360,
) -> float:
    """RPV01 of a forward-starting CDS using curve objects.

    Identical to risky_annuity() with the forward start date.
    """
    return risky_annuity(
        start_date, end_date, discount_curve, survival_curve,
        frequency=frequency, day_count=day_count,
    )


class StandardCDS(CDS):
    """Post-Big Bang standard CDS with market conventions.

    Auto-sets standard coupon, recovery, IMM dates based on grade (IG/HY).
    Carries the standard coupon and computes upfront internally.

    Usage:
        cds = StandardCDS.from_market(ref, maturity_years=5,
                                       par_spread_bps=60, grade="IG")
    """

    _SERIAL_TYPE = "standard_cds"

    def __init__(
        self,
        start: date,
        end: date,
        spread: float,
        standard_coupon: float = 0.01,
        grade: str = "IG",
        notional: float = 1_000_000.0,
        recovery: float = 0.4,
        frequency: Frequency = Frequency.QUARTERLY,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
        protection_day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
        steps_per_year: int = 4,
    ):
        super().__init__(
            start=start, end=end, spread=spread,
            notional=notional, recovery=recovery,
            frequency=frequency, day_count=day_count,
            protection_day_count=protection_day_count,
            steps_per_year=steps_per_year,
        )
        self.standard_coupon = standard_coupon
        self.grade = grade

    @classmethod
    def from_market(
        cls,
        reference_date: date,
        maturity_years: int = 5,
        par_spread_bps: float = 60.0,
        grade: str = "IG",
        notional: float = 1_000_000.0,
    ) -> StandardCDS:
        """Build from market quote with standard conventions.

        Auto-sets:
        - Standard coupon: 100bp (IG) or 500bp (HY)
        - Recovery: 40% (IG) or 25% (HY)
        - Dates: IMM-snapped from cds_conventions
        """
        from pricebook.cds_conventions import (
            standard_cds_dates, STANDARD_COUPONS_BPS, STANDARD_RECOVERY,
        )
        dates = standard_cds_dates(reference_date, maturity_years)
        start = dates[0]
        end = dates[-1]
        coupon_bps = STANDARD_COUPONS_BPS.get(grade, 100)
        recovery = STANDARD_RECOVERY.get(grade, 0.40)
        spread = par_spread_bps / 10_000

        return cls(
            start=start, end=end, spread=spread,
            standard_coupon=coupon_bps / 10_000, grade=grade,
            notional=notional, recovery=recovery,
        )

    def upfront_pct(self, discount_curve: DiscountCurve, survival_curve: SurvivalCurve) -> float:
        """Upfront as % of notional at standard coupon."""
        return self.to_upfront(discount_curve, survival_curve, self.standard_coupon)["upfront_pct"]

    # ---- Serialisation ----

    def to_dict(self) -> dict:
        from pricebook.serialisable import _serialise_atom
        return {"type": self._SERIAL_TYPE, "params": {
            "start": self.start.isoformat(), "end": self.end.isoformat(),
            "spread": self.spread, "standard_coupon": self.standard_coupon,
            "grade": self.grade, "notional": self.notional,
            "recovery": self.recovery,
            "frequency": _serialise_atom(self.frequency),
            "day_count": _serialise_atom(self.day_count),
            "protection_day_count": _serialise_atom(self.protection_day_count),
            "steps_per_year": self.steps_per_year,
        }}

    @classmethod
    def from_dict(cls, d: dict) -> StandardCDS:
        from pricebook.serialisable import _deserialise_atom
        p = d["params"]
        return cls(
            start=date.fromisoformat(p["start"]), end=date.fromisoformat(p["end"]),
            spread=p["spread"], standard_coupon=p.get("standard_coupon", 0.01),
            grade=p.get("grade", "IG"), notional=p.get("notional", 1_000_000.0),
            recovery=p.get("recovery", 0.4),
            frequency=Frequency(p.get("frequency", 3)),
            day_count=DayCountConvention(p.get("day_count", "ACT/360")),
            protection_day_count=DayCountConvention(p.get("protection_day_count", "ACT_365_FIXED")),
            steps_per_year=p.get("steps_per_year", 4),
        )


from pricebook.serialisable import _register as _reg_std
_reg_std(StandardCDS)


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
_serialisable("cds", ["start", "end", "spread", "notional", "recovery", "frequency", "day_count", "protection_day_count", "steps_per_year"])(CDS)
