"""CMS swaps, CMS caps/floors, CMS spread options, and range accruals.

A CMS (Constant Maturity Swap) leg pays a long-dated swap rate observed
at each fixing. Due to the convexity of the swap rate payoff, the
expected CMS rate exceeds the forward swap rate by a convexity adjustment.

    from pricebook.cms import (
        cms_convexity_adjustment, CMSLeg, cms_cap, cms_spread_option,
        range_accrual,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from dateutil.relativedelta import relativedelta

from pricebook.black76 import black76_price, OptionType
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.schedule import Frequency, StubType, generate_schedule
from pricebook.swap import InterestRateSwap, SwapDirection


# ---- CMS convexity adjustment ----

def cms_convexity_adjustment(
    forward_swap_rate: float,
    annuity: float,
    swap_tenor_years: float,
    vol: float,
    time_to_fixing: float,
    duration: float | None = None,
) -> float:
    """CMS convexity adjustment via linear Terminal Swap Rate (TSR) model.

    The CMS rate exceeds the forward swap rate by a positive convexity premium:
        adjustment ≈ S² × σ² × T × duration / annuity

    where duration is the modified duration of the underlying swap (the
    magnitude of dA/dS scaled by 1/A).

    This captures the fact that the swap rate payoff is a non-linear
    function of discount factors. The adjustment is always non-negative.

    Args:
        forward_swap_rate: forward par swap rate.
        annuity: PV of the fixed leg annuity.
        swap_tenor_years: tenor of the underlying swap.
        vol: swaption vol at the CMS fixing.
        time_to_fixing: time to the CMS observation date.
        duration: modified duration. If None, approximated from tenor and rate.
    """
    if time_to_fixing <= 0 or vol <= 0 or abs(annuity) < 1e-15:
        return 0.0

    if duration is None:
        freq = 2.0  # semi-annual
        duration = swap_tenor_years / (1 + forward_swap_rate / freq)

    return forward_swap_rate ** 2 * vol ** 2 * time_to_fixing * duration / annuity


# ---- Re-exports from dedicated modules (backward compatibility) ----
# Canonical locations: pricebook.cash_settlement, pricebook.credit_adjustment

from pricebook.cash_settlement import cash_annuity  # noqa: F401
from pricebook.credit_adjustment import (  # noqa: F401
    cra_discount,
    risky_annuity,
    risky_swap_rate,
    linear_swap_rate_calibrate,
    displaced_lognormal_cross_moment,
)


# ---- CMS Leg ----

@dataclass
class CMSCashflow:
    """One CMS fixing/payment."""
    fixing_date: date
    payment_date: date
    accrual_start: date
    accrual_end: date
    year_fraction: float
    forward_rate: float
    convexity_adj: float
    cms_rate: float


class CMSLeg:
    """Floating leg that pays a CMS rate (e.g. 10Y swap rate) each period.

    Args:
        start: leg start date.
        end: leg end date.
        cms_tenor: tenor of the underlying swap rate (in years).
        notional: notional amount.
        frequency: payment frequency.
        spread: additive spread over the CMS rate.
    """

    def __init__(
        self,
        start: date,
        end: date,
        cms_tenor: int,
        notional: float = 1_000_000.0,
        frequency: Frequency = Frequency.QUARTERLY,
        spread: float = 0.0,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
    ):
        self.start = start
        self.end = end
        self.cms_tenor = cms_tenor
        self.notional = notional
        self.frequency = frequency
        self.spread = spread
        self.day_count = day_count

        schedule = generate_schedule(start, end, frequency)
        self.periods: list[tuple[date, date]] = []
        for i in range(1, len(schedule)):
            self.periods.append((schedule[i - 1], schedule[i]))

    def cashflows(
        self,
        curve: DiscountCurve,
        vol: float = 0.0,
    ) -> list[CMSCashflow]:
        """Compute CMS cashflows with convexity adjustments."""
        ref = curve.reference_date
        result = []
        for accrual_start, accrual_end in self.periods:
            if accrual_end <= ref:
                continue

            yf = year_fraction(accrual_start, accrual_end, self.day_count)
            fixing = accrual_start
            payment = accrual_end

            # Forward swap rate for CMS tenor starting at fixing
            swap_end = fixing + relativedelta(years=self.cms_tenor)
            if fixing <= ref:
                fixing_eff = ref
            else:
                fixing_eff = fixing

            fwd_swap = InterestRateSwap(
                fixing_eff, swap_end, fixed_rate=0.05,
                direction=SwapDirection.PAYER, notional=1_000_000.0,
            )
            fwd_rate = fwd_swap.par_rate(curve)
            ann = fwd_swap.fixed_leg.annuity(curve)

            t_fix = year_fraction(ref, fixing, DayCountConvention.ACT_365_FIXED)
            t_fix = max(t_fix, 0.0)

            adj = cms_convexity_adjustment(
                fwd_rate, ann, self.cms_tenor, vol, t_fix,
            )
            cms_rate = fwd_rate + adj

            result.append(CMSCashflow(
                fixing_date=fixing,
                payment_date=payment,
                accrual_start=accrual_start,
                accrual_end=accrual_end,
                year_fraction=yf,
                forward_rate=fwd_rate,
                convexity_adj=adj,
                cms_rate=cms_rate,
            ))

        return result

    def pv(self, curve: DiscountCurve, vol: float = 0.0) -> float:
        """PV of the CMS leg."""
        total = 0.0
        for cf in self.cashflows(curve, vol):
            df = curve.df(cf.payment_date)
            total += self.notional * (cf.cms_rate + self.spread) * cf.year_fraction * df
        return total


# ---- CMS cap/floor ----

def cms_cap(
    start: date,
    end: date,
    strike: float,
    cms_tenor: int,
    curve: DiscountCurve,
    vol: float,
    notional: float = 1_000_000.0,
    frequency: Frequency = Frequency.QUARTERLY,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """Price a CMS cap (or floor) as a strip of CMS caplets.

    Each caplet pays max(CMS_rate - K, 0) × yf × notional.
    Priced via Black-76 on the convexity-adjusted CMS forward.
    """
    ref = curve.reference_date
    schedule = generate_schedule(start, end, frequency)
    total = 0.0

    for i in range(1, len(schedule)):
        accrual_start = schedule[i - 1]
        accrual_end = schedule[i]
        if accrual_end <= ref:
            continue

        yf = year_fraction(accrual_start, accrual_end, DayCountConvention.ACT_360)
        fixing = accrual_start
        swap_end = fixing + relativedelta(years=cms_tenor)

        if fixing <= ref:
            fixing_eff = ref
        else:
            fixing_eff = fixing

        fwd_swap = InterestRateSwap(
            fixing_eff, swap_end, fixed_rate=0.05,
            direction=SwapDirection.PAYER, notional=1_000_000.0,
        )
        fwd_rate = fwd_swap.par_rate(curve)
        ann = fwd_swap.fixed_leg.annuity(curve)

        t_fix = max(year_fraction(ref, fixing, DayCountConvention.ACT_365_FIXED), 1e-6)
        adj = cms_convexity_adjustment(fwd_rate, ann, cms_tenor, vol, t_fix)
        cms_fwd = fwd_rate + adj

        df = curve.df(accrual_end)
        caplet = black76_price(cms_fwd, strike, vol, t_fix, df, option_type)
        total += notional * yf * caplet

    return total


# ---- CMS spread option ----

def cms_spread_option(
    start: date,
    end: date,
    cms_tenor_long: int,
    cms_tenor_short: int,
    strike: float,
    curve: DiscountCurve,
    vol_long: float,
    vol_short: float,
    correlation: float = 0.9,
    notional: float = 1_000_000.0,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """Price a CMS spread option: payoff on (CMS_long - CMS_short - K).

    Uses Margrabe-like approximation: spread vol ≈ sqrt(v1² + v2² - 2ρv1v2).
    """
    ref = curve.reference_date
    schedule = generate_schedule(start, end, Frequency.QUARTERLY)
    total = 0.0

    for i in range(1, len(schedule)):
        accrual_start = schedule[i - 1]
        accrual_end = schedule[i]
        if accrual_end <= ref:
            continue

        yf = year_fraction(accrual_start, accrual_end, DayCountConvention.ACT_360)
        fixing = accrual_start if accrual_start > ref else ref

        swap_end_l = fixing + relativedelta(years=cms_tenor_long)
        swap_end_s = fixing + relativedelta(years=cms_tenor_short)

        fwd_l = InterestRateSwap(fixing, swap_end_l, 0.05, SwapDirection.PAYER).par_rate(curve)
        fwd_s = InterestRateSwap(fixing, swap_end_s, 0.05, SwapDirection.PAYER).par_rate(curve)
        spread_fwd = fwd_l - fwd_s

        t_fix = max(year_fraction(ref, accrual_start, DayCountConvention.ACT_365_FIXED), 1e-6)

        spread_vol = math.sqrt(max(
            vol_long ** 2 + vol_short ** 2 - 2 * correlation * vol_long * vol_short,
            1e-10,
        ))

        df = curve.df(accrual_end)

        # Black-76 requires positive forward and strike; use Bachelier for near-zero
        if strike <= 0 or spread_fwd <= 0:
            from pricebook.black76 import bachelier_price
            # Convert lognormal vol to absolute (normal) vol via fwd-rate scaling
            # Use a representative rate to scale: max(|fwd|, 0.01)
            absolute_vol = spread_vol * max(abs(spread_fwd), 0.01)
            pv = bachelier_price(spread_fwd, strike, absolute_vol, t_fix, df, option_type)
        else:
            pv = black76_price(spread_fwd, strike, spread_vol, t_fix, df, option_type)
        total += notional * yf * pv

    return total


# ---- Range accrual ----

def range_accrual(
    start: date,
    end: date,
    coupon_rate: float,
    lower_bound: float,
    upper_bound: float,
    curve: DiscountCurve,
    vol: float,
    notional: float = 1_000_000.0,
    frequency: Frequency = Frequency.QUARTERLY,
    obs_per_period: int = 60,
) -> float:
    """Price a range accrual note.

    Coupon accrues only on days when the reference rate is within [L, U].
    Each observation is a digital option: P(L ≤ rate ≤ U).

    Digital P(rate ≤ K) ≈ N(d2) in Black-76, approximated via
    tight call spread.

    Args:
        coupon_rate: annualised coupon rate (paid proportionally).
        lower_bound: lower barrier for accrual.
        upper_bound: upper barrier for accrual.
        obs_per_period: number of observation days per accrual period.
    """
    ref = curve.reference_date
    schedule = generate_schedule(start, end, frequency)
    total = 0.0

    for i in range(1, len(schedule)):
        accrual_start = schedule[i - 1]
        accrual_end = schedule[i]
        if accrual_end <= ref:
            continue

        yf = year_fraction(accrual_start, accrual_end, DayCountConvention.ACT_360)
        df = curve.df(accrual_end)

        # Forward rate for this period (using ACT/360 day count)
        df1 = curve.df(accrual_start)
        df2 = curve.df(accrual_end)
        fwd = (df1 - df2) / (yf * df2)

        # Average time to observations
        t_mid = max(
            year_fraction(ref, accrual_start, DayCountConvention.ACT_365_FIXED),
            1e-6,
        )

        # P(rate in [L, U]) via digital approximation
        # P(rate ≤ U) - P(rate ≤ L) using Black-76 N(d2)
        prob = _digital_range_prob(fwd, lower_bound, upper_bound, vol, t_mid)

        total += notional * coupon_rate * yf * prob * df

    return total


def _digital_range_prob(
    forward: float,
    lower: float,
    upper: float,
    vol: float,
    T: float,
) -> float:
    """Probability that rate falls within [lower, upper] under Black-76.

    P(L ≤ F ≤ U) = N(d2_upper) - N(d2_lower)
    where d2 = (ln(F/K) - 0.5σ²T) / (σ√T)
    """
    from scipy.stats import norm

    if vol <= 0 or T <= 0:
        return 1.0 if lower <= forward <= upper else 0.0

    if upper >= 1e6:
        # No upper barrier
        if lower <= 0:
            return 1.0
        d2_lower = (math.log(forward / lower) - 0.5 * vol ** 2 * T) / (vol * math.sqrt(T))
        return float(norm.cdf(d2_lower))

    if lower <= 0:
        # No lower barrier
        d2_upper = (math.log(forward / upper) - 0.5 * vol ** 2 * T) / (vol * math.sqrt(T))
        return float(norm.cdf(-d2_upper))

    d2_upper = (math.log(forward / upper) - 0.5 * vol ** 2 * T) / (vol * math.sqrt(T))
    d2_lower = (math.log(forward / lower) - 0.5 * vol ** 2 * T) / (vol * math.sqrt(T))

    return float(norm.cdf(-d2_upper) - norm.cdf(-d2_lower))
