"""Interest rate swap."""

from __future__ import annotations

from datetime import date
from dataclasses import dataclass
from enum import Enum

from pricebook.core.day_count import DayCountConvention
from pricebook.core.discount_curve import DiscountCurve
from pricebook.fixed_income.fixed_leg import FixedLeg
from pricebook.core.fixings import FixingsStore
from pricebook.fixed_income.floating_leg import FloatingLeg
from pricebook.core.schedule import Frequency, StubType
from pricebook.core.calendar import Calendar, BusinessDayConvention


class SwapDirection(Enum):
    PAYER = "payer"        # pay fixed, receive floating
    RECEIVER = "receiver"  # receive fixed, pay floating


def _map_notional_to_schedule(
    fixed_leg: "FixedLeg",
    start: date,
    end: date,
    float_frequency: Frequency,
    calendar=None,
    convention=None,
    stub=None,
    eom=True,
) -> list[float]:
    """Map a fixed-leg notional schedule to floating-leg periods.

    Each floating period inherits the notional of the fixed period
    whose accrual window contains the floating period's start date.
    This is the market convention for amortising swaps where the
    fixed and floating legs have different frequencies.
    """
    from pricebook.core.calendar import BusinessDayConvention
    from pricebook.core.schedule import StubType, generate_schedule

    _conv = convention or BusinessDayConvention.MODIFIED_FOLLOWING
    _stub = stub or StubType.SHORT_FRONT
    float_schedule = generate_schedule(start, end, float_frequency, calendar, _conv, _stub, eom)

    fixed_periods = fixed_leg.cashflows
    float_notionals = []

    for i in range(1, len(float_schedule)):
        float_start = float_schedule[i - 1]
        # Find the fixed period containing this floating period start
        matched = fixed_periods[-1].notional  # fallback to last
        for cf in fixed_periods:
            if cf.accrual_start <= float_start < cf.accrual_end:
                matched = cf.notional
                break
        float_notionals.append(matched)

    return float_notionals


class InterestRateSwap:
    """
    A vanilla interest rate swap: fixed leg vs floating leg.

    Payer swap: pay fixed, receive floating -> PV = PV(float) - PV(fixed)
    Receiver swap: receive fixed, pay floating -> PV = PV(fixed) - PV(float)

    Supports dual-curve pricing:
        - discount_curve: used for discounting all cashflows
        - projection_curve: used for computing floating forward rates
    Single-curve is the special case where both are the same.
    """

    def __init__(
        self,
        start: date,
        end: date,
        fixed_rate: float,
        direction: SwapDirection = SwapDirection.PAYER,
        notional: float | list[float] = 1_000_000.0,
        fixed_frequency: Frequency = Frequency.SEMI_ANNUAL,
        float_frequency: Frequency = Frequency.QUARTERLY,
        fixed_day_count: DayCountConvention = DayCountConvention.THIRTY_360,
        float_day_count: DayCountConvention = DayCountConvention.ACT_360,
        spread: float = 0.0,
        calendar: Calendar | None = None,
        convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
        stub: StubType = StubType.SHORT_FRONT,
        eom: bool = True,
        payment_delay_days: int = 0,
        observation_shift_days: int = 0,
    ):
        self.start = start
        self.end = end
        self.fixed_rate = fixed_rate
        self.direction = direction
        self.fixed_frequency = fixed_frequency
        self.float_frequency = float_frequency
        self.fixed_day_count = fixed_day_count
        self.float_day_count = float_day_count
        self.spread = spread
        self.calendar = calendar
        self.convention = convention
        self.stub = stub
        self.eom = eom
        self.payment_delay_days = payment_delay_days
        self.observation_shift_days = observation_shift_days

        self.fixed_leg = FixedLeg(
            start, end, fixed_rate, fixed_frequency,
            notional=notional, day_count=fixed_day_count,
            calendar=calendar, convention=convention, stub=stub, eom=eom,
            payment_delay_days=payment_delay_days,
        )

        # For the floating leg: if notional is a list, map fixed-leg periods
        # to floating-leg periods (which may have different frequency).
        # Each floating period inherits the notional of the fixed period
        # it falls within — market convention for amortising swaps.
        if isinstance(notional, list):
            float_notional = _map_notional_to_schedule(
                self.fixed_leg, start, end, float_frequency,
                calendar, convention, stub, eom,
            )
        else:
            float_notional = notional

        self.floating_leg = FloatingLeg(
            start, end, float_frequency,
            notional=float_notional, spread=spread, day_count=float_day_count,
            calendar=calendar, convention=convention, stub=stub, eom=eom,
            payment_delay_days=payment_delay_days,
            observation_shift_days=observation_shift_days,
        )

        # Store scalar notional (face amount) — always float
        self.notional = self.fixed_leg.notional

    def pv(
        self,
        curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
        fixings: FixingsStore | None = None,
        rate_name: str | None = None,
    ) -> float:
        """
        Present value of the swap.

        Args:
            curve: discount curve.
            projection_curve: forward projection curve. If None, single-curve pricing.
            fixings: historical fixings for seasoned swaps.
            rate_name: index name in fixings store (e.g. "SOFR").
        """
        pv_fixed = self.fixed_leg.pv(curve)
        pv_float = self.floating_leg.pv(curve, projection_curve, fixings, rate_name)
        if self.direction == SwapDirection.PAYER:
            return pv_float - pv_fixed
        return pv_fixed - pv_float

    def par_rate(
        self,
        curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
    ) -> float:
        """
        The fixed rate that makes PV = 0.

        par_rate = PV_float / weighted_annuity

        For uniform notional N: weighted_annuity = N * annuity, so this
        equals PV_float / (N * annuity) — identical to the classic formula.
        For variable notional (amortising/accreting), uses the correct
        notional-weighted sum.
        """
        w_annuity = self.fixed_leg.weighted_annuity(curve)
        if abs(w_annuity) < 1e-15:
            return 0.0
        pv_float = self.floating_leg.pv(curve, projection_curve)
        return pv_float / w_annuity

    def annuity(self, curve: DiscountCurve) -> float:
        """Fixed leg annuity (PV01): sum of year_frac × df for each period.

        This is the PV of the fixed leg per unit of fixed rate.
        Fundamental for par rate, DV01, and risk decomposition.
        """
        return self.fixed_leg.annuity(curve)

    @property
    def notional_schedule(self) -> list[float]:
        """Per-period notional schedule (from fixed leg)."""
        return list(self.fixed_leg.notional_schedule)

    @property
    def average_notional(self) -> float:
        """Arithmetic mean of the notional schedule."""
        ns = self.fixed_leg.notional_schedule
        return sum(ns) / len(ns)

    def pv_ctx(
        self,
        ctx,
        projection_curve_name: str | None = None,
    ) -> float:
        """Price the swap from a PricingContext."""
        curve = ctx.discount_curve
        projection_curve = (
            ctx.get_projection_curve(projection_curve_name)
            if projection_curve_name is not None
            else None
        )
        return self.pv(curve, projection_curve)

    def dv01(
        self,
        curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
        shift: float = 0.0001,
    ) -> float:
        """Parallel DV01: PV change for a 1bp parallel shift in rates."""
        pv_base = self.pv(curve, projection_curve)
        bumped = curve.bumped(shift)
        bumped_proj = projection_curve.bumped(shift) if projection_curve is not None else None
        return self.pv(bumped, bumped_proj) - pv_base

    def cashflow_schedule(
        self,
        curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
    ) -> list[dict]:
        """Produce a full cashflow schedule for both legs.

        Returns a list of dicts with: leg, accrual_start, accrual_end,
        payment_date, rate, spread, year_frac, notional, amount, df, pv.

        Fix T4-SWAP2: pre-fix the floating row reported ``rate`` as the
        pure forward (no spread) but ``amount`` already INCLUDED the
        spread, so the natural verification
        ``rate · year_frac · notional ≈ amount`` was wrong by exactly
        ``spread · year_frac · notional``.  On a 1mm notional swap with
        50 bp spread and a 3-month period that's ~$1,236 of "missing"
        amount per coupon — easy for a downstream consumer (P&L attribution,
        cashflow reconciliation, regulatory reporting) to misread.

        The fix adds two new fields per row:
          - ``spread``: the spread component (0 for the fixed leg).
          - ``notional``: the period notional (for amortising / accreting).

        With these, ``(rate + spread) · year_frac · notional == amount``
        holds exactly for both legs, and the row is fully self-describing.
        Existing ``rate`` / ``amount`` semantics are unchanged.
        """
        proj = projection_curve if projection_curve is not None else curve
        rows = []
        for cf in self.fixed_leg.cashflows:
            df = curve.df(cf.payment_date)
            rows.append({
                "leg": "fixed",
                "accrual_start": cf.accrual_start,
                "accrual_end": cf.accrual_end,
                "payment_date": cf.payment_date,
                "rate": cf.rate,
                "spread": 0.0,
                "year_frac": cf.year_frac,
                "notional": cf.notional,
                "amount": cf.amount,
                "df": df,
                "pv": cf.amount * df,
            })
        for cf in self.floating_leg.cashflows:
            fwd = cf.forward_rate(proj)
            amount = cf.amount(proj)
            df = curve.df(cf.payment_date)
            rows.append({
                "leg": "float",
                "accrual_start": cf.accrual_start,
                "accrual_end": cf.accrual_end,
                "payment_date": cf.payment_date,
                "rate": fwd,
                "spread": cf.spread,
                "year_frac": cf.year_frac,
                "notional": cf.notional,
                "amount": amount,
                "df": df,
                "pv": amount * df,
            })
        return sorted(rows, key=lambda r: r["payment_date"])

    # ── Factory classmethods ──

    @classmethod
    def amortising(
        cls,
        start: date,
        end: date,
        fixed_rate: float,
        initial_notional: float = 1_000_000.0,
        direction: SwapDirection = SwapDirection.PAYER,
        **kwargs,
    ) -> "InterestRateSwap":
        """Amortising swap: notional decreases linearly to zero by maturity.

        Period i (0-indexed, i=0..n-1) outstanding notional is
        ``initial · (1 - i/n)``.  Standard market convention: period 0
        carries the full initial notional, each subsequent period is
        smaller by ``initial/n``, and after the final period (i.e. at
        maturity) the principal is fully repaid (notional = 0).  Note the
        FINAL period's outstanding notional is ``initial/n``, NOT zero —
        the zero is the post-maturity state, not a period value.

            swap = InterestRateSwap.amortising(start, end, 0.04, 1_000_000)
        """
        from pricebook.core.schedule import generate_schedule
        freq = kwargs.get("fixed_frequency", Frequency.SEMI_ANNUAL)
        schedule = generate_schedule(start, end, freq)
        n = len(schedule) - 1
        notionals = [initial_notional * (1.0 - i / n) for i in range(n)]
        return cls(start, end, fixed_rate, direction, notional=notionals, **kwargs)

    @classmethod
    def accreting(
        cls,
        start: date,
        end: date,
        fixed_rate: float,
        initial_notional: float = 500_000.0,
        final_notional: float = 1_000_000.0,
        direction: SwapDirection = SwapDirection.PAYER,
        **kwargs,
    ) -> "InterestRateSwap":
        """Accreting swap: notional increases linearly from initial to final.

        Period i (0-indexed, i=0..n-1) has notional
        ``initial + (final - initial) · i / (n - 1)`` for ``n >= 2``.

        Fix T4-SWAP1: pre-fix the n=1 branch divided by ``max(n-1, 1) = 1``
        and looped ``i in range(1)``, so the single-period notional was
        ``initial + 0·… = initial`` — ``final_notional`` was silently
        dropped.  Calling ``accreting(...500_000, 1_000_000)`` with a
        single-period schedule returned ``[500_000]`` with no warning.

        Correct: for n=1, "accreting" is degenerate (no two endpoints to
        interpolate between).  We use the **average** of the requested
        endpoints — it is the only choice that honours BOTH inputs and is
        symmetric in ``initial`` / ``final``.

            swap = InterestRateSwap.accreting(start, end, 0.04, 500_000, 1_000_000)
        """
        from pricebook.core.schedule import generate_schedule
        freq = kwargs.get("fixed_frequency", Frequency.SEMI_ANNUAL)
        schedule = generate_schedule(start, end, freq)
        n = len(schedule) - 1
        if n <= 1:
            notionals = [0.5 * (initial_notional + final_notional)]
        else:
            notionals = [
                initial_notional + (final_notional - initial_notional) * i / (n - 1)
                for i in range(n)
            ]
        return cls(start, end, fixed_rate, direction, notional=notionals, **kwargs)

    @classmethod
    def roller_coaster(
        cls,
        start: date,
        end: date,
        fixed_rate: float,
        notional_schedule: list[float],
        direction: SwapDirection = SwapDirection.PAYER,
        **kwargs,
    ) -> "InterestRateSwap":
        """Roller-coaster swap: arbitrary notional schedule.

            swap = InterestRateSwap.roller_coaster(start, end, 0.04, [1e6, 2e6, 1e6])
        """
        return cls(start, end, fixed_rate, direction, notional=notional_schedule, **kwargs)


@classmethod
def _irs_from_convention(cls, conv, start, end, fixed_rate,
                         direction=SwapDirection.PAYER, notional=1_000_000.0, spread=0.0):
    """Create InterestRateSwap from a CurrencyConventions object.

    The convention provides: fixed/float frequency, fixed/float day_count.
    The caller provides: start, end, fixed_rate, direction, notional, spread.
    """
    return cls(
        start, end, fixed_rate, direction, notional,
        fixed_frequency=conv.fixed_frequency,
        float_frequency=conv.float_frequency,
        fixed_day_count=conv.fixed_day_count,
        float_day_count=conv.float_day_count,
        spread=spread,
    )

InterestRateSwap.from_convention = _irs_from_convention


def create_swap(currency: str, start, end, fixed_rate,
                direction=SwapDirection.PAYER, notional=1_000_000.0, spread=0.0):
    """Create an InterestRateSwap with correct conventions for the currency.

    Convenience wrapper around InterestRateSwap.from_convention().
    """
    from pricebook.curves.curve_builder import get_conventions as _get_curve_conv
    conv = _get_curve_conv(currency)
    return InterestRateSwap.from_convention(conv, start, end, fixed_rate, direction, notional, spread)


from pricebook.core.serialisable import serialisable as _serialisable
_serialisable("irs", ["start", "end", "fixed_rate", "direction", "notional", "fixed_frequency", "float_frequency", "fixed_day_count", "float_day_count", "spread"])(InterestRateSwap)
