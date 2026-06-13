"""
European swaption — the right to enter an interest rate swap.

A payer swaption gives the right to enter a payer swap (pay fixed).
A receiver swaption gives the right to enter a receiver swap (receive fixed).

Priced with Black-76 on the forward swap rate:

    price = annuity * Black76(forward_swap_rate, strike, vol, T_expiry)

where:
    forward_swap_rate = par rate of the forward-starting underlying swap
    annuity = sum of year_frac * df for the fixed leg of the underlying
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pricebook.models.black76 import OptionType
from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.greeks import Greeks
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.pricing_context import PricingContext
from pricebook.core.schedule import Frequency, StubType
from pricebook.fixed_income.swap import InterestRateSwap, SwapDirection
from pricebook.core.calendar import Calendar, BusinessDayConvention


class SwaptionType(Enum):
    PAYER = "payer"        # right to pay fixed (call on swap rate)
    RECEIVER = "receiver"  # right to receive fixed (put on swap rate)


class Swaption:
    """
    European swaption on a vanilla interest rate swap.

    Args:
        expiry: option expiry date (= start of the underlying swap).
        swap_end: maturity of the underlying swap.
        strike: the fixed rate of the underlying swap.
        swaption_type: PAYER or RECEIVER.
        notional: notional of the underlying swap.
        fixed_frequency: coupon frequency of the fixed leg.
        float_frequency: coupon frequency of the floating leg.
        fixed_day_count: day count for the fixed leg.
        float_day_count: day count for the floating leg.
        calendar: business day calendar.
        convention: business day convention.
        stub: stub type for schedule generation.
        eom: end-of-month rule.
    """

    def __init__(
        self,
        expiry: date,
        swap_end: date,
        strike: float,
        swaption_type: SwaptionType = SwaptionType.PAYER,
        notional: float = 1_000_000.0,
        fixed_frequency: Frequency = Frequency.SEMI_ANNUAL,
        float_frequency: Frequency = Frequency.QUARTERLY,
        fixed_day_count: DayCountConvention = DayCountConvention.THIRTY_360,
        float_day_count: DayCountConvention = DayCountConvention.ACT_360,
        calendar: Calendar | None = None,
        convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
        stub: StubType = StubType.SHORT_FRONT,
        eom: bool = True,
    ):
        if expiry >= swap_end:
            raise ValueError("expiry must be before swap_end")

        self.expiry = expiry
        self.swap_end = swap_end
        self.strike = strike
        self.swaption_type = swaption_type
        self.notional = notional
        self.fixed_frequency = fixed_frequency
        self.float_frequency = float_frequency
        self.fixed_day_count = fixed_day_count
        self.float_day_count = float_day_count
        self.calendar = calendar
        self.convention = convention
        self.stub = stub
        self.eom = eom

        # Build the underlying forward-starting swap (payer by convention)
        self.underlying = InterestRateSwap(
            start=expiry,
            end=swap_end,
            fixed_rate=strike,
            direction=SwapDirection.PAYER,
            notional=notional,
            fixed_frequency=fixed_frequency,
            float_frequency=float_frequency,
            fixed_day_count=fixed_day_count,
            float_day_count=float_day_count,
            calendar=calendar,
            convention=convention,
            stub=stub,
            eom=eom,
        )

    def forward_swap_rate(
        self,
        curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
    ) -> float:
        """Par rate of the underlying forward-starting swap."""
        return self.underlying.par_rate(curve, projection_curve)

    def annuity(self, curve: DiscountCurve) -> float:
        """Annuity of the underlying swap's fixed leg."""
        return self.underlying.fixed_leg.annuity(curve)

    def pv_ctx(
        self,
        ctx: PricingContext,
        vol_surface_name: str = "ir",
        projection_curve_name: str | None = None,
    ) -> float:
        """Price the swaption from a PricingContext (Black-76 via vol surface)."""
        if ctx.discount_curve is None:
            raise ValueError("PricingContext must have a discount_curve")

        vol_surface = ctx.get_vol_surface(vol_surface_name)
        projection_curve = (
            ctx.get_projection_curve(projection_curve_name)
            if projection_curve_name is not None
            else None
        )

        from pricebook.models.models import Black76Model
        vol = vol_surface.vol(self.expiry, self.strike)
        return self.price(
            Black76Model(vol=vol),
            ctx.discount_curve,
            projection_curve,
            valuation_date=ctx.valuation_date,
        )

    def greeks(
        self,
        model,
        curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
        valuation_date: date | None = None,
    ) -> Greeks:
        """Swaption Greeks using a pluggable model.

        If model implements ``greeks_ir_option()``, uses analytical greeks.
        Otherwise falls back to bump-and-reprice.

        Args:
            model: any IROptionModel (Black76Model, BachelierModel, SABRModel, etc.).
            curve: discount curve.
            projection_curve: forward projection curve (None = single-curve).
            valuation_date: date for computing time to expiry.
        """
        if valuation_date is None:
            valuation_date = curve.reference_date

        fwd = self.forward_swap_rate(curve, projection_curve)
        ann = self.annuity(curve)

        if self.expiry <= valuation_date:
            intrinsic = self.price(model, curve, projection_curve, valuation_date)
            return Greeks(price=intrinsic)

        T = year_fraction(valuation_date, self.expiry, DayCountConvention.ACT_365_FIXED)
        option_type = (
            OptionType.CALL if self.swaption_type == SwaptionType.PAYER
            else OptionType.PUT
        )

        if hasattr(model, "greeks_ir_option"):
            raw = model.greeks_ir_option(fwd, self.strike, ann, T, option_type)
            return Greeks(
                price=self.notional * raw.price,
                delta=self.notional * raw.delta,
                gamma=self.notional * raw.gamma,
                vega=self.notional * raw.vega,
                theta=self.notional * raw.theta,
            )

        # Fallback: bump-and-reprice
        base = self.price(model, curve, projection_curve, valuation_date)
        bump = 0.0001
        up = self.price(model, curve.bumped(bump), projection_curve, valuation_date)
        dn = self.price(model, curve.bumped(-bump), projection_curve, valuation_date)
        delta = (up - dn) / (2 * bump)
        gamma = (up - 2 * base + dn) / (bump ** 2)
        return Greeks(price=base, delta=delta, gamma=gamma)

    def price(
        self,
        model,
        curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
        valuation_date: date | None = None,
    ) -> float:
        """Price using a pluggable model (Black76Model, BachelierModel, SABRModel, etc.).

        The model handles the option-pricing step. The swaption computes its own
        forward swap rate, annuity, and time to expiry.

        For tree-based models (HullWhiteTreeModel), the model receives the full
        swaption object via duck-typed ``price_swaption()``.

        Args:
            model: any object implementing ``price_ir_option(forward, strike, annuity, T, option_type)``
                   or ``price_swaption(swaption, curve)`` for tree models.
            curve: discount curve.
            projection_curve: forward projection curve (None = single-curve).
            valuation_date: date for computing time to expiry.
        """
        if valuation_date is None:
            valuation_date = curve.reference_date

        # Duck-type: HW-style models get the full swaption + all context
        if hasattr(model, "price_swaption"):
            return self.notional * model.price_swaption(
                self, curve, projection_curve, valuation_date)

        if not hasattr(model, "price_ir_option"):
            raise TypeError(
                f"{type(model).__name__} does not implement price_ir_option() "
                f"or price_swaption() — cannot price a swaption with this model"
            )

        fwd = self.forward_swap_rate(curve, projection_curve)
        ann = self.annuity(curve)

        if self.expiry <= valuation_date:
            if self.swaption_type == SwaptionType.PAYER:
                return self.notional * ann * max(fwd - self.strike, 0.0)
            return self.notional * ann * max(self.strike - fwd, 0.0)

        T = year_fraction(valuation_date, self.expiry, DayCountConvention.ACT_365_FIXED)
        option_type = (
            OptionType.CALL if self.swaption_type == SwaptionType.PAYER
            else OptionType.PUT
        )

        return self.notional * model.price_ir_option(fwd, self.strike, ann, T, option_type)


from pricebook.core.serialisable import serialisable as _serialisable
_serialisable("swaption", ["expiry", "swap_end", "strike", "swaption_type", "notional", "fixed_frequency", "float_frequency", "fixed_day_count", "float_day_count"])(Swaption)

@classmethod
def _swaption_from_convention(cls, conv, expiry, swap_end, strike,
                               swaption_type=None, notional=1_000_000.0):
    """Create Swaption from CurrencyConventions (swap freq/dc from convention)."""
    if swaption_type is None:
        from pricebook.options.swaption import SwaptionType
        swaption_type = SwaptionType.PAYER
    return cls(expiry, swap_end, strike, swaption_type, notional,
               fixed_frequency=conv.fixed_frequency, float_frequency=conv.float_frequency,
               fixed_day_count=conv.fixed_day_count, float_day_count=conv.float_day_count)

Swaption.from_convention = _swaption_from_convention


# ═══════════════════════════════════════════════════════════════
# SABR-HW blended pricing
# ═══════════════════════════════════════════════════════════════


def price_swaption_sabr_hw(
    swaption: Swaption,
    sabr_cube,
    hw_model,
    curve,
    blend_half_life: float = 5.0,
) -> float:
    """Price a swaption blending SABR smile with Hull-White term structure.

    At short expiry, SABR dominates (smile accuracy).
    At long expiry, HW dominates (mean reversion term structure).

    Weighting: vol = w_sabr × sabr_vol + (1-w_sabr) × hw_vol
    where w_sabr = exp(-expiry / blend_half_life).

    Args:
        swaption: Swaption to price.
        sabr_cube: SwaptionVolCube with SABR smile.
        hw_model: calibrated HullWhite model.
        curve: discount curve.
        blend_half_life: years at which SABR weight = 0.5.

    Returns:
        Blended swaption price.
    """
    import math
    from pricebook.core.day_count import DayCountConvention, year_fraction
    from pricebook.models.black76 import black76_price, OptionType

    # Fix T4-SW1: pre-fix `blend_half_life` was not validated.  Passing 0
    # caused `math.exp(-T / 0.0)` to raise `ZeroDivisionError` deep inside
    # the pricer with no diagnostic context.  Validate upfront.
    if blend_half_life <= 0:
        raise ValueError(
            f"blend_half_life must be > 0 (got {blend_half_life}); "
            f"use a positive years value (default 5.0)."
        )

    ref = curve.reference_date
    T = year_fraction(ref, swaption.expiry, DayCountConvention.ACT_365_FIXED)

    # Forward swap rate and annuity (also used by the T≤0 intrinsic branch).
    fwd = swaption.forward_swap_rate(curve)
    ann = swaption.annuity(curve)

    if T <= 0:
        # Fix T2.15: pre-fix this returned 0.0 unconditionally at T=0, even
        # if the swaption had positive intrinsic value (e.g. a payer with
        # strike below the current forward swap rate).  Correct behaviour
        # at expiry is the intrinsic value:
        #   payer:    annuity · max(fwd − K, 0)
        #   receiver: annuity · max(K − fwd, 0)
        if swaption.swaption_type == SwaptionType.PAYER:
            intrinsic = max(fwd - swaption.strike, 0.0)
        else:
            intrinsic = max(swaption.strike - fwd, 0.0)
        return ann * intrinsic * swaption.notional

    # SABR vol (from cube)
    tenor = year_fraction(swaption.expiry, swaption.swap_end,
                           DayCountConvention.ACT_365_FIXED)
    sabr_vol = sabr_cube.vol(swaption.expiry, tenor, swaption.strike)

    # HW vol (from tree → implied vol)
    from pricebook.models.hw_calibration import _hw_implied_vol
    hw_vol = _hw_implied_vol(hw_model.a, hw_model.sigma, curve,
                              T, tenor, swaption.strike, n_steps=30)

    # Blending weight
    w_sabr = math.exp(-T / blend_half_life)

    # Blended vol — Fix T4-SW2: pre-fix when BOTH SABR and HW vols failed
    # (returned ≤ 0), the fallback silently substituted a hard-coded 1%
    # volatility and produced an essentially arbitrary price with no
    # warning.  Fail loudly instead so the caller can diagnose the
    # upstream SABR/HW failure.
    if sabr_vol > 0 and hw_vol > 0:
        blended_vol = w_sabr * sabr_vol + (1 - w_sabr) * hw_vol
    elif sabr_vol > 0:
        blended_vol = sabr_vol
    elif hw_vol > 0:
        blended_vol = hw_vol
    else:
        raise ValueError(
            "price_swaption_sabr_hw: both SABR and HW vols returned "
            f"non-positive values (sabr_vol={sabr_vol}, hw_vol={hw_vol}); "
            "the cube/HW calibration upstream is degenerate — investigate "
            "the inputs rather than relying on a silent fallback."
        )

    # Price via Black-76
    opt_type = OptionType.CALL if swaption.swaption_type == SwaptionType.PAYER else OptionType.PUT
    price = ann * black76_price(fwd, swaption.strike, blended_vol, T, 1.0, opt_type)

    return price * swaption.notional
