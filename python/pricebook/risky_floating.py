"""Risky floating payments: credit-risky floating leg and FRN.

Layer 0: RiskyFloatingLeg — FloatingLeg × survival probability per period.
Layer 1: CreditRiskyFRN — full FRN instrument with default risk.

    from pricebook.risky_floating import risky_floating_pv, CreditRiskyFRN

    # Standalone risky leg PV
    pv = risky_floating_pv(floating_leg, disc, proj, surv, recovery=0.4)

    # Full instrument
    frn = CreditRiskyFRN(start, end, spread=0.005, recovery=0.4)
    price = frn.dirty_price(disc, proj, surv)

References:
    O'Kane (2008). Modelling Single-name and Multi-name Credit Derivatives.
    Ch. 4 — Risky floating rate notes.
    Schönbucher (2003). Credit Derivatives Pricing Models. Ch. 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.floating_leg import FloatingLeg
from pricebook.schedule import Frequency, StubType
from pricebook.calendar import Calendar, BusinessDayConvention
from pricebook.survival_curve import SurvivalCurve
from pricebook.solvers import brentq
from pricebook.serialisable import _register, _serialise_atom


# ---------------------------------------------------------------------------
# Layer 0: Risky floating leg PV
# ---------------------------------------------------------------------------

@dataclass
class RiskyFloatingResult:
    """Decomposed risky floating payment PV."""
    total_pv: float
    coupon_pv: float           # survival-weighted coupon PV
    accrued_on_default: float  # expected accrued at default
    principal_pv: float        # survival-weighted principal
    recovery_pv: float         # recovery on default

    def to_dict(self) -> dict:
        return {"total_pv": self.total_pv, "coupon_pv": self.coupon_pv,
                "accrued_on_default": self.accrued_on_default,
                "principal_pv": self.principal_pv, "recovery_pv": self.recovery_pv}


def risky_floating_pv(
    floating_leg: FloatingLeg,
    discount_curve: DiscountCurve,
    projection_curve: DiscountCurve | None,
    survival_curve: SurvivalCurve,
    notional: float = 1_000_000.0,
    recovery: float = 0.4,
) -> RiskyFloatingResult:
    """PV of a risky floating leg with survival weighting.

    Each coupon is weighted by Q(t_i). Includes accrued premium on default
    (mid-period approximation) and recovery on principal.

    PV = coupon_pv + accrued_on_default + principal_pv + recovery_pv

    where:
        coupon_pv = Σ (fwd_i + spread) × τ_i × df(t_i) × Q(t_i)
        accrued_on_default = Σ ½(fwd_i + spread) × τ_i × df_mid × ΔPD_i
        principal_pv = notional × df(T) × Q(T)
        recovery_pv = R × notional × Σ df(t_i) × ΔPD_i

    Args:
        floating_leg: the risk-free FloatingLeg (already built with schedule).
        discount_curve: OIS discount curve.
        projection_curve: forward rate projection curve (None = single curve).
        survival_curve: issuer's survival curve.
        notional: face value / principal.
        recovery: recovery rate on default.
    """
    proj = projection_curve if projection_curve is not None else discount_curve

    coupon_pv = 0.0
    accrued_pv = 0.0
    recovery_pv = 0.0

    for cf in floating_leg.cashflows:
        fwd = cf.forward_rate(proj)
        rate = fwd + cf.spread
        yf = cf.year_frac
        df_end = discount_curve.df(cf.payment_date)
        surv_end = survival_curve.survival(cf.payment_date)
        surv_start = survival_curve.survival(cf.accrual_start)
        dpd = surv_start - surv_end  # default probability this period

        # Coupon: received if alive at payment date
        coupon_pv += notional * rate * yf * df_end * surv_end

        # Accrued on default: half-period approximation
        # If default occurs mid-period, accrued ≈ ½ × coupon
        mid_date = cf.accrual_start
        df_mid = discount_curve.df(cf.accrual_start)  # approximate
        accrued_pv += 0.5 * notional * rate * yf * df_mid * dpd

        # Recovery: on default, get R × notional
        recovery_pv += recovery * notional * df_end * dpd

    # Principal at maturity: received if alive
    end_date = floating_leg.cashflows[-1].payment_date if floating_leg.cashflows else floating_leg.end
    principal_pv = notional * discount_curve.df(end_date) * survival_curve.survival(end_date)

    total = coupon_pv + accrued_pv + principal_pv + recovery_pv

    return RiskyFloatingResult(
        total_pv=total, coupon_pv=coupon_pv,
        accrued_on_default=accrued_pv,
        principal_pv=principal_pv, recovery_pv=recovery_pv,
    )


# ---------------------------------------------------------------------------
# Layer 1: CreditRiskyFRN instrument
# ---------------------------------------------------------------------------

class CreditRiskyFRN:
    """Credit-risky floating-rate note.

    Pays floating coupons (forward + spread) conditional on survival.
    On default: recovery × notional. At maturity: notional if alive.

    This is the credit-risky version of FloatingRateNote.

    Args:
        start: issue date.
        end: maturity.
        spread: credit spread over floating index.
        notional: face value.
        recovery: recovery rate on default.
        frequency: coupon frequency.
        day_count: accrual convention.
    """

    _SERIAL_TYPE = "credit_risky_frn"

    def __init__(
        self,
        start: date,
        end: date,
        spread: float = 0.005,
        notional: float = 1_000_000.0,
        recovery: float = 0.4,
        frequency: Frequency = Frequency.QUARTERLY,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
        calendar: Calendar | None = None,
        convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
    ):
        self.start = start
        self.end = end
        self.spread = spread
        self.notional = notional
        self.recovery = recovery
        self.frequency = frequency
        self.day_count = day_count
        self.calendar = calendar
        self.convention = convention

        self.floating_leg = FloatingLeg(
            start, end, frequency,
            notional=notional, spread=spread, day_count=day_count,
            calendar=calendar, convention=convention,
        )

    def dirty_price(
        self,
        discount_curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
        survival_curve: SurvivalCurve | None = None,
    ) -> float:
        """Full price per 100 face.

        If survival_curve is None, returns risk-free FRN price.
        """
        if survival_curve is None:
            # Risk-free
            pv_coupons = self.floating_leg.pv(discount_curve, projection_curve)
            pv_principal = self.notional * discount_curve.df(self.end)
            return (pv_coupons + pv_principal) / self.notional * 100.0

        result = risky_floating_pv(
            self.floating_leg, discount_curve, projection_curve,
            survival_curve, self.notional, self.recovery,
        )
        return result.total_pv / self.notional * 100.0

    def price_decomposition(
        self,
        discount_curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
        survival_curve: SurvivalCurve | None = None,
    ) -> RiskyFloatingResult:
        """Full decomposition: coupon, accrued, principal, recovery."""
        if survival_curve is None:
            sc = SurvivalCurve.flat(discount_curve.reference_date, 0.0001)
        else:
            sc = survival_curve
        return risky_floating_pv(
            self.floating_leg, discount_curve, projection_curve,
            sc, self.notional, self.recovery,
        )

    def credit_spread_sensitivity(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
        projection_curve: DiscountCurve | None = None,
        shift_bps: float = 1.0,
    ) -> float:
        """CS01: price change per 1bp parallel spread shift."""
        base = self.dirty_price(discount_curve, projection_curve, survival_curve)
        bumped_sc = survival_curve.bumped(shift_bps / 10_000)
        up = self.dirty_price(discount_curve, projection_curve, bumped_sc)
        return (up - base) / shift_bps

    def z_spread(
        self,
        market_price: float,
        discount_curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
        survival_curve: SurvivalCurve | None = None,
    ) -> float:
        """Z-spread: constant spread to discount curve that reprices to market.

        For a credit-risky FRN, z-spread > 0 reflects both credit risk
        and any additional spread.
        """
        def objective(z: float) -> float:
            shifted = discount_curve.bumped(-z)  # shift discount down = higher rate
            return self.dirty_price(shifted, projection_curve, survival_curve) - market_price

        return brentq(objective, -0.05, 0.10)

    def discount_margin(
        self,
        market_price: float,
        discount_curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
        survival_curve: SurvivalCurve | None = None,
    ) -> float:
        """Discount margin: spread adjustment to reprice at market."""
        def objective(dm: float) -> float:
            shifted_frn = CreditRiskyFRN(
                self.start, self.end, spread=self.spread + dm,
                notional=self.notional, recovery=self.recovery,
                frequency=self.frequency, day_count=self.day_count,
            )
            return shifted_frn.dirty_price(discount_curve, projection_curve,
                                            survival_curve) - market_price

        return brentq(objective, -0.05, 0.10)

    def pv_ctx(self, ctx) -> float:
        sc = None
        if ctx.credit_curves:
            sc = next(iter(ctx.credit_curves.values()))
        proj = None
        if ctx.projection_curves:
            proj = next(iter(ctx.projection_curves.values()))
        return self.dirty_price(ctx.discount_curve, proj, sc) / 100.0 * self.notional

    # ---- Serialisation ----

    def to_dict(self) -> dict[str, Any]:
        return {"type": self._SERIAL_TYPE, "params": {
            "start": self.start.isoformat(), "end": self.end.isoformat(),
            "spread": self.spread, "notional": self.notional,
            "recovery": self.recovery,
            "frequency": _serialise_atom(self.frequency),
            "day_count": _serialise_atom(self.day_count),
        }}

    @classmethod
    def from_dict(cls, d: dict) -> CreditRiskyFRN:
        p = d["params"]
        return cls(
            start=date.fromisoformat(p["start"]), end=date.fromisoformat(p["end"]),
            spread=p.get("spread", 0.005), notional=p.get("notional", 1_000_000.0),
            recovery=p.get("recovery", 0.4),
            frequency=Frequency(p.get("frequency", 3)),
            day_count=DayCountConvention(p.get("day_count", "ACT/360")),
        )


_register(CreditRiskyFRN)
