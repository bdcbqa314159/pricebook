"""Perpetual bonds and step-up coupon bonds.

Perpetuals have no maturity — they pay coupons forever.
Step-up bonds have coupon escalation at specific dates (common in AT1/Tier 2).

    from pricebook.fixed_income.perpetual import (
        PerpetualBond, StepUpBond, price_perpetual,
    )

References:
    Tuckman & Serrat (2012). Fixed Income Securities, Ch 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.serialisable import serialisable as _serialisable


@dataclass
class PerpetualBond:
    """A perpetual bond (consol).

    Pays coupon forever, no maturity. Optionally callable.
    """
    coupon: float                    # annual coupon rate
    face_value: float = 100.0
    first_call_years: float | None = None  # callable perpetual
    call_price: float = 100.0
    step_up_bp: float = 0.0         # coupon step-up after call date (bp)
    currency: str = "EUR"

    def to_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class PerpetualPricingResult:
    """Result of perpetual bond pricing."""
    clean_price: float
    yield_current: float             # coupon / price
    yield_to_call: float | None      # if callable
    duration: float                  # Macaulay duration approximation
    call_value: float                # value of the call option to the issuer

    def to_dict(self) -> dict:
        return dict(vars(self))


def price_perpetual(
    perp: PerpetualBond,
    discount_curve: DiscountCurve,
    call_probability: float = 0.90,
    credit_spread: float = 0.0,
    max_years: int = 50,
) -> PerpetualPricingResult:
    """Price a perpetual bond.

    For a plain perpetual: PV = Σ coupon × df(t) for t = 1, 2, ...
    For callable: blend call scenario and extension scenario.

    Args:
        perp: perpetual bond specification.
        discount_curve: risk-free discount curve.
        call_probability: probability of call at first call date.
        credit_spread: additional spread for credit risk.
        max_years: cap for numerical summation (default 50).
    """
    ref = discount_curve.reference_date
    coupon = perp.coupon * perp.face_value
    freq = 2  # semi-annual
    cpn_per = coupon / freq

    if perp.first_call_years is not None:
        # Callable perpetual: price to call + extension
        T_call = perp.first_call_years
        step_up_rate = perp.coupon + perp.step_up_bp / 10_000
        cpn_ext = step_up_rate * perp.face_value / freq

        # Scenario 1: called
        pv_called = 0.0
        n_call = int(T_call * freq)
        for i in range(1, n_call + 1):
            t = i / freq
            df = _df_spread(discount_curve, ref, t, credit_spread)
            pv_called += cpn_per * df
        pv_called += perp.call_price * _df_spread(discount_curve, ref, T_call, credit_spread)

        # Scenario 2: not called, extends with step-up
        pv_ext = 0.0
        n_ext = max_years * freq
        for i in range(1, n_ext + 1):
            t = i / freq
            df = _df_spread(discount_curve, ref, t, credit_spread)
            c = cpn_per if t <= T_call else cpn_ext
            pv_ext += c * df

        price = call_probability * pv_called + (1 - call_probability) * pv_ext
        call_value = pv_ext - pv_called  # how much issuer saves by calling
    else:
        # Plain perpetual (no call)
        price = 0.0
        n = max_years * freq
        for i in range(1, n + 1):
            t = i / freq
            df = _df_spread(discount_curve, ref, t, credit_spread)
            price += cpn_per * df
        call_value = 0.0

    # Analytics
    yield_current = coupon / price if price > 0 else 0.0

    # Yield to call
    ytc = None
    if perp.first_call_years is not None:
        ytc = _solve_ytc_perp(price, perp.coupon, perp.first_call_years,
                               perp.face_value, perp.call_price, freq)

    # Duration: approx (1+y)/y for perpetual
    y = yield_current
    duration = (1 + y) / y if y > 0 else max_years

    return PerpetualPricingResult(
        clean_price=price,
        yield_current=yield_current,
        yield_to_call=ytc,
        duration=duration,
        call_value=call_value,
    )


class StepUpBond:
    """Bond with coupon step-up at specific dates.

    Common in bank capital instruments: coupon increases if not called.
    """

    def __init__(
        self,
        issue_date: date,
        maturity_years: float,
        initial_coupon: float,
        step_up_schedule: list[tuple[float, float]],
        face_value: float = 100.0,
    ):
        """
        Args:
            issue_date: issue date.
            maturity_years: maturity in years (or large number for quasi-perpetual).
            initial_coupon: coupon rate before first step-up.
            step_up_schedule: [(year, new_coupon), ...] e.g. [(5, 0.08), (10, 0.09)].
            face_value: face value.
        """
        self.issue_date = issue_date
        self.maturity_years = maturity_years
        self.initial_coupon = initial_coupon
        self.step_up_schedule = sorted(step_up_schedule, key=lambda x: x[0])
        self.face_value = face_value

    def coupon_at(self, t: float) -> float:
        """Coupon rate at time t years from issue."""
        rate = self.initial_coupon
        for step_t, step_rate in self.step_up_schedule:
            if t >= step_t:
                rate = step_rate
        return rate

    def price(self, discount_curve: DiscountCurve, credit_spread: float = 0.0) -> float:
        """Price the step-up bond."""
        ref = discount_curve.reference_date
        freq = 2
        n = int(self.maturity_years * freq)
        pv = 0.0
        for i in range(1, n + 1):
            t = i / freq
            cpn_rate = self.coupon_at(t)
            cpn = cpn_rate / freq * self.face_value
            df = _df_spread(discount_curve, ref, t, credit_spread)
            pv += cpn * df
        # Principal
        pv += self.face_value * _df_spread(discount_curve, ref, self.maturity_years, credit_spread)
        return pv

    def to_dict(self) -> dict:
        return {
            "issue_date": self.issue_date.isoformat(),
            "maturity_years": self.maturity_years,
            "initial_coupon": self.initial_coupon,
            "step_up_schedule": self.step_up_schedule,
            "face_value": self.face_value,
        }


def _df_spread(curve: DiscountCurve, ref: date, t: float, spread: float) -> float:
    """Discount factor with additional credit spread."""
    d = date.fromordinal(ref.toordinal() + int(t * 365))
    return curve.df(d) * math.exp(-spread * t)


def _solve_ytc_perp(price, coupon, call_years, face, call_price, freq):
    from pricebook.core.solvers import brentq
    def obj(y):
        pv = 0.0
        n = int(call_years * freq)
        c = coupon / freq * face
        for i in range(1, n + 1):
            pv += c / (1 + y / freq) ** i
        pv += call_price / (1 + y / freq) ** n
        return pv - price
    try:
        return brentq(obj, -0.10, 1.0)
    except ValueError:
        return coupon

_serialisable("step_up_bond", ['issue_date', 'maturity_years', 'initial_coupon', 'step_up_schedule', 'face_value'])(StepUpBond)
