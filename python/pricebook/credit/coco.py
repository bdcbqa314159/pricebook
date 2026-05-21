"""Contingent Convertible (CoCo) / AT1 bonds.

CoCo bonds are hybrid capital instruments that convert to equity or suffer
principal write-down when the issuer's capital ratio breaches a trigger.

    from pricebook.credit.coco import (
        CoCoBond, CoCoTriggerType, CoCoLossAbsorption, price_coco,
    )

    coco = CoCoBond(
        coupon=0.07, maturity_years=None,  # perpetual AT1
        trigger_level=0.0525,  # 5.25% CET1 ratio
        trigger_type=CoCoTriggerType.CET1_RATIO,
        loss_absorption=CoCoLossAbsorption.FULL_WRITE_DOWN,
    )
    price = price_coco(coco, discount_curve, trigger_prob_curve)

Pricing approaches:
1. Credit-derivative approach: PV = coupon annuity × (1 - trigger_prob) + recovery × trigger_prob
2. Equity-derivative approach: trigger linked to stock price via Merton
3. Structural approach: Black-Cox first-passage to CET1 barrier

We implement approach 1 (market-standard for trading desks).

References:
    De Spiegeleer & Schoutens (2012). Pricing Contingent Convertibles.
    Wilkens & Bethke (2014). Contingent Convertible Bonds: A First Empirical
    Assessment of Selected Pricing Models. Financial Analysts Journal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from pricebook.core.discount_curve import DiscountCurve


class CoCoTriggerType(Enum):
    """Type of conversion/write-down trigger."""
    CET1_RATIO = "cet1_ratio"          # Common Equity Tier 1 ratio
    TOTAL_CAPITAL = "total_capital"     # Total capital ratio
    STOCK_PRICE = "stock_price"         # Stock price barrier
    REGULATORY = "regulatory"           # PONV (point of non-viability) — regulator discretion


class CoCoLossAbsorption(Enum):
    """How loss is absorbed on trigger breach."""
    FULL_WRITE_DOWN = "full_write_down"             # Principal → 0
    PARTIAL_WRITE_DOWN = "partial_write_down"       # Principal reduced to restore ratio
    EQUITY_CONVERSION = "equity_conversion"          # Convert to shares at fixed price
    EQUITY_CONVERSION_FLOOR = "equity_conversion_floor"  # Convert at max(price, floor)


@dataclass
class CoCoBond:
    """CoCo / AT1 bond specification."""
    coupon: float                                    # annual coupon (e.g. 0.07 = 7%)
    trigger_level: float                             # e.g. 0.0525 for 5.25% CET1
    trigger_type: CoCoTriggerType = CoCoTriggerType.CET1_RATIO
    loss_absorption: CoCoLossAbsorption = CoCoLossAbsorption.FULL_WRITE_DOWN
    maturity_years: float | None = None              # None = perpetual (AT1)
    first_call_years: float = 5.0                    # first call date
    call_price: float = 100.0                        # call at par
    face_value: float = 100.0
    conversion_price: float | None = None            # for equity conversion
    write_down_pct: float = 1.0                      # 1.0 = full, 0.5 = 50%
    coupon_cancellation: bool = True                 # AT1: coupon can be skipped
    currency: str = "EUR"

    def to_dict(self) -> dict:
        return {
            "coupon": self.coupon,
            "trigger_level": self.trigger_level,
            "trigger_type": self.trigger_type.value,
            "loss_absorption": self.loss_absorption.value,
            "maturity_years": self.maturity_years,
            "first_call_years": self.first_call_years,
            "face_value": self.face_value,
            "coupon_cancellation": self.coupon_cancellation,
        }


@dataclass
class CoCoPricingResult:
    """Result of CoCo pricing."""
    clean_price: float
    coupon_pv: float                 # PV of coupon stream
    redemption_pv: float             # PV of call/maturity redemption
    trigger_loss_pv: float           # PV of loss on trigger breach
    yield_to_call: float             # yield assuming called at first call
    yield_to_worst: float            # min of YTC and YTM
    credit_spread_bp: float          # implied spread over risk-free
    trigger_prob_5y: float           # cumulative trigger probability to 5Y

    def to_dict(self) -> dict:
        return vars(self)


def price_coco(
    coco: CoCoBond,
    discount_curve: DiscountCurve,
    trigger_intensity: float,
    call_probability: float = 0.90,
    coupon_skip_prob: float = 0.0,
) -> CoCoPricingResult:
    """Price a CoCo bond using the credit-derivative approach.

    The CoCo is decomposed into:
    1. Coupon annuity (survival-weighted, adjusted for skip probability)
    2. Redemption at call/maturity (survival-weighted)
    3. Loss on trigger breach (trigger probability × loss given trigger)

    Args:
        coco: CoCo bond specification.
        discount_curve: risk-free discount curve.
        trigger_intensity: annual trigger intensity (hazard-like, e.g. 0.02 = 2%/yr).
        call_probability: probability bond is called at first call (market convention ~90%).
        coupon_skip_prob: probability of coupon cancellation per period.

    Returns:
        CoCoPricingResult with price breakdown and analytics.
    """
    if trigger_intensity < 0:
        raise ValueError(f"trigger_intensity must be >= 0, got {trigger_intensity}")
    face = coco.face_value
    T_call = coco.first_call_years
    T_mat = coco.maturity_years or T_call  # perpetual AT1: price to call
    freq = 2  # semi-annual coupons (AT1 standard)
    coupon_per_period = coco.coupon / freq * face

    # Loss given trigger
    if coco.loss_absorption in (CoCoLossAbsorption.FULL_WRITE_DOWN,):
        lgt = face * coco.write_down_pct
    elif coco.loss_absorption == CoCoLossAbsorption.PARTIAL_WRITE_DOWN:
        lgt = face * coco.write_down_pct
    else:
        # Equity conversion: loss depends on conversion price vs market
        # Approximate as 50% loss (shares received worth ~50% of face)
        lgt = face * 0.50

    recovery_on_trigger = face - lgt

    # Extension horizon for not-called scenario
    T_ext = T_mat + 5.0 if coco.maturity_years is None else T_mat
    ref = discount_curve.reference_date

    # Coupons to call date (both scenarios share this)
    coupon_pv_to_call = 0.0
    n_call = int(T_call * freq)
    for i in range(1, n_call + 1):
        t = i / freq
        df = _df_at(discount_curve, ref, t)
        q_survive = math.exp(-trigger_intensity * t)
        skip_adj = 1.0 - coupon_skip_prob
        coupon_pv_to_call += coupon_per_period * df * q_survive * skip_adj

    # Additional coupons in not-called extension (T_call → T_ext)
    coupon_pv_ext = 0.0
    n_ext = int(T_ext * freq)
    for i in range(n_call + 1, n_ext + 1):
        t = i / freq
        df = _df_at(discount_curve, ref, t)
        q_survive = math.exp(-trigger_intensity * t)
        skip_adj = 1.0 - coupon_skip_prob
        coupon_pv_ext += coupon_per_period * df * q_survive * skip_adj

    # Blended coupon PV
    coupon_pv = coupon_pv_to_call + (1 - call_probability) * coupon_pv_ext

    # Redemption: call at T_call or maturity/extension at T_ext
    df_call = _df_at(discount_curve, ref, T_call)
    df_ext = _df_at(discount_curve, ref, T_ext)
    q_call = math.exp(-trigger_intensity * T_call)
    q_ext = math.exp(-trigger_intensity * T_ext)

    redemption_pv = (
        call_probability * coco.call_price * df_call * q_call
        + (1 - call_probability) * face * df_ext * q_ext
    )

    # Trigger loss: expected PV of loss on trigger breach over full horizon
    trigger_loss_pv = 0.0
    dt = 0.25
    for t in np.arange(dt, T_ext + dt, dt):
        df = _df_at(discount_curve, ref, float(t))
        q = math.exp(-trigger_intensity * float(t))
        q_prev = math.exp(-trigger_intensity * float(t - dt))
        p_trigger = q_prev - q
        trigger_loss_pv += p_trigger * (face - recovery_on_trigger) * df

    clean_price = coupon_pv + redemption_pv - trigger_loss_pv

    # Analytics
    trigger_5y = 1.0 - math.exp(-trigger_intensity * 5.0)

    # Yield to call (approximate)
    ytc = _solve_ytc(clean_price, coco.coupon, T_call, face, freq)

    # Credit spread
    rf_5y = -math.log(_df_at(discount_curve, ref, 5.0)) / 5.0 if T_call > 0 else 0.0
    spread_bp = max((ytc - rf_5y) * 10_000, 0.0)

    return CoCoPricingResult(
        clean_price=clean_price,
        coupon_pv=coupon_pv,
        redemption_pv=redemption_pv,
        trigger_loss_pv=trigger_loss_pv,
        yield_to_call=ytc,
        yield_to_worst=ytc,
        credit_spread_bp=spread_bp,
        trigger_prob_5y=trigger_5y,
    )


def _df_at(curve: DiscountCurve, ref, t_years: float) -> float:
    """Discount factor at t years from reference."""
    from datetime import date, timedelta
    d = date.fromordinal(ref.toordinal() + int(t_years * 365))
    return curve.df(d)


def _solve_ytc(price: float, coupon: float, call_years: float, face: float, freq: int) -> float:
    """Solve for yield-to-call."""
    from pricebook.core.solvers import brentq

    def objective(y: float) -> float:
        pv = 0.0
        n = int(call_years * freq)
        c = coupon / freq * face
        for i in range(1, n + 1):
            pv += c / (1 + y / freq) ** i
        pv += face / (1 + y / freq) ** n
        return pv - price

    try:
        return brentq(objective, -0.10, 1.0)
    except ValueError:
        return coupon  # fallback
