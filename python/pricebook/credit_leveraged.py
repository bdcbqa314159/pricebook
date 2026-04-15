"""Leveraged credit structures: LCDS, digital CLN, CMCDS, CDS options, TRS.

Phase C9 slices 231-233 — final credit deepening phase.

* :func:`leveraged_cds` — first-loss on a single name.
* :func:`digital_cln_leveraged` — digital recovery CLN with leverage.
* :func:`constant_maturity_cds` — CMCDS with convexity adjustment.
* :func:`cds_straddle` — straddle on CDS spread.
* :func:`credit_trs` — total return swap on a credit index.

References:
    O'Kane, *Modelling Single-name and Multi-name Credit Derivatives*,
    Wiley, 2008, Ch. 13-16.
    Schönbucher, *Credit Derivatives Pricing Models*, Ch. 10-11.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.cds_swaption import cds_swaption_black, forward_cds_spread


# ---- Leveraged CDS ----

@dataclass
class LeveragedCDSResult:
    """Leveraged CDS pricing result."""
    premium: float
    leveraged_spread: float
    standard_spread: float
    leverage: float
    max_loss: float


def leveraged_cds(
    notional: float,
    leverage: float,
    flat_hazard: float,
    flat_rate: float = 0.05,
    recovery: float = 0.4,
    maturity_years: int = 5,
    frequency: int = 4,
) -> LeveragedCDSResult:
    """Price a leveraged CDS (first-loss on a single name).

    The protection buyer pays a leveraged spread on notional,
    but losses are capped at notional × leverage × (1−R).
    The seller takes first-loss up to leverage × notional.

    Leveraged spread = standard_spread × leverage
    (in practice, modified by the loss cap).

    Args:
        leverage: leverage factor (e.g. 3.0 = 3× exposure).
    """
    dt = 1.0 / frequency
    n_periods = maturity_years * frequency

    # Standard CDS spread ≈ λ(1−R)
    standard_spread = flat_hazard * (1 - recovery)

    # Leveraged protection PV: min(loss, cap) where cap = leverage × notional × (1-R)
    max_loss = leverage * notional * (1 - recovery)

    protection_pv = 0.0
    annuity = 0.0

    for i in range(1, n_periods + 1):
        t = i * dt
        df = math.exp(-flat_rate * t)
        surv = math.exp(-flat_hazard * t)
        surv_prev = math.exp(-flat_hazard * (t - dt))
        dpd = surv_prev - surv

        # Loss on default: min(leverage × notional × (1−R), max_loss)
        # For single name LCDS, loss = leverage × notional × (1−R) always ≤ max_loss
        loss = leverage * notional * (1 - recovery)
        protection_pv += df * dpd * loss

        annuity += df * surv * notional * dt

    leveraged_spread = protection_pv / annuity if annuity > 0 else 0.0
    premium = protection_pv - leveraged_spread * annuity  # at par, PV ≈ 0

    return LeveragedCDSResult(
        premium, leveraged_spread, standard_spread, leverage, max_loss,
    )


# ---- Digital CLN with leverage ----

@dataclass
class DigitalCLNResult:
    """Leveraged digital CLN result."""
    price: float
    coupon_pv: float
    digital_loss_pv: float
    leverage: float


def digital_cln_leveraged(
    notional: float,
    coupon_rate: float,
    leverage: float,
    maturity_years: int = 5,
    flat_rate: float = 0.05,
    flat_hazard: float = 0.02,
    frequency: int = 4,
) -> DigitalCLNResult:
    """Digital CLN with leverage: pays 0 or full notional on default.

    On default: investor loses leverage × notional (capped at notional).
    No recovery uncertainty — payout is digital.
    Premium is higher than vanilla CLN due to binary payoff.

    Args:
        leverage: multiplier on digital loss (capped at 1.0 per unit).
    """
    dt = 1.0 / frequency
    n_periods = maturity_years * frequency
    effective_leverage = min(leverage, 1.0 / (1e-10 + 0))  # no cap needed for digital

    coupon_pv = 0.0
    digital_loss_pv = 0.0

    for i in range(1, n_periods + 1):
        t = i * dt
        df = math.exp(-flat_rate * t)
        surv = math.exp(-flat_hazard * t)
        surv_prev = math.exp(-flat_hazard * (t - dt))
        dpd = surv_prev - surv

        coupon_pv += df * surv * coupon_rate * notional * dt
        # Digital loss: on default, lose min(leverage, 1) × notional
        digital_loss_pv += df * dpd * min(leverage, 1.0) * notional

    # Principal at maturity (conditional on survival)
    df_T = math.exp(-flat_rate * maturity_years)
    surv_T = math.exp(-flat_hazard * maturity_years)
    principal_pv = df_T * surv_T * notional

    price = (coupon_pv + principal_pv - digital_loss_pv) / notional * 100

    return DigitalCLNResult(price, coupon_pv, digital_loss_pv, leverage)


# ---- Constant-maturity CDS ----

@dataclass
class CMCDSResult:
    """Constant-maturity CDS result."""
    fair_spread: float
    convexity_adjustment: float
    forward_spread: float


def constant_maturity_cds(
    reference_maturity: int,
    flat_hazard: float,
    flat_rate: float = 0.05,
    recovery: float = 0.4,
    spread_vol: float = 0.40,
    reset_frequency: int = 4,
    payment_years: int = 5,
) -> CMCDSResult:
    """Price a constant-maturity CDS (CMCDS).

    The CMCDS premium resets periodically to the prevailing par CDS
    spread of a fixed-maturity reference CDS. Analogous to CMS swaps.

    The fair CMCDS spread differs from the forward spread due to a
    convexity adjustment: E[spread] under payment measure ≠ forward.

    Convexity ≈ 0.5 × σ² × T × ∂²(annuity)/∂spread² / annuity.

    Args:
        reference_maturity: maturity of the reference CDS (e.g. 5Y).
        spread_vol: volatility of the CDS spread.
        reset_frequency: how often the spread resets.
        payment_years: total CMCDS tenor.
    """
    fwd = forward_cds_spread(
        0.0, float(reference_maturity), flat_hazard, flat_rate, recovery,
    )
    forward_spread = fwd.forward_spread

    # Convexity adjustment (simplified TSR-like)
    # For a linear annuity approximation: ∂A/∂s ≈ −duration × A
    # Convexity ≈ 0.5 σ² T × (duration²) × A / A
    duration = sum(
        i / reset_frequency * math.exp(-flat_rate * i / reset_frequency)
        * math.exp(-flat_hazard * i / reset_frequency)
        for i in range(1, reference_maturity * reset_frequency + 1)
    )
    annuity_val = fwd.risky_annuity
    if annuity_val > 0:
        convexity = 0.5 * spread_vol**2 * payment_years * (duration / annuity_val)**2 * forward_spread
    else:
        convexity = 0.0

    fair_spread = forward_spread + convexity

    return CMCDSResult(fair_spread, convexity, forward_spread)


# ---- CDS straddle ----

@dataclass
class CDSStraddleResult:
    """CDS straddle (payer + receiver swaption)."""
    premium: float
    payer_premium: float
    receiver_premium: float
    breakeven_move: float


def cds_straddle(
    flat_hazard: float,
    strike_spread: float | None = None,
    spread_vol: float = 0.40,
    expiry: float = 1.0,
    cds_maturity: float = 6.0,
    flat_rate: float = 0.05,
    recovery: float = 0.4,
    notional: float = 1_000_000,
) -> CDSStraddleResult:
    """Price a CDS straddle: long payer + long receiver swaption.

    ATM straddle (strike = forward) profits from any large spread move.

    Args:
        strike_spread: swaption strike. If None, uses ATM (forward spread).
    """
    fwd = forward_cds_spread(expiry, cds_maturity, flat_hazard, flat_rate, recovery)

    if strike_spread is None:
        strike_spread = fwd.forward_spread

    payer = cds_swaption_black(
        fwd.forward_spread, strike_spread, spread_vol, expiry,
        fwd.risky_annuity, fwd.survival_to_start, notional, "payer",
    )
    receiver = cds_swaption_black(
        fwd.forward_spread, strike_spread, spread_vol, expiry,
        fwd.risky_annuity, fwd.survival_to_start, notional, "receiver",
    )

    total = payer.premium + receiver.premium

    # Breakeven: spread must move by premium / (annuity × notional × survival)
    denom = fwd.risky_annuity * notional * fwd.survival_to_start
    breakeven = total / denom if denom > 0 else 0.0

    return CDSStraddleResult(total, payer.premium, receiver.premium, breakeven)


# ---- Credit total return swap ----

@dataclass
class CreditTRSResult:
    """Credit total return swap result."""
    trs_pv: float
    total_return: float
    funding_cost: float
    credit_pnl: float


def credit_trs(
    index_notional: float,
    index_spread_start: float,
    index_spread_end: float,
    funding_rate: float,
    trs_spread: float,
    period_years: float = 0.25,
    flat_rate: float = 0.05,
) -> CreditTRSResult:
    """Price a total return swap on a credit index.

    Total return receiver gets: spread income + price change.
    Total return payer gets: funding rate + TRS spread.

    PV = (total_return − funding_cost) × notional × period.

    Args:
        index_spread_start: index spread at period start.
        index_spread_end: index spread at period end.
        funding_rate: floating rate paid by TR receiver.
        trs_spread: additional spread paid by TR receiver.
    """
    # Spread tightening = positive return for TR receiver
    # Approximate price change: −duration × Δspread
    duration = 4.0  # typical IG index duration
    price_change = -duration * (index_spread_end - index_spread_start) / 10_000

    # Carry: spread income
    carry = index_spread_start / 10_000 * period_years

    total_return = price_change + carry
    funding_cost = (funding_rate + trs_spread) * period_years

    credit_pnl = total_return - funding_cost
    trs_pv = credit_pnl * index_notional

    return CreditTRSResult(trs_pv, total_return, funding_cost, credit_pnl)
