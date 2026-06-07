"""Options on credit tranches (CDO tranches).

Prices options on CDO tranche spreads using Black-76 (log-normal) and
Bachelier (normal) models.  The Bachelier model is preferred when tranche
spreads are tight and can approach zero (or go negative for super-senior
tranches after mark-to-market).

    from pricebook.credit.tranche_option import (
        tranche_option_black, tranche_option_bachelier,
        tranche_forward_spread, tranche_option_greeks, tranche_straddle,
    )

    # ATM payer on 5Y [3%-6%] tranche, 100 bps spread, 60 vol, 1Y expiry
    result = tranche_option_black(
        tranche_spread=0.0100, strike_spread=0.0100, vol=0.60,
        T=1.0, r=0.05, annuity=4.2, is_payer=True,
    )
    print(result.price, result.delta)

References:
    O'Kane, D. (2008). *Modelling Single-name and Multi-name Credit
        Derivatives*. Wiley Finance.
    Elizalde, A. (2005). *Credit Default Swap Valuation*. CEMFI Working Paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---- Results ----------------------------------------------------------------

@dataclass
class TrancheOptionResult:
    """Pricing result for a single tranche option."""
    price: float
    delta: float
    gamma: float
    vega: float
    tranche_spread_forward: float
    annuity: float


@dataclass
class StraddleResult:
    """Result for an ATM straddle on a tranche spread."""
    price: float
    breakeven_up: float
    breakeven_down: float
    max_loss: float


# ---- Internal helpers -------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ---- Forward spread ---------------------------------------------------------

def tranche_forward_spread(
    current_spread: float,
    funding_cost: float,
    T: float,
    expected_losses_to_T: float = 0.0,
) -> float:
    """Forward tranche spread adjusted for funding cost and expected losses.

    The forward spread accounts for the cost of carry on the protection leg
    and for any expected tranche losses that will have crystallised by expiry.

    Args:
        current_spread: Current par tranche spread (annualised, decimal).
        funding_cost: Risk-free or repo rate used to carry the position (decimal).
        T: Time to option expiry in years.
        expected_losses_to_T: Expected fractional tranche losses before expiry
            (reduces the notional the option is written on).

    Returns:
        Forward tranche spread (decimal).
    """
    carry_adjustment = math.exp(funding_cost * T)
    survival_factor = max(1.0 - expected_losses_to_T, 1e-8)
    return current_spread * carry_adjustment / survival_factor


# ---- Black-76 ---------------------------------------------------------------

def tranche_option_black(
    tranche_spread: float,
    strike_spread: float,
    vol: float,
    T: float,
    r: float,
    annuity: float,
    is_payer: bool = True,
) -> TrancheOptionResult:
    """Black-76 option on tranche spread.

    A *payer* option gives the right to buy protection at the strike spread
    (profitable when the tranche spread widens beyond the strike).
    A *receiver* option gives the right to sell protection at the strike spread.

    Args:
        tranche_spread: Forward tranche spread (decimal, e.g. 0.01 = 100 bps).
        strike_spread: Option strike spread (decimal).
        vol: Log-normal implied volatility of the tranche spread.
        T: Time to expiry in years.
        r: Risk-free rate (decimal).
        annuity: Risky annuity (RPV01) of the underlying tranche.
        is_payer: True for payer (long protection) option.

    Returns:
        TrancheOptionResult with price and first-order greeks.
    """
    if T <= 0.0:
        intrinsic = max(tranche_spread - strike_spread, 0.0) if is_payer \
            else max(strike_spread - tranche_spread, 0.0)
        pv = math.exp(-r * T) * annuity * intrinsic
        return TrancheOptionResult(
            price=pv, delta=float(is_payer), gamma=0.0, vega=0.0,
            tranche_spread_forward=tranche_spread, annuity=annuity,
        )

    if tranche_spread <= 0 or strike_spread <= 0:
        raise ValueError(
            "Black-76 requires positive spreads; use tranche_option_bachelier "
            "for spreads near zero or negative."
        )
    sqrt_T = math.sqrt(T)
    d1 = (math.log(tranche_spread / strike_spread) + 0.5 * vol ** 2 * T) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T
    df = math.exp(-r * T)

    if is_payer:
        price = df * annuity * (
            tranche_spread * _norm_cdf(d1) - strike_spread * _norm_cdf(d2)
        )
        delta = df * annuity * _norm_cdf(d1)
    else:
        price = df * annuity * (
            strike_spread * _norm_cdf(-d2) - tranche_spread * _norm_cdf(-d1)
        )
        delta = -df * annuity * _norm_cdf(-d1)

    pdf_d1 = _norm_pdf(d1)
    gamma = df * annuity * pdf_d1 / (tranche_spread * vol * sqrt_T)
    vega = df * annuity * tranche_spread * pdf_d1 * sqrt_T

    return TrancheOptionResult(
        price=price, delta=delta, gamma=gamma, vega=vega,
        tranche_spread_forward=tranche_spread, annuity=annuity,
    )


# ---- Bachelier (normal model) -----------------------------------------------

def tranche_option_bachelier(
    tranche_spread: float,
    strike_spread: float,
    vol_normal: float,
    T: float,
    r: float,
    annuity: float,
    is_payer: bool = True,
) -> TrancheOptionResult:
    """Bachelier (normal) model option on tranche spread.

    Preferred when tranche spreads are tight and the log-normal assumption
    breaks down (e.g. super-senior tranches, spreads near zero).

    Args:
        tranche_spread: Forward tranche spread (decimal).
        strike_spread: Option strike spread (decimal).
        vol_normal: Normal (basis-point) volatility of the tranche spread.
        T: Time to expiry in years.
        r: Risk-free rate (decimal).
        annuity: Risky annuity (RPV01) of the underlying tranche.
        is_payer: True for payer option.

    Returns:
        TrancheOptionResult with price and first-order greeks.
    """
    if T <= 0.0:
        intrinsic = max(tranche_spread - strike_spread, 0.0) if is_payer \
            else max(strike_spread - tranche_spread, 0.0)
        pv = math.exp(-r * T) * annuity * intrinsic
        return TrancheOptionResult(
            price=pv, delta=float(is_payer), gamma=0.0, vega=0.0,
            tranche_spread_forward=tranche_spread, annuity=annuity,
        )

    sqrt_T = math.sqrt(T)
    sigma_T = vol_normal * sqrt_T
    moneyness = tranche_spread - strike_spread
    d = moneyness / sigma_T
    df = math.exp(-r * T)
    phi = _norm_pdf(d)
    Phi = _norm_cdf(d)

    sign = 1.0 if is_payer else -1.0
    price = df * annuity * (sign * moneyness * _norm_cdf(sign * d) + sigma_T * phi)
    delta = df * annuity * sign * _norm_cdf(sign * d)
    gamma = df * annuity * phi / sigma_T
    vega = df * annuity * phi * sqrt_T

    return TrancheOptionResult(
        price=price, delta=delta, gamma=gamma, vega=vega,
        tranche_spread_forward=tranche_spread, annuity=annuity,
    )


# ---- Numerical Greeks -------------------------------------------------------

def tranche_option_greeks(
    tranche_spread: float,
    strike_spread: float,
    vol: float,
    T: float,
    r: float,
    annuity: float,
    is_payer: bool = True,
    bump: float = 0.0001,
) -> dict[str, float]:
    """Numerical Greeks for a Black-76 tranche option.

    Args:
        tranche_spread: Forward tranche spread (decimal).
        strike_spread: Strike spread (decimal).
        vol: Log-normal implied volatility.
        T: Time to expiry in years.
        r: Risk-free rate (decimal).
        annuity: Risky annuity.
        is_payer: True for payer option.
        bump: Spread bump for finite differences (default 1 bp).

    Returns:
        Dict with keys: spread_delta, spread_gamma, vega, theta.
    """
    def _price(s: float, v: float, t: float) -> float:
        return tranche_option_black(s, strike_spread, v, t, r, annuity, is_payer).price

    base = _price(tranche_spread, vol, T)
    up = _price(tranche_spread + bump, vol, T)
    dn = _price(tranche_spread - bump, vol, T)

    spread_delta = (up - dn) / (2.0 * bump)
    spread_gamma = (up - 2.0 * base + dn) / (bump ** 2)

    vol_bump = 0.01
    vega = (_price(tranche_spread, vol + vol_bump, T) - base) / vol_bump

    dt = min(1.0 / 252.0, T * 0.5)
    theta = (_price(tranche_spread, vol, T - dt) - base) / dt if T > dt else 0.0

    return {
        "spread_delta": spread_delta,
        "spread_gamma": spread_gamma,
        "vega": vega,
        "theta": theta,
    }


# ---- Straddle ---------------------------------------------------------------

def tranche_straddle(
    tranche_spread: float,
    strike_spread: float,
    vol: float,
    T: float,
    r: float,
    annuity: float,
) -> StraddleResult:
    """ATM straddle on a tranche spread (payer + receiver at same strike).

    Args:
        tranche_spread: Forward tranche spread (decimal).
        strike_spread: Strike spread (typically set to tranche_spread for ATM).
        vol: Log-normal implied volatility.
        T: Time to expiry in years.
        r: Risk-free rate (decimal).
        annuity: Risky annuity.

    Returns:
        StraddleResult with combined price, breakeven spreads, and max_loss.
    """
    payer = tranche_option_black(tranche_spread, strike_spread, vol, T, r, annuity, is_payer=True)
    receiver = tranche_option_black(tranche_spread, strike_spread, vol, T, r, annuity, is_payer=False)

    price = payer.price + receiver.price
    max_loss = price  # premium paid upfront

    # Breakeven: strike ± straddle_price / annuity
    premium_per_spread = price / annuity if annuity > 0.0 else 0.0
    breakeven_up = strike_spread + premium_per_spread
    breakeven_down = max(strike_spread - premium_per_spread, 0.0)

    return StraddleResult(
        price=price,
        breakeven_up=breakeven_up,
        breakeven_down=breakeven_down,
        max_loss=max_loss,
    )
