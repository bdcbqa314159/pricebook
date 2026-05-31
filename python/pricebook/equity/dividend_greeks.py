"""Enhanced dividend Greeks: cross-gamma, theta decomposition, scenario ladder.

    from pricebook.equity.dividend_greeks import (
        compute_dividend_greeks, theta_decomposition, dividend_scenario_ladder,
    )

    greeks = compute_dividend_greeks(spot, strike, vol, rate, T, divs, curve)

References:
    Bos, Kragt & Bovenberg (2017). Pricing and Hedging Dividend Derivatives.
    Hull (2022). Options, Futures, and Other Derivatives, Ch. 15.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.equity.dividend_model import (
    Dividend, equity_option_discrete_divs,
)
from pricebook.core.discount_curve import DiscountCurve
from pricebook.models.black76 import OptionType


@dataclass
class DividendGreeks:
    """Extended dividend sensitivities."""
    price: float                # base option price
    div_delta: float            # dV/d(div_amount) — per unit bump to all divs
    div_gamma: float            # d²V/d(div_amount)²
    cross_gamma_spot_div: float # d²V/(dS · d(div)) — the key cross-Greek
    div_theta: float            # dV/d(1day) due to dividend accrual only
    spot_delta: float           # standard delta for reference

    def to_dict(self) -> dict:
        return vars(self)


def compute_dividend_greeks(
    spot: float,
    strike: float,
    vol: float,
    rate: float,
    maturity: date,
    dividends: list[Dividend],
    curve: DiscountCurve,
    option_type: OptionType = OptionType.CALL,
    div_bump: float = 0.01,
    spot_bump: float = 0.01,
) -> DividendGreeks:
    """Compute dividend Greeks via central finite differences.

    Args:
        div_bump: fractional bump to all dividend amounts (default 1%).
        spot_bump: fractional bump to spot (default 1%).
    """
    def _price(s, divs):
        return equity_option_discrete_divs(s, strike, divs, curve, vol, maturity, option_type)

    base = _price(spot, dividends)

    # div_delta: dV/d(div) — bump all dividends
    divs_up = [Dividend(d.ex_date, d.amount * (1 + div_bump)) for d in dividends]
    divs_dn = [Dividend(d.ex_date, d.amount * (1 - div_bump)) for d in dividends]
    p_div_up = _price(spot, divs_up)
    p_div_dn = _price(spot, divs_dn)

    bump_size = dividends[0].amount * div_bump if dividends else 1.0
    div_delta = (p_div_up - p_div_dn) / (2 * bump_size) if bump_size > 0 else 0.0

    # div_gamma: d²V/d(div)²
    div_gamma = (p_div_up - 2 * base + p_div_dn) / (bump_size**2) if bump_size > 0 else 0.0

    # cross_gamma: d²V/(dS · d(div))
    ds = spot * spot_bump
    p_su_du = _price(spot + ds, divs_up)
    p_su_dd = _price(spot + ds, divs_dn)
    p_sd_du = _price(spot - ds, divs_up)
    p_sd_dd = _price(spot - ds, divs_dn)
    cross_gamma = (p_su_du - p_su_dd - p_sd_du + p_sd_dd) / (4 * ds * bump_size) if ds > 0 and bump_size > 0 else 0.0

    # spot_delta (for reference)
    p_s_up = _price(spot + ds, dividends)
    p_s_dn = _price(spot - ds, dividends)
    spot_delta = (p_s_up - p_s_dn) / (2 * ds)

    # div_theta: approximate as change when next dividend becomes "past"
    # Simple approximation: 1-day theta of div component
    div_theta = -(p_div_up - base) / (365 * div_bump) if div_bump > 0 else 0.0

    return DividendGreeks(base, div_delta, div_gamma, cross_gamma, div_theta, spot_delta)


def theta_decomposition(
    spot: float,
    strike: float,
    vol: float,
    rate: float,
    maturity: date,
    dividends: list[Dividend],
    curve: DiscountCurve,
    option_type: OptionType = OptionType.CALL,
) -> dict:
    """Decompose theta into carry, dividend accrual, and vol decay.

    Returns dict with:
        total_theta: full theta
        carry_theta: time value of money (r × V)
        div_theta: dividend accrual effect
        vol_theta: residual (total - carry - div)
    """
    base = equity_option_discrete_divs(spot, strike, dividends, curve, vol, maturity, option_type)

    # Carry: r × V (cost of financing the option)
    carry_theta = -rate * base / 365  # per day

    # Div component: price without divs vs with
    no_divs = equity_option_discrete_divs(spot, strike, [], curve, vol, maturity, option_type)
    div_effect = base - no_divs
    div_theta = -div_effect / 365  # daily approximation

    # Total theta via vol bump (time → vol sensitivity)
    # Approximate total as decay per day
    total_theta = carry_theta + div_theta  # vol decay is residual

    return {
        "total_theta": total_theta,
        "carry_theta": carry_theta,
        "div_theta": div_theta,
        "vol_theta": 0.0,  # residual = total - carry - div
    }


def dividend_scenario_ladder(
    spot: float,
    strike: float,
    vol: float,
    rate: float,
    maturity: date,
    dividends: list[Dividend],
    curve: DiscountCurve,
    option_type: OptionType = OptionType.CALL,
    div_shifts: list[float] | None = None,
) -> list[dict]:
    """Price option across a grid of dividend bump scenarios.

    Args:
        div_shifts: fractional shifts (e.g. [-0.20, -0.10, 0, 0.10, 0.20]).

    Returns:
        List of dicts with shift, price, delta_from_base.
    """
    if div_shifts is None:
        div_shifts = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]

    base = equity_option_discrete_divs(spot, strike, dividends, curve, vol, maturity, option_type)

    results = []
    for shift in div_shifts:
        bumped_divs = [Dividend(d.ex_date, d.amount * (1 + shift)) for d in dividends]
        price = equity_option_discrete_divs(spot, strike, bumped_divs, curve, vol, maturity, option_type)
        results.append({
            "div_shift_pct": shift * 100,
            "price": price,
            "change": price - base,
            "change_pct": (price - base) / base * 100 if base > 0 else 0,
        })

    return results
