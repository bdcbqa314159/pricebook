"""Dividend modelling: implied dividends, dividend swaps, and dividend risk.

Extract implied dividends from options via put-call parity, price
dividend swaps/forwards, and compute dividend sensitivity.

    from pricebook.dividend_desk import (
        implied_dividends, DividendSwap, dividend_risk,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.dividend_model import Dividend, pv_dividends, dividend_adjusted_forward


# ---- Implied dividends from put-call parity ----

def implied_dividend_pv(
    spot: float,
    call_price: float,
    put_price: float,
    strike: float,
    df: float,
) -> float:
    """Extract PV of dividends from put-call parity.

    C - P = (S - PV(divs)) × df_to_expiry × (F/S) - K × df
    Simplifying: C - P = S - PV(divs) - K × df  (for forward delivery)

    PV(divs) = S - K × df - (C - P)
    """
    return spot - strike * df - (call_price - put_price)


def implied_dividends_term_structure(
    spot: float,
    options: list[tuple[date, float, float, float, float]],
    reference_date: date,
    discount_curve: DiscountCurve,
) -> list[tuple[date, float]]:
    """Extract implied dividend term structure from options at multiple expiries.

    Args:
        options: list of (expiry, strike, call_price, put_price, df).
            Should be sorted by expiry.

    Returns:
        List of (expiry, cumulative_implied_div_pv) sorted by expiry.
    """
    result = []
    for expiry, strike, call, put, df in sorted(options, key=lambda x: x[0]):
        pv_div = implied_dividend_pv(spot, call, put, strike, df)
        result.append((expiry, max(pv_div, 0.0)))
    return result


def strip_discrete_dividends(
    cumulative_pvs: list[tuple[date, float]],
    discount_curve: DiscountCurve,
) -> list[Dividend]:
    """Convert cumulative PV term structure to discrete dividends.

    Incremental dividend between consecutive expiries, discounted back.
    """
    dividends = []
    prev_pv = 0.0
    for i, (d, cum_pv) in enumerate(cumulative_pvs):
        incremental = cum_pv - prev_pv
        if incremental > 0:
            df = discount_curve.df(d)
            if df > 0:
                amount = incremental / df
                dividends.append(Dividend(d, amount))
        prev_pv = cum_pv
    return dividends


# ---- Dividend swap ----

@dataclass
class DividendSwap:
    """Swap exchanging realised dividends for a fixed amount.

    Buyer receives: sum of actual dividends over the period.
    Buyer pays: fixed_div × notional at maturity.
    PV = PV(expected divs) × notional - fixed_div × df(T) × notional.

    Convention: dividends are per-share amounts, notional = number of shares.
    fixed_div is the per-share fixed amount.
    """
    start: date
    end: date
    fixed_div: float
    notional: float = 1.0

    def pv(
        self,
        dividends: list[Dividend],
        discount_curve: DiscountCurve,
    ) -> float:
        """PV of the dividend swap (receiver of realised divs)."""
        pv_divs = sum(
            d.amount * discount_curve.df(d.ex_date)
            for d in dividends
            if self.start <= d.ex_date <= self.end
        )
        pv_fixed = self.fixed_div * discount_curve.df(self.end) * self.notional
        return pv_divs * self.notional - pv_fixed

    def fair_fixed(
        self,
        dividends: list[Dividend],
        discount_curve: DiscountCurve,
    ) -> float:
        """Fixed dividend rate that makes PV = 0."""
        pv_divs = sum(
            d.amount * discount_curve.df(d.ex_date)
            for d in dividends
            if self.start <= d.ex_date <= self.end
        )
        df_T = discount_curve.df(self.end)
        if abs(df_T) < 1e-15:
            return 0.0
        return pv_divs / df_T


# ---- Dividend forward ----

def dividend_forward(
    dividends: list[Dividend],
    start: date,
    end: date,
    discount_curve: DiscountCurve,
) -> float:
    """Forward price of cumulative dividends over [start, end].

    FV = PV(divs) / df(end).
    """
    pv = sum(
        d.amount * discount_curve.df(d.ex_date)
        for d in dividends
        if start <= d.ex_date <= end
    )
    df = discount_curve.df(end)
    return pv / df if abs(df) > 1e-15 else 0.0


# ---- Dividend risk ----

@dataclass
class DividendRisk:
    """Sensitivity of an equity forward to dividend assumptions."""
    forward_price: float
    div_delta: float  # dF/d(div_pv) per unit PV change
    div_rho: float    # dF/d(1bp div yield change)


def dividend_risk(
    spot: float,
    dividends: list[Dividend],
    discount_curve: DiscountCurve,
    maturity: date,
    div_bump: float = 0.01,
) -> DividendRisk:
    """Compute sensitivity of equity forward to dividend changes.

    Args:
        div_bump: fractional bump to all dividend amounts (e.g. 0.01 = +1%).
    """
    fwd_base = dividend_adjusted_forward(spot, dividends, discount_curve, maturity)

    # Bump all dividends up
    bumped_divs = [Dividend(d.ex_date, d.amount * (1 + div_bump)) for d in dividends]
    fwd_up = dividend_adjusted_forward(spot, bumped_divs, discount_curve, maturity)

    div_delta = (fwd_up - fwd_base) / div_bump if div_bump > 0 else 0.0

    # Rho: sensitivity to a 1bp change in continuous div yield
    pv_divs = pv_dividends(dividends, discount_curve, maturity)
    df_T = discount_curve.df(maturity)
    # d(forward)/d(div_yield) ≈ -S/df × T (from F = S×exp((r-q)T))
    T = year_fraction(discount_curve.reference_date, maturity, DayCountConvention.ACT_365_FIXED)
    div_rho = -spot * T / df_T * 0.0001 if df_T > 0 else 0.0

    return DividendRisk(
        forward_price=fwd_base,
        div_delta=div_delta,
        div_rho=div_rho,
    )
