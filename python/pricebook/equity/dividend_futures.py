"""Dividend futures, swaps, and options.

Pricing for single-stock and index dividend derivatives.
Total return futures vs price return distinction.

* :func:`dividend_future_price` — fair value of dividend futures.
* :func:`dividend_swap_fair_value` — dividend swap fixed rate.
* :func:`dividend_option_price` — option on dividend.
* :func:`total_return_future` — TR future vs price return.

References:
    Buehler et al., *Discrete Dividend Modeling*, in *Equity Derivatives*,
    Risk Books, 2010.
    Eurex, *Dividend Derivatives Product Guide*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.models.black76 import (
    OptionType, black76_price, black76_delta, black76_vega,
)


@dataclass
class DividendFutureResult:
    """Dividend future pricing result."""
    fair_value: float           # PV of expected dividends
    implied_dividend: float     # market-implied dividend
    spot: float
    forward: float
    dividend_yield: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def dividend_future_price(
    spot: float,
    forward: float,
    rate: float,
    T: float,
) -> DividendFutureResult:
    """Fair value of a dividend future.

    From the cost-of-carry relationship:
    F = S × exp(rT) − FV(divs)
    ⟹ FV(divs) = S × exp(rT) − F
    ⟹ PV(divs) = (S × exp(rT) − F) × exp(−rT) = S − F × exp(−rT)

    Args:
        spot: current spot price.
        forward: equity forward price (or futures price).
        rate: risk-free rate.
        T: time to expiry (years).
    """
    df = math.exp(-rate * T)
    implied_div = spot - forward * df
    div_yield = implied_div / spot / T if spot > 0 and T > 0 else 0

    return DividendFutureResult(
        fair_value=implied_div,
        implied_dividend=implied_div,
        spot=spot,
        forward=forward,
        dividend_yield=div_yield,
    )


@dataclass
class DividendSwapResult:
    """Dividend swap pricing result."""
    fixed_rate: float           # fair fixed dividend amount
    pv: float                   # MTM PV (for existing swap)
    notional: float
    realised_dividends: float   # if provided

    def to_dict(self) -> dict:
        return dict(vars(self))


def dividend_swap_fair_value(
    expected_dividends: list[float],
    payment_dates_years: list[float],
    rate: float = 0.04,
    notional: float = 1_000_000.0,
    realised: float | None = None,
    fixed_rate: float | None = None,
) -> DividendSwapResult:
    """Dividend swap fair fixed rate.

    Float leg pays realised dividends. Fixed leg pays agreed amount.
    Fair fixed = PV(expected dividends) / PV(annuity).

    Args:
        expected_dividends: expected dividend per period.
        payment_dates_years: payment timing (years from now).
        rate: risk-free rate.
        notional: swap notional.
        realised: if given, compute MTM PV of existing swap.
        fixed_rate: agreed fixed rate (for MTM calculation).
    """
    float_pv = sum(
        d * math.exp(-rate * t)
        for d, t in zip(expected_dividends, payment_dates_years)
    )

    annuity = sum(math.exp(-rate * t) for t in payment_dates_years)
    fair_fixed = float_pv / annuity if annuity > 0 else 0

    if realised is not None and fixed_rate is not None:
        pv = (realised - fixed_rate) * notional
    else:
        pv = 0.0

    return DividendSwapResult(
        fixed_rate=fair_fixed,
        pv=pv,
        notional=notional,
        realised_dividends=realised or 0,
    )


@dataclass
class DividendOptionResult:
    """Option on dividend result."""
    price: float
    delta: float
    vega: float
    forward_dividend: float
    strike: float
    vol: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def dividend_option_price(
    forward_dividend: float,
    strike: float,
    vol: float,
    T: float,
    rate: float = 0.04,
    option_type: str = "call",
    notional: float = 1_000_000.0,
) -> DividendOptionResult:
    """Price an option on dividends via Black-76.

    Treats dividend as a forward-settled instrument.

    Args:
        forward_dividend: expected total dividend over period.
        strike: dividend strike.
        vol: implied vol of dividend.
        T: time to expiry.
    """
    df = math.exp(-rate * T)
    otype = OptionType.CALL if option_type.lower() == "call" else OptionType.PUT

    price = black76_price(forward_dividend, strike, vol, T, df, otype) * notional
    delta = black76_delta(forward_dividend, strike, vol, T, df, otype)
    vega = black76_vega(forward_dividend, strike, vol, T, df) * 0.01 * notional

    return DividendOptionResult(
        price=price, delta=delta, vega=vega,
        forward_dividend=forward_dividend,
        strike=strike, vol=vol,
    )


@dataclass
class TotalReturnFutureResult:
    """Total return future pricing result."""
    tr_futures_price: float
    price_futures_price: float
    dividend_component: float
    financing_spread: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def total_return_future(
    spot: float,
    T: float,
    rate: float = 0.04,
    div_yield: float = 0.02,
    financing_spread: float = 0.001,
) -> TotalReturnFutureResult:
    """Total return future vs price return future.

    Price return: F_PR = S × exp((r − q) × T)
    Total return: F_TR = S × exp((r − q + q) × T) × exp(−spread × T)
                       = S × exp((r − spread) × T)

    The difference is the reinvested dividend component.

    Args:
        spot: current spot.
        T: time to expiry.
        rate: risk-free rate.
        div_yield: continuous dividend yield.
        financing_spread: TR futures financing spread.
    """
    pr_futures = spot * math.exp((rate - div_yield) * T)
    tr_futures = spot * math.exp((rate - financing_spread) * T)
    div_component = tr_futures - pr_futures

    return TotalReturnFutureResult(
        tr_futures_price=tr_futures,
        price_futures_price=pr_futures,
        dividend_component=div_component,
        financing_spread=financing_spread,
    )
