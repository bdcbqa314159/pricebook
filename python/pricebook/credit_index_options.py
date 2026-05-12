"""Credit index options: options on CDX/iTraxx spread.

* :func:`credit_index_option` — Black-76 on index spread.
* :func:`index_option_greeks` — delta, gamma, vega on index spread.

References:
    O'Kane (2008). Modelling Single-name and Multi-name Credit Derivatives, Ch. 17.
    Pedersen (2003). Valuation of Portfolio Credit Default Swaptions.
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook.black76 import (
    black76_price, black76_delta, black76_gamma, black76_vega, black76_theta,
    OptionType,
)


@dataclass
class CreditIndexOptionResult:
    price: float
    forward_spread: float
    annuity: float
    spread_vol: float
    is_payer: bool
    def to_dict(self) -> dict:
        return vars(self)


def credit_index_option(
    forward_spread: float,
    strike_spread: float,
    spread_vol: float,
    T: float,
    annuity: float,
    notional: float = 10_000_000,
    is_payer: bool = True,
) -> CreditIndexOptionResult:
    """Option on a credit index spread (CDX, iTraxx).

    Payer option: right to buy protection at strike spread.
        Payoff = annuity x notional x max(spread_T - K, 0).
    Receiver option: right to sell protection at strike spread.
        Payoff = annuity x notional x max(K - spread_T, 0).

    Priced via Black-76 on the forward spread with risky annuity as numeraire.

    Args:
        forward_spread: forward index spread.
        strike_spread: option strike spread.
        spread_vol: lognormal vol of index spread.
        T: option expiry.
        annuity: risky annuity (RPV01) of the underlying index.
    """
    opt = OptionType.CALL if is_payer else OptionType.PUT
    # Black-76: forward=spread, strike=K, vol=spread_vol, df=annuity
    unit_price = black76_price(forward_spread, strike_spread, spread_vol, T,
                                annuity, opt)
    price = unit_price * notional

    return CreditIndexOptionResult(float(price), forward_spread, annuity,
                                    spread_vol, is_payer)


@dataclass
class CreditIndexOptionGreeks:
    price: float
    delta: float      # dPV/dSpread per 1bp
    gamma: float      # d2PV/dSpread2
    vega: float       # dPV/dVol per 1%
    theta: float      # dPV/dT per day
    def to_dict(self) -> dict:
        return vars(self)


def index_option_greeks(
    forward_spread: float,
    strike_spread: float,
    spread_vol: float,
    T: float,
    annuity: float,
    notional: float = 10_000_000,
    is_payer: bool = True,
) -> CreditIndexOptionGreeks:
    """Greeks for credit index option."""
    opt = OptionType.CALL if is_payer else OptionType.PUT
    price = black76_price(forward_spread, strike_spread, spread_vol, T, annuity, opt)
    delta = black76_delta(forward_spread, strike_spread, spread_vol, T, annuity, opt)
    gamma = black76_gamma(forward_spread, strike_spread, spread_vol, T, annuity)
    vega = black76_vega(forward_spread, strike_spread, spread_vol, T, annuity)
    theta = black76_theta(forward_spread, strike_spread, spread_vol, T, annuity)

    return CreditIndexOptionGreeks(
        price=float(price * notional),
        delta=float(delta * notional * 0.0001),
        gamma=float(gamma * notional * 0.0001**2),
        vega=float(vega * notional * 0.01),
        theta=float(theta * notional / 365),
    )
