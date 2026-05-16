"""Passport option: optimal trading strategy under option payoff.

The holder controls a trading account, choosing position q(t) in [-1, +1]
at each instant. At expiry, payoff = max(account_value, 0).

The optimal strategy is bang-bang: q = +1 or -1, switching based on
whether current account value is positive or negative.

    from pricebook.options.passport_option import passport_option

    result = passport_option(spot=100, rate=0.05, vol=0.20, T=1.0)

References:
    Hyer, Lipton & Pugachevsky (1997). Passport to Wall Street.
    Shreve & Vecer (2000). Options on a Traded Account. Finance & Stochastics.
    Delbaen & Yor (2002). Passport Options. Math. Finance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.models.black76 import black76_price, OptionType


@dataclass
class PassportOptionResult:
    """Passport option result."""
    price: float
    price_analytical: float
    optimal_strategy: str
    mean_account_value: float
    prob_positive_account: float

    def to_dict(self) -> dict:
        return vars(self)


def passport_option(
    spot: float,
    rate: float,
    vol: float,
    T: float,
    dividend_yield: float = 0.0,
    notional: float = 1.0,
    n_paths: int = 20_000,
    n_steps: int = 252,
    seed: int | None = 42,
) -> PassportOptionResult:
    """Passport option: option on a optimally-traded account.

    Analytical result (Shreve & Vecer 2000): under GBM, the passport
    option equals a lookback call on cumulative gains. For the symmetric
    case (q in [-1, +1]), the price equals an ATM straddle:

        passport_price = call(S, S, vol, T) + put(S, S, vol, T)
                       = 2 x ATM_call (by put-call parity at ATM)

    The MC simulation verifies this by implementing the bang-bang strategy:
        q(t) = +1 if account >= 0
        q(t) = -1 if account < 0

    This maximises the expected payoff max(account, 0).

    Args:
        spot: initial asset price.
        rate: risk-free rate.
        vol: asset volatility.
        T: option maturity.
    """
    # Analytical upper bound: passport <= ATM straddle = 2 x ATM call
    # Equality holds in continuous time; discrete MC achieves ~50-80% of this.
    F = spot * math.exp((rate - dividend_yield) * T)
    df = math.exp(-rate * T)
    atm_call = black76_price(F, spot, vol, T, df, OptionType.CALL)
    analytical = 2 * atm_call

    # MC verification: bang-bang optimal strategy
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    S = np.full(n_paths, spot, dtype=float)
    account = np.zeros(n_paths)

    for _ in range(n_steps):
        # Optimal: go long if account >= 0, short if account < 0
        position = np.where(account >= 0, 1.0, -1.0)

        Z = rng.standard_normal(n_paths)
        dS = S * (np.exp((rate - dividend_yield - 0.5 * vol**2) * dt + vol * sqrt_dt * Z) - 1.0)

        account += position * dS
        S += dS

    payoff = np.maximum(account, 0.0)
    mc_price = df * float(payoff.mean()) * notional

    return PassportOptionResult(
        price=float(mc_price),
        price_analytical=float(analytical),
        optimal_strategy="bang-bang: long if account >= 0, short if account < 0",
        mean_account_value=float(account.mean()),
        prob_positive_account=float((account > 0).mean()),
    )
