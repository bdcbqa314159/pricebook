"""Crypto options: Deribit-style with inverse contract support.

* :func:`crypto_option_price` — price crypto option (linear or inverse).
* :func:`inverse_greeks` — Greeks in both BTC and USD terms.
* :func:`dvol_index` — DVOL-style implied vol index.

References:
    Deribit, *Options Contract Specifications*.
    Clark, *FX Option Pricing*, Ch. 2 (inverse contracts analogy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.models.black76 import (
    OptionType, black76_price, black76_delta, black76_gamma,
    black76_vega, black76_theta,
)


@dataclass
class CryptoOptionResult:
    """Crypto option pricing result."""
    price_usd: float            # premium in USD
    price_crypto: float         # premium in underlying crypto
    delta_usd: float
    delta_crypto: float
    gamma: float
    vega: float                 # per 1% vol
    theta: float                # per day
    iv: float
    contract_type: str          # "linear" or "inverse"

    def to_dict(self) -> dict:
        return vars(self)


def crypto_option_price(
    spot: float,
    strike: float,
    vol: float,
    T: float,
    rate: float = 0.0,
    option_type: str = "call",
    contract_type: str = "linear",
) -> CryptoOptionResult:
    """Price a crypto option (Deribit-style).

    Linear: standard BS, premium in USD.
    Inverse: premium in crypto. 1 BTC of notional, settled in BTC.

    For inverse contracts:
    price_btc = BS_price_usd / spot
    delta_btc = delta_usd − price_btc  (quanto adjustment)

    Args:
        spot: crypto spot price in USD.
        strike: strike in USD.
        vol: implied volatility.
        T: time to expiry (years).
        rate: risk-free rate (often 0 for crypto).
        option_type: "call" or "put".
        contract_type: "linear" (USD) or "inverse" (crypto settled).
    """
    fwd = spot * math.exp(rate * T)
    df = math.exp(-rate * T)
    otype = OptionType.CALL if option_type == "call" else OptionType.PUT

    # BS price in USD
    price_usd = black76_price(fwd, strike, vol, T, df, otype)
    delta_usd = black76_delta(fwd, strike, vol, T, df, otype)
    gamma = black76_gamma(fwd, strike, vol, T, df)
    vega = black76_vega(fwd, strike, vol, T, df) * 0.01
    theta = black76_theta(fwd, strike, vol, T, df, otype) / 365.0

    if contract_type == "inverse":
        # Inverse: premium in crypto
        price_crypto = price_usd / spot if spot > 0 else 0
        # Inverse delta: adjusted for settlement in crypto
        delta_crypto = delta_usd - price_crypto
    else:
        price_crypto = price_usd / spot if spot > 0 else 0
        delta_crypto = delta_usd

    return CryptoOptionResult(
        price_usd=price_usd,
        price_crypto=price_crypto,
        delta_usd=delta_usd,
        delta_crypto=delta_crypto,
        gamma=gamma,
        vega=vega,
        theta=theta,
        iv=vol,
        contract_type=contract_type,
    )


def dvol_index(
    atm_vols: list[float],
    weights: list[float] | None = None,
) -> float:
    """DVOL-style implied volatility index.

    Weighted average of ATM vols across expirations,
    annualised. Similar to VIX but for crypto.

    DVOL = √(Σ w_i × σ_i² × T_i / Σ w_i × T_i) × 100

    Simplified: weighted RMS of ATM vols.

    Args:
        atm_vols: ATM implied vols per expiry (decimal).
        weights: optional weights (default: equal).
    """
    if not atm_vols:
        return 0.0
    w = weights or [1.0] * len(atm_vols)
    total_w = sum(w)
    if total_w <= 0:
        return 0.0
    weighted_var = sum(wi * vi**2 for wi, vi in zip(w, atm_vols)) / total_w
    return math.sqrt(weighted_var) * 100


def put_call_parity_inverse(
    call_btc: float,
    put_btc: float,
    spot: float,
    strike: float,
    T: float,
    rate: float = 0.0,
) -> float:
    """Check put-call parity for inverse contracts.

    Inverse parity: C_btc − P_btc = e^{-rT}/spot − e^{-rT}/strike

    Returns the parity violation (should be ~0).
    """
    df = math.exp(-rate * T)
    lhs = call_btc - put_btc
    rhs = df / spot - df / strike if spot > 0 and strike > 0 else 0
    return lhs - rhs
