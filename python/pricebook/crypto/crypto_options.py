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


# ═══════════════════════════════════════════════════════════════
# CD1: Move Contracts, Variance Swaps, Greeks P&L, Portfolio
# ═══════════════════════════════════════════════════════════════

@dataclass
class MoveContractResult:
    """Move contract (absolute value option) result."""
    price: float
    delta: float
    gamma: float
    breakeven_move_pct: float

    def to_dict(self) -> dict:
        return vars(self)


def move_contract(
    spot: float,
    vol: float,
    T: float,
    rate: float = 0.0,
) -> MoveContractResult:
    """Move contract: pays |S_T − S_0|.

    Equivalent to a straddle at spot (ATM call + ATM put).
    Used on Deribit as $MOVE.

    price ≈ spot × vol × √(2T/π) (for ATM, approximate).
    """
    fwd = spot * math.exp(rate * T)
    df = math.exp(-rate * T)
    otype_c = OptionType.CALL
    otype_p = OptionType.PUT

    call = black76_price(fwd, spot, vol, T, df, otype_c)
    put = black76_price(fwd, spot, vol, T, df, otype_p)
    price = call + put

    delta_c = black76_delta(fwd, spot, vol, T, df, otype_c)
    delta_p = black76_delta(fwd, spot, vol, T, df, otype_p)
    delta = delta_c + delta_p  # near zero for ATM

    gamma = black76_gamma(fwd, spot, vol, T, df) * 2

    breakeven = price / spot * 100 if spot > 0 else 0

    return MoveContractResult(price=price, delta=delta, gamma=gamma,
                               breakeven_move_pct=breakeven)


@dataclass
class CryptoVarianceSwapResult:
    """Crypto variance/vol swap result."""
    fair_variance: float
    fair_vol: float
    vega_notional: float
    pv: float

    def to_dict(self) -> dict:
        return vars(self)


def crypto_variance_swap(
    atm_vol: float,
    skew_adjustment: float = 0.0,
    T: float = 30 / 365,
    vega_notional: float = 10_000.0,
    realised_vol: float | None = None,
) -> CryptoVarianceSwapResult:
    """Crypto variance swap pricing.

    fair_var ≈ ATM² + skew_adjustment (from OTM puts).
    For crypto: skew is steep (puts expensive), so fair var > ATM².

    Args:
        atm_vol: ATM implied vol.
        skew_adjustment: convexity from smile (~0.02-0.05 for crypto).
        T: expiry in years.
        vega_notional: notional per vol point.
        realised_vol: if given, compute MTM PV.
    """
    fair_var = atm_vol**2 + skew_adjustment
    fair_vol = math.sqrt(max(fair_var, 0))

    if realised_vol is not None:
        pv = vega_notional / (2 * fair_vol) * (realised_vol**2 - fair_var) if fair_vol > 0 else 0
    else:
        pv = 0.0

    return CryptoVarianceSwapResult(fair_var, fair_vol * 100, vega_notional, pv)


@dataclass
class GreeksPnLExplain:
    """Greeks-based P&L attribution for crypto options."""
    delta_pnl: float        # from spot move
    gamma_pnl: float        # from convexity (realized vol)
    vega_pnl: float         # from vol move
    theta_pnl: float        # time decay
    unexplained: float
    total_pnl: float

    def to_dict(self) -> dict:
        return vars(self)


def greeks_pnl_explain(
    delta: float,
    gamma: float,
    vega: float,
    theta: float,
    spot_move: float,
    vol_move: float,
    dt_days: float = 1.0,
    actual_pnl: float | None = None,
) -> GreeksPnLExplain:
    """Greeks-based P&L decomposition for crypto options.

    delta_pnl = delta × ΔS
    gamma_pnl = ½ × gamma × ΔS²
    vega_pnl = vega × Δσ
    theta_pnl = theta × Δt

    Args:
        delta: position delta.
        gamma: position gamma.
        vega: position vega (per 1% vol).
        theta: position theta (per day).
        spot_move: spot price change (absolute).
        vol_move: vol change (absolute, e.g. 0.02 for 2%).
        dt_days: number of days.
        actual_pnl: if given, compute unexplained.
    """
    d_pnl = delta * spot_move
    g_pnl = 0.5 * gamma * spot_move**2
    v_pnl = vega * vol_move * 100  # vega is per 1%, vol_move is decimal
    t_pnl = theta * dt_days

    explained = d_pnl + g_pnl + v_pnl + t_pnl
    unexplained = (actual_pnl - explained) if actual_pnl is not None else 0

    return GreeksPnLExplain(d_pnl, g_pnl, v_pnl, t_pnl, unexplained,
                             actual_pnl if actual_pnl is not None else explained)


@dataclass
class PortfolioGreeks:
    """Aggregated Greeks for a crypto options portfolio."""
    net_delta: float
    net_gamma: float
    net_vega: float
    net_theta: float
    n_positions: int
    delta_btc: float        # delta in BTC terms
    gamma_usd_1pct: float   # gamma P&L for 1% spot move

    def to_dict(self) -> dict:
        return vars(self)


def aggregate_greeks(
    positions: list[dict],
    spot: float,
) -> PortfolioGreeks:
    """Aggregate Greeks across a crypto options portfolio.

    Each position dict: {"delta", "gamma", "vega", "theta", "quantity"}.

    Args:
        positions: list of position dicts with Greeks.
        spot: current spot for dollar conversion.
    """
    d = sum(p.get("delta", 0) * p.get("quantity", 1) for p in positions)
    g = sum(p.get("gamma", 0) * p.get("quantity", 1) for p in positions)
    v = sum(p.get("vega", 0) * p.get("quantity", 1) for p in positions)
    t = sum(p.get("theta", 0) * p.get("quantity", 1) for p in positions)

    delta_btc = d  # already in crypto units for inverse
    gamma_1pct = 0.5 * g * (spot * 0.01)**2  # P&L from 1% move

    return PortfolioGreeks(d, g, v, t, len(positions), delta_btc, gamma_1pct)
