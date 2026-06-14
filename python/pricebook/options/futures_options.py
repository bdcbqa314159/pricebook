"""Futures options: unified product for options on any futures contract.

Wraps Black-76 / Bachelier with contract specifications, daily settlement
mechanics, and full Greeks. Covers equity index, commodity, IR, and bond
futures options.

* :class:`FuturesOptionSpec` — contract specification.
* :class:`FuturesOption` — option on a futures contract.
* :class:`FuturesOptionResult` — pricing result with Greeks.
* :func:`futures_option_strip` — price a strip of options across expiries.
* :func:`futures_option_vol_surface` — build vol surface from option prices.

References:
    Black, *The Pricing of Commodity Contracts*, JFE, 1976.
    Hull, *Options, Futures, and Other Derivatives*, 11th ed., Ch. 18.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum

import numpy as np

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.models.black76 import (
    OptionType, black76_price, black76_delta, black76_gamma,
    black76_vega, black76_theta, bachelier_price,
    bachelier_delta, bachelier_gamma, bachelier_vega, bachelier_theta,
    _norm_cdf,
)


class FuturesAsset(Enum):
    """Futures asset class."""
    EQUITY_INDEX = "equity_index"
    COMMODITY = "commodity"
    INTEREST_RATE = "interest_rate"
    BOND = "bond"
    FX = "fx"


class ExerciseStyle(Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


class SettlementMethod(Enum):
    CASH = "cash"
    PHYSICAL = "physical"  # delivers futures position


@dataclass
class FuturesOptionSpec:
    """Contract specification for a futures option.

    Attributes:
        ticker: contract identifier (e.g. "ES", "CL", "ZN").
        asset_class: equity/commodity/IR/bond/FX.
        multiplier: dollar value per point.
        tick_size: minimum price increment.
        tick_value: dollar value per tick.
        exercise: European or American.
        settlement: cash or physical delivery of futures.
    """
    ticker: str
    asset_class: FuturesAsset
    multiplier: float
    tick_size: float
    tick_value: float
    exercise: ExerciseStyle = ExerciseStyle.AMERICAN
    settlement: SettlementMethod = SettlementMethod.PHYSICAL

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "asset_class": self.asset_class.value,
            "multiplier": self.multiplier,
            "tick_size": self.tick_size,
            "exercise": self.exercise.value,
        }


# Standard contract specs
EQUITY_INDEX_SPECS = {
    "ES": FuturesOptionSpec("ES", FuturesAsset.EQUITY_INDEX, 50.0, 0.25, 12.50,
                            ExerciseStyle.AMERICAN, SettlementMethod.CASH),
    "NQ": FuturesOptionSpec("NQ", FuturesAsset.EQUITY_INDEX, 20.0, 0.25, 5.00,
                            ExerciseStyle.AMERICAN, SettlementMethod.CASH),
    "YM": FuturesOptionSpec("YM", FuturesAsset.EQUITY_INDEX, 5.0, 1.0, 5.00,
                            ExerciseStyle.AMERICAN, SettlementMethod.CASH),
    "RTY": FuturesOptionSpec("RTY", FuturesAsset.EQUITY_INDEX, 50.0, 0.10, 5.00,
                             ExerciseStyle.AMERICAN, SettlementMethod.CASH),
}

COMMODITY_SPECS = {
    "CL": FuturesOptionSpec("CL", FuturesAsset.COMMODITY, 1000.0, 0.01, 10.00,
                            ExerciseStyle.AMERICAN, SettlementMethod.PHYSICAL),
    "GC": FuturesOptionSpec("GC", FuturesAsset.COMMODITY, 100.0, 0.10, 10.00,
                            ExerciseStyle.AMERICAN, SettlementMethod.PHYSICAL),
    "SI": FuturesOptionSpec("SI", FuturesAsset.COMMODITY, 5000.0, 0.005, 25.00,
                            ExerciseStyle.AMERICAN, SettlementMethod.PHYSICAL),
    "NG": FuturesOptionSpec("NG", FuturesAsset.COMMODITY, 10000.0, 0.001, 10.00,
                            ExerciseStyle.AMERICAN, SettlementMethod.PHYSICAL),
    "ZC": FuturesOptionSpec("ZC", FuturesAsset.COMMODITY, 50.0, 0.125, 6.25,
                            ExerciseStyle.AMERICAN, SettlementMethod.PHYSICAL),
    "ZW": FuturesOptionSpec("ZW", FuturesAsset.COMMODITY, 50.0, 0.125, 6.25,
                            ExerciseStyle.AMERICAN, SettlementMethod.PHYSICAL),
    "ZS": FuturesOptionSpec("ZS", FuturesAsset.COMMODITY, 50.0, 0.125, 6.25,
                            ExerciseStyle.AMERICAN, SettlementMethod.PHYSICAL),
}

IR_SPECS = {
    "ZN": FuturesOptionSpec("ZN", FuturesAsset.BOND, 1000.0, 1/64, 15.625,
                            ExerciseStyle.AMERICAN, SettlementMethod.PHYSICAL),
    "ZB": FuturesOptionSpec("ZB", FuturesAsset.BOND, 1000.0, 1/64, 15.625,
                            ExerciseStyle.AMERICAN, SettlementMethod.PHYSICAL),
    "ZF": FuturesOptionSpec("ZF", FuturesAsset.BOND, 1000.0, 1/128, 7.8125,
                            ExerciseStyle.AMERICAN, SettlementMethod.PHYSICAL),
    "SR3": FuturesOptionSpec("SR3", FuturesAsset.INTEREST_RATE, 2500.0, 0.0025, 6.25,
                             ExerciseStyle.AMERICAN, SettlementMethod.CASH),
}

ALL_SPECS = {**EQUITY_INDEX_SPECS, **COMMODITY_SPECS, **IR_SPECS}


def get_spec(ticker: str) -> FuturesOptionSpec:
    """Look up contract spec by ticker."""
    t = ticker.upper()
    if t in ALL_SPECS:
        return ALL_SPECS[t]
    raise ValueError(f"Unknown futures ticker: {ticker}")


# ---- Pricing result ----

@dataclass
class FuturesOptionResult:
    """Futures option pricing result with full Greeks."""
    price: float            # option premium per unit
    price_total: float      # premium × multiplier × n_contracts
    delta: float            # ∂V/∂F
    gamma: float            # ∂²V/∂F²
    vega: float             # ∂V/∂σ (per 1% vol)
    theta: float            # ∂V/∂t (per day)
    # Derived
    delta_dollars: float    # delta × multiplier × n_contracts
    gamma_dollars: float    # gamma × multiplier × n_contracts
    vega_dollars: float     # vega per 1% × multiplier × n_contracts
    theta_dollars: float    # theta per day × multiplier × n_contracts
    # Inputs
    futures_price: float
    strike: float
    vol: float
    expiry_years: float
    option_type: str
    model: str

    def to_dict(self) -> dict:
        return vars(self)


# ---- Futures option product ----

class FuturesOption:
    """Option on a futures contract.

    Prices via Black-76 (lognormal) or Bachelier (normal).
    Supports American exercise via Barone-Adesi-Whaley approximation.

    Args:
        spec: contract specification (or ticker string).
        futures_price: current futures price.
        strike: option strike price.
        expiry: option expiry date.
        vol: implied volatility (lognormal for Black-76, normal for Bachelier).
        option_type: "call" or "put".
        n_contracts: number of contracts.
        model: "black76" or "bachelier".
    """

    def __init__(
        self,
        spec: FuturesOptionSpec | str,
        futures_price: float,
        strike: float,
        expiry: date,
        vol: float,
        option_type: str = "call",
        n_contracts: int = 1,
        model: str = "black76",
    ):
        self.spec = get_spec(spec) if isinstance(spec, str) else spec
        self.futures_price = futures_price
        self.strike = strike
        self.expiry = expiry
        self.vol = vol
        self.option_type = OptionType.CALL if option_type.lower() == "call" else OptionType.PUT
        self.n_contracts = n_contracts
        self.model = model

    def price(
        self,
        valuation_date: date,
        rate: float = 0.04,
    ) -> FuturesOptionResult:
        """Price the futures option.

        Args:
            valuation_date: pricing date.
            rate: risk-free rate for discounting premium.
        """
        T = year_fraction(valuation_date, self.expiry, DayCountConvention.ACT_365_FIXED)
        df = math.exp(-rate * max(T, 0))
        F = self.futures_price
        K = self.strike
        sigma = self.vol
        mult = self.spec.multiplier
        nc = self.n_contracts

        # American exercise adjustment
        if self.spec.exercise == ExerciseStyle.AMERICAN and T > 0 and self.model == "black76":
            premium = _baw_futures_option(F, K, sigma, T, rate, self.option_type)
        elif self.model == "bachelier":
            premium = bachelier_price(F, K, sigma, T, df, self.option_type)
        else:
            premium = black76_price(F, K, sigma, T, df, self.option_type)

        # Greeks: pick analytical family matching the pricing model.  Pre-fix
        # the Bachelier branch silently fell through to Black-76 Greeks, which
        # disagree numerically (lognormal vs normal d1/d2) — wrong delta/gamma
        # for products that *must* be priced under a normal model (e.g.
        # short-rate futures where rates can go negative).
        if self.model == "bachelier":
            delta = bachelier_delta(F, K, sigma, T, df, self.option_type)
            gamma = bachelier_gamma(F, K, sigma, T, df)
            vega = bachelier_vega(F, K, sigma, T, df) * 0.01
            theta = bachelier_theta(F, K, sigma, T, df, self.option_type) / 365.0
        else:
            delta = black76_delta(F, K, sigma, T, df, self.option_type)
            gamma = black76_gamma(F, K, sigma, T, df)
            vega = black76_vega(F, K, sigma, T, df) * 0.01
            theta = black76_theta(F, K, sigma, T, df, self.option_type) / 365.0

        return FuturesOptionResult(
            price=premium,
            price_total=premium * mult * nc,
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            delta_dollars=delta * mult * nc,
            gamma_dollars=gamma * mult * nc,
            vega_dollars=vega * mult * nc,
            theta_dollars=theta * mult * nc,
            futures_price=F,
            strike=K,
            vol=sigma,
            expiry_years=T,
            option_type=self.option_type.value,
            model=self.model,
        )

    def intrinsic(self) -> float:
        """Intrinsic value."""
        if self.option_type == OptionType.CALL:
            return max(self.futures_price - self.strike, 0)
        return max(self.strike - self.futures_price, 0)

    def moneyness(self) -> float:
        """Moneyness: F/K for calls, K/F for puts."""
        if self.option_type == OptionType.CALL:
            return self.futures_price / self.strike if self.strike > 0 else 0
        return self.strike / self.futures_price if self.futures_price > 0 else 0

    def to_dict(self) -> dict:
        return {
            "ticker": self.spec.ticker,
            "futures_price": self.futures_price,
            "strike": self.strike,
            "expiry": self.expiry.isoformat(),
            "vol": self.vol,
            "option_type": self.option_type.value,
            "n_contracts": self.n_contracts,
        }


# ---- Strip and surface ----

def futures_option_strip(
    spec: FuturesOptionSpec | str,
    futures_prices: list[float],
    strike: float,
    expiry_dates: list[date],
    vols: list[float],
    valuation_date: date,
    option_type: str = "call",
    rate: float = 0.04,
) -> list[FuturesOptionResult]:
    """Price a strip of futures options across expiry dates.

    Args:
        futures_prices: futures price per expiry.
        strike: common strike.
        expiry_dates: option expiry dates.
        vols: implied vol per expiry.
        valuation_date: pricing date.
    """
    results = []
    for fp, exp, vol in zip(futures_prices, expiry_dates, vols):
        opt = FuturesOption(spec, fp, strike, exp, vol, option_type)
        results.append(opt.price(valuation_date, rate))
    return results


@dataclass
class FuturesVolPoint:
    """Single point on a futures vol surface."""
    expiry_years: float
    strike: float
    vol: float
    option_type: str


def futures_option_vol_surface(
    futures_price: float,
    strikes: list[float],
    expiry_years: list[float],
    vols: list[list[float]],
) -> list[list[FuturesVolPoint]]:
    """Build a vol surface grid from observed vols.

    Args:
        futures_price: current futures price.
        strikes: strike levels.
        expiry_years: expiry tenors.
        vols: 2D grid vols[expiry_idx][strike_idx].

    Returns:
        2D grid of FuturesVolPoint.
    """
    surface = []
    for i, T in enumerate(expiry_years):
        row = []
        for j, K in enumerate(strikes):
            otype = "call" if K >= futures_price else "put"
            row.append(FuturesVolPoint(T, K, vols[i][j], otype))
        surface.append(row)
    return surface


def interpolate_vol(
    surface: list[list[FuturesVolPoint]],
    target_expiry: float,
    target_strike: float,
) -> float:
    """Bilinear interpolation on a futures vol surface."""
    expiries = [row[0].expiry_years for row in surface]
    strikes = [p.strike for p in surface[0]]

    # Find bracketing expiry
    ei = 0
    for i in range(len(expiries) - 1):
        if expiries[i] <= target_expiry <= expiries[i + 1]:
            ei = i
            break
    else:
        ei = len(expiries) - 2 if target_expiry > expiries[-1] else 0

    # Find bracketing strike
    si = 0
    for j in range(len(strikes) - 1):
        if strikes[j] <= target_strike <= strikes[j + 1]:
            si = j
            break
    else:
        si = len(strikes) - 2 if target_strike > strikes[-1] else 0

    e0, e1 = expiries[ei], expiries[min(ei + 1, len(expiries) - 1)]
    s0, s1 = strikes[si], strikes[min(si + 1, len(strikes) - 1)]

    we = (target_expiry - e0) / (e1 - e0) if e1 > e0 else 0.0
    ws = (target_strike - s0) / (s1 - s0) if s1 > s0 else 0.0

    v00 = surface[ei][si].vol
    v01 = surface[ei][min(si + 1, len(strikes) - 1)].vol
    v10 = surface[min(ei + 1, len(expiries) - 1)][si].vol
    v11 = surface[min(ei + 1, len(expiries) - 1)][min(si + 1, len(strikes) - 1)].vol

    return (1 - we) * ((1 - ws) * v00 + ws * v01) + we * ((1 - ws) * v10 + ws * v11)


# ---- Barone-Adesi-Whaley for American futures options ----

def _baw_futures_option(
    F: float,
    K: float,
    sigma: float,
    T: float,
    r: float,
    option_type: OptionType,
) -> float:
    """Barone-Adesi-Whaley approximation for American options on futures.

    For futures (cost-of-carry b=0), the approximation simplifies.
    American futures call = European call (no early exercise premium
    when b=0 and r>0). American put has early exercise value.

    Reference: Barone-Adesi & Whaley (1987), JF.
    """
    df = math.exp(-r * T)
    euro = black76_price(F, K, sigma, T, df, option_type)

    if T <= 0 or sigma <= 0:
        return euro

    # For futures with b=0: American call = European call when r >= 0
    if option_type == OptionType.CALL and r >= 0:
        return euro

    # American put (or call with r < 0): compute early exercise premium
    M = 2 * r / (sigma * sigma)
    q2 = (-(1) + math.sqrt(1 + 4 * M / (1 - df))) / 2 if df < 1 else 0

    if option_type == OptionType.PUT:
        if q2 >= 0:
            return euro  # no early exercise benefit

        # Find critical futures price S* where exercise is optimal
        S_star = _baw_critical_price_put(F, K, sigma, T, r, q2)
        if F <= S_star:
            return K - F  # exercise immediately

        A2 = -(S_star / q2) * (1 - df * _norm_cdf(-_d1(S_star, K, sigma, T)))
        return euro + A2 * (F / S_star) ** q2
    else:
        # American call with r < 0
        return euro  # rare case, European approximation sufficient


def _d1(F: float, K: float, sigma: float, T: float) -> float:
    if F <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return 0.0
    return (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))


def _baw_critical_price_put(
    F: float, K: float, sigma: float, T: float, r: float, q2: float,
    tol: float = 1e-6, max_iter: int = 100,
) -> float:
    """Newton-Raphson for BAW critical put price."""
    df = math.exp(-r * T)
    S = K * 0.8  # initial guess below strike
    for _ in range(max_iter):
        d1 = _d1(S, K, sigma, T)
        euro_put = black76_price(S, K, sigma, T, df, OptionType.PUT)
        lhs = K - S - euro_put
        rhs = -(S / q2) * (1 - df * _norm_cdf(-d1))

        diff = lhs - rhs
        if abs(diff) < tol:
            break

        # Derivative (simplified)
        d_lhs = -1 + df * _norm_cdf(-d1)
        d_rhs = -(1 / q2) * (1 - df * _norm_cdf(-d1))
        deriv = d_lhs - d_rhs
        if abs(deriv) < 1e-15:
            break
        S -= diff / deriv
        S = max(S, K * 0.01)
        S = min(S, K * 2.0)

    return S
