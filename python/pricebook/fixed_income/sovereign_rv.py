"""Cross-market sovereign relative value framework.

Decomposes sovereign bond spreads into fundamental, technical, and liquidity
components, and computes Z-scores across markets for relative value.

    from pricebook.fixed_income.sovereign_rv import (
        sovereign_spread_decomposition, cross_market_rv_scores,
        SovereignRVInput,
    )

References:
    Ilmanen (2011). Expected Returns, Ch 14 (Sovereign Bonds).
    Ang & Longstaff (2013). Systemic sovereign credit risk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np


@dataclass
class SovereignRVInput:
    """Input data for a single sovereign market."""
    country_code: str           # ISO 2-letter
    currency: str               # ISO 3-letter
    spread_bp: float            # Sovereign spread vs benchmark (bp)
    debt_to_gdp: float          # % (e.g. 80.0 for 80%)
    fiscal_balance_gdp: float   # % (negative = deficit)
    cds_spread_bp: float        # 5Y CDS spread (bp)
    current_account_gdp: float  # % (negative = deficit)
    fx_vol_3m: float            # 3-month FX implied vol (%)
    reserves_months_imports: float  # FX reserves in months of imports
    rating_notch: int           # Numeric rating (1=AAA, 21=C)
    bid_ask_bp: float = 5.0     # Bid-ask spread (liquidity proxy)
    turnover_ratio: float = 1.0 # Trading volume / outstanding (liquidity)

    def to_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class SpreadDecomposition:
    """Decomposed sovereign spread."""
    country_code: str
    total_spread_bp: float
    fundamental_bp: float       # Driven by macro fundamentals
    credit_bp: float            # CDS-implied credit component
    liquidity_bp: float         # Residual liquidity premium
    technical_bp: float         # Positioning / flow effects (residual)

    def to_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class RVScore:
    """Relative value score for a sovereign market."""
    country_code: str
    currency: str
    spread_bp: float
    z_score: float              # Standard deviations from cross-section mean
    percentile: float           # Rank percentile (0-100)
    fundamental_z: float        # Z-score of fundamental component
    liquidity_z: float          # Z-score of liquidity component
    rating_notch: int
    signal: str                 # "CHEAP", "FAIR", "RICH"

    def to_dict(self) -> dict:
        return dict(vars(self))


def sovereign_spread_decomposition(
    market: SovereignRVInput,
) -> SpreadDecomposition:
    """Decompose a sovereign spread into components.

    credit_bp ≈ CDS spread (market-implied default risk)
    liquidity_bp ≈ f(bid_ask, turnover) — residual non-credit spread
    fundamental_bp ≈ regression-fitted from macro variables
    technical_bp = total - credit - liquidity - fundamental

    The fundamental component uses a simplified model:
        fundamental ≈ β₁×debt/gdp + β₂×fiscal + β₃×ca + β₄×rating

    Coefficients are illustrative (cross-section regression on EM sovereigns).
    """
    # Credit: directly from CDS
    credit = market.cds_spread_bp

    # Liquidity: proportional to bid-ask and inverse turnover
    liquidity = market.bid_ask_bp * 2.0 + max(0, (1.0 / max(market.turnover_ratio, 0.01) - 1.0)) * 5.0

    # Fundamental: simplified macro model
    fundamental = (
        0.5 * max(market.debt_to_gdp - 40, 0)        # penalty above 40% D/GDP
        + 3.0 * max(-market.fiscal_balance_gdp - 2, 0) # penalty for deficit > 2%
        + 2.0 * max(-market.current_account_gdp - 3, 0) # CA deficit penalty
        + 5.0 * max(market.rating_notch - 5, 0)        # rating penalty below AA
        + 0.5 * max(market.fx_vol_3m - 8, 0)           # FX vol penalty
        - 2.0 * max(market.reserves_months_imports - 3, 0) # reserves credit
    )
    fundamental = max(fundamental, 0.0)

    # Technical: residual
    technical = market.spread_bp - credit - liquidity - fundamental
    # Clamp negative technical to zero and re-attribute
    if technical < 0:
        fundamental = market.spread_bp - credit - liquidity
        fundamental = max(fundamental, 0.0)
        technical = market.spread_bp - credit - liquidity - fundamental

    return SpreadDecomposition(
        country_code=market.country_code,
        total_spread_bp=market.spread_bp,
        fundamental_bp=fundamental,
        credit_bp=credit,
        liquidity_bp=liquidity,
        technical_bp=technical,
    )


def cross_market_rv_scores(
    markets: list[SovereignRVInput],
) -> list[RVScore]:
    """Compute relative value Z-scores across a set of sovereign markets.

    Steps:
    1. Decompose each market's spread.
    2. Compute cross-sectional Z-score of total spread.
    3. Compute fundamental and liquidity Z-scores.
    4. Classify as CHEAP (z > 1), RICH (z < -1), or FAIR.

    Returns list of RVScore sorted by z_score (cheapest first).
    """
    if len(markets) < 2:
        raise ValueError("Need at least 2 markets for cross-section RV")

    decomps = [sovereign_spread_decomposition(m) for m in markets]

    spreads = np.array([m.spread_bp for m in markets])
    mean_s = float(np.mean(spreads))
    std_s = float(np.std(spreads, ddof=1))
    if std_s < 1e-10:
        std_s = 1.0

    fund_vals = np.array([d.fundamental_bp for d in decomps])
    mean_f = float(np.mean(fund_vals))
    std_f = float(np.std(fund_vals, ddof=1)) or 1.0

    liq_vals = np.array([d.liquidity_bp for d in decomps])
    mean_l = float(np.mean(liq_vals))
    std_l = float(np.std(liq_vals, ddof=1)) or 1.0

    # Rank for percentiles
    sorted_spreads = np.sort(spreads)

    scores = []
    for i, m in enumerate(markets):
        z = (m.spread_bp - mean_s) / std_s
        z_f = (decomps[i].fundamental_bp - mean_f) / std_f
        z_l = (decomps[i].liquidity_bp - mean_l) / std_l

        # Percentile
        rank = float(np.searchsorted(sorted_spreads, m.spread_bp, side="right"))
        pct = rank / len(markets) * 100.0

        # Signal
        if z > 1.0:
            signal = "CHEAP"
        elif z < -1.0:
            signal = "RICH"
        else:
            signal = "FAIR"

        scores.append(RVScore(
            country_code=m.country_code,
            currency=m.currency,
            spread_bp=m.spread_bp,
            z_score=z,
            percentile=pct,
            fundamental_z=z_f,
            liquidity_z=z_l,
            rating_notch=m.rating_notch,
            signal=signal,
        ))

    scores.sort(key=lambda s: -s.z_score)  # cheapest first
    return scores
