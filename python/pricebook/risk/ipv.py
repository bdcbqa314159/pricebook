"""Independent Price Verification (IPV): automated fair value hierarchy,
matrix pricing integration, and prudent valuation workflow.

    from pricebook.risk.ipv import ipv_single_trade, ipv_portfolio, IPVResult

References:
    BCBS 287 (2014). Supervisory guidance for managing risks associated with
    the settlement of foreign exchange transactions.
    EBA/RTS/2017/01. Regulatory Technical Standards on prudent valuation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

from pricebook.risk.prudent_valuation import (
    market_price_uncertainty_ava,
    close_out_cost_ava,
    model_risk_ava,
    concentration_ava,
    investing_funding_ava,
    future_admin_cost_ava,
    compute_prudent_value,
)


# ═══════════════════════════════════════════════════════════════
# Fair Value Hierarchy
# ═══════════════════════════════════════════════════════════════

class FairValueLevel(Enum):
    LEVEL_1 = "level_1"   # observable market price
    LEVEL_2 = "level_2"   # comparable / matrix-priced
    LEVEL_3 = "level_3"   # model-only


# BCBS 287 / EBA indicative bid-ask widths by asset class (bp)
BCBS287_BID_ASK: dict[str, dict] = {
    "govt_bond_aaa":   {"normal_bp": 1,  "stressed_bp": 5,   "quality": "high"},
    "govt_bond_other": {"normal_bp": 3,  "stressed_bp": 15,  "quality": "high"},
    "corp_bond_ig":    {"normal_bp": 10, "stressed_bp": 50,  "quality": "medium"},
    "corp_bond_hy":    {"normal_bp": 30, "stressed_bp": 150, "quality": "low"},
    "irs":             {"normal_bp": 0.5,"stressed_bp": 3,   "quality": "high"},
    "cds_ig":          {"normal_bp": 3,  "stressed_bp": 20,  "quality": "medium"},
    "cds_hy":          {"normal_bp": 10, "stressed_bp": 60,  "quality": "low"},
    "fx_g10":          {"normal_bp": 1,  "stressed_bp": 5,   "quality": "high"},
    "fx_em":           {"normal_bp": 10, "stressed_bp": 50,  "quality": "medium"},
    "equity_large":    {"normal_bp": 5,  "stressed_bp": 20,  "quality": "high"},
    "equity_small":    {"normal_bp": 20, "stressed_bp": 80,  "quality": "medium"},
    "commodity_liquid":{"normal_bp": 5,  "stressed_bp": 30,  "quality": "medium"},
    "commodity_illiq": {"normal_bp": 30, "stressed_bp": 100, "quality": "low"},
    "structured":      {"normal_bp": 50, "stressed_bp": 200, "quality": "low"},
    "cln":             {"normal_bp": 30, "stressed_bp": 100, "quality": "low"},
}


# ═══════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════

@dataclass
class IPVResult:
    """Per-trade IPV result."""
    trade_id: str
    instrument_type: str
    fair_value_level: FairValueLevel
    model_price: float
    market_price: float | None
    matrix_price: float | None
    ipv_price: float
    ipv_source: str
    bid: float
    ask: float
    mid: float
    prudent_value: float
    total_ava: float
    ava_breakdown: dict
    variance_to_model_bp: float
    threshold_breach: bool

    def to_dict(self) -> dict:
        d = vars(self).copy()
        d["fair_value_level"] = self.fair_value_level.value
        return d


@dataclass
class IPVReport:
    """Portfolio-level IPV report."""
    reference_date: date
    n_trades: int
    results: list[IPVResult]
    total_model_pv: float
    total_ipv_pv: float
    total_ava: float
    total_prudent_value: float
    level_summary: dict[str, int]
    breach_count: int

    def to_dict(self) -> dict:
        return {
            "reference_date": self.reference_date.isoformat(),
            "n_trades": self.n_trades,
            "total_model_pv": self.total_model_pv,
            "total_ipv_pv": self.total_ipv_pv,
            "total_ava": self.total_ava,
            "total_prudent_value": self.total_prudent_value,
            "level_summary": self.level_summary,
            "breach_count": self.breach_count,
            "results": [r.to_dict() for r in self.results],
        }


# ═══════════════════════════════════════════════════════════════
# Core Functions
# ═══════════════════════════════════════════════════════════════

def classify_fair_value_level(
    has_market_price: bool,
    has_comparables: bool,
    n_quotes: int,
) -> FairValueLevel:
    """Classify into IFRS 13 / BCBS fair value hierarchy."""
    if has_market_price and n_quotes >= 2:
        return FairValueLevel.LEVEL_1
    if has_comparables or (has_market_price and n_quotes == 1):
        return FairValueLevel.LEVEL_2
    return FairValueLevel.LEVEL_3


def ipv_single_trade(
    trade_id: str,
    instrument_type: str,
    model_price: float,
    notional: float,
    asset_class: str = "corp_bond_ig",
    *,
    market_price: float | None = None,
    market_bid: float | None = None,
    market_ask: float | None = None,
    n_quotes: int = 0,
    matrix_price: float | None = None,
    model_prices: list[float] | None = None,
    illiquidity_premium_bp: float = 0.0,
    maturity_years: float = 5.0,
    daily_volume: float = 0.0,
    position_days: float = 1.0,
    cva: float = 0.0,
    collateralised: bool = True,
    complexity_score: int = 2,
    variance_threshold_bp: float = 50.0,
    stressed: bool = False,
    direction: int = 1,
) -> IPVResult:
    """Run IPV for a single trade.

    Price hierarchy: Level 1 (market) > Level 2 (matrix) > Level 3 (model).
    Computes all applicable AVAs using existing prudent_valuation functions.

    Args:
        direction: +1 for long position, −1 for short.  Drives the
            sign of AVA application to mid (long → prudent below mid;
            short → prudent above mid, since the prudent liability
            value is conservatively *larger* than the mid quote).
            Default +1 preserves pre-fix long-only behaviour.

    Fix T4-RISK13: pre-fix had no concept of position direction.
    ``prudent_value = mid - ava`` was hardcoded long.  Short positions
    were silently mispriced — the prudent value of a liability is
    *above* mid, not below.  Also, any negative-notional input
    propagated through to the AVA functions as negative-magnitude
    AVAs (since e.g. close_out_cost_ava multiplies by notional).
    Now: take ``abs(notional)`` for AVA size, apply ``direction`` for
    AVA-vs-mid sign.
    """
    if direction not in (-1, +1):
        raise ValueError(f"direction must be +1 (long) or -1 (short); got {direction}")
    abs_notional = abs(notional)
    has_market = market_price is not None
    has_matrix = matrix_price is not None
    level = classify_fair_value_level(has_market, has_matrix, n_quotes)

    # Select IPV price
    if level == FairValueLevel.LEVEL_1:
        ipv_price = market_price
        ipv_source = "market"
    elif level == FairValueLevel.LEVEL_2:
        ipv_price = matrix_price if has_matrix else market_price
        ipv_source = "matrix" if has_matrix else "market"
    else:
        ipv_price = model_price
        ipv_source = "model"

    # Bid/ask: from market or BCBS table
    table = BCBS287_BID_ASK.get(asset_class, BCBS287_BID_ASK["structured"])
    spread_key = "stressed_bp" if stressed else "normal_bp"

    if market_bid is not None and market_ask is not None:
        bid = market_bid
        ask = market_ask
    else:
        half_spread = ipv_price * table[spread_key] / 20_000
        bid = ipv_price - half_spread
        ask = ipv_price + half_spread

    mid = (bid + ask) / 2

    # AVA computation using existing prudent_valuation functions
    ava = {}

    # 1. Market price uncertainty
    mpu = market_price_uncertainty_ava(mid, bid, ask, max(n_quotes, 1))
    ava["market_price_uncertainty"] = mpu.ava

    # 2. Close-out cost
    coc = close_out_cost_ava(abs_notional, asset_class, daily_volume, position_days)
    ava["close_out_cost"] = coc.ava

    # 3. Model risk (if multiple model prices available)
    if model_prices and len(model_prices) >= 2:
        mr = model_risk_ava(model_prices)
        ava["model_risk"] = mr.ava
    else:
        ava["model_risk"] = 0.0

    # 4. Concentration
    if daily_volume > 0 and abs_notional > 0:
        conc = concentration_ava(abs_notional, daily_volume * 252, mid)
        ava["concentration"] = conc.ava
    else:
        ava["concentration"] = 0.0

    # 5. Investing & funding (illiquidity)
    if illiquidity_premium_bp > 0:
        ifava = investing_funding_ava(abs_notional, illiquidity_premium_bp, maturity_years)
        ava["investing_funding"] = ifava.ava
    else:
        ava["investing_funding"] = 0.0

    # 6. Future admin cost
    fac = future_admin_cost_ava(abs_notional, complexity_score, maturity_years)
    ava["future_admin_cost"] = fac.ava

    total_ava = sum(ava.values())

    # Diversification benefit (EBA simplified: 50%)
    diversified_ava = total_ava * 0.50
    # Long → prudent below mid; short → prudent above mid (liability).
    prudent_value = mid - direction * diversified_ava

    # Variance check — uses absolute notional so short positions don't
    # silently produce negative variance_bp (which would never breach).
    if abs_notional > 0:
        variance_bp = abs(ipv_price - model_price) / abs_notional * 10_000
    else:
        variance_bp = 0.0

    return IPVResult(
        trade_id=trade_id,
        instrument_type=instrument_type,
        fair_value_level=level,
        model_price=model_price,
        market_price=market_price,
        matrix_price=matrix_price,
        ipv_price=ipv_price,
        ipv_source=ipv_source,
        bid=bid, ask=ask, mid=mid,
        prudent_value=prudent_value,
        total_ava=diversified_ava,
        ava_breakdown=ava,
        variance_to_model_bp=variance_bp,
        threshold_breach=variance_bp > variance_threshold_bp,
    )


def ipv_portfolio(
    trades: list[dict],
    reference_date: date,
) -> IPVReport:
    """Run IPV across a portfolio.

    Args:
        trades: list of kwarg dicts for ipv_single_trade().
        reference_date: valuation date.
    """
    results = [ipv_single_trade(**t) for t in trades]

    total_model = sum(r.model_price for r in results)
    total_ipv = sum(r.ipv_price for r in results)
    total_ava = sum(r.total_ava for r in results)
    total_prudent = sum(r.prudent_value for r in results)

    level_summary = {}
    for r in results:
        lv = r.fair_value_level.value
        level_summary[lv] = level_summary.get(lv, 0) + 1

    breach_count = sum(1 for r in results if r.threshold_breach)

    return IPVReport(
        reference_date=reference_date,
        n_trades=len(results),
        results=results,
        total_model_pv=total_model,
        total_ipv_pv=total_ipv,
        total_ava=total_ava,
        total_prudent_value=total_prudent,
        level_summary=level_summary,
        breach_count=breach_count,
    )
