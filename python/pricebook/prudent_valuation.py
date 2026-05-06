"""Prudent valuation: Additional Valuation Adjustments (AVA) per EBA RTS.

Implements the 6 quantifiable AVA categories from CRR Art 105 /
EBA RTS 2014/2017 for fair-valued positions:

1. Market Price Uncertainty (MPU) — spread of available prices
2. Close-Out Cost (COC) — bid-ask exit cost, size-dependent
3. Model Risk (MR) — spread of model valuations
4. Concentration (CONC) — additional cost for large positions
5. Unearned Credit Spread (UCS) — day-1 CVA on uncollateralised
6. Investing & Funding Cost (IFC) — liquidity premium for illiquid

Prudent value = mid_price - total_AVA

    from pricebook.prudent_valuation import (
        PrudentValuationReport, compute_prudent_value,
        MarketPriceUncertaintyAVA, CloseOutCostAVA, ModelRiskAVA,
        ConcentrationAVA, UnearnedCreditSpreadAVA, InvestingFundingAVA,
    )

References:
    EBA (2014). RTS on prudent valuation (EU 2016/101).
    EBA (2017). Final draft RTS on prudent valuation (EBA/RTS/2017/03).
    CRR Art 105. Prudent valuation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Individual AVA categories
# ---------------------------------------------------------------------------

@dataclass
class MarketPriceUncertaintyAVA:
    """AVA for market price uncertainty.

    Based on the spread of available prices/quotes for the same position.
    AVA = (mid - prudent_price) where prudent_price is the 90th percentile
    of the price distribution (EBA RTS Art 9).
    """
    mid_price: float
    bid_price: float
    ask_price: float
    n_quotes: int
    ava: float               # the adjustment (always >= 0)
    confidence_level: float  # typically 0.90

    def to_dict(self) -> dict:
        return {"mid": self.mid_price, "bid": self.bid_price, "ask": self.ask_price,
                "n_quotes": self.n_quotes, "ava": self.ava,
                "confidence": self.confidence_level}


def market_price_uncertainty_ava(
    mid_price: float,
    bid_price: float,
    ask_price: float,
    n_quotes: int = 3,
    confidence: float = 0.90,
) -> MarketPriceUncertaintyAVA:
    """Compute market price uncertainty AVA.

    For liquid instruments with multiple quotes:
        AVA = mid - bid (for long positions, conservative exit at bid)
    Scaled by confidence factor when few quotes available.

    Args:
        mid_price: mid-market price.
        bid_price: best bid price.
        ask_price: best ask price.
        n_quotes: number of independent quotes available.
        confidence: confidence level (EBA: 90%).
    """
    half_spread = (ask_price - bid_price) / 2
    # Prudent value adjustment: half the bid-ask, scaled for quote reliability
    reliability_factor = min(n_quotes / 5.0, 1.0)  # full confidence at 5+ quotes
    ava = half_spread * confidence / reliability_factor if reliability_factor > 0 else half_spread

    return MarketPriceUncertaintyAVA(
        mid_price=mid_price, bid_price=bid_price, ask_price=ask_price,
        n_quotes=n_quotes, ava=max(ava, 0.0), confidence_level=confidence,
    )


@dataclass
class CloseOutCostAVA:
    """AVA for close-out cost — cost to exit at market bid, not mid.

    Size-dependent: larger positions face wider exit spreads.
    """
    base_spread_bp: float    # normal bid-ask spread (bp)
    size_adjustment_bp: float  # additional spread for position size
    notional: float
    ava: float

    def to_dict(self) -> dict:
        return {"base_spread_bp": self.base_spread_bp,
                "size_adj_bp": self.size_adjustment_bp,
                "notional": self.notional, "ava": self.ava}


# Typical bid-ask by asset class (bp of notional)
_ASSET_CLASS_SPREADS = {
    "irs": 0.5,       # very liquid
    "cds": 3.0,       # liquid IG
    "cds_hy": 10.0,   # high yield
    "bond_gov": 1.0,  # government
    "bond_ig": 3.0,   # investment grade
    "bond_hy": 15.0,  # high yield
    "fx": 0.5,        # major pairs
    "equity": 5.0,    # single stock
    "commodity": 5.0,
    "swaption": 5.0,
    "repo": 1.0,
    "trs": 5.0,
    "cln": 20.0,      # illiquid
    "structured": 30.0,  # very illiquid
    "private": 50.0,  # private placement
}


def close_out_cost_ava(
    notional: float,
    asset_class: str = "bond_ig",
    daily_volume: float = 0.0,
    position_days: float = 1.0,
) -> CloseOutCostAVA:
    """Compute close-out cost AVA.

    Args:
        notional: position notional.
        asset_class: key into _ASSET_CLASS_SPREADS.
        daily_volume: average daily trading volume (0 = unknown → use base).
        position_days: days to exit position (size / daily_volume).
    """
    base_bp = _ASSET_CLASS_SPREADS.get(asset_class, 10.0)

    # Size adjustment: if position > 1 day's volume, spread widens
    if daily_volume > 0:
        position_days = max(notional / daily_volume, 1.0)
        size_adj_bp = base_bp * max(math.log(position_days), 0.0)
    else:
        size_adj_bp = base_bp * 0.5  # default 50% premium for unknown liquidity

    total_bp = base_bp + size_adj_bp
    ava = notional * total_bp / 10_000

    return CloseOutCostAVA(
        base_spread_bp=base_bp,
        size_adjustment_bp=size_adj_bp,
        notional=notional,
        ava=max(ava, 0.0),
    )


@dataclass
class ModelRiskAVA:
    """AVA for model risk — spread of valuations across models."""
    prices: list[float]        # prices from different models
    mid_price: float           # selected mid price
    model_spread: float        # max - min across models
    ava: float

    def to_dict(self) -> dict:
        return {"n_models": len(self.prices), "mid": self.mid_price,
                "model_spread": self.model_spread, "ava": self.ava}


def model_risk_ava(
    model_prices: list[float],
    confidence: float = 0.90,
) -> ModelRiskAVA:
    """Compute model risk AVA from multiple model valuations.

    AVA = half the range of model prices × confidence factor.
    EBA: use the range of plausible models at 90% confidence.

    Args:
        model_prices: list of prices from independent models.
        confidence: confidence level (EBA: 90%).
    """
    if len(model_prices) < 2:
        return ModelRiskAVA(model_prices, model_prices[0] if model_prices else 0.0, 0.0, 0.0)

    mid = sum(model_prices) / len(model_prices)
    spread = max(model_prices) - min(model_prices)
    ava = spread / 2 * confidence

    return ModelRiskAVA(
        prices=list(model_prices), mid_price=mid,
        model_spread=spread, ava=max(ava, 0.0),
    )


@dataclass
class ConcentrationAVA:
    """AVA for concentration — additional exit cost for large positions."""
    notional: float
    market_size: float         # total market outstanding
    concentration_pct: float   # position / market
    ava: float

    def to_dict(self) -> dict:
        return {"notional": self.notional, "market_size": self.market_size,
                "concentration_pct": self.concentration_pct, "ava": self.ava}


def concentration_ava(
    notional: float,
    market_size: float,
    base_spread_bp: float = 5.0,
) -> ConcentrationAVA:
    """Compute concentration AVA.

    Additional exit cost when position is large relative to the market.
    AVA scales quadratically with concentration (convex market impact).

    Args:
        notional: position notional.
        market_size: total market outstanding for this instrument/sector.
        base_spread_bp: baseline spread (bp).
    """
    if market_size <= 0:
        return ConcentrationAVA(notional, 0, 1.0, notional * base_spread_bp / 10_000)

    conc = notional / market_size
    # Quadratic impact: cost = base × (1 + κ × conc²) where κ calibrated to 10x at 10%
    kappa = 1000.0  # at 10% concentration → 10x base spread
    impact_multiplier = 1.0 + kappa * conc ** 2
    ava = notional * base_spread_bp * impact_multiplier / 10_000

    return ConcentrationAVA(
        notional=notional, market_size=market_size,
        concentration_pct=conc, ava=max(ava, 0.0),
    )


@dataclass
class UnearnedCreditSpreadAVA:
    """AVA for unearned credit spread — day-1 CVA on uncollateralised trades."""
    cva: float               # CVA amount
    collateralised: bool
    ava: float

    def to_dict(self) -> dict:
        return {"cva": self.cva, "collateralised": self.collateralised,
                "ava": self.ava}


def unearned_credit_spread_ava(
    cva: float,
    collateralised: bool = False,
) -> UnearnedCreditSpreadAVA:
    """Compute unearned credit spread AVA.

    For uncollateralised trades, the day-1 CVA represents credit risk
    that cannot be immediately hedged → AVA = full CVA.
    For collateralised trades, AVA is reduced by the collateral benefit.

    Args:
        cva: credit valuation adjustment (positive = cost).
        collateralised: whether the trade has CSA/margin.
    """
    if collateralised:
        ava = cva * 0.10  # 90% collateral benefit
    else:
        ava = cva

    return UnearnedCreditSpreadAVA(
        cva=cva, collateralised=collateralised, ava=max(ava, 0.0),
    )


@dataclass
class InvestingFundingAVA:
    """AVA for investing and funding cost — illiquidity premium."""
    illiquidity_premium_bp: float
    notional: float
    maturity_years: float
    ava: float

    def to_dict(self) -> dict:
        return {"illiquidity_bp": self.illiquidity_premium_bp,
                "notional": self.notional, "maturity_years": self.maturity_years,
                "ava": self.ava}


def investing_funding_ava(
    notional: float,
    illiquidity_premium_bp: float,
    maturity_years: float,
) -> InvestingFundingAVA:
    """Compute investing/funding cost AVA.

    Cost of funding an illiquid position over its remaining life.
    AVA = notional × illiquidity_premium × min(maturity, 1) (1-year horizon).

    Args:
        notional: position notional.
        illiquidity_premium_bp: illiquidity spread (from LiquidityPremiumModel).
        maturity_years: remaining maturity.
    """
    horizon = min(maturity_years, 1.0)  # EBA: max 1-year funding horizon
    ava = notional * illiquidity_premium_bp / 10_000 * horizon

    return InvestingFundingAVA(
        illiquidity_premium_bp=illiquidity_premium_bp,
        notional=notional, maturity_years=maturity_years,
        ava=max(ava, 0.0),
    )


# ---------------------------------------------------------------------------
# Aggregation + Prudent Value Report
# ---------------------------------------------------------------------------

@dataclass
class PrudentValuationReport:
    """Full prudent valuation report for a position."""
    mid_price: float
    mpu_ava: float            # market price uncertainty
    coc_ava: float            # close-out cost
    mr_ava: float             # model risk
    conc_ava: float           # concentration
    ucs_ava: float            # unearned credit spread
    ifc_ava: float            # investing/funding cost
    total_ava: float          # sum (before diversification)
    diversification_benefit: float  # typically 50% under EBA simplified
    total_ava_diversified: float
    prudent_value: float      # mid_price - total_ava_diversified

    def to_dict(self) -> dict:
        return {
            "mid_price": self.mid_price,
            "mpu": self.mpu_ava, "coc": self.coc_ava, "mr": self.mr_ava,
            "conc": self.conc_ava, "ucs": self.ucs_ava, "ifc": self.ifc_ava,
            "total_ava_gross": self.total_ava,
            "diversification": self.diversification_benefit,
            "total_ava": self.total_ava_diversified,
            "prudent_value": self.prudent_value,
        }


def compute_prudent_value(
    mid_price: float,
    mpu: MarketPriceUncertaintyAVA | None = None,
    coc: CloseOutCostAVA | None = None,
    mr: ModelRiskAVA | None = None,
    conc: ConcentrationAVA | None = None,
    ucs: UnearnedCreditSpreadAVA | None = None,
    ifc: InvestingFundingAVA | None = None,
    diversification_pct: float = 0.50,
) -> PrudentValuationReport:
    """Compute prudent value from individual AVAs.

    Prudent value = mid_price - total_AVA (after diversification benefit).

    EBA simplified approach allows 50% diversification across AVA categories
    (CRR Art 105(9), EBA RTS Art 8).

    Args:
        mid_price: fair value (mid-market).
        mpu, coc, mr, conc, ucs, ifc: individual AVA results (None = 0).
        diversification_pct: diversification benefit (EBA: 50%).
    """
    mpu_val = mpu.ava if mpu else 0.0
    coc_val = coc.ava if coc else 0.0
    mr_val = mr.ava if mr else 0.0
    conc_val = conc.ava if conc else 0.0
    ucs_val = ucs.ava if ucs else 0.0
    ifc_val = ifc.ava if ifc else 0.0

    total_gross = mpu_val + coc_val + mr_val + conc_val + ucs_val + ifc_val

    # Diversification: sqrt-of-sum-of-squares (correlation assumption)
    # Simplified: apply flat percentage reduction
    diversification = total_gross * diversification_pct
    total_net = total_gross - diversification

    return PrudentValuationReport(
        mid_price=mid_price,
        mpu_ava=mpu_val, coc_ava=coc_val, mr_ava=mr_val,
        conc_ava=conc_val, ucs_ava=ucs_val, ifc_ava=ifc_val,
        total_ava=total_gross,
        diversification_benefit=diversification,
        total_ava_diversified=total_net,
        prudent_value=mid_price - total_net,
    )
