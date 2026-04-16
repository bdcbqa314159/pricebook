"""Sovereign bond trading: peripheral spreads, basis, auctions, OTR/OFR.

* :class:`SovereignSpreadCurve` — term structure of sovereign spreads.
* :func:`sovereign_basis` — bond vs CDS vs futures basis.
* :func:`cross_country_rv` — cross-sovereign relative value.
* :class:`AuctionResult` — auction bid-cover, tail, concession analytics.
* :func:`otr_ofr_analysis` — on-the-run vs off-the-run premium.

References:
    Beber, Brandt & Kavajecz, *Flight-to-Quality or Flight-to-Liquidity?*,
    RFS, 2009.
    Ang, Papanikolaou & Westerfield, *Portfolio Choice with Illiquid Assets*,
    Mgmt. Sci., 2014.
    Pasquariello & Vega, *Strategic Cross-Trading in the U.S. Stock Market*,
    RF, 2013.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---- Sovereign spread curve ----

@dataclass
class SovereignSpreadCurve:
    """Term structure of sovereign spreads over a benchmark."""
    country: str
    benchmark: str
    tenors: np.ndarray
    spreads_bps: np.ndarray

    def spread_at(self, T: float) -> float:
        """Spread (in bps) at tenor T, interpolated."""
        if T <= self.tenors[0]:
            return float(self.spreads_bps[0])
        if T >= self.tenors[-1]:
            return float(self.spreads_bps[-1])
        return float(np.interp(T, self.tenors, self.spreads_bps))

    def z_score(self, historical_spreads_bps: list[float], T: float) -> float:
        """Z-score of current spread vs historical distribution."""
        arr = np.array(historical_spreads_bps)
        mu = float(arr.mean())
        sigma = float(arr.std())
        current = self.spread_at(T)
        return (current - mu) / max(sigma, 1e-10)


def build_spread_curve(
    country: str,
    benchmark: str,
    tenors: list[float],
    country_yields_pct: list[float],
    benchmark_yields_pct: list[float],
) -> SovereignSpreadCurve:
    """Build sovereign spread curve from yields."""
    T = np.array(tenors)
    spreads_bps = (np.array(country_yields_pct) - np.array(benchmark_yields_pct)) * 100
    return SovereignSpreadCurve(country, benchmark, T, spreads_bps)


# ---- Sovereign basis ----

@dataclass
class SovereignBasisResult:
    """Sovereign basis: bond vs CDS vs futures."""
    bond_yield_bps: float
    cds_spread_bps: float
    futures_implied_bps: float
    bond_cds_basis_bps: float       # bond - CDS (negative basis typical)
    bond_futures_basis_bps: float
    cds_futures_basis_bps: float


def sovereign_basis(
    bond_yield_pct: float,
    swap_rate_pct: float,
    cds_spread_pct: float,
    futures_implied_yield_pct: float,
) -> SovereignBasisResult:
    """Decompose sovereign basis.

    Bond-CDS basis = (bond yield - swap) - CDS spread.
    Bond-futures basis = (bond yield - futures-implied yield) × 100.
    CDS-futures basis = CDS - (futures-implied - swap).

    Args:
        bond_yield_pct: bond yield (%).
        swap_rate_pct: matched-maturity swap rate (%).
        cds_spread_pct: CDS spread (%).
        futures_implied_yield_pct: futures-implied yield (%).
    """
    bond_yield_bps = bond_yield_pct * 100
    cds_bps = cds_spread_pct * 100
    fut_bps = futures_implied_yield_pct * 100
    swap_bps = swap_rate_pct * 100

    asw_spread = bond_yield_bps - swap_bps
    bond_cds_basis = asw_spread - cds_bps
    bond_futures_basis = bond_yield_bps - fut_bps
    cds_futures_basis = cds_bps - (fut_bps - swap_bps)

    return SovereignBasisResult(
        bond_yield_bps=bond_yield_bps,
        cds_spread_bps=cds_bps,
        futures_implied_bps=fut_bps,
        bond_cds_basis_bps=float(bond_cds_basis),
        bond_futures_basis_bps=float(bond_futures_basis),
        cds_futures_basis_bps=float(cds_futures_basis),
    )


# ---- Cross-country RV ----

@dataclass
class CrossCountryRV:
    """Cross-country sovereign RV analysis."""
    pair: tuple[str, str]           # (country1, country2)
    tenor: float
    yield_diff_bps: float
    z_score: float                  # z-score of current diff vs history
    mean_historical: float
    std_historical: float


def cross_country_rv(
    country1: str,
    country2: str,
    tenor: float,
    yield1_pct: float,
    yield2_pct: float,
    historical_diffs_bps: list[float],
) -> CrossCountryRV:
    """Current cross-country yield spread vs historical distribution."""
    diff = (yield1_pct - yield2_pct) * 100
    hist = np.array(historical_diffs_bps)
    mu = float(hist.mean())
    sig = float(hist.std())
    z = (diff - mu) / max(sig, 1e-10)

    return CrossCountryRV(
        pair=(country1, country2),
        tenor=tenor,
        yield_diff_bps=float(diff),
        z_score=float(z),
        mean_historical=mu,
        std_historical=sig,
    )


# ---- Auction analytics ----

@dataclass
class AuctionResult:
    """Auction analytics result."""
    instrument: str
    amount_offered: float
    bids_received: float
    bid_cover_ratio: float
    stop_out_yield: float
    wi_yield_at_auction: float      # when-issued yield
    tail_bps: float                 # stop-out - average bid
    concession_bps: float           # wi yield - secondary curve yield
    quality_score: float            # simple 0-10 score


def auction_analytics(
    instrument: str,
    amount_offered: float,
    bids_received: float,
    stop_out_yield_pct: float,
    average_bid_yield_pct: float,
    wi_yield_pct: float,
    secondary_curve_yield_pct: float,
) -> AuctionResult:
    """Analyse a sovereign auction.

    Key measures:
    - Bid-cover: bids_received / amount_offered (higher = better demand).
    - Tail: stop_out − average (positive = poor demand, demand concentrated at high yield).
    - Concession: WI yield − secondary curve (positive = auction at a discount).
    """
    bid_cover = bids_received / max(amount_offered, 1e-10)
    tail_bps = (stop_out_yield_pct - average_bid_yield_pct) * 100
    concession_bps = (wi_yield_pct - secondary_curve_yield_pct) * 100

    # Simple quality score: high bid-cover, low tail, low concession → high score
    quality = 5.0
    if bid_cover > 2.5:
        quality += 2
    elif bid_cover < 1.5:
        quality -= 2
    if abs(tail_bps) < 1:
        quality += 1
    elif abs(tail_bps) > 3:
        quality -= 1
    if abs(concession_bps) < 1:
        quality += 1
    elif abs(concession_bps) > 3:
        quality -= 1
    quality = max(0, min(10, quality))

    return AuctionResult(
        instrument=instrument,
        amount_offered=amount_offered,
        bids_received=bids_received,
        bid_cover_ratio=float(bid_cover),
        stop_out_yield=stop_out_yield_pct,
        wi_yield_at_auction=wi_yield_pct,
        tail_bps=float(tail_bps),
        concession_bps=float(concession_bps),
        quality_score=float(quality),
    )


# ---- OTR / OFR ----

@dataclass
class OTROFRResult:
    """On-the-run vs off-the-run comparison."""
    tenor: float
    otr_yield_pct: float
    ofr_yield_pct: float
    otr_premium_bps: float      # OFR yield - OTR yield (positive = OTR richer)
    is_squeeze: bool            # flagged if premium unusually large


def otr_ofr_analysis(
    tenor: float,
    otr_yield_pct: float,
    ofr_yield_pct: float,
    typical_premium_bps: float = 2.0,
    squeeze_threshold_bps: float = 10.0,
) -> OTROFRResult:
    """Compare OTR vs OFR yields; detect squeezes.

    OTR (on-the-run): most recently auctioned bond — highest liquidity.
    OFR (off-the-run): earlier-issued, slightly less liquid.

    OTR typically trades 1-3 bps richer (lower yield) than OFR at similar maturity.
    Squeeze: OTR much richer than normal → liquidity / repo special.
    """
    premium_bps = (ofr_yield_pct - otr_yield_pct) * 100
    is_squeeze = premium_bps > squeeze_threshold_bps

    return OTROFRResult(
        tenor=tenor,
        otr_yield_pct=otr_yield_pct,
        ofr_yield_pct=ofr_yield_pct,
        otr_premium_bps=float(premium_bps),
        is_squeeze=bool(is_squeeze),
    )
