"""Government bond trading: OTR/OFR spreads, auction analytics, basis trading.

Builds on :mod:`pricebook.bond_futures` (``bond_futures_basis``,
``cheapest_to_deliver``, ``implied_repo_rate``) with desk-level
monitors and analytics.

* :func:`otr_ofr_spread` — on-the-run vs off-the-run yield spread.
* :func:`when_issued_price` — estimate WI price from the yield curve.
* :class:`AuctionResult` — tail, bid-to-cover, dealer allocation.
* :func:`basis_decomposition` — gross basis = carry + net basis (optionality).
* :func:`ctd_switch_monitor` — track which bond is cheapest-to-deliver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date


# ---- OTR / OFR spread ----

@dataclass
class OTROFRSpread:
    """On-the-run vs off-the-run yield spread for a tenor."""
    tenor_label: str
    otr_yield: float
    ofr_yield: float
    spread_bps: float
    z_score: float | None
    signal: str


def otr_ofr_spread(
    tenor_label: str,
    otr_yield: float,
    ofr_yield: float,
    history_bps: list[float] | None = None,
    threshold: float = 2.0,
) -> OTROFRSpread:
    """Compute the OTR-OFR yield spread and z-score vs history.

    Convention: ``spread = ofr_yield - otr_yield`` (positive = OTR trades
    rich, which is the normal liquidity premium).
    """
    spread = (ofr_yield - otr_yield) * 10_000  # in bps

    z = None
    signal = "fair"
    if history_bps and len(history_bps) >= 2:
        mean = sum(history_bps) / len(history_bps)
        var = sum((h - mean) ** 2 for h in history_bps) / len(history_bps)
        std = math.sqrt(var) if var > 0 else 0.0
        if std > 1e-12:
            z = (spread - mean) / std
            if abs(z) >= threshold:
                signal = "wide" if z > 0 else "tight"

    return OTROFRSpread(
        tenor_label=tenor_label,
        otr_yield=otr_yield,
        ofr_yield=ofr_yield,
        spread_bps=spread,
        z_score=z,
        signal=signal,
    )


# ---- When-issued (WI) pricing ----

@dataclass
class WhenIssuedEstimate:
    """Pre-auction WI price estimate."""
    estimated_yield: float
    estimated_price: float
    interpolation_basis: str


def when_issued_price(
    short_tenor_yield: float,
    long_tenor_yield: float,
    short_tenor_years: float,
    long_tenor_years: float,
    target_tenor_years: float,
    coupon_rate: float,
    face_value: float = 100.0,
) -> WhenIssuedEstimate:
    """Estimate a WI yield and price by interpolating the yield curve.

    Linearly interpolates between two bracketing tenors to estimate the
    WI yield, then converts to a rough price using the annuity formula.

    Args:
        short_tenor_yield / long_tenor_yield: bracketing yields.
        short_tenor_years / long_tenor_years: their maturities.
        target_tenor_years: the WI bond's maturity.
        coupon_rate: expected coupon of the new issue.
        face_value: face value (default 100).
    """
    span = long_tenor_years - short_tenor_years
    if span <= 0:
        wi_yield = short_tenor_yield
    else:
        w = (target_tenor_years - short_tenor_years) / span
        wi_yield = short_tenor_yield + w * (long_tenor_yield - short_tenor_yield)

    # Rough price from yield (simple annuity approximation)
    if wi_yield <= 0:
        price = face_value + coupon_rate * target_tenor_years * face_value
    else:
        annuity = (1 - (1 + wi_yield) ** (-target_tenor_years)) / wi_yield
        price = coupon_rate * face_value * annuity + face_value * (1 + wi_yield) ** (-target_tenor_years)

    return WhenIssuedEstimate(
        estimated_yield=wi_yield,
        estimated_price=price / face_value * 100.0,
        interpolation_basis=f"{short_tenor_years}Y-{long_tenor_years}Y",
    )


# ---- Auction analytics ----

@dataclass
class AuctionResult:
    """Post-auction analytics."""
    issue_tenor: str
    high_yield: float
    bid_to_cover: float
    tail_bps: float
    dealer_pct: float
    indirect_pct: float
    direct_pct: float

    @property
    def well_received(self) -> bool:
        """Heuristic: auction is well-received if bid-to-cover > 2.3
        and tail < 1bp."""
        return self.bid_to_cover > 2.3 and self.tail_bps < 1.0


def auction_analytics(
    issue_tenor: str,
    high_yield: float,
    when_issued_yield: float,
    total_bids: float,
    accepted_amount: float,
    dealer_amount: float,
    indirect_amount: float,
    direct_amount: float,
) -> AuctionResult:
    """Compute post-auction statistics.

    Args:
        high_yield: auction stop-out yield.
        when_issued_yield: WI yield just before auction.
        total_bids: total amount of bids submitted.
        accepted_amount: amount accepted (= issue size).
        dealer_amount / indirect_amount / direct_amount: allocation.
    """
    tail = (high_yield - when_issued_yield) * 10_000  # bps
    btc = total_bids / accepted_amount if accepted_amount > 0 else 0.0
    total_alloc = dealer_amount + indirect_amount + direct_amount
    if total_alloc > 0:
        dealer_pct = dealer_amount / total_alloc * 100
        indirect_pct = indirect_amount / total_alloc * 100
        direct_pct = direct_amount / total_alloc * 100
    else:
        dealer_pct = indirect_pct = direct_pct = 0.0

    return AuctionResult(
        issue_tenor=issue_tenor,
        high_yield=high_yield,
        bid_to_cover=btc,
        tail_bps=tail,
        dealer_pct=dealer_pct,
        indirect_pct=indirect_pct,
        direct_pct=direct_pct,
    )


# ---- Basis decomposition ----

@dataclass
class BasisDecomposition:
    """Gross basis = carry + net basis (optionality)."""
    bond_name: str
    gross_basis: float
    carry: float
    net_basis: float
    implied_repo: float


def basis_decomposition(
    bond_name: str,
    bond_price: float,
    futures_price: float,
    cf: float,
    coupon_rate: float,
    repo_rate: float,
    days_to_delivery: int,
    face_value: float = 100.0,
) -> BasisDecomposition:
    """Decompose the bond futures basis into carry + optionality.

    Wraps the formula from ``bond_futures_basis`` with an implied repo
    calculation.
    """
    gross = bond_price - cf * futures_price
    dt = days_to_delivery / 365.0
    coupon_income = coupon_rate * face_value * dt
    financing = bond_price * repo_rate * dt
    carry = coupon_income - financing
    net = gross - carry

    # Implied repo: rate that makes gross basis = carry
    if dt > 0 and bond_price > 0:
        implied_repo = (coupon_income - (bond_price - cf * futures_price)) / (bond_price * dt)
    else:
        implied_repo = 0.0

    return BasisDecomposition(
        bond_name=bond_name,
        gross_basis=gross,
        carry=carry,
        net_basis=net,
        implied_repo=implied_repo,
    )


# ---- CTD switch monitor ----

@dataclass
class CTDSwitchEntry:
    """One deliverable bond with its basis metrics."""
    bond_name: str
    implied_repo: float
    gross_basis: float
    net_basis: float
    is_ctd: bool


def ctd_switch_monitor(
    deliverables: list[BasisDecomposition],
) -> list[CTDSwitchEntry]:
    """Identify the CTD bond and rank all deliverables.

    The CTD is the bond with the highest implied repo rate (cheapest
    to deliver = most attractive to the short).
    """
    if not deliverables:
        return []

    best_repo = max(d.implied_repo for d in deliverables)

    return [
        CTDSwitchEntry(
            bond_name=d.bond_name,
            implied_repo=d.implied_repo,
            gross_basis=d.gross_basis,
            net_basis=d.net_basis,
            is_ctd=(abs(d.implied_repo - best_repo) < 1e-12),
        )
        for d in sorted(deliverables, key=lambda x: -x.implied_repo)
    ]
