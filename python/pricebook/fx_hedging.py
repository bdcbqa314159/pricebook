"""FX hedging: delta hedge, cross-hedge, triangular arb, NDF settlement.

* :func:`fx_delta_hedge` — spot or forward hedge quantity.
* :func:`fx_cross_hedge` — proxy hedge (e.g. NOK via SEK).
* :func:`triangular_arb_monitor` — synthetic cross vs direct.
* :func:`ndf_settlement` — cash settlement from FX fixing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.settlement import add_business_days


# ---- Delta hedge ----

def fx_delta_hedge(
    book_delta: float,
    hedge_delta_per_unit: float = 1.0,
) -> float:
    """Quantity to trade to flatten FX delta.

    ``quantity = -book_delta / hedge_delta_per_unit``.
    """
    if abs(hedge_delta_per_unit) < 1e-15:
        return 0.0
    return -book_delta / hedge_delta_per_unit


# ---- Cross-hedge ----

@dataclass
class CrossHedgeResult:
    """Proxy hedge recommendation."""
    target_pair: str
    proxy_pair: str
    hedge_ratio: float
    proxy_quantity: float


def fx_cross_hedge(
    target_pair: str,
    target_delta: float,
    proxy_pair: str,
    correlation: float,
    proxy_vol: float,
    target_vol: float,
) -> CrossHedgeResult:
    """Cross-hedge: hedge one pair using a correlated proxy.

    Optimal hedge ratio (minimum-variance):
        h = ρ × σ_target / σ_proxy

    Args:
        target_delta: delta to hedge (in target pair notional).
        correlation: historical correlation between the two pairs.
        proxy_vol / target_vol: annualised vols.
    """
    if proxy_vol <= 0:
        h = 0.0
    else:
        h = correlation * target_vol / proxy_vol
    qty = -target_delta * h
    return CrossHedgeResult(target_pair, proxy_pair, h, qty)


# ---- Triangular arbitrage ----

@dataclass
class TriangularArbResult:
    """Triangular arbitrage monitor."""
    pair_ab: str
    pair_bc: str
    pair_ac: str
    direct_rate: float
    synthetic_rate: float
    arb_bps: float
    is_arb: bool


def triangular_arb_monitor(
    pair_ab: str,
    rate_ab: float,
    pair_bc: str,
    rate_bc: float,
    pair_ac: str,
    rate_ac: float,
    threshold_bps: float = 1.0,
) -> TriangularArbResult:
    """Check for triangular arbitrage: A/C direct vs A/B × B/C synthetic.

    ``synthetic_ac = rate_ab × rate_bc``.
    ``arb = (direct − synthetic) / direct × 10000`` (in bps).
    """
    synthetic = rate_ab * rate_bc
    if rate_ac <= 0:
        arb_bps = 0.0
    else:
        arb_bps = (rate_ac - synthetic) / rate_ac * 10_000
    return TriangularArbResult(
        pair_ab, pair_bc, pair_ac,
        direct_rate=rate_ac,
        synthetic_rate=synthetic,
        arb_bps=arb_bps,
        is_arb=abs(arb_bps) > threshold_bps,
    )


# ---- NDF settlement ----

@dataclass
class NDFSettlementResult:
    """NDF cash settlement."""
    pair: str
    contracted_rate: float
    fixing_rate: float
    notional: float
    settlement_amount: float
    settlement_date: date


def ndf_settlement(
    pair: str,
    contracted_rate: float,
    fixing_rate: float,
    notional: float,
    fixing_date: date,
    calendar: object | None = None,
) -> NDFSettlementResult:
    """Compute NDF cash settlement.

    ``settlement = (fixing − contracted) × notional``.
    Settlement date is T+2 from the fixing date.
    """
    amount = (fixing_rate - contracted_rate) * notional
    settle_date = add_business_days(fixing_date, 2, calendar)
    return NDFSettlementResult(
        pair=pair,
        contracted_rate=contracted_rate,
        fixing_rate=fixing_rate,
        notional=notional,
        settlement_amount=amount,
        settlement_date=settle_date,
    )
