"""FX carry strategies: G10 carry, EM carry, NDF carry, carry-adjusted forward.

* :func:`carry_signal` — annualised carry from forward points.
* :func:`carry_adjusted_forward` — break-even spot move.
* :func:`g10_carry_ranking` — rank G10 pairs by carry.
* :func:`ndf_carry` — NDF carry from rate differential.
* :func:`carry_volatility_ratio` — carry / vol (Sharpe-like).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---- G10 carry ----

@dataclass
class CarrySignal:
    """Carry for a single FX pair."""
    pair: str
    spot: float
    forward: float
    forward_points: float
    annualised_carry: float  # as a fraction of spot
    carry_direction: str     # "long_base" or "short_base"


def carry_signal(
    pair: str,
    spot: float,
    forward: float,
    days_to_maturity: int = 90,
) -> CarrySignal:
    """Compute carry from the forward-spot differential.

    ``annualised_carry = (forward - spot) / spot × (365 / days)``.
    Positive carry for a long-base position means the forward is above
    spot (base currency at a premium — foreign rate > domestic rate).
    The carry trade is to *short* the premium currency.
    """
    fwd_pts = forward - spot
    ann = fwd_pts / spot * (365.0 / days_to_maturity) if days_to_maturity > 0 and spot != 0 else 0.0
    direction = "long_base" if ann < 0 else "short_base"
    return CarrySignal(pair, spot, forward, fwd_pts, ann, direction)


def carry_adjusted_forward(
    spot: float,
    forward: float,
    days_to_maturity: int,
) -> float:
    """Break-even spot move: the spot change that zeroes the carry.

    ``break_even = forward - spot``. If the base currency depreciates
    by more than this, the carry trade loses money.
    """
    return forward - spot


@dataclass
class CarryRanking:
    """Ranked FX pairs by carry attractiveness."""
    pair: str
    annualised_carry: float
    rank: int


def g10_carry_ranking(
    pairs: list[tuple[str, float, float, int]],
) -> list[CarryRanking]:
    """Rank G10 pairs by annualised carry (most negative first = best carry).

    The best carry trade is *short* the pair with the most negative
    annualised carry (short the high-yielding base).

    Args:
        pairs: list of ``(pair, spot, forward, days_to_maturity)``.
    """
    signals = [carry_signal(p, s, f, d) for p, s, f, d in pairs]
    signals.sort(key=lambda s: s.annualised_carry)
    return [
        CarryRanking(s.pair, s.annualised_carry, rank=i + 1)
        for i, s in enumerate(signals)
    ]


# ---- EM / NDF carry ----

@dataclass
class NDFCarry:
    """NDF carry from rate differential."""
    pair: str
    domestic_rate: float
    foreign_rate: float
    rate_differential: float
    annualised_carry: float
    ndf_points: float


def ndf_carry(
    pair: str,
    spot: float,
    domestic_rate: float,
    foreign_rate: float,
    days: int = 90,
) -> NDFCarry:
    """NDF carry = rate differential implied by the NDF.

    ``NDF = spot × (1 + r_d × T) / (1 + r_f × T)``
    ``carry ≈ (r_f − r_d)`` annualised.
    """
    T = days / 365.0
    if T <= 0:
        return NDFCarry(pair, domestic_rate, foreign_rate, 0.0, 0.0, 0.0)
    ndf = spot * (1 + domestic_rate * T) / (1 + foreign_rate * T) if (1 + foreign_rate * T) != 0 else spot
    diff = foreign_rate - domestic_rate
    pts = ndf - spot
    return NDFCarry(pair, domestic_rate, foreign_rate, diff, diff, pts)


# ---- Carry / vol ratio ----

def carry_volatility_ratio(
    annualised_carry: float,
    annualised_vol: float,
) -> float:
    """Carry-to-vol ratio (Sharpe-like measure).

    Higher ratio = more carry per unit of risk. Typical threshold:
    ratio > 0.5 is attractive.
    """
    if annualised_vol <= 0:
        return 0.0
    return abs(annualised_carry) / annualised_vol
