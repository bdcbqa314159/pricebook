"""Dynamic haircuts — credit-driven, stochastic, with rating triggers.

    from pricebook.risk.dynamic_haircuts import (
        DynamicHaircutModel, haircut_stress_scenarios,
        credit_spread_to_haircut, rating_trigger_impact,
    )

References:
    BCBS (2013). Margin Requirements, Annex B (procyclicality).
    Lo (2016). Gap Risk in Secured Lending.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class DynamicHaircutResult:
    """Result of dynamic haircut computation."""
    base_haircut: float
    spread_driven_add_on: float
    vol_driven_add_on: float
    rating_trigger_add_on: float
    procyclicality_buffer: float
    total_haircut: float

    def to_dict(self) -> dict:
        return vars(self)


class DynamicHaircutModel:
    """Stochastic haircut model driven by spread + vol + rating.

    h(t) = h_base + f(spread) + g(vol) + rating_step + procyclicality_buffer

    where:
    - f(spread) = duration × spread_change × confidence_multiplier
    - g(vol) = duration × vol × √(MPOR/252) × z(99%)
    - rating_step = step function on downgrade
    - procyclicality = BCBS 261 25% buffer
    """

    def __init__(
        self,
        base_haircut: float,
        duration: float,
        current_spread_bp: float,
        spread_vol: float = 0.30,
        mpor_days: int = 10,
        confidence: float = 0.99,
        procyclicality_buffer_pct: float = 0.25,
    ):
        self.base_haircut = base_haircut
        self.duration = duration
        self.current_spread_bp = current_spread_bp
        self.spread_vol = spread_vol
        self.mpor_days = mpor_days
        self.confidence = confidence
        self.procyclicality_buffer_pct = procyclicality_buffer_pct

    def compute(
        self,
        spread_shock_bp: float = 0.0,
        vol_shock_pct: float = 0.0,
        rating_downgrade_notches: int = 0,
    ) -> DynamicHaircutResult:
        """Compute dynamic haircut under given conditions.

        Args:
            spread_shock_bp: spread widening (bp).
            vol_shock_pct: vol increase (additive, e.g. 0.10 = +10%).
            rating_downgrade_notches: number of notches downgraded.
        """
        z = norm.ppf(self.confidence)
        sqrt_mpor = math.sqrt(self.mpor_days / 252.0)

        # Spread-driven add-on
        spread_add = self.duration * spread_shock_bp / 10_000

        # Vol-driven add-on
        effective_vol = self.spread_vol + vol_shock_pct
        vol_add = self.duration * (self.current_spread_bp / 10_000) * effective_vol * sqrt_mpor * z

        # Rating trigger
        rating_add = _rating_step(rating_downgrade_notches)

        # Procyclicality buffer (BCBS 261)
        subtotal = self.base_haircut + spread_add + vol_add + rating_add
        procyc = subtotal * self.procyclicality_buffer_pct

        total = subtotal + procyc

        return DynamicHaircutResult(
            base_haircut=self.base_haircut,
            spread_driven_add_on=spread_add,
            vol_driven_add_on=vol_add,
            rating_trigger_add_on=rating_add,
            procyclicality_buffer=procyc,
            total_haircut=min(total, 1.0),
        )

    def to_dict(self) -> dict:
        return {
            "base_haircut": self.base_haircut,
            "duration": self.duration,
            "current_spread_bp": self.current_spread_bp,
            "spread_vol": self.spread_vol,
            "mpor_days": self.mpor_days,
        }


def haircut_stress_scenarios(
    model: DynamicHaircutModel,
) -> list[dict]:
    """Run standard haircut stress scenarios.

    Scenarios: base, spread +100bp, spread +300bp, vol ×2,
    1-notch downgrade, 3-notch downgrade, combined stress.
    """
    scenarios = [
        ("base", 0, 0.0, 0),
        ("spread_+100bp", 100, 0.0, 0),
        ("spread_+300bp", 300, 0.0, 0),
        ("vol_x2", 0, model.spread_vol, 0),
        ("downgrade_1", 0, 0.0, 1),
        ("downgrade_3", 0, 0.0, 3),
        ("combined_stress", 200, model.spread_vol * 0.5, 2),
    ]
    results = []
    for name, spread, vol, rating in scenarios:
        r = model.compute(spread, vol, rating)
        results.append({
            "scenario": name,
            "total_haircut": r.total_haircut,
            "spread_add_on": r.spread_driven_add_on,
            "vol_add_on": r.vol_driven_add_on,
            "rating_add_on": r.rating_trigger_add_on,
        })
    return results


def credit_spread_to_haircut(
    spread_bp: float,
    duration: float,
    base_haircut: float = 0.02,
    mpor_days: int = 10,
) -> float:
    """Continuous mapping from CDS spread → haircut add-on.

    haircut = base + duration × spread × vol × √(MPOR/252) × z(99%)

    Simplified for quick lookup.
    """
    z = norm.ppf(0.99)
    spread_vol = 0.30  # assumed
    sqrt_mpor = math.sqrt(mpor_days / 252.0)
    add_on = duration * (spread_bp / 10_000) * spread_vol * sqrt_mpor * z
    return base_haircut + add_on


def rating_trigger_impact(
    current_haircut: float,
    downgrade_notches: int,
) -> float:
    """Haircut after rating trigger (step function).

    Each notch adds an incremental haircut based on regulatory tables.
    """
    return current_haircut + _rating_step(downgrade_notches)


# Rating downgrade haircut steps (cumulative by notch)
_RATING_STEPS = {
    0: 0.00,
    1: 0.01,   # 1 notch: +1%
    2: 0.03,   # 2 notches: +3%
    3: 0.06,   # 3 notches: +6% (fallen angel territory)
    4: 0.10,   # 4 notches: +10%
    5: 0.15,   # 5+ notches: +15%
}


def _rating_step(notches: int) -> float:
    if notches <= 0:
        return 0.0
    return _RATING_STEPS.get(min(notches, 5), 0.15)
