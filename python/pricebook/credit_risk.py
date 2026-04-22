"""Credit risk measures and credit book management.

CS01, spread DV01, jump-to-default risk for CDS trades.
CreditBook extends Book for credit-specific aggregation.

    from pricebook.credit_risk import cs01, jump_to_default, CreditBook

    cs = cs01(cds, discount_curve, survival_curve)
    jtd = jump_to_default(cds, discount_curve)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

from pricebook.cds import CDS
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.pricing_context import PricingContext
from pricebook.trade import Trade


# ---- Survival curve bumping ----

def _bump_survival_curve(
    curve: SurvivalCurve,
    shift: float,
) -> SurvivalCurve:
    """Parallel bump: shift all hazard rates by `shift`, rebuild survival probs."""
    ref = curve.reference_date
    pillar_dates = curve._pillar_dates
    times = [float(t) for t in curve._times if t > 0]
    survs = [float(s) for t, s in zip(curve._times, curve._survs) if t > 0]

    # Convert to hazard rates, bump, convert back
    new_survs = []
    prev_t = 0.0
    prev_q = 1.0
    for t, q in zip(times, survs):
        dt = t - prev_t
        if dt > 0 and q > 0 and prev_q > 0:
            h = -math.log(q / prev_q) / dt
            h_bumped = max(h + shift, 1e-10)
            new_q = prev_q * math.exp(-h_bumped * dt)
        else:
            new_q = q
        new_survs.append(new_q)
        prev_t = t
        prev_q = new_q

    return SurvivalCurve(ref, pillar_dates, new_survs)


def _bump_survival_curve_at(
    curve: SurvivalCurve,
    pillar_idx: int,
    shift: float,
) -> SurvivalCurve:
    """Bump hazard rate at a single pillar."""
    ref = curve.reference_date
    pillar_dates = curve._pillar_dates
    times = [float(t) for t in curve._times if t > 0]
    survs = [float(s) for t, s in zip(curve._times, curve._survs) if t > 0]

    new_survs = list(survs)
    if pillar_idx < len(times):
        t = times[pillar_idx]
        prev_t = times[pillar_idx - 1] if pillar_idx > 0 else 0.0
        prev_q = survs[pillar_idx - 1] if pillar_idx > 0 else 1.0
        q = survs[pillar_idx]
        dt = t - prev_t
        if dt > 0 and q > 0 and prev_q > 0:
            h = -math.log(q / prev_q) / dt
            h_bumped = max(h + shift, 1e-10)
            new_survs[pillar_idx] = prev_q * math.exp(-h_bumped * dt)

    return SurvivalCurve(ref, pillar_dates, new_survs)


# ---- CS01 ----

def cs01(
    cds: CDS,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    shift_bps: float = 1.0,
) -> float:
    """CS01: PV change per 1bp parallel shift in credit spreads.

    Bumps all hazard rates by shift_bps basis points and measures PV change.
    """
    base_pv = cds.pv(discount_curve, survival_curve)
    shift = shift_bps / 10000.0
    bumped = _bump_survival_curve(survival_curve, shift)
    return (cds.pv(discount_curve, bumped) - base_pv) / shift_bps


def spread_dv01(
    cds: CDS,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    shift_bps: float = 1.0,
) -> list[tuple[int, float]]:
    """Per-pillar credit spread sensitivity (key-rate CS01).

    Returns list of (pillar_index, sensitivity_per_bp).
    """
    base_pv = cds.pv(discount_curve, survival_curve)
    shift = shift_bps / 10000.0
    n_pillars = len([t for t in survival_curve._times if t > 0])

    results = []
    for i in range(n_pillars):
        bumped = _bump_survival_curve_at(survival_curve, i, shift)
        delta = (cds.pv(discount_curve, bumped) - base_pv) / shift_bps
        results.append((i, delta))

    return results


# ---- Jump-to-default ----

def jump_to_default(
    cds: CDS,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
) -> float:
    """Jump-to-default risk: PV change if name defaults immediately.

    For protection buyer: receives (1-R) * notional, stops paying premium.
    JTD = (1-R) * notional - accrued_premium - current_PV
    """
    lgd = (1.0 - cds.recovery) * cds.notional
    current_pv = cds.pv(discount_curve, survival_curve)
    # On immediate default, protection buyer receives LGD
    # and loses the current mark-to-market position
    return lgd - current_pv


# ---- Credit Book ----

@dataclass
class CreditPosition:
    """Net exposure for one name."""
    name: str
    net_notional: float
    trade_count: int
    cs01: float
    jtd: float


class CreditBook:
    """A book of credit trades with credit-specific risk aggregation.

    Args:
        name: book name.
        discount_curve: shared discount curve for risk calculations.
    """

    def __init__(self, name: str):
        self.name = name
        self._trades: list[tuple[Trade, str, SurvivalCurve]] = []  # (trade, credit_name, surv_curve)

    def add(
        self,
        trade: Trade,
        credit_name: str,
        survival_curve: SurvivalCurve,
    ) -> None:
        """Add a credit trade with its reference entity and survival curve."""
        self._trades.append((trade, credit_name, survival_curve))

    def __len__(self) -> int:
        return len(self._trades)

    def pv(self, discount_curve: DiscountCurve) -> float:
        """Aggregate PV."""
        total = 0.0
        for trade, _, surv in self._trades:
            inst = trade.instrument
            if isinstance(inst, CDS):
                total += trade.direction * trade.notional_scale * inst.pv(discount_curve, surv)
        return total

    def total_cs01(self, discount_curve: DiscountCurve, shift_bps: float = 1.0) -> float:
        """Aggregate CS01 across all trades."""
        total = 0.0
        for trade, _, surv in self._trades:
            inst = trade.instrument
            if isinstance(inst, CDS):
                total += trade.direction * trade.notional_scale * cs01(inst, discount_curve, surv, shift_bps)
        return total

    def cs01_by_name(self, discount_curve: DiscountCurve, shift_bps: float = 1.0) -> dict[str, float]:
        """CS01 broken down by credit name."""
        result: dict[str, float] = {}
        for trade, name, surv in self._trades:
            inst = trade.instrument
            if isinstance(inst, CDS):
                val = trade.direction * trade.notional_scale * cs01(inst, discount_curve, surv, shift_bps)
                result[name] = result.get(name, 0.0) + val
        return result

    def jtd_by_name(self, discount_curve: DiscountCurve) -> dict[str, float]:
        """Jump-to-default risk broken down by credit name."""
        result: dict[str, float] = {}
        for trade, name, surv in self._trades:
            inst = trade.instrument
            if isinstance(inst, CDS):
                val = trade.direction * trade.notional_scale * jump_to_default(inst, discount_curve, surv)
                result[name] = result.get(name, 0.0) + val
        return result

    def positions(self, discount_curve: DiscountCurve) -> list[CreditPosition]:
        """Aggregate positions by credit name."""
        names: dict[str, dict[str, Any]] = {}
        for trade, name, surv in self._trades:
            inst = trade.instrument
            if isinstance(inst, CDS):
                if name not in names:
                    names[name] = {"notional": 0.0, "count": 0, "cs01": 0.0, "jtd": 0.0}
                signed_notional = trade.direction * trade.notional_scale * inst.notional
                names[name]["notional"] += signed_notional
                names[name]["count"] += 1
                names[name]["cs01"] += trade.direction * trade.notional_scale * cs01(inst, discount_curve, surv)
                names[name]["jtd"] += trade.direction * trade.notional_scale * jump_to_default(inst, discount_curve, surv)

        return [
            CreditPosition(name, d["notional"], d["count"], d["cs01"], d["jtd"])
            for name, d in sorted(names.items())
        ]
