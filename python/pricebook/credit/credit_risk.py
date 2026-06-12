"""Credit risk measures and credit book management.

CS01, spread DV01, jump-to-default risk for CDS trades.
CreditBook extends Book for credit-specific aggregation.

    from pricebook.credit.credit_risk import cs01, jump_to_default, CreditBook

    cs = cs01(cds, discount_curve, survival_curve)
    jtd = jump_to_default(cds, discount_curve)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

from pricebook.credit.cds import CDS
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve
from pricebook.core.pricing_context import PricingContext
from pricebook.core.trade import Trade


# ---- Survival curve bumping ----

def _bump_survival_curve(
    curve: SurvivalCurve,
    shift: float,
) -> SurvivalCurve:
    """Parallel bump: shift all per-segment hazard rates by `shift`, rebuild survival.

    Fix T4-CR2: pre-fix this routine extracted segment-i hazard as
    ``h_i = -log(q_old_i / prev_q_NEW) / dt`` where ``prev_q_NEW`` had
    already been bumped by the previous iteration's shift.  This made the
    extracted `h_i` partially absorb the upstream shift, so the bump applied
    to it was effectively smaller for later pillars.  Empirically on a flat
    2% hazard curve with shift=1bp, the 5y survival shifted by only ~half
    of the expected `-shift·t`.

    Correct: extract all hazards from the ORIGINAL curve first (using OLD
    `prev_q` at each step), bump them all by `shift`, then reconstruct
    survivals forward.  This makes the per-pillar `_bump_survival_curve_at`
    and the parallel `_bump_survival_curve` mathematically consistent
    (sum of per-pillar key-rate CS01s = parallel CS01).
    """
    ref = curve.reference_date
    pillar_dates = curve._pillar_dates
    times = [float(t) for t in curve._times if t > 0]
    survs = [float(s) for t, s in zip(curve._times, curve._survs) if t > 0]

    # Extract original per-segment hazards using OLD prev_q at each step.
    hazards = []
    prev_t = 0.0
    prev_q = 1.0
    for t, q in zip(times, survs):
        dt = t - prev_t
        if dt > 0 and q > 0 and prev_q > 0:
            h = -math.log(q / prev_q) / dt
        else:
            h = 0.0
        hazards.append(h)
        prev_t = t
        prev_q = q   # OLD q, not new — the key fix.

    # Bump every segment hazard.
    hazards = [max(h + shift, 1e-10) for h in hazards]

    # Reconstruct survivals forward.
    new_survs = []
    prev_t = 0.0
    prev_q = 1.0
    for h_i, t in zip(hazards, times):
        dt = t - prev_t
        new_q = prev_q * math.exp(-h_i * dt)
        new_survs.append(new_q)
        prev_t = t
        prev_q = new_q

    return SurvivalCurve(ref, pillar_dates, new_survs)


def _bump_survival_curve_at(
    curve: SurvivalCurve,
    pillar_idx: int,
    shift: float,
) -> SurvivalCurve:
    """Bump hazard rate in segment `pillar_idx` only; propagate to downstream survivals.

    Fix T4-CR1: pre-fix only updated `survs[pillar_idx]` and left every
    later survival `survs[pillar_idx+1:]` at its ORIGINAL value.  Since the
    segment-(i+1) hazard is `-log(Q_{i+1} / Q_i) / dt`, holding Q_{i+1}
    fixed while shifting Q_i meant segment i+1's hazard ALSO shifted by
    `−shift` — contaminating the next segment with the OPPOSITE-sign
    perturbation.  Key-rate CS01 reports were therefore wrong on every
    pillar except the last.

    Correct semantics: bump the per-segment hazard at index `pillar_idx`,
    keep every other segment hazard at its original value, then propagate
    forward by chaining `Q_i = Q_{i-1} · exp(-h_i · dt_i)`.
    """
    ref = curve.reference_date
    pillar_dates = curve._pillar_dates
    times = [float(t) for t in curve._times if t > 0]
    survs = [float(s) for t, s in zip(curve._times, curve._survs) if t > 0]

    # Extract per-segment hazards from survival probs.
    hazards = []
    prev_t = 0.0
    prev_q = 1.0
    for t, q in zip(times, survs):
        dt = t - prev_t
        if dt > 0 and q > 0 and prev_q > 0:
            h = -math.log(q / prev_q) / dt
        else:
            h = 0.0
        hazards.append(h)
        prev_t = t
        prev_q = q

    # Bump only segment `pillar_idx`.
    if 0 <= pillar_idx < len(hazards):
        hazards[pillar_idx] = max(hazards[pillar_idx] + shift, 1e-10)

    # Propagate forward to recover ALL downstream survivals.
    new_survs = []
    prev_t = 0.0
    prev_q = 1.0
    for h_i, t in zip(hazards, times):
        dt = t - prev_t
        new_q = prev_q * math.exp(-h_i * dt)
        new_survs.append(new_q)
        prev_t = t
        prev_q = new_q

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



    def to_dict(self) -> dict:
        return vars(self)
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
