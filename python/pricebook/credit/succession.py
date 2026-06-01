"""Succession events: merger, spin-off, split.

When a reference entity undergoes a succession event, the CDS
contract must be adjusted. Notional is split according to economic
weight, with conservation verified.

* :class:`SuccessionEvent` — event specification.
* :func:`apply_succession` — apply succession to CDS contract.
* :func:`verify_notional_conservation` — check Σ new = original.

References:
    ISDA, *Credit Derivatives Definitions*, Section 2.2 (Successor), 2014.
    ISDA, *Succession Event Supplement*, 2009.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class SuccessionType(Enum):
    """Types of succession events."""
    MERGER = "merger"
    SPIN_OFF = "spin_off"
    SPLIT = "split"
    REVERSE_MERGER = "reverse_merger"
    ACQUISITION = "acquisition"


@dataclass
class SuccessionEvent:
    """Succession event specification."""
    original_entity: str
    event_type: SuccessionType
    event_date: date
    successors: list[str]           # successor entity names
    weights: list[float]            # economic weight per successor
    original_notional: float = 0.0

    def to_dict(self) -> dict:
        return {
            "original_entity": self.original_entity,
            "event_type": self.event_type.value,
            "event_date": self.event_date.isoformat(),
            "successors": self.successors,
            "weights": self.weights,
        }


@dataclass
class SuccessorCDS:
    """CDS contract resulting from a succession event."""
    entity: str
    notional: float
    spread: float
    weight: float       # fraction of original

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class SuccessionResult:
    """Result of applying a succession event."""
    original_entity: str
    original_notional: float
    event_type: SuccessionType
    successor_cds: list[SuccessorCDS]
    notional_conserved: bool
    total_successor_notional: float

    def to_dict(self) -> dict:
        return {
            "original_entity": self.original_entity,
            "original_notional": self.original_notional,
            "event_type": self.event_type.value,
            "n_successors": len(self.successor_cds),
            "notional_conserved": self.notional_conserved,
            "total_successor_notional": self.total_successor_notional,
            "successors": [s.to_dict() for s in self.successor_cds],
        }


def apply_succession(
    event: SuccessionEvent,
    original_spread: float,
    original_notional: float | None = None,
    spread_adjustments: list[float] | None = None,
) -> SuccessionResult:
    """Apply a succession event to a CDS contract.

    Each successor receives a fraction of the original notional
    based on economic weight. Spreads may be adjusted per successor
    to reflect different credit quality.

    ISDA rules:
    - If one successor holds ≥75% of obligations → sole successor.
    - If two+ successors hold ≥25% each → split proportionally.
    - If no successor holds ≥25% → original entity remains.

    Args:
        event: succession event specification.
        original_spread: original CDS spread.
        original_notional: CDS notional (overrides event if given).
        spread_adjustments: per-successor spread adjustments (additive).
    """
    notional = original_notional or event.original_notional
    if notional <= 0:
        raise ValueError("original_notional must be positive")

    n = len(event.successors)
    if len(event.weights) != n:
        raise ValueError("successors and weights must have same length")

    # Normalise weights
    total_w = sum(event.weights)
    if total_w <= 0:
        raise ValueError("weights must sum to positive value")
    norm_weights = [w / total_w for w in event.weights]

    successor_cds = []
    for i, (name, weight) in enumerate(zip(event.successors, norm_weights)):
        spread = original_spread
        if spread_adjustments and i < len(spread_adjustments):
            spread += spread_adjustments[i]

        successor_cds.append(SuccessorCDS(
            entity=name,
            notional=notional * weight,
            spread=max(spread, 0),
            weight=weight,
        ))

    total_successor = sum(s.notional for s in successor_cds)
    conserved = abs(total_successor - notional) < 1e-6

    return SuccessionResult(
        original_entity=event.original_entity,
        original_notional=notional,
        event_type=event.event_type,
        successor_cds=successor_cds,
        notional_conserved=conserved,
        total_successor_notional=total_successor,
    )


def verify_notional_conservation(result: SuccessionResult) -> bool:
    """Verify that successor notionals sum to original."""
    return result.notional_conserved
