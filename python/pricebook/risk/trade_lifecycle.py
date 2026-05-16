"""Trade lifecycle: amendments, exercises, novations, and audit history."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any

from pricebook.trade import Trade


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


class EventType(Enum):
    CREATED = "created"
    AMENDED = "amended"
    EXERCISED = "exercised"
    NOVATED = "novated"


@dataclass
class LifecycleEvent:
    """A single lifecycle event on a trade."""

    event_type: EventType
    event_date: date
    details: dict[str, Any] = field(default_factory=dict)
    version: int = 0


# ---------------------------------------------------------------------------
# Managed trade with versioned history
# ---------------------------------------------------------------------------


class ManagedTrade:
    """A trade with full lifecycle management and audit trail.

    Wraps a Trade and tracks amendments, exercises, and novations as
    versioned events. The original trade is never mutated; each change
    produces a new version.
    """

    def __init__(self, trade: Trade, creation_date: date | None = None):
        self._versions: list[Trade] = [copy.deepcopy(trade)]
        self._events: list[LifecycleEvent] = []
        cdate = creation_date or trade.trade_date or date.today()
        self._events.append(LifecycleEvent(
            event_type=EventType.CREATED,
            event_date=cdate,
            details={"trade_id": trade.trade_id},
            version=0,
        ))

    @property
    def current(self) -> Trade:
        """The current (latest) version of the trade."""
        return self._versions[-1]

    @property
    def version(self) -> int:
        return len(self._versions) - 1

    @property
    def history(self) -> list[LifecycleEvent]:
        return list(self._events)

    @property
    def is_exercised(self) -> bool:
        return any(e.event_type == EventType.EXERCISED for e in self._events)

    def get_version(self, v: int) -> Trade:
        if v < 0 or v >= len(self._versions):
            raise IndexError(f"Version {v} not found (have 0..{len(self._versions)-1})")
        return self._versions[v]

    # ----- Amendments -----

    def amend(
        self,
        event_date: date,
        notional_scale: float | None = None,
        direction: int | None = None,
        counterparty: str | None = None,
        instrument: object | None = None,
    ) -> Trade:
        """Amend the trade, creating a new version. Returns the new version."""
        new_trade = copy.deepcopy(self.current)
        changes: dict[str, Any] = {}

        if notional_scale is not None:
            changes["notional_scale"] = (new_trade.notional_scale, notional_scale)
            new_trade.notional_scale = notional_scale
        if direction is not None:
            changes["direction"] = (new_trade.direction, direction)
            new_trade.direction = direction
        if counterparty is not None:
            changes["counterparty"] = (new_trade.counterparty, counterparty)
            new_trade.counterparty = counterparty
        if instrument is not None:
            changes["instrument"] = "replaced"
            new_trade.instrument = instrument

        if not changes:
            raise ValueError("No amendments specified")

        new_version = len(self._versions)
        self._versions.append(new_trade)
        self._events.append(LifecycleEvent(
            event_type=EventType.AMENDED,
            event_date=event_date,
            details=changes,
            version=new_version,
        ))
        return new_trade

    # ----- Exercise -----

    def exercise(
        self,
        event_date: date,
        underlying_instrument: object,
    ) -> Trade:
        """Exercise an option, replacing the instrument with its underlying.

        For example, exercising a swaption produces a swap.
        """
        if self.is_exercised:
            raise ValueError("Trade already exercised")

        new_trade = copy.deepcopy(self.current)
        old_type = type(new_trade.instrument).__name__
        new_trade.instrument = underlying_instrument

        new_version = len(self._versions)
        self._versions.append(new_trade)
        self._events.append(LifecycleEvent(
            event_type=EventType.EXERCISED,
            event_date=event_date,
            details={
                "old_instrument": old_type,
                "new_instrument": type(underlying_instrument).__name__,
            },
            version=new_version,
        ))
        return new_trade

    # ----- Novation -----

    def novate(
        self,
        event_date: date,
        new_counterparty: str,
    ) -> Trade:
        """Transfer the trade to a new counterparty. Economics unchanged."""
        old_cp = self.current.counterparty
        new_trade = copy.deepcopy(self.current)
        new_trade.counterparty = new_counterparty

        new_version = len(self._versions)
        self._versions.append(new_trade)
        self._events.append(LifecycleEvent(
            event_type=EventType.NOVATED,
            event_date=event_date,
            details={
                "old_counterparty": old_cp,
                "new_counterparty": new_counterparty,
            },
            version=new_version,
        ))
        return new_trade
