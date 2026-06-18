"""Middle office operations: trade status, settlement, confirmation, margin calls.

    from pricebook.risk.trade_operations import (
        TradeStatus, TradeStatusTracker, generate_margin_calls,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


# ═══════════════════════════════════════════════════════════════
# Trade Status State Machine
# ═══════════════════════════════════════════════════════════════

class TradeStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    ALLOCATED = "allocated"
    SETTLED = "settled"
    MATURED = "matured"
    TERMINATED = "terminated"
    DEFAULTED = "defaulted"


VALID_TRANSITIONS: dict[TradeStatus, list[TradeStatus]] = {
    TradeStatus.PENDING: [TradeStatus.CONFIRMED, TradeStatus.TERMINATED],
    TradeStatus.CONFIRMED: [TradeStatus.ALLOCATED, TradeStatus.TERMINATED],
    TradeStatus.ALLOCATED: [TradeStatus.SETTLED, TradeStatus.TERMINATED],
    TradeStatus.SETTLED: [TradeStatus.MATURED, TradeStatus.TERMINATED, TradeStatus.DEFAULTED],
    TradeStatus.MATURED: [],
    TradeStatus.TERMINATED: [],
    TradeStatus.DEFAULTED: [],
}


@dataclass
class AuditEntry:
    """Immutable audit record."""
    timestamp: date
    trade_id: str
    action: str
    old_value: str
    new_value: str
    user: str
    reason: str = ""

    def to_dict(self) -> dict:
        return {**vars(self), "timestamp": self.timestamp.isoformat()}


class TradeStatusTracker:
    """State machine for trade status with audit trail."""

    def __init__(self, trade_id: str, created_date: date, user: str = "system"):
        self.trade_id = trade_id
        self.status = TradeStatus.PENDING
        self.created_date = created_date
        self.history: list[AuditEntry] = [
            AuditEntry(created_date, trade_id, "created", "", "pending", user)
        ]

    def can_transition(self, target: TradeStatus) -> bool:
        return target in VALID_TRANSITIONS.get(self.status, [])

    def transition(self, new_status: TradeStatus, user: str,
                   reason: str = "", as_of: date | None = None) -> None:
        if not self.can_transition(new_status):
            raise ValueError(
                f"Cannot transition {self.trade_id} from {self.status.value} to {new_status.value}"
            )
        old = self.status.value
        self.status = new_status
        self.history.append(AuditEntry(
            as_of or date.today(), self.trade_id,
            f"status_change", old, new_status.value, user, reason,
        ))

    @property
    def is_terminal(self) -> bool:
        return self.status in (TradeStatus.MATURED, TradeStatus.TERMINATED, TradeStatus.DEFAULTED)

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "status": self.status.value,
            "created_date": self.created_date.isoformat(),
            "history": [e.to_dict() for e in self.history],
        }


# ═══════════════════════════════════════════════════════════════
# Settlement
# ═══════════════════════════════════════════════════════════════

@dataclass
class SettlementInstruction:
    """What to pay/receive, when, where."""
    trade_id: str
    settlement_date: date
    pay_currency: str
    pay_amount: float
    receive_currency: str
    receive_amount: float
    pay_account: str = ""
    receive_account: str = ""
    status: str = "pending"

    def to_dict(self) -> dict:
        d = vars(self).copy()
        d["settlement_date"] = self.settlement_date.isoformat()
        return d


def generate_settlement(
    trade_id: str,
    settlement_date: date,
    pay_amount: float,
    receive_amount: float,
    pay_currency: str = "USD",
    receive_currency: str = "USD",
) -> SettlementInstruction:
    """Generate settlement instruction."""
    return SettlementInstruction(
        trade_id=trade_id,
        settlement_date=settlement_date,
        pay_currency=pay_currency,
        pay_amount=pay_amount,
        receive_currency=receive_currency,
        receive_amount=receive_amount,
    )


# ═══════════════════════════════════════════════════════════════
# Confirmation
# ═══════════════════════════════════════════════════════════════

@dataclass
class ConfirmationRecord:
    """Trade confirmation tracking."""
    trade_id: str
    counterparty: str
    sent_date: date | None = None
    received_date: date | None = None
    matched: bool = False
    discrepancies: list[str] = field(default_factory=list)
    status: str = "unmatched"

    def to_dict(self) -> dict:
        return dict(vars(self))


def match_confirmation(
    record: ConfirmationRecord,
    our_terms: dict,
    counterparty_terms: dict,
) -> ConfirmationRecord:
    """Compare terms, set matched/disputed status."""
    discrepancies = []
    for key in our_terms:
        if key in counterparty_terms and our_terms[key] != counterparty_terms[key]:
            discrepancies.append(f"{key}: ours={our_terms[key]}, theirs={counterparty_terms[key]}")

    record.discrepancies = discrepancies
    if not discrepancies:
        record.matched = True
        record.status = "matched"
    else:
        record.status = "disputed"
    return record


# ═══════════════════════════════════════════════════════════════
# Margin Calls
# ═══════════════════════════════════════════════════════════════

@dataclass
class DailyMarginCall:
    """A single margin call."""
    counterparty: str
    call_date: date
    exposure: float
    required_collateral: float
    current_collateral: float
    margin_call_amount: float
    currency: str = "USD"

    def to_dict(self) -> dict:
        d = vars(self).copy()
        d["call_date"] = self.call_date.isoformat()
        return d


@dataclass
class MarginCallReport:
    """Daily margin call report."""
    call_date: date
    calls: list[DailyMarginCall]
    total_calls_out: float
    total_returns: float
    net_call: float

    def to_dict(self) -> dict:
        return {
            "call_date": self.call_date.isoformat(),
            "calls": [c.to_dict() for c in self.calls],
            "total_calls_out": self.total_calls_out,
            "total_returns": self.total_returns,
            "net_call": self.net_call,
        }


def generate_margin_calls(
    exposures: dict[str, float],
    thresholds: dict[str, float],
    current_collateral: dict[str, float],
    call_date: date,
    mta: float = 500_000,
) -> MarginCallReport:
    """Compute daily margin calls.

    Args:
        exposures: {counterparty: exposure (positive = we are owed)}.
        thresholds: {counterparty: CSA threshold}.
        current_collateral: {counterparty: collateral currently posted to us}.
        call_date: margin call date.
        mta: minimum transfer amount.
    """
    calls = []
    total_out = 0.0
    total_ret = 0.0

    for cp, exposure in exposures.items():
        threshold = thresholds.get(cp, 0.0)
        required = max(exposure - threshold, 0.0)
        current = current_collateral.get(cp, 0.0)
        delta = required - current

        if abs(delta) < mta:
            delta = 0.0

        calls.append(DailyMarginCall(
            counterparty=cp,
            call_date=call_date,
            exposure=exposure,
            required_collateral=required,
            current_collateral=current,
            margin_call_amount=delta,
        ))

        if delta > 0:
            total_out += delta
        else:
            total_ret += abs(delta)

    return MarginCallReport(
        call_date=call_date,
        calls=calls,
        total_calls_out=total_out,
        total_returns=total_ret,
        net_call=total_out - total_ret,
    )
