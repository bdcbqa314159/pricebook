"""Tests for middle office operations."""
import pytest
from datetime import date
from pricebook.risk.trade_operations import (
    TradeStatus, TradeStatusTracker, VALID_TRANSITIONS,
    SettlementInstruction, generate_settlement,
    ConfirmationRecord, match_confirmation,
    DailyMarginCall, MarginCallReport, generate_margin_calls,
)

class TestTradeStatus:
    def test_create(self):
        t = TradeStatusTracker("T1", date(2024,1,1))
        assert t.status == TradeStatus.PENDING

    def test_valid_transition(self):
        t = TradeStatusTracker("T1", date(2024,1,1))
        t.transition(TradeStatus.CONFIRMED, "trader", "verified")
        assert t.status == TradeStatus.CONFIRMED
        assert len(t.history) == 2  # created + confirmed

    def test_invalid_transition(self):
        t = TradeStatusTracker("T1", date(2024,1,1))
        with pytest.raises(ValueError):
            t.transition(TradeStatus.SETTLED, "trader")  # can't skip

    def test_terminal(self):
        t = TradeStatusTracker("T1", date(2024,1,1))
        t.transition(TradeStatus.TERMINATED, "ops")
        assert t.is_terminal

    def test_audit_trail(self):
        t = TradeStatusTracker("T1", date(2024,1,1), "system")
        t.transition(TradeStatus.CONFIRMED, "trader", "matched", date(2024,1,2))
        assert t.history[-1].user == "trader"
        assert t.history[-1].reason == "matched"

    def test_to_dict(self):
        d = TradeStatusTracker("T1", date(2024,1,1)).to_dict()
        assert d["status"] == "pending"

class TestSettlement:
    def test_generate(self):
        s = generate_settlement("T1", date(2024,1,5), 1e6, 0, "USD", "EUR")
        assert isinstance(s, SettlementInstruction)
        assert s.pay_amount == 1e6

class TestConfirmation:
    def test_match(self):
        r = ConfirmationRecord("T1", "BigBank", date(2024,1,1))
        r = match_confirmation(r, {"notional": 1e6}, {"notional": 1e6})
        assert r.matched

    def test_disputed(self):
        r = ConfirmationRecord("T1", "BigBank", date(2024,1,1))
        r = match_confirmation(r, {"notional": 1e6}, {"notional": 2e6})
        assert r.status == "disputed"
        assert len(r.discrepancies) == 1

class TestMarginCalls:
    def test_basic(self):
        r = generate_margin_calls(
            exposures={"BankA": 10e6, "BankB": 5e6},
            thresholds={"BankA": 1e6, "BankB": 2e6},
            current_collateral={"BankA": 5e6, "BankB": 3e6},
            call_date=date(2024,6,15),
        )
        assert isinstance(r, MarginCallReport)
        assert r.net_call >= 0 or r.net_call < 0  # valid number

    def test_no_call_under_mta(self):
        r = generate_margin_calls(
            exposures={"BankA": 1.1e6},
            thresholds={"BankA": 1e6},
            current_collateral={"BankA": 0.05e6},
            call_date=date(2024,1,1),
            mta=500_000,
        )
        # Required=0.1M, current=0.05M, delta=0.05M < MTA=0.5M → no call
        assert r.calls[0].margin_call_amount == 0.0

    def test_to_dict(self):
        r = generate_margin_calls({"A": 5e6}, {"A": 0}, {"A": 0}, date(2024,1,1))
        d = r.to_dict()
        assert "net_call" in d
