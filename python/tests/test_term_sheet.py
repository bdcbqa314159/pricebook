"""Tests for term sheet generator."""
import pytest
from datetime import date
from pricebook.desks.term_sheet import generate_term_sheet, TermSheet, _format_table, _format_number

class FakeInstrument:
    def __init__(self, notional=1e6, spread=0.03):
        self.notional = notional
        self.spread = spread
    def to_dict(self):
        return {"notional": self.notional, "spread": self.spread, "start": "2024-01-01", "end": "2029-01-01"}

class TestTermSheet:
    def test_basic(self):
        ts = generate_term_sheet(FakeInstrument(), pv=500000, generated_date=date(2024,6,15))
        assert isinstance(ts, TermSheet)
        assert len(ts.sections) >= 3
        assert ts.instrument_type == "FakeInstrument"

    def test_markdown(self):
        ts = generate_term_sheet(FakeInstrument(), generated_date=date(2024,1,1))
        md = ts.to_markdown()
        assert "# FakeInstrument Term Sheet" in md
        assert "Key Terms" in md

    def test_with_metadata(self):
        ts = generate_term_sheet(FakeInstrument(), metadata={"counterparty": "BigBank"},
                                  generated_date=date(2024,1,1))
        assert "BigBank" in ts.to_markdown()

    def test_with_scenarios(self):
        scenarios = [{"scenario": "+100bp", "pv": 490000}, {"scenario": "-100bp", "pv": 510000}]
        ts = generate_term_sheet(FakeInstrument(), scenarios=scenarios, generated_date=date(2024,1,1))
        assert len(ts.sections) == 4  # summary + terms + risk + scenarios

    def test_to_dict(self):
        d = generate_term_sheet(FakeInstrument(), generated_date=date(2024,1,1)).to_dict()
        assert "sections" in d
        assert "instrument_type" in d

class TestHelpers:
    def test_format_table(self):
        t = _format_table(["A","B"], [["1","2"],["3","4"]])
        assert "| A | B |" in t
        assert "| 1 | 2 |" in t

    def test_format_number(self):
        assert _format_number(1_000_000) == "1,000,000"
        assert "." in _format_number(0.0001)
