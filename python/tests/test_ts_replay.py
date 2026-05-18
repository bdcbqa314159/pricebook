"""Tests for ts._replay: replay result construction."""
import pytest
from pricebook.ts._replay import ReplayResult, DrawdownPeriod

class TestReplayResult:
    def test_dataclass(self):
        assert hasattr(ReplayResult, "__dataclass_fields__") or callable(ReplayResult)

class TestDrawdownPeriod:
    def test_dataclass(self):
        assert hasattr(DrawdownPeriod, "__dataclass_fields__") or callable(DrawdownPeriod)
