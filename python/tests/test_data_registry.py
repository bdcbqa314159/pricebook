"""Tests for the convention data-registry loader.

Focus on the loader's *contract at the edges* — especially that a present but
corrupt/schema-drifted file fails loud rather than silently reverting to
hardcoded defaults (which would use the wrong conventions in a pricing run).
"""

import json
import warnings

import pytest

from pricebook.core import data_registry
from pricebook.core.market_conventions import EquityIndexSpec

_GOOD = {
    "ticker": "SPX", "name": "S&P 500", "exchange": "CBOE", "currency": "USD",
    "settlement_lag": 2, "option_style": "european", "option_multiplier": 100.0,
    "dividend_frequency": "quarterly", "ex_date_rule": "T-1",
}


class TestLoadConventions:
    def test_missing_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(data_registry, "DATA_DIR", tmp_path)
        assert data_registry.load_conventions("nope.json", EquityIndexSpec) == []

    def test_empty_file_returns_empty(self, tmp_path, monkeypatch):
        """A genuinely empty file legitimately means 'use defaults' → []."""
        monkeypatch.setattr(data_registry, "DATA_DIR", tmp_path)
        (tmp_path / "e.json").write_text("[]")
        assert data_registry.load_conventions("e.json", EquityIndexSpec) == []

    def test_all_entries_invalid_fails_loud(self, tmp_path, monkeypatch):
        """A present file whose entries all fail must raise — never silently []
        (which the registry would replace with defaults)."""
        monkeypatch.setattr(data_registry, "DATA_DIR", tmp_path)
        (tmp_path / "bad.json").write_text('[{"garbage": 1}, {"nope": 2}]')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="corrupt or its schema"):
                data_registry.load_conventions("bad.json", EquityIndexSpec)

    def test_partial_failure_keeps_valid_entries(self, tmp_path, monkeypatch):
        """One bad row among good ones is skipped-with-warning, not fatal."""
        monkeypatch.setattr(data_registry, "DATA_DIR", tmp_path)
        (tmp_path / "mix.json").write_text(json.dumps([_GOOD, {"garbage": 1}]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            items = data_registry.load_conventions("mix.json", EquityIndexSpec)
        assert len(items) == 1 and items[0].ticker == "SPX"

    def test_path_traversal_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(data_registry, "DATA_DIR", tmp_path)
        with pytest.raises(ValueError, match="Invalid filename"):
            data_registry.load_conventions("../secrets.json", EquityIndexSpec)


class TestLoadRegistry:
    def test_falls_back_to_defaults_when_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(data_registry, "DATA_DIR", tmp_path)
        defaults = {"SPX": EquityIndexSpec.from_dict(_GOOD)}
        out = data_registry.load_registry("nope.json", EquityIndexSpec, lambda e: e.ticker, defaults)
        assert out is defaults

    def test_key_fn_none_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(data_registry, "DATA_DIR", tmp_path)
        with pytest.raises(ValueError, match="key_fn must not be None"):
            data_registry.load_registry("nope.json", EquityIndexSpec, None, {})
