"""Tests for IPV."""
import pytest
from datetime import date
from pricebook.risk.ipv import (
    FairValueLevel, classify_fair_value_level, ipv_single_trade, ipv_portfolio,
    BCBS287_BID_ASK,
)

class TestClassify:
    def test_level1(self):
        assert classify_fair_value_level(True, False, 5) == FairValueLevel.LEVEL_1
    def test_level2(self):
        assert classify_fair_value_level(False, True, 0) == FairValueLevel.LEVEL_2
    def test_level3(self):
        assert classify_fair_value_level(False, False, 0) == FairValueLevel.LEVEL_3

class TestIPVTrade:
    def test_level1_market(self):
        r = ipv_single_trade("T1", "bond", 100.5, 1e6, market_price=100.3, n_quotes=3)
        assert r.fair_value_level == FairValueLevel.LEVEL_1
        assert r.ipv_price == 100.3

    def test_level2_matrix(self):
        r = ipv_single_trade("T2", "bond", 98.0, 1e6, matrix_price=97.5)
        assert r.ipv_source == "matrix"

    def test_level3_model(self):
        r = ipv_single_trade("T3", "structured", 95.0, 1e6)
        assert r.ipv_source == "model"

    def test_ava_positive(self):
        r = ipv_single_trade("T4", "bond", 100.0, 1e6, market_price=100.0, n_quotes=3)
        assert r.total_ava >= 0

    def test_threshold_breach(self):
        r = ipv_single_trade("T5", "bond", 100.0, 100.0,
                              market_price=90.0, n_quotes=3, variance_threshold_bp=10.0)
        assert r.threshold_breach

    def test_to_dict(self):
        d = ipv_single_trade("T8", "bond", 100, 1e6).to_dict()
        assert "fair_value_level" in d

class TestIPVPortfolio:
    def test_aggregation(self):
        trades = [
            {"trade_id": "T1", "instrument_type": "bond", "model_price": 100.0,
             "notional": 1e6, "market_price": 100.1, "n_quotes": 3},
            {"trade_id": "T2", "instrument_type": "cds", "model_price": 50000,
             "notional": 5e6, "matrix_price": 49500},
        ]
        r = ipv_portfolio(trades, date(2024, 6, 15))
        assert r.n_trades == 2
        assert r.total_ava >= 0

class TestBCBS287:
    def test_tables(self):
        assert len(BCBS287_BID_ASK) >= 10
        for ac, d in BCBS287_BID_ASK.items():
            assert d["stressed_bp"] >= d["normal_bp"]
