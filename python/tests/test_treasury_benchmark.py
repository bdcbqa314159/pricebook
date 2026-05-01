"""Tests for TreasuryBenchmark and analytics dataclass serialisation."""

from __future__ import annotations

from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.bond import FixedRateBond
from pricebook.schedule import Frequency
from pricebook.treasury_benchmark import (
    TreasuryBenchmark, AUCTION_SCHEDULE, TLOCK_TENORS,
    create_benchmark_set,
)
from pricebook.govt_bond_trading import (
    otr_ofr_spread, when_issued_price, auction_analytics,
    basis_decomposition, ctd_switch_monitor,
)
from pricebook.bond_futures import bond_futures_basis


REF = date(2024, 7, 15)


# ---- TreasuryBenchmark ----

class TestBenchmark:

    def test_create_set(self):
        bms = create_benchmark_set()
        assert "10Y" in bms
        assert "2Y" in bms
        assert len(bms) == len(TLOCK_TENORS)

    def test_otr_ofr_spread(self):
        bm = TreasuryBenchmark(
            tenor="10Y", otr_yield=0.042, ofr_yield=0.0425,
        )
        assert bm.otr_ofr_spread_bps == pytest.approx(5.0, abs=0.1)

    def test_specialness(self):
        bm = TreasuryBenchmark(tenor="10Y", specialness_bps=30)
        assert bm.is_special
        bm2 = TreasuryBenchmark(tenor="10Y", specialness_bps=10)
        assert not bm2.is_special

    def test_adjusted_repo(self):
        bm = TreasuryBenchmark(tenor="10Y", specialness_bps=15)
        adj = bm.adjusted_repo_rate(gc_rate=0.045)
        assert adj == pytest.approx(0.045 - 0.0015)

    def test_next_auction_10y(self):
        bm = TreasuryBenchmark(tenor="10Y")
        nxt = bm.next_auction(date(2024, 3, 1))
        # 10Y auctions in Feb, May, Aug, Nov → next after Mar 1 is May
        assert nxt is not None
        assert nxt.month == 5

    def test_next_auction_2y(self):
        bm = TreasuryBenchmark(tenor="2Y")
        nxt = bm.next_auction(date(2024, 7, 16))
        # 2Y auctions monthly → next is Aug 15
        assert nxt is not None
        assert nxt.month == 8

    def test_days_to_auction(self):
        bm = TreasuryBenchmark(tenor="10Y")
        days = bm.days_to_next_auction(date(2024, 5, 1))
        assert days is not None
        assert days > 0

    def test_to_dict(self):
        bond = FixedRateBond.treasury_note(
            REF - relativedelta(years=1), REF + relativedelta(years=9), 0.04,
        )
        bm = TreasuryBenchmark(
            tenor="10Y", otr_cusip="ABC123", otr_bond=bond,
            otr_yield=0.042, ofr_yield=0.0425, specialness_bps=15,
            last_auction=date(2024, 5, 15),
        )
        d = bm.to_dict()
        assert d["tenor"] == "10Y"
        assert d["otr_cusip"] == "ABC123"
        assert "otr_bond" in d
        assert d["last_auction"] == "2024-05-15"


# ---- Analytics dataclass to_dict ----

class TestAnalyticsToDict:

    def test_otr_ofr_spread_to_dict(self):
        result = otr_ofr_spread("10Y", 0.042, 0.0425)
        d = result.to_dict()
        assert "spread_bps" in d
        assert "signal" in d

    def test_when_issued_to_dict(self):
        result = when_issued_price(0.04, 0.045, 5.0, 10.0, 7.0, 0.04)
        d = result.to_dict()
        assert "estimated_yield" in d
        assert "estimated_price" in d

    def test_auction_to_dict(self):
        result = auction_analytics(
            "10Y", 0.042, 0.0415, 80e9, 35e9, 15e9, 14e9, 6e9,
        )
        d = result.to_dict()
        assert "bid_to_cover" in d
        assert "well_received" in d

    def test_basis_decomposition_to_dict(self):
        result = basis_decomposition("UST 10Y", 100.5, 99.0, 0.85, 0.04, 0.04, 90)
        d = result.to_dict()
        assert "gross_basis" in d
        assert "implied_repo" in d

    def test_ctd_switch_to_dict(self):
        bd1 = basis_decomposition("Bond A", 100.5, 99.0, 0.85, 0.04, 0.04, 90)
        bd2 = basis_decomposition("Bond B", 101.0, 99.0, 0.86, 0.035, 0.04, 90)
        entries = ctd_switch_monitor([bd1, bd2])
        for e in entries:
            d = e.to_dict()
            assert "is_ctd" in d

    def test_bond_futures_basis_to_dict(self):
        result = bond_futures_basis(100.5, 99.0, 0.85, 0.04, 90, 2.0)
        d = result.to_dict()
        assert "gross_basis" in d
        assert "net_basis" in d
