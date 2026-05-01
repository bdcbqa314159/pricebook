"""Tests for TreasuryBenchmark and analytics dataclass serialisation."""

from __future__ import annotations

from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.bond import FixedRateBond
from pricebook.schedule import Frequency
from pricebook.treasury_benchmark import (
    TreasuryBenchmark, AUCTION_SCHEDULE, TLOCK_TENORS,
    create_benchmark_set, when_issued_bond,
    SpecialnessProfile, ctd_switch_analysis,
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


# ---- When-issued ----

class TestWhenIssued:

    def test_from_wi_yield(self):
        """Coupon = floor(WI_yield to nearest 1/8%)."""
        from pricebook.day_count import DayCountConvention
        bond = when_issued_bond(REF, 10, wi_yield=0.0425)
        # 4.25% → 4.25% (already 1/8%)
        assert bond.coupon_rate == pytest.approx(0.0425, abs=0.002)
        assert bond.day_count == DayCountConvention.ACT_ACT_ICMA
        assert bond.settlement_days == 1

    def test_from_coupon(self):
        bond = when_issued_bond(REF, 5, estimated_coupon=0.04)
        assert bond.coupon_rate == 0.04

    def test_maturity(self):
        bond = when_issued_bond(REF, 10, estimated_coupon=0.04)
        expected_mat = REF + relativedelta(years=10)
        assert bond.maturity == expected_mat

    def test_no_args_raises(self):
        with pytest.raises(ValueError, match="Must provide"):
            when_issued_bond(REF, 10)

    def test_prices_on_curve(self):
        from tests.conftest import make_flat_curve
        bond = when_issued_bond(REF, 10, wi_yield=0.04)
        dc = make_flat_curve(REF, 0.04)
        price = bond.dirty_price(dc)
        assert price > 0


# ---- Specialness dynamics ----

class TestSpecialness:

    def test_overnight_specialness(self):
        sp = SpecialnessProfile(
            tenor="10Y", gc_rate=0.045,
            special_rates={1: 0.035, 7: 0.037, 30: 0.040},
        )
        assert sp.overnight_specialness_bps == pytest.approx(100, abs=1)

    def test_specialness_at_tenor(self):
        sp = SpecialnessProfile(
            tenor="10Y", gc_rate=0.045,
            special_rates={1: 0.035, 30: 0.040, 90: 0.043},
        )
        # At 30 days: 45 - 40 = 50bp
        assert sp.specialness_at(30) == pytest.approx(50, abs=1)

    def test_specialness_interpolation(self):
        sp = SpecialnessProfile(
            tenor="10Y", gc_rate=0.045,
            special_rates={1: 0.035, 90: 0.043},
        )
        # Midpoint ~45 days: interpolated between 1d and 90d
        mid = sp.specialness_at(45)
        assert 20 < mid < 100  # between O/N and 90d

    def test_forward_specialness(self):
        sp = SpecialnessProfile(
            tenor="10Y", gc_rate=0.045,
            special_rates={1: 0.035, 30: 0.040, 90: 0.043},
        )
        fwd = sp.forward_specialness(30, 90)
        assert fwd >= 0  # forward specialness should be non-negative

    def test_decay_projection(self):
        sp = SpecialnessProfile(tenor="10Y", gc_rate=0.045, special_rates={})
        decay = sp.expected_specialness_decay(60, 100.0)
        # First point: full specialness
        assert decay[0]["specialness_bps"] == pytest.approx(100.0)
        # Last point: collapsed post-auction
        assert decay[-1]["specialness_bps"] < 20

    def test_to_dict(self):
        sp = SpecialnessProfile(
            tenor="10Y", gc_rate=0.045,
            special_rates={1: 0.035, 30: 0.040},
        )
        d = sp.to_dict()
        assert "tenor" in d
        assert "overnight_specialness_bps" in d


# ---- CTD switching ----

class TestCTDSwitch:

    def test_no_deliverables(self):
        assert ctd_switch_analysis([]) == []

    def test_single_bond_no_switch(self):
        deliverables = [
            {"name": "A", "price": 100, "cf": 0.85, "coupon": 0.04,
             "repo_rate": 0.04, "days": 90, "duration": 8.0, "futures_price": 99},
        ]
        transitions = ctd_switch_analysis(deliverables)
        assert len(transitions) == 0  # only 1 bond → no switch

    def test_two_bonds_may_switch(self):
        deliverables = [
            {"name": "Long", "price": 105, "cf": 0.90, "coupon": 0.05,
             "repo_rate": 0.04, "days": 90, "duration": 12.0, "futures_price": 99},
            {"name": "Short", "price": 98, "cf": 0.82, "coupon": 0.03,
             "repo_rate": 0.04, "days": 90, "duration": 4.0, "futures_price": 99},
        ]
        transitions = ctd_switch_analysis(deliverables, yield_range=(-0.02, 0.02))
        # With different durations, CTD should switch at some yield level
        # (long duration bond becomes CTD in falling rates, short in rising)
        for t in transitions:
            assert t.current_ctd != t.new_ctd
            assert t.probability in ("high", "medium", "low")
