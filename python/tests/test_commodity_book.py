"""Tests for commodity book."""

import pytest
from datetime import date

from pricebook.commodity_book import (
    CommodityBook,
    CommodityLimits,
    CommodityPosition,
    CommoditySectorExposure,
    TermStructureBucket,
    commodity_tenor_bucket,
)
from pricebook.swaption import Swaption
from pricebook.trade import Trade


REF = date(2024, 1, 15)


def _instr():
    """Stand-in instrument; CommodityBook only needs the trade for direction."""
    return Swaption(date(2025, 1, 15), date(2030, 1, 15),
                    strike=0.05, notional=1_000_000)


def _trade(direction=1, scale=1.0, trade_id="t"):
    return Trade(_instr(), direction=direction, notional_scale=scale, trade_id=trade_id)


# ---- Tenor bucket helper ----

class TestTenorBucket:
    def test_unknown_when_none(self):
        assert commodity_tenor_bucket(REF, None) == "unknown"

    def test_front(self):
        assert commodity_tenor_bucket(REF, date(2024, 2, 1)) == "front"

    def test_six_month(self):
        assert commodity_tenor_bucket(REF, date(2024, 6, 1)) == "≤6M"

    def test_one_year(self):
        assert commodity_tenor_bucket(REF, date(2024, 12, 1)) == "≤1Y"

    def test_two_year(self):
        assert commodity_tenor_bucket(REF, date(2025, 12, 1)) == "≤2Y"

    def test_long_dated(self):
        assert commodity_tenor_bucket(REF, date(2027, 1, 1)) == ">2Y"


# ---- Step 1: book + aggregations ----

class TestCommodityBook:
    def test_create_empty(self):
        book = CommodityBook("Energy", REF)
        assert book.name == "Energy"
        assert len(book) == 0
        assert book.n_commodities == 0
        assert book.n_sectors == 0

    def test_add_trade(self):
        book = CommodityBook("Energy", REF)
        book.add(_trade(), commodity="WTI", sector="energy", unit="bbl",
                 quantity=10_000, reference_price=72.0)
        assert len(book) == 1
        assert book.n_commodities == 1
        assert book.n_sectors == 1


class TestPositionsByCommodity:
    def test_single_long(self):
        book = CommodityBook("test", REF)
        book.add(_trade(direction=1), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0)
        positions = book.positions_by_commodity()
        assert len(positions) == 1
        p = positions[0]
        assert p.commodity == "WTI"
        assert p.sector == "energy"
        assert p.net_quantity == pytest.approx(10_000)
        assert p.long_quantity == pytest.approx(10_000)
        assert p.short_quantity == pytest.approx(0)
        assert p.net_notional == pytest.approx(720_000)
        assert p.long_notional == pytest.approx(720_000)
        assert p.trade_count == 1

    def test_long_short_netting(self):
        book = CommodityBook("test", REF)
        book.add(_trade(direction=1), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0)
        book.add(_trade(direction=-1), commodity="WTI", sector="energy",
                 quantity=4_000, reference_price=72.0)
        positions = book.positions_by_commodity()
        assert len(positions) == 1
        p = positions[0]
        assert p.net_quantity == pytest.approx(6_000)
        assert p.long_quantity == pytest.approx(10_000)
        assert p.short_quantity == pytest.approx(4_000)
        assert p.net_notional == pytest.approx(6_000 * 72.0)
        assert p.trade_count == 2

    def test_multiple_commodities(self):
        book = CommodityBook("test", REF)
        book.add(_trade(), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0)
        book.add(_trade(), commodity="Brent", sector="energy",
                 quantity=8_000, reference_price=78.0)
        positions = book.positions_by_commodity()
        assert len(positions) == 2
        assert {p.commodity for p in positions} == {"WTI", "Brent"}

    def test_notional_scale_applied(self):
        book = CommodityBook("test", REF)
        book.add(_trade(scale=2.0), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0)
        positions = book.positions_by_commodity()
        # 10_000 × 72 × 2.0 scale
        assert positions[0].net_notional == pytest.approx(1_440_000)


class TestExposuresBySector:
    def test_single_sector(self):
        book = CommodityBook("test", REF)
        book.add(_trade(), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0)
        book.add(_trade(), commodity="Brent", sector="energy",
                 quantity=10_000, reference_price=78.0)
        exposures = book.exposures_by_sector()
        assert len(exposures) == 1
        assert exposures[0].sector == "energy"
        assert exposures[0].net_notional == pytest.approx(720_000 + 780_000)
        assert exposures[0].n_commodities == 2

    def test_multiple_sectors(self):
        book = CommodityBook("test", REF)
        book.add(_trade(), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0)
        book.add(_trade(), commodity="GOLD", sector="metals",
                 quantity=500, reference_price=2_000.0)
        book.add(_trade(), commodity="WHEAT", sector="agri",
                 quantity=20_000, reference_price=6.0)
        exposures = book.exposures_by_sector()
        assert len(exposures) == 3
        assert {e.sector for e in exposures} == {"energy", "metals", "agri"}

    def test_long_short_within_sector(self):
        book = CommodityBook("test", REF)
        book.add(_trade(direction=1), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0)
        book.add(_trade(direction=-1), commodity="Brent", sector="energy",
                 quantity=5_000, reference_price=78.0)
        exposures = book.exposures_by_sector()
        assert len(exposures) == 1
        assert exposures[0].long_notional == pytest.approx(720_000)
        assert exposures[0].short_notional == pytest.approx(390_000)
        assert exposures[0].net_notional == pytest.approx(720_000 - 390_000)


class TestExposuresByTenor:
    def test_buckets_by_delivery(self):
        book = CommodityBook("test", REF)
        book.add(_trade(), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0,
                 delivery_date=date(2024, 2, 1))  # front
        book.add(_trade(), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0,
                 delivery_date=date(2024, 7, 1))  # ≤6M
        book.add(_trade(), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0,
                 delivery_date=date(2025, 7, 15))  # ≤2Y
        exposures = book.exposures_by_tenor()
        labels = {e.bucket_label for e in exposures}
        assert "front" in labels
        assert "≤6M" in labels
        assert "≤2Y" in labels

    def test_aggregates_within_bucket(self):
        book = CommodityBook("test", REF)
        book.add(_trade(), commodity="WTI", sector="energy",
                 quantity=5_000, reference_price=72.0,
                 delivery_date=date(2024, 2, 1))
        book.add(_trade(), commodity="WTI", sector="energy",
                 quantity=3_000, reference_price=72.0,
                 delivery_date=date(2024, 2, 10))
        exposures = book.exposures_by_tenor()
        front = next(e for e in exposures if e.bucket_label == "front")
        assert front.net_notional == pytest.approx(8_000 * 72.0)
        assert front.n_positions == 2


class TestNetGrossNotional:
    def test_net_with_offsetting(self):
        book = CommodityBook("test", REF)
        book.add(_trade(direction=1), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0)
        book.add(_trade(direction=-1), commodity="Brent", sector="energy",
                 quantity=4_000, reference_price=72.0)
        # 720_000 − 288_000 = 432_000
        assert book.net_notional() == pytest.approx(432_000)

    def test_gross(self):
        book = CommodityBook("test", REF)
        book.add(_trade(direction=1), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0)
        book.add(_trade(direction=-1), commodity="Brent", sector="energy",
                 quantity=4_000, reference_price=72.0)
        # |720K| + |288K|
        assert book.gross_notional() == pytest.approx(1_008_000)


# ---- Step 2: limits ----

class TestCommodityLimits:
    def test_per_commodity_breach(self):
        limits = CommodityLimits(
            max_notional_per_commodity={"WTI": 500_000},
        )
        book = CommodityBook("test", REF, limits=limits)
        book.add(_trade(), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0)
        breaches = book.check_limits()
        assert len(breaches) == 1
        assert breaches[0].limit_type == "per_commodity"
        assert breaches[0].limit_name == "WTI"

    def test_per_commodity_ok(self):
        limits = CommodityLimits(
            max_notional_per_commodity={"WTI": 1_000_000},
        )
        book = CommodityBook("test", REF, limits=limits)
        book.add(_trade(), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0)
        assert book.check_limits() == []

    def test_per_sector_breach(self):
        limits = CommodityLimits(
            max_notional_per_sector={"energy": 1_000_000},
        )
        book = CommodityBook("test", REF, limits=limits)
        book.add(_trade(), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0)
        book.add(_trade(), commodity="Brent", sector="energy",
                 quantity=10_000, reference_price=78.0)
        breaches = book.check_limits()
        assert any(b.limit_type == "per_sector" for b in breaches)

    def test_net_notional_breach(self):
        limits = CommodityLimits(max_net_notional=500_000)
        book = CommodityBook("test", REF, limits=limits)
        book.add(_trade(), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0)
        breaches = book.check_limits()
        assert any(b.limit_type == "net_notional" for b in breaches)

    def test_gross_notional_breach(self):
        limits = CommodityLimits(max_gross_notional=1_000_000)
        book = CommodityBook("test", REF, limits=limits)
        # Net = 0, gross = 2 × 720K = 1.44M
        book.add(_trade(direction=1), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0)
        book.add(_trade(direction=-1), commodity="Brent", sector="energy",
                 quantity=10_000, reference_price=72.0)
        breaches = book.check_limits()
        assert any(b.limit_type == "gross_notional" for b in breaches)

    def test_per_tenor_breach(self):
        limits = CommodityLimits(
            max_notional_per_tenor={"front": 500_000},
        )
        book = CommodityBook("test", REF, limits=limits)
        book.add(_trade(), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0,
                 delivery_date=date(2024, 2, 1))  # front
        breaches = book.check_limits()
        assert any(b.limit_type == "per_tenor" for b in breaches)

    def test_no_breaches(self):
        limits = CommodityLimits(
            max_net_notional=10_000_000,
            max_gross_notional=20_000_000,
            max_notional_per_commodity={"WTI": 5_000_000},
            max_notional_per_sector={"energy": 10_000_000},
        )
        book = CommodityBook("test", REF, limits=limits)
        book.add(_trade(), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0)
        assert book.check_limits() == []

    def test_breach_records_actual_value(self):
        limits = CommodityLimits(max_notional_per_commodity={"WTI": 500_000})
        book = CommodityBook("test", REF, limits=limits)
        book.add(_trade(), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0)
        breaches = book.check_limits()
        assert breaches[0].actual_value == pytest.approx(720_000)
        assert breaches[0].limit_value == pytest.approx(500_000)
