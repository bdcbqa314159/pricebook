"""Tests for bond book."""

import pytest
from datetime import date

from pricebook.bond_book import (
    BondBook,
    BondLimits,
    BondPosition,
    BondSectorExposure,
    BondTenorBucket,
    bond_tenor_bucket,
)
from pricebook.swaption import Swaption
from pricebook.trade import Trade


REF = date(2024, 1, 15)


def _instr():
    return Swaption(date(2025, 1, 15), date(2030, 1, 15),
                    strike=0.05, notional=1_000_000)


def _trade(direction=1, scale=1.0, trade_id="t"):
    return Trade(_instr(), direction=direction, notional_scale=scale, trade_id=trade_id)


# ---- Tenor bucket ----

class TestBondTenorBucket:
    def test_unknown(self):
        assert bond_tenor_bucket(REF, None) == "unknown"

    def test_short(self):
        assert bond_tenor_bucket(REF, date(2024, 7, 1)) == "≤1Y"

    def test_two_year(self):
        assert bond_tenor_bucket(REF, date(2026, 1, 1)) == "1-2Y"

    def test_five_year(self):
        assert bond_tenor_bucket(REF, date(2029, 1, 1)) == "3-5Y"

    def test_ten_year(self):
        assert bond_tenor_bucket(REF, date(2034, 1, 1)) == "7-10Y"

    def test_thirty_year(self):
        assert bond_tenor_bucket(REF, date(2054, 1, 1)) == "20-30Y"

    def test_ultra_long(self):
        assert bond_tenor_bucket(REF, date(2060, 1, 1)) == "30Y+"


# ---- Step 1: book + aggregation ----

class TestBondBook:
    def test_create_empty(self):
        book = BondBook("test", REF)
        assert len(book) == 0
        assert book.n_issuers == 0
        assert book.n_sectors == 0

    def test_add_trade(self):
        book = BondBook("test", REF)
        book.add(_trade(), issuer="UST", sector="govt",
                 face_amount=10_000_000, dirty_price=98.5,
                 maturity=date(2034, 1, 15), dv01_per_million=85.0,
                 duration=8.2)
        assert len(book) == 1
        assert book.n_issuers == 1


class TestPositionsByIssuer:
    def test_single_long(self):
        book = BondBook("test", REF)
        book.add(_trade(), issuer="UST", sector="govt",
                 face_amount=10_000_000, dirty_price=98.5,
                 dv01_per_million=85.0, duration=8.2)
        positions = book.positions_by_issuer()
        assert len(positions) == 1
        p = positions[0]
        assert p.issuer == "UST"
        assert p.net_face == pytest.approx(10_000_000)
        assert p.long_face == pytest.approx(10_000_000)
        assert p.short_face == pytest.approx(0)
        assert p.net_market_value == pytest.approx(10_000_000 * 98.5 / 100)
        assert p.net_dv01 == pytest.approx(10_000_000 * 85.0 / 1_000_000)
        assert p.weighted_duration == pytest.approx(8.2)

    def test_long_short_netting(self):
        book = BondBook("test", REF)
        book.add(_trade(direction=1), issuer="UST", sector="govt",
                 face_amount=10_000_000, dirty_price=98.5,
                 dv01_per_million=85.0, duration=8.2)
        book.add(_trade(direction=-1), issuer="UST", sector="govt",
                 face_amount=4_000_000, dirty_price=98.5,
                 dv01_per_million=85.0, duration=8.2)
        positions = book.positions_by_issuer()
        assert positions[0].net_face == pytest.approx(6_000_000)
        assert positions[0].long_face == pytest.approx(10_000_000)
        assert positions[0].short_face == pytest.approx(4_000_000)
        assert positions[0].trade_count == 2

    def test_multiple_issuers(self):
        book = BondBook("test", REF)
        book.add(_trade(), issuer="UST", sector="govt",
                 face_amount=10_000_000)
        book.add(_trade(), issuer="DBR", sector="govt",
                 face_amount=5_000_000)
        positions = book.positions_by_issuer()
        assert len(positions) == 2
        assert {p.issuer for p in positions} == {"UST", "DBR"}


class TestPositionsBySector:
    def test_multiple_sectors(self):
        book = BondBook("test", REF)
        book.add(_trade(), issuer="UST", sector="govt", face_amount=10_000_000)
        book.add(_trade(), issuer="AAPL", sector="ig", face_amount=5_000_000)
        book.add(_trade(), issuer="HY_CO", sector="hy", face_amount=3_000_000)
        exposures = book.positions_by_sector()
        assert len(exposures) == 3
        assert {e.sector for e in exposures} == {"govt", "ig", "hy"}

    def test_sector_aggregates_issuers(self):
        book = BondBook("test", REF)
        book.add(_trade(), issuer="UST", sector="govt", face_amount=10_000_000)
        book.add(_trade(), issuer="DBR", sector="govt", face_amount=5_000_000)
        exposures = book.positions_by_sector()
        assert len(exposures) == 1
        assert exposures[0].net_face == pytest.approx(15_000_000)
        assert exposures[0].n_issuers == 2


class TestPositionsByTenor:
    def test_buckets(self):
        book = BondBook("test", REF)
        book.add(_trade(), issuer="UST", face_amount=10_000_000,
                 maturity=date(2025, 7, 15))  # ~1.5Y → 1-2Y
        book.add(_trade(), issuer="UST", face_amount=5_000_000,
                 maturity=date(2032, 1, 15))  # ~8Y → 7-10Y
        buckets = book.positions_by_tenor()
        labels = {b.bucket_label for b in buckets}
        assert "1-2Y" in labels
        assert "7-10Y" in labels


class TestBookAggregates:
    def test_net_dv01(self):
        book = BondBook("test", REF)
        book.add(_trade(direction=1), issuer="UST",
                 face_amount=10_000_000, dv01_per_million=85.0)
        book.add(_trade(direction=-1), issuer="DBR",
                 face_amount=5_000_000, dv01_per_million=60.0)
        # 10M × 85/1M - 5M × 60/1M = 850 - 300 = 550
        assert book.net_dv01() == pytest.approx(550.0)

    def test_net_market_value(self):
        book = BondBook("test", REF)
        book.add(_trade(), issuer="UST", face_amount=10_000_000,
                 dirty_price=98.5)
        assert book.net_market_value() == pytest.approx(9_850_000)

    def test_weighted_duration(self):
        book = BondBook("test", REF)
        book.add(_trade(), issuer="UST", face_amount=10_000_000,
                 dirty_price=100.0, duration=5.0)
        book.add(_trade(), issuer="DBR", face_amount=10_000_000,
                 dirty_price=100.0, duration=10.0)
        # Equal market values → duration = (5+10)/2 = 7.5
        assert book.weighted_duration() == pytest.approx(7.5)


# ---- Step 2: limits ----

class TestBondLimits:
    def test_per_issuer_breach(self):
        limits = BondLimits(max_face_per_issuer={"UST": 5_000_000})
        book = BondBook("test", REF, limits=limits)
        book.add(_trade(), issuer="UST", face_amount=10_000_000)
        breaches = book.check_limits()
        assert len(breaches) == 1
        assert breaches[0].limit_type == "per_issuer"

    def test_per_issuer_ok(self):
        limits = BondLimits(max_face_per_issuer={"UST": 20_000_000})
        book = BondBook("test", REF, limits=limits)
        book.add(_trade(), issuer="UST", face_amount=10_000_000)
        assert book.check_limits() == []

    def test_per_sector_breach(self):
        limits = BondLimits(max_face_per_sector={"govt": 12_000_000})
        book = BondBook("test", REF, limits=limits)
        book.add(_trade(), issuer="UST", sector="govt", face_amount=10_000_000)
        book.add(_trade(), issuer="DBR", sector="govt", face_amount=5_000_000)
        breaches = book.check_limits()
        assert any(b.limit_type == "per_sector" for b in breaches)

    def test_dv01_breach(self):
        limits = BondLimits(max_dv01=500.0)
        book = BondBook("test", REF, limits=limits)
        book.add(_trade(), issuer="UST", face_amount=10_000_000,
                 dv01_per_million=85.0)
        # DV01 = 850 > 500
        breaches = book.check_limits()
        assert any(b.limit_type == "dv01" for b in breaches)

    def test_per_tenor_dv01_breach(self):
        limits = BondLimits(max_dv01_per_tenor={"7-10Y": 400.0})
        book = BondBook("test", REF, limits=limits)
        book.add(_trade(), issuer="UST", face_amount=10_000_000,
                 dv01_per_million=85.0, maturity=date(2032, 1, 15))
        breaches = book.check_limits()
        assert any(b.limit_type == "per_tenor_dv01" for b in breaches)

    def test_duration_breach(self):
        limits = BondLimits(max_duration=6.0)
        book = BondBook("test", REF, limits=limits)
        book.add(_trade(), issuer="UST", face_amount=10_000_000,
                 dirty_price=100.0, duration=8.2)
        breaches = book.check_limits()
        assert any(b.limit_type == "duration" for b in breaches)

    def test_no_breaches(self):
        limits = BondLimits(
            max_face_per_issuer={"UST": 20_000_000},
            max_dv01=1_000, max_duration=10.0,
        )
        book = BondBook("test", REF, limits=limits)
        book.add(_trade(), issuer="UST", face_amount=10_000_000,
                 dv01_per_million=85.0, dirty_price=100.0, duration=8.2)
        assert book.check_limits() == []
