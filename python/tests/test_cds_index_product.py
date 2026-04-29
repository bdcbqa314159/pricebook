"""Tests for CDSIndexProduct: pricing, basis, roll, serialisation."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta

import pytest

from pricebook.cds_index_product import CDSIndexProduct, IndexResult
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.serialisable import from_dict

REF = date(2026, 4, 28)


def _disc():
    return DiscountCurve.flat(REF, 0.03)


def _survival_curves(n=5, base_hazard=0.02):
    """Create n survival curves with slightly different hazards."""
    return [SurvivalCurve.flat(REF, base_hazard + 0.002 * i) for i in range(n)]


class TestCDSIndexProduct:

    def test_from_spec(self):
        p = CDSIndexProduct.from_spec("CDX.NA.IG", series=42,
                                       market_spread=0.005,
                                       reference_date=REF)
        assert p.index_name == "CDX.NA.IG"
        assert p.n_names == 125
        assert p.standard_coupon == 0.01
        assert p.recovery == 0.4

    def test_price(self):
        p = CDSIndexProduct(index_name="test", n_names=5,
                             market_spread=0.005, notional=5_000_000,
                             start=REF, end=REF + timedelta(days=1825))
        scs = _survival_curves(5)
        r = p.price(_disc(), scs)
        assert math.isfinite(r.pv)
        assert r.n_constituents == 5
        assert r.intrinsic_spread > 0
        assert math.isfinite(r.index_basis)

    def test_intrinsic_spread(self):
        p = CDSIndexProduct(index_name="test", n_names=5,
                             market_spread=0.005, start=REF,
                             end=REF + timedelta(days=1825))
        scs = _survival_curves(5)
        intr = p.intrinsic_spread(_disc(), scs)
        assert intr > 0

    def test_wrong_n_curves_raises(self):
        p = CDSIndexProduct(index_name="test", n_names=5,
                             market_spread=0.005, start=REF,
                             end=REF + timedelta(days=1825))
        with pytest.raises(ValueError, match="Expected 5"):
            p.price(_disc(), _survival_curves(3))

    def test_cheapest_to_protect(self):
        p = CDSIndexProduct(index_name="test", n_names=5,
                             market_spread=0.005, start=REF,
                             end=REF + timedelta(days=1825))
        scs = _survival_curves(5)
        names = ["A", "B", "C", "D", "E"]
        ctp = p.cheapest_to_protect(_disc(), scs, names)
        assert "widest_name" in ctp
        assert "tightest_name" in ctp
        assert ctp["widest_spread_bp"] > ctp["tightest_spread_bp"]
        # Last curve has highest hazard → widest spread
        assert ctp["widest_name"] == "E"

    def test_next_roll(self):
        p = CDSIndexProduct.from_spec("CDX.NA.IG", reference_date=REF)
        roll = p.next_roll_date(REF)
        assert roll > REF

    def test_result_dict(self):
        p = CDSIndexProduct(index_name="test", n_names=3,
                             market_spread=0.005, start=REF,
                             end=REF + timedelta(days=1825))
        r = p.price(_disc(), _survival_curves(3))
        d = r.to_dict()
        assert "index_basis" in d
        assert "intrinsic_spread" in d


class TestCDSIndexSerialisation:

    def test_round_trip(self):
        p = CDSIndexProduct(index_name="CDX.NA.IG", series=42,
                             market_spread=0.006, notional=50_000_000,
                             start=REF, end=REF + timedelta(days=1825))
        d = p.to_dict()
        assert d["type"] == "cds_index_product"
        p2 = from_dict(d)
        assert p2.index_name == "CDX.NA.IG"
        assert p2.series == 42
        assert p2.market_spread == 0.006

    def test_json(self):
        p = CDSIndexProduct(index_name="iTraxx Europe", market_spread=0.004)
        s = json.dumps(p.to_dict())
        p2 = from_dict(json.loads(s))
        assert p2.index_name == "iTraxx Europe"

    def test_from_spec_serial(self):
        p = CDSIndexProduct.from_spec("CDX.NA.IG", reference_date=REF)
        d = p.to_dict()
        p2 = from_dict(d)
        assert p2.standard_coupon == p.standard_coupon
        assert p2.n_names == 125
