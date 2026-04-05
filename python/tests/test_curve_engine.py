"""Tests for curve engine: CurveDefinition, CurveBuilder, CurveSet."""

import math
import pytest
from datetime import date

from pricebook.curve_engine import (
    CurveDefinition,
    CurveRole,
    CurveSet,
    InstrumentSpec,
    ExtrapolationPolicy,
    build_curve,
)
from pricebook.market_data import MarketDataSnapshot, Quote, QuoteType, MissingQuoteError


REF = date(2024, 1, 15)


def _usd_snapshot():
    snap = MarketDataSnapshot(REF)
    # Deposits
    for tenor, rate in [("1M", 0.053), ("3M", 0.052), ("6M", 0.051)]:
        snap.add(Quote(QuoteType.DEPOSIT_RATE, tenor, rate, "USD"))
    # Swaps
    for tenor, rate in [("1Y", 0.050), ("2Y", 0.048), ("3Y", 0.047),
                        ("5Y", 0.046), ("7Y", 0.045), ("10Y", 0.044),
                        ("15Y", 0.043), ("20Y", 0.042), ("30Y", 0.041)]:
        snap.add(Quote(QuoteType.SWAP_RATE, tenor, rate, "USD"))
    return snap


# ---------------------------------------------------------------------------
# Step 1: CurveDefinition
# ---------------------------------------------------------------------------


class TestCurveDefinition:
    def test_usd_ois_preset(self):
        defn = CurveDefinition.usd_ois()
        assert defn.name == "USD_OIS"
        assert defn.currency == "USD"
        assert defn.role == CurveRole.DISCOUNT
        assert len(defn.instruments) == 12

    def test_eur_estr_preset(self):
        defn = CurveDefinition.eur_estr()
        assert defn.name == "EUR_ESTR"
        assert defn.currency == "EUR"

    def test_roundtrip_dict(self):
        defn = CurveDefinition.usd_ois()
        d = defn.to_dict()
        defn2 = CurveDefinition.from_dict(d)
        assert defn2.name == defn.name
        assert defn2.currency == defn.currency
        assert len(defn2.instruments) == len(defn.instruments)

    def test_custom_definition(self):
        defn = CurveDefinition(
            name="TEST",
            instruments=[
                InstrumentSpec(QuoteType.DEPOSIT_RATE, "3M"),
                InstrumentSpec(QuoteType.SWAP_RATE, "5Y"),
            ],
        )
        assert len(defn.instruments) == 2


# ---------------------------------------------------------------------------
# Step 2: CurveBuilder
# ---------------------------------------------------------------------------


class TestBuildCurve:
    def test_builds_from_snapshot(self):
        defn = CurveDefinition.usd_ois()
        snap = _usd_snapshot()
        curve = build_curve(defn, snap)
        assert curve.reference_date == REF
        d5y = date.fromordinal(REF.toordinal() + int(5 * 365))
        df = curve.df(d5y)
        assert 0.7 < df < 0.85

    def test_dfs_reasonable(self):
        defn = CurveDefinition.usd_ois()
        snap = _usd_snapshot()
        curve = build_curve(defn, snap)
        # Short end DF near 1, long end smaller
        d1m = date.fromordinal(REF.toordinal() + 30)
        d30y = date.fromordinal(REF.toordinal() + int(30 * 365))
        assert curve.df(d1m) > 0.99
        assert curve.df(d30y) < 0.5

    def test_missing_quote_raises(self):
        defn = CurveDefinition(
            name="EMPTY",
            instruments=[InstrumentSpec(QuoteType.DEPOSIT_RATE, "99Y")],
        )
        snap = _usd_snapshot()
        with pytest.raises(MissingQuoteError):
            build_curve(defn, snap)

    def test_smith_wilson_extrapolation(self):
        defn = CurveDefinition(
            name="USD_SW",
            currency="USD",
            instruments=[
                InstrumentSpec(QuoteType.DEPOSIT_RATE, "3M"),
                InstrumentSpec(QuoteType.DEPOSIT_RATE, "6M"),
                InstrumentSpec(QuoteType.SWAP_RATE, "1Y"),
                InstrumentSpec(QuoteType.SWAP_RATE, "5Y"),
                InstrumentSpec(QuoteType.SWAP_RATE, "10Y"),
                InstrumentSpec(QuoteType.SWAP_RATE, "20Y"),
            ],
            extrapolation=ExtrapolationPolicy.SMITH_WILSON,
        )
        snap = _usd_snapshot()
        curve = build_curve(defn, snap)
        # Should have extrapolation beyond 20Y
        d50y = date.fromordinal(REF.toordinal() + int(50 * 365))
        assert curve.df(d50y) > 0

    def test_reprices_deposits(self):
        """Built curve should give reasonable forward rates."""
        defn = CurveDefinition.usd_ois()
        snap = _usd_snapshot()
        curve = build_curve(defn, snap)
        d3m = date.fromordinal(REF.toordinal() + 91)
        fwd = curve.forward_rate(REF, d3m)
        assert 0.04 < fwd < 0.06


# ---------------------------------------------------------------------------
# Step 3: CurveSet
# ---------------------------------------------------------------------------


class TestCurveSet:
    def test_build_from_definitions(self):
        ois_defn = CurveDefinition.usd_ois()
        proj_defn = CurveDefinition(
            name="USD_SOFR_3M",
            currency="USD",
            role=CurveRole.PROJECTION,
            instruments=[
                InstrumentSpec(QuoteType.DEPOSIT_RATE, "3M"),
                InstrumentSpec(QuoteType.DEPOSIT_RATE, "6M"),
                InstrumentSpec(QuoteType.SWAP_RATE, "1Y"),
                InstrumentSpec(QuoteType.SWAP_RATE, "2Y"),
                InstrumentSpec(QuoteType.SWAP_RATE, "5Y"),
                InstrumentSpec(QuoteType.SWAP_RATE, "10Y"),
            ],
        )
        snap = _usd_snapshot()
        cs = CurveSet.from_definitions("USD", "USD", [ois_defn, proj_defn], snap)

        assert cs.discount_curve is not None
        assert len(cs.projection_curves) == 1
        assert "USD_SOFR_3M" in cs.projection_curves

    def test_to_pricing_context(self):
        defn = CurveDefinition.usd_ois()
        snap = _usd_snapshot()
        cs = CurveSet.from_definitions("USD", "USD", [defn], snap)
        ctx = cs.to_pricing_context()

        assert ctx.valuation_date == REF
        assert ctx.discount_curve is not None

    def test_add_curve(self):
        from pricebook.discount_curve import DiscountCurve
        cs = CurveSet(name="TEST", currency="USD")
        curve = DiscountCurve.flat(REF, 0.05)
        cs.add("OIS", curve, CurveRole.DISCOUNT)
        assert cs.discount_curve is curve
        assert len(cs.curves) == 1

    def test_empty_projection(self):
        defn = CurveDefinition.usd_ois()
        snap = _usd_snapshot()
        cs = CurveSet.from_definitions("USD", "USD", [defn], snap)
        assert len(cs.projection_curves) == 0  # only discount, no projection
