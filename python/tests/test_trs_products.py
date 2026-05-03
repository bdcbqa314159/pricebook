"""Tests for TRS product types: commodity, FX, cross-currency bond."""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.trs import (
    TotalReturnSwap, FundingLegSpec,
    CommodityUnderlying, FXUnderlying, XccySpec,
)
from pricebook.bond import FixedRateBond
from tests.conftest import make_flat_curve


REF = date(2024, 7, 15)
END = REF + relativedelta(years=1)


# ── Commodity TRS ──

class TestCommodityTRS:

    def _make(self, spot=75.0, storage=0.02, cy=0.01, spread=0.005):
        return TotalReturnSwap(
            underlying=CommodityUnderlying("WTI", spot, storage, cy),
            notional=100_000,
            start=REF, end=END,
            funding=FundingLegSpec(spread=spread),
            initial_price=spot,
        )

    def test_type_detection(self):
        trs = self._make()
        assert trs._underlying_type == "commodity"

    def test_price_finite(self):
        trs = self._make()
        curve = make_flat_curve(REF, 0.04)
        r = trs.price(curve)
        assert math.isfinite(r.value)

    def test_forward_increases_with_storage(self):
        """Higher storage cost → higher forward → higher performance."""
        curve = make_flat_curve(REF, 0.04)
        trs_low = self._make(storage=0.01, cy=0.0)
        trs_high = self._make(storage=0.05, cy=0.0)
        pv_low = trs_low.price(curve).price_return
        pv_high = trs_high.price(curve).price_return
        assert pv_high > pv_low

    def test_forward_decreases_with_convenience_yield(self):
        """Higher convenience yield → lower forward."""
        curve = make_flat_curve(REF, 0.04)
        trs_low = self._make(cy=0.01, storage=0.0)
        trs_high = self._make(cy=0.05, storage=0.0)
        pv_low = trs_low.price(curve).price_return
        pv_high = trs_high.price(curve).price_return
        assert pv_low > pv_high

    def test_forward_hand_calc(self):
        """Forward = spot x exp((r - cy + sc) x T)."""
        spot = 75.0
        r = 0.04
        sc = 0.02
        cy = 0.01
        T = 1.0  # ~1 year
        expected_fwd = spot * math.exp((r - cy + sc) * T)

        trs = self._make(spot=spot, storage=sc, cy=cy)
        curve = make_flat_curve(REF, r)
        result = trs.price(curve)
        # price_return = (forward - initial) * quantity
        quantity = 100_000 / spot
        expected_return = (expected_fwd - spot) * quantity
        assert abs(result.price_return - expected_return) / abs(expected_return) < 0.01

    def test_greeks_delta(self):
        trs = self._make()
        curve = make_flat_curve(REF, 0.04)
        g = trs.greeks(curve)
        assert g["delta"] != 0

    def test_income_zero(self):
        """Commodities have no income (no coupons/dividends)."""
        trs = self._make()
        curve = make_flat_curve(REF, 0.04)
        r = trs.price(curve)
        assert r.income_return == 0.0


# ── FX TRS ──

class TestFXTRS:

    def _make(self, spot=1.08, spread=0.005):
        return TotalReturnSwap(
            underlying=FXUnderlying("EUR", "USD", spot),
            notional=10_000_000,
            start=REF, end=END,
            funding=FundingLegSpec(spread=spread),
            initial_price=spot,
        )

    def test_type_detection(self):
        trs = self._make()
        assert trs._underlying_type == "fx"

    def test_price_finite(self):
        trs = self._make()
        usd_curve = make_flat_curve(REF, 0.05)
        eur_curve = make_flat_curve(REF, 0.03)
        r = trs.price(usd_curve, eur_curve)
        assert math.isfinite(r.value)

    def test_forward_interest_rate_parity(self):
        """FX forward = spot x df_base / df_quote.
        Higher USD rates → USD depreciates → EUR/USD forward > spot."""
        trs = self._make(spot=1.08)
        usd_curve = make_flat_curve(REF, 0.05)
        eur_curve = make_flat_curve(REF, 0.03)
        r = trs.price(usd_curve, eur_curve)
        # USD rates > EUR rates → EUR/USD forward > spot → positive performance
        assert r.price_return > 0

    def test_same_rates_forward_equals_spot(self):
        """If both currencies have same rate, forward ≈ spot."""
        trs = self._make(spot=1.08)
        curve = make_flat_curve(REF, 0.04)
        r = trs.price(curve, curve)
        # Forward ≈ spot, so price_return ≈ 0
        assert abs(r.price_return) < 1000  # small relative to 10M notional

    def test_greeks_delta(self):
        trs = self._make()
        usd_curve = make_flat_curve(REF, 0.05)
        eur_curve = make_flat_curve(REF, 0.03)
        g = trs.greeks(usd_curve, eur_curve)
        assert g["delta"] != 0

    def test_income_zero(self):
        trs = self._make()
        curve = make_flat_curve(REF, 0.04)
        r = trs.price(curve)
        assert r.income_return == 0.0


# ── Cross-currency Bond TRS ──

class TestXccyBondTRS:

    def _make(self, fx_rate=1.08, fx_haircut=0.08):
        bond = FixedRateBond.treasury_note(
            date(2024, 2, 15), date(2034, 2, 15), 0.04125)
        return TotalReturnSwap(
            underlying=bond, notional=10_000_000,
            start=REF, end=REF + relativedelta(months=6),
            funding=FundingLegSpec(spread=0.005),
            repo_spread=0.005, initial_price=102.0,
            xccy=XccySpec(
                fx_rate=fx_rate, asset_currency="EUR",
                funding_currency="USD", fx_haircut=fx_haircut),
        )

    def test_type_still_bond(self):
        trs = self._make()
        assert trs._underlying_type == "bond"

    def test_price_finite(self):
        trs = self._make()
        curve = make_flat_curve(REF, 0.04)
        r = trs.price(curve)
        assert math.isfinite(r.value)

    def test_xccy_different_from_domestic(self):
        """Xccy TRS value differs from domestic due to FX conversion + haircut."""
        bond = FixedRateBond.treasury_note(
            date(2024, 2, 15), date(2034, 2, 15), 0.04125)
        domestic = TotalReturnSwap(
            underlying=bond, notional=10_000_000,
            start=REF, end=REF + relativedelta(months=6),
            funding=FundingLegSpec(spread=0.005),
            repo_spread=0.005, initial_price=102.0,
        )
        xccy = self._make()
        curve = make_flat_curve(REF, 0.04)
        pv_dom = domestic.price(curve).value
        pv_xccy = xccy.price(curve).value
        assert pv_dom != pv_xccy

    def test_fx_haircut_reduces_value(self):
        """Higher FX haircut → additional cost → lower PV."""
        curve = make_flat_curve(REF, 0.04)
        trs_low = self._make(fx_haircut=0.0)
        trs_high = self._make(fx_haircut=0.10)
        pv_low = trs_low.price(curve).value
        pv_high = trs_high.price(curve).value
        assert pv_low > pv_high


# ── Dividend swap (already in dividend_desk, verify integration) ──

class TestDividendSwapExists:

    def test_import(self):
        from pricebook.dividend_desk import DividendSwap
        assert DividendSwap is not None

    def test_pv(self):
        from pricebook.dividend_desk import DividendSwap
        from pricebook.dividend_model import Dividend
        ds = DividendSwap(
            start=REF, end=END,
            fixed_div=2.0, notional=1000,
        )
        divs = [
            Dividend(amount=0.50, ex_date=REF + relativedelta(months=3)),
            Dividend(amount=0.55, ex_date=REF + relativedelta(months=6)),
            Dividend(amount=0.50, ex_date=REF + relativedelta(months=9)),
            Dividend(amount=0.55, ex_date=REF + relativedelta(months=12)),
        ]
        curve = make_flat_curve(REF, 0.04)
        pv = ds.pv(divs, curve)
        assert math.isfinite(pv)

    def test_fair_fixed(self):
        from pricebook.dividend_desk import DividendSwap
        from pricebook.dividend_model import Dividend
        ds = DividendSwap(start=REF, end=END, fixed_div=0.0, notional=1000)
        divs = [
            Dividend(amount=0.50, ex_date=REF + relativedelta(months=3)),
            Dividend(amount=0.50, ex_date=REF + relativedelta(months=9)),
        ]
        curve = make_flat_curve(REF, 0.04)
        fair = ds.fair_fixed(divs, curve)
        assert fair > 0
