"""Tests for multi-currency PricingContext and FX-translated aggregation."""

import pytest
from datetime import date

from pricebook.discount_curve import DiscountCurve
from pricebook.equity_book import EquityBook
from pricebook.commodity_book import CommodityBook
from pricebook.pricing_context import PricingContext
from pricebook.swaption import Swaption
from pricebook.trade import Trade
from pricebook.vol_surface import FlatVol


REF = date(2024, 1, 15)


def _usd_curve():
    return DiscountCurve.flat(REF, 0.05)


def _eur_curve():
    return DiscountCurve.flat(REF, 0.03)


def _ctx_multiccy():
    return PricingContext(
        valuation_date=REF,
        discount_curve=_usd_curve(),
        discount_curves={"USD": _usd_curve(), "EUR": _eur_curve()},
        vol_surfaces={"ir": FlatVol(0.20)},
        fx_spots={("EUR", "USD"): 1.0850, ("GBP", "USD"): 1.2700},
        reporting_currency="USD",
    )


# ---- Step 1: multi-curve PricingContext ----

class TestMultiCurveContext:
    def test_backward_compat_discount_curve(self):
        """Old code using ctx.discount_curve still works."""
        ctx = PricingContext.simple(REF, rate=0.05)
        assert ctx.discount_curve is not None
        assert ctx.discount_curve.df(date(2025, 1, 15)) > 0

    def test_get_discount_curve_by_ccy(self):
        ctx = _ctx_multiccy()
        usd = ctx.get_discount_curve("USD")
        eur = ctx.get_discount_curve("EUR")
        assert usd is not None
        assert eur is not None
        # USD rate higher → lower DF
        assert usd.df(date(2025, 1, 15)) < eur.df(date(2025, 1, 15))

    def test_get_discount_curve_fallback_to_default(self):
        ctx = PricingContext(
            valuation_date=REF,
            discount_curve=_usd_curve(),
        )
        # No discount_curves dict, but discount_curve exists
        curve = ctx.get_discount_curve("JPY")
        assert curve is ctx.discount_curve

    def test_get_discount_curve_none_ccy_uses_default(self):
        ctx = _ctx_multiccy()
        curve = ctx.get_discount_curve(None)
        assert curve is ctx.discount_curve

    def test_get_discount_curve_missing_raises(self):
        ctx = PricingContext(valuation_date=REF)
        with pytest.raises(KeyError):
            ctx.get_discount_curve("USD")

    def test_inflation_curves(self):
        # Use a dummy object as inflation curve
        dummy_cpi = {"type": "cpi", "base": 300.0}
        ctx = PricingContext(
            valuation_date=REF,
            inflation_curves={"USD": dummy_cpi},
        )
        assert ctx.get_inflation_curve("USD") is dummy_cpi
        with pytest.raises(KeyError):
            ctx.get_inflation_curve("EUR")

    def test_repo_curves(self):
        repo = DiscountCurve.flat(REF, 0.045)
        ctx = PricingContext(
            valuation_date=REF,
            repo_curves={"USD": repo},
        )
        assert ctx.get_repo_curve("USD") is repo
        with pytest.raises(KeyError):
            ctx.get_repo_curve("EUR")

    def test_reporting_currency_default(self):
        ctx = PricingContext(valuation_date=REF)
        assert ctx.reporting_currency == "USD"

    def test_reporting_currency_custom(self):
        ctx = PricingContext(valuation_date=REF, reporting_currency="EUR")
        assert ctx.reporting_currency == "EUR"

    def test_replace_preserves_new_fields(self):
        ctx = _ctx_multiccy()
        ctx2 = ctx.replace(reporting_currency="EUR")
        assert ctx2.reporting_currency == "EUR"
        assert ctx2.discount_curves == ctx.discount_curves
        assert ctx2.inflation_curves == ctx.inflation_curves
        assert ctx2.repo_curves == ctx.repo_curves

    def test_replace_backward_compat(self):
        ctx = PricingContext.simple(REF, rate=0.05, vol=0.20)
        ctx2 = ctx.replace(discount_curve=DiscountCurve.flat(REF, 0.06))
        assert ctx2.discount_curve is not ctx.discount_curve


# ---- FX translation ----

class TestFXTranslation:
    def test_same_currency_rate_one(self):
        ctx = _ctx_multiccy()
        assert ctx.fx_rate("USD", "USD") == 1.0

    def test_direct_lookup(self):
        ctx = _ctx_multiccy()
        assert ctx.fx_rate("EUR", "USD") == pytest.approx(1.0850)

    def test_inverse_lookup(self):
        ctx = _ctx_multiccy()
        assert ctx.fx_rate("USD", "EUR") == pytest.approx(1.0 / 1.0850)

    def test_missing_pair_raises(self):
        ctx = _ctx_multiccy()
        with pytest.raises(KeyError):
            ctx.fx_rate("JPY", "USD")

    def test_fx_translate_to_reporting(self):
        ctx = _ctx_multiccy()
        # 1M EUR → USD at 1.085
        result = ctx.fx_translate(1_000_000, "EUR")
        assert result == pytest.approx(1_085_000)

    def test_fx_translate_same_ccy(self):
        ctx = _ctx_multiccy()
        assert ctx.fx_translate(1_000_000, "USD") == pytest.approx(1_000_000)

    def test_fx_translate_explicit_target(self):
        ctx = _ctx_multiccy()
        # 1M GBP → USD at 1.27
        result = ctx.fx_translate(1_000_000, "GBP", "USD")
        assert result == pytest.approx(1_270_000)


# ---- Step 2: FX-translated book aggregation ----

class TestMultiCurrencyBookPV:
    def _swaption(self):
        return Swaption(date(2025, 1, 15), date(2030, 1, 15),
                        strike=0.05, notional=1_000_000)

    def _trade(self, trade_id="t"):
        return Trade(self._swaption(), trade_id=trade_id)

    def test_equity_book_fx_translated_pv(self):
        """Multi-ccy equity book: USD + EUR trades aggregated in USD."""
        ctx = _ctx_multiccy()
        book = EquityBook("test", currency="USD")

        book.add(self._trade(trade_id="usd_t"), ticker="AAPL",
                 currency="USD")
        book.add(self._trade(trade_id="eur_t"), ticker="SAP",
                 currency="EUR")

        # Single-ccy PV (no translation — existing behavior)
        raw_pv = book.pv(ctx)

        # FX-translated: each trade PV × fx_rate(trade_ccy, reporting_ccy)
        translated = 0.0
        for entry in book.entries:
            inst = entry.trade.instrument
            if hasattr(inst, "pv_ctx"):
                trade_pv = entry.trade.pv(ctx)
                translated += ctx.fx_translate(trade_pv, entry.currency)

        # Raw PV sums without translation (both priced on same USD curve)
        # so raw = translated only when all trades are USD
        # Here both are priced on the same USD curve, but one is "EUR"
        # so the translated EUR leg gets ×1.085
        assert translated != pytest.approx(raw_pv)
        # The EUR leg should be scaled up
        assert translated > raw_pv

    def test_commodity_book_fx_translated_notional(self):
        """Commodity book with mixed currencies."""
        ctx = _ctx_multiccy()

        book = CommodityBook("test", REF)
        book.add(self._trade(), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0, currency="USD")
        book.add(self._trade(), commodity="Brent", sector="energy",
                 quantity=10_000, reference_price=78.0, currency="EUR")

        # Translate notionals to USD
        usd_notional = 10_000 * 72.0  # already USD
        eur_notional = 10_000 * 78.0  # in EUR
        total_usd = usd_notional + ctx.fx_translate(eur_notional, "EUR")
        expected = 720_000 + 780_000 * 1.085
        assert total_usd == pytest.approx(expected)

    def test_fx_translate_round_trip(self):
        """USD → EUR → USD returns the original amount."""
        ctx = _ctx_multiccy()
        original = 1_000_000.0
        in_eur = ctx.fx_translate(original, "USD", "EUR")
        back = ctx.fx_translate(in_eur, "EUR", "USD")
        assert back == pytest.approx(original)
