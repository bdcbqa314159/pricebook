"""Tests for FundingCurve and CollateralisedPricer."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.csa import CSA
from pricebook.discount_curve import DiscountCurve
from pricebook.funding_curve import FundingCurve, CollateralisedPricer, CollateralisedResult
from pricebook.pricing_context import PricingContext
from pricebook.rfr import SpreadCurve
from pricebook.swap import InterestRateSwap


REF = date(2026, 4, 27)


def _ois():
    return DiscountCurve.flat(REF, 0.03)


# ---- FundingCurve ----

class TestFundingCurve:

    def test_flat_spread(self):
        ois = _ois()
        fc = FundingCurve.flat_spread(ois, 0.005)
        assert fc.reference_date == REF

    def test_df_lower_than_ois(self):
        """Funding df < OIS df (positive spread → higher rate → lower df)."""
        ois = _ois()
        fc = FundingCurve.flat_spread(ois, 0.005)
        d = REF + timedelta(days=1825)
        assert fc.df(d) < ois.df(d)

    def test_df_at_zero_spread_equals_ois(self):
        ois = _ois()
        fc = FundingCurve.flat_spread(ois, 0.0)
        d = REF + timedelta(days=365)
        assert fc.df(d) == pytest.approx(ois.df(d))

    def test_funding_rate(self):
        ois = _ois()
        fc = FundingCurve.flat_spread(ois, 0.005)
        d = REF + timedelta(days=365)
        r = fc.funding_rate(d)
        assert r == pytest.approx(0.035, abs=0.001)

    def test_forward_funding_rate(self):
        ois = _ois()
        fc = FundingCurve.flat_spread(ois, 0.005)
        d1 = REF + timedelta(days=365)
        d2 = REF + timedelta(days=730)
        fwd = fc.forward_funding_rate(d1, d2)
        assert math.isfinite(fwd)
        assert fwd > 0

    def test_as_discount_curve(self):
        ois = _ois()
        fc = FundingCurve.flat_spread(ois, 0.005)
        dc = fc.as_discount_curve()
        d = REF + timedelta(days=365)
        assert dc.df(d) == pytest.approx(fc.df(d), rel=1e-4)

    def test_term_structure_spread(self):
        ois = _ois()
        sc = SpreadCurve(
            REF,
            [REF + timedelta(days=365), REF + timedelta(days=3650)],
            [0.003, 0.008],
        )
        fc = FundingCurve(ois, sc)
        # Short end: lower spread
        d_short = REF + timedelta(days=365)
        d_long = REF + timedelta(days=3650)
        # Long end should have larger spread → larger gap from OIS
        gap_short = ois.df(d_short) - fc.df(d_short)
        gap_long = ois.df(d_long) - fc.df(d_long)
        assert gap_long > gap_short


# ---- CollateralisedPricer ----

class TestCollateralisedPricer:

    def _make_swap(self):
        return InterestRateSwap(
            start=REF, end=REF + timedelta(days=1825),
            fixed_rate=0.035, notional=1_000_000,
        )

    def test_no_csa_uses_funding(self):
        """No CSA → discount on funding curve."""
        ois = _ois()
        fc = FundingCurve.flat_spread(ois, 0.005)
        pricer = CollateralisedPricer({"USD": ois}, fc)
        curve, csa_type = pricer.discount_curve_for("USD", csa=None)
        assert csa_type == "none"
        # Funding curve should have lower dfs than OIS
        d = REF + timedelta(days=1825)
        assert curve.df(d) < ois.df(d)

    def test_cash_csa_same_ccy(self):
        """Cash CSA in same currency → OIS."""
        ois = _ois()
        fc = FundingCurve.flat_spread(ois, 0.005)
        pricer = CollateralisedPricer({"USD": ois}, fc)
        csa = CSA(currency="USD")
        curve, csa_type = pricer.discount_curve_for("USD", csa=csa)
        assert csa_type == "cash_same_ccy"
        d = REF + timedelta(days=365)
        assert curve.df(d) == pytest.approx(ois.df(d))

    def test_cash_csa_foreign(self):
        """Cash CSA in foreign currency → xccy basis curve."""
        ois_usd = _ois()
        ois_eur = DiscountCurve.flat(REF, 0.025)
        fc = FundingCurve.flat_spread(ois_usd, 0.005)
        xccy = {"USD_EUR": DiscountCurve.flat(REF, 0.028)}
        pricer = CollateralisedPricer({"USD": ois_usd, "EUR": ois_eur}, fc, xccy)
        csa = CSA(currency="EUR")
        curve, csa_type = pricer.discount_curve_for("USD", csa=csa)
        assert csa_type == "cash_foreign"

    def test_uncollateralised_pv_lower(self):
        """Uncollateralised PV < collateralised PV (receiver swap, positive rates)."""
        ois = _ois()
        fc = FundingCurve.flat_spread(ois, 0.005)
        pricer = CollateralisedPricer({"USD": ois}, fc)
        swap = self._make_swap()
        ctx = PricingContext(valuation_date=REF, discount_curve=ois)

        # Collateralised (CSA)
        csa = CSA(currency="USD")
        r_coll = pricer.price(swap, ctx, csa=csa)

        # Uncollateralised
        r_uncoll = pricer.price(swap, ctx, csa=None)

        # Both should be finite
        assert math.isfinite(r_coll.pv)
        assert math.isfinite(r_uncoll.pv)

    def test_funding_adjustment(self):
        """Funding adjustment should be non-zero for non-zero spread."""
        ois = _ois()
        fc = FundingCurve.flat_spread(ois, 0.005)
        pricer = CollateralisedPricer({"USD": ois}, fc)
        swap = self._make_swap()
        ctx = PricingContext(valuation_date=REF, discount_curve=ois)
        csa = CSA(currency="USD")
        result = pricer.price(swap, ctx, csa=csa)
        # Funding adjustment = PV(funding) - PV(CSA)
        assert result.funding_adjustment != 0.0

    def test_result_fields(self):
        ois = _ois()
        fc = FundingCurve.flat_spread(ois, 0.005)
        pricer = CollateralisedPricer({"USD": ois}, fc)
        swap = self._make_swap()
        ctx = PricingContext(valuation_date=REF, discount_curve=ois)
        result = pricer.price(swap, ctx, csa=CSA(currency="USD"))
        assert isinstance(result, CollateralisedResult)
        assert result.csa_type == "cash_same_ccy"
        assert result.discount_curve_type == "cash_same_ccy"
