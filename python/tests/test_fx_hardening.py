"""Tests for FX hardening (FXH1-FXH5)."""

from datetime import date

import pytest

from pricebook.currency import Currency, CurrencyPair
from pricebook.fx_forward import FXForward
from pricebook.fx_forward_curve import FXForwardCurve
from pricebook.fx_swap import FXSwap
from pricebook.ndf import NDF
from tests.conftest import make_flat_curve


# ---- FXH1: FXForward sensitivities ----

class TestFXForwardSensitivities:
    def test_fx_delta_positive_for_buyer(self):
        """Forward buyer benefits from spot increase → delta > 0."""
        ref = date(2026, 4, 21)
        base = make_flat_curve(ref, rate=0.03)
        quote = make_flat_curve(ref, rate=0.04)
        pair = CurrencyPair.from_currencies(Currency.EUR, Currency.USD)
        fwd = FXForward(pair, date(2027, 4, 21), 1.10)
        delta = fwd.fx_delta(1.10, base, quote)
        assert delta > 0

    def test_dv01_base(self):
        """Sensitivity to base currency rate shift."""
        ref = date(2026, 4, 21)
        base = make_flat_curve(ref, rate=0.03)
        quote = make_flat_curve(ref, rate=0.04)
        pair = CurrencyPair.from_currencies(Currency.EUR, Currency.USD)
        fwd = FXForward(pair, date(2027, 4, 21), 1.10)
        dv01 = fwd.dv01_base(1.10, base, quote)
        assert isinstance(dv01, float)
        assert dv01 != 0.0

    def test_dv01_quote(self):
        ref = date(2026, 4, 21)
        base = make_flat_curve(ref, rate=0.03)
        quote = make_flat_curve(ref, rate=0.04)
        pair = CurrencyPair.from_currencies(Currency.EUR, Currency.USD)
        fwd = FXForward(pair, date(2027, 4, 21), 1.10)
        dv01 = fwd.dv01_quote(1.10, base, quote)
        assert dv01 != 0.0


# ---- FXH2: FXSwap sensitivities ----

class TestFXSwapSensitivities:
    def test_swap_fx_delta(self):
        ref = date(2026, 4, 21)
        base = make_flat_curve(ref, rate=0.03)
        quote = make_flat_curve(ref, rate=0.04)
        pair = CurrencyPair.from_currencies(Currency.EUR, Currency.USD)
        swap = FXSwap(pair, date(2026, 7, 21), date(2027, 4, 21), 1.10, 1.11)
        delta = swap.fx_delta(1.10, base, quote)
        assert isinstance(delta, float)

    def test_swap_dv01(self):
        ref = date(2026, 4, 21)
        base = make_flat_curve(ref, rate=0.03)
        quote = make_flat_curve(ref, rate=0.04)
        pair = CurrencyPair.from_currencies(Currency.EUR, Currency.USD)
        swap = FXSwap(pair, date(2026, 7, 21), date(2027, 4, 21), 1.10, 1.11)
        dv01 = swap.dv01(1.10, base, quote)
        assert isinstance(dv01, float)


# ---- FXH3: CurrencyPair.forward_rate_from_curves ----

class TestCIPFromCurves:
    def test_matches_fx_forward(self):
        """CurrencyPair curve-based CIP should match FXForward."""
        ref = date(2026, 4, 21)
        base = make_flat_curve(ref, rate=0.03)
        quote = make_flat_curve(ref, rate=0.04)
        pair = CurrencyPair.from_currencies(Currency.EUR, Currency.USD)
        mat = date(2027, 4, 21)

        from_pair = pair.forward_rate_from_curves(1.10, mat, base, quote)
        from_fwd = FXForward.forward_rate(1.10, mat, base, quote)
        assert from_pair == pytest.approx(from_fwd, rel=1e-12)


# ---- FXH4: NDF ----

class TestNDF:
    def test_ndf_forward(self):
        ref = date(2026, 4, 21)
        usd = make_flat_curve(ref, rate=0.04)
        cny = make_flat_curve(ref, rate=0.02)
        ndf = NDF("USD/CNY", date(2027, 4, 21), 7.25)
        fwd = ndf.forward_rate(7.20, usd, cny)
        # USD (base) rate > CNY (quote) rate → df_base < df_quote → F < S
        assert fwd < 7.20

    def test_ndf_pv_at_forward(self):
        """NDF at forward strike has PV ≈ 0."""
        ref = date(2026, 4, 21)
        usd = make_flat_curve(ref, rate=0.04)
        cny = make_flat_curve(ref, rate=0.02)
        fwd = 7.20 * usd.df(date(2027, 4, 21)) / cny.df(date(2027, 4, 21))
        ndf = NDF("USD/CNY", date(2027, 4, 21), fwd)
        pv = ndf.pv(7.20, usd, cny)
        assert abs(pv) < 1.0

    def test_ndf_settlement(self):
        ndf = NDF("USD/CNY", date(2027, 4, 21), 7.25, notional=1_000_000)
        settle = ndf.settlement_amount(7.30)
        # fixing > contracted → buyer receives
        assert settle > 0
        assert settle == pytest.approx(1_000_000 * (7.30 - 7.25))

    def test_ndf_delta(self):
        ref = date(2026, 4, 21)
        usd = make_flat_curve(ref, rate=0.04)
        cny = make_flat_curve(ref, rate=0.02)
        ndf = NDF("USD/CNY", date(2027, 4, 21), 7.25)
        delta = ndf.fx_delta(7.20, usd, cny)
        assert delta > 0


# ---- FXH5: FXForwardCurve ----

class TestFXForwardCurve:
    def test_from_curves(self):
        ref = date(2026, 4, 21)
        eur = make_flat_curve(ref, rate=0.03)
        usd = make_flat_curve(ref, rate=0.04)
        tenors = [date(2026, 7, 21), date(2026, 10, 21), date(2027, 4, 21)]
        curve = FXForwardCurve.from_curves("EUR/USD", 1.10, ref, eur, usd, tenors)
        # EUR rate < USD rate → forward > spot (USD at premium)
        assert curve.forward(date(2027, 4, 21)) > 1.10

    def test_forward_points(self):
        ref = date(2026, 4, 21)
        eur = make_flat_curve(ref, rate=0.03)
        usd = make_flat_curve(ref, rate=0.04)
        tenors = [date(2026, 7, 21), date(2027, 4, 21)]
        curve = FXForwardCurve.from_curves("EUR/USD", 1.10, ref, eur, usd, tenors)
        pts = curve.forward_points(date(2027, 4, 21))
        assert pts > 0  # EUR/USD forward points positive when USD rate > EUR rate

    def test_forward_points_curve(self):
        ref = date(2026, 4, 21)
        eur = make_flat_curve(ref, rate=0.03)
        usd = make_flat_curve(ref, rate=0.04)
        tenors = [date(2026, 7, 21), date(2026, 10, 21), date(2027, 4, 21)]
        curve = FXForwardCurve.from_curves("EUR/USD", 1.10, ref, eur, usd, tenors)
        pts_curve = curve.forward_points_curve()
        assert len(pts_curve) == 3
        # Points should increase with tenor
        assert pts_curve[2][1] > pts_curve[0][1]

    def test_spot_stored(self):
        ref = date(2026, 4, 21)
        curve = FXForwardCurve("EUR/USD", ref, 1.10,
                               [date(2027, 4, 21)], [1.12])
        assert curve.spot == 1.10
