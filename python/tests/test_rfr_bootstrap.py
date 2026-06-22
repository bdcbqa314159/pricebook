"""Tests for multi-RFR OIS curve bootstrap."""

import math
import pytest
from datetime import date

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.curves.rfr_bootstrap import (
    bootstrap_rfr, RFRCurveInputs, RFRCurveResult,
    get_rfr_ois_conventions, list_rfr_ois_currencies,
)
from pricebook.fixed_income.rfr_futures import RFRFutureSpec


REF = date(2024, 1, 15)


def _make_ois_swaps(ref, rate_curve):
    """Helper: create OIS swap inputs from a rate term structure."""
    return [(date(ref.year + y, ref.month, ref.day), r) for y, r in rate_curve]


class TestConventions:
    def test_list_currencies(self):
        ccys = list_rfr_ois_currencies()
        assert "USD" in ccys
        assert "EUR" in ccys
        assert "GBP" in ccys
        assert len(ccys) == 7

    def test_usd_conventions(self):
        c = get_rfr_ois_conventions("USD")
        assert c.rfr_name == "SOFR"
        assert c.deposit_dc == DayCountConvention.ACT_360

    def test_gbp_conventions(self):
        c = get_rfr_ois_conventions("GBP")
        assert c.rfr_name == "SONIA"
        assert c.deposit_dc == DayCountConvention.ACT_365_FIXED

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="No RFR OIS"):
            get_rfr_ois_conventions("ZAR")


class TestBootstrapUSD:
    def test_deposits_only(self):
        """Bootstrap from deposits only → short-end curve."""
        inputs = RFRCurveInputs(
            overnight_rate=0.053,
            deposits=[(date(2024, 4, 15), 0.052), (date(2024, 7, 15), 0.051)],
        )
        result = bootstrap_rfr("USD", REF, inputs)
        assert isinstance(result, RFRCurveResult)
        assert result.currency == "USD"
        assert result.rfr_name == "SOFR"
        assert result.curve.df(date(2024, 7, 15)) < 1.0
        assert result.curve.df(date(2024, 7, 15)) > 0.95

    def test_w8_no_round_trip_warning(self):
        """W8 regression: prior to pinning the swap pillar at the
        schedule's actual end date (rather than the unadjusted swap
        maturity), business-day-roll could push the schedule end date
        a day past the pillar — at solve time df(schedule_end) was
        extrapolated, after later swaps were added it was interpolated
        — giving ~2e-6 PV round-trip residual on every USD OIS curve.
        """
        import warnings as _warnings
        inputs = RFRCurveInputs(
            overnight_rate=0.053,
            deposits=[(date(2024, 4, 15), 0.052)],
            ois_swaps=_make_ois_swaps(REF, [
                (1, 0.050), (2, 0.048), (3, 0.046),
                (5, 0.043), (7, 0.042), (10, 0.041),
            ]),
        )
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", RuntimeWarning)
            bootstrap_rfr("USD", REF, inputs)

    def test_deposits_and_swaps(self):
        """Full curve: deposits + OIS swaps."""
        inputs = RFRCurveInputs(
            overnight_rate=0.053,
            deposits=[(date(2024, 4, 15), 0.052)],
            ois_swaps=_make_ois_swaps(REF, [
                (1, 0.050), (2, 0.048), (3, 0.046),
                (5, 0.043), (7, 0.042), (10, 0.041),
            ]),
        )
        result = bootstrap_rfr("USD", REF, inputs)
        assert result.n_instruments >= 8
        # 10Y DF should be reasonable
        df_10y = result.curve.df(date(2034, 1, 15))
        assert 0.5 < df_10y < 0.8

    def test_round_trip_deposits(self):
        """Deposits should reprice within 1bp."""
        inputs = RFRCurveInputs(
            overnight_rate=0.05,
            deposits=[(date(2024, 4, 15), 0.049), (date(2024, 7, 15), 0.048)],
        )
        result = bootstrap_rfr("USD", REF, inputs)
        assert result.round_trip_max_error_bp < 1.0

    def test_with_futures(self):
        """Bootstrap with futures → short-end from futures, long-end from swaps."""
        futures = [
            RFRFutureSpec("USD", "SOFR", "3M", date(2024, 3, 1),
                           date(2024, 3, 1), date(2024, 5, 31), price=94.80),
            RFRFutureSpec("USD", "SOFR", "3M", date(2024, 6, 1),
                           date(2024, 6, 1), date(2024, 8, 31), price=94.90),
        ]
        inputs = RFRCurveInputs(
            overnight_rate=0.053,
            futures_3m=futures,
            ois_swaps=_make_ois_swaps(REF, [(2, 0.048), (5, 0.043), (10, 0.041)]),
        )
        result = bootstrap_rfr("USD", REF, inputs)
        assert len(result.convexity_adjustments) == 2
        assert all(ca > 0 for ca in result.convexity_adjustments.values())

    def test_pillar_zero_rates_positive(self):
        inputs = RFRCurveInputs(
            overnight_rate=0.05,
            ois_swaps=_make_ois_swaps(REF, [(1, 0.048), (5, 0.045)]),
        )
        result = bootstrap_rfr("USD", REF, inputs)
        assert all(z > 0 for z in result.pillar_zero_rates)


class TestBootstrapMultiCurrency:
    def test_eur_estr(self):
        inputs = RFRCurveInputs(
            overnight_rate=0.039,
            ois_swaps=_make_ois_swaps(REF, [(1, 0.037), (5, 0.032), (10, 0.030)]),
        )
        result = bootstrap_rfr("EUR", REF, inputs)
        assert result.rfr_name == "ESTR"
        assert result.currency == "EUR"

    def test_gbp_sonia(self):
        inputs = RFRCurveInputs(
            overnight_rate=0.052,
            ois_swaps=_make_ois_swaps(REF, [(1, 0.050), (5, 0.045), (10, 0.042)]),
        )
        result = bootstrap_rfr("GBP", REF, inputs)
        assert result.rfr_name == "SONIA"

    def test_jpy_tona(self):
        inputs = RFRCurveInputs(
            overnight_rate=0.001,
            ois_swaps=_make_ois_swaps(REF, [(1, 0.002), (5, 0.005), (10, 0.008)]),
        )
        result = bootstrap_rfr("JPY", REF, inputs)
        assert result.rfr_name == "TONA"

    def test_chf_saron(self):
        inputs = RFRCurveInputs(
            overnight_rate=0.017,
            ois_swaps=_make_ois_swaps(REF, [(1, 0.015), (5, 0.012)]),
        )
        result = bootstrap_rfr("CHF", REF, inputs)
        assert result.rfr_name == "SARON"

    def test_all_g7_currencies(self):
        """Every G7 currency should bootstrap successfully."""
        for ccy in ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD"]:
            inputs = RFRCurveInputs(
                overnight_rate=0.03,
                ois_swaps=_make_ois_swaps(REF, [(1, 0.03), (5, 0.03)]),
            )
            result = bootstrap_rfr(ccy, REF, inputs)
            assert result.curve.df(date(2029, 1, 15)) > 0


class TestEdgeCases:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            bootstrap_rfr("USD", REF, RFRCurveInputs())

    def test_term_rates(self):
        """Term SOFR rates as deposits."""
        inputs = RFRCurveInputs(
            overnight_rate=0.053,
            term_rates=[("1M", 0.052), ("3M", 0.051), ("6M", 0.050)],
        )
        result = bootstrap_rfr("USD", REF, inputs)
        assert result.n_instruments >= 4

    def test_to_dict(self):
        inputs = RFRCurveInputs(overnight_rate=0.05,
                                 ois_swaps=_make_ois_swaps(REF, [(1, 0.05)]))
        result = bootstrap_rfr("USD", REF, inputs)
        d = result.to_dict()
        assert "currency" in d
        assert "pillar_zero_rates" in d

    def test_inputs_to_dict(self):
        inputs = RFRCurveInputs(overnight_rate=0.05)
        d = inputs.to_dict()
        assert d["overnight_rate"] == 0.05


class TestRFRProvenance:
    """Bootstrapper campaign Tier 1 — bootstrap_rfr inherits the curve's record."""

    def _result(self, method="sequential"):
        inputs = RFRCurveInputs(
            overnight_rate=0.053,
            deposits=[(date(2024, 4, 15), 0.052)],
            ois_swaps=_make_ois_swaps(REF, [
                (1, 0.050), (2, 0.048), (3, 0.046), (5, 0.043), (10, 0.041),
            ]),
        )
        return bootstrap_rfr("USD", REF, inputs, method=method)

    def test_sequential_surfaces_record(self):
        cr = self._result("sequential").calibration_result
        assert cr is not None and cr.fit.model_class == "discount_curve_bootstrap"

    def test_global_surfaces_record(self):
        # B-rfr-global fixed (v1.143.0): the global method path now works and
        # surfaces the discount_curve_global record through RFRCurveResult.
        cr = self._result("global").calibration_result
        assert cr is not None and cr.fit.model_class == "discount_curve_global"

    def test_record_persists(self):
        from pricebook.db.db import PricebookDB
        cr = self._result().calibration_result
        with PricebookDB(":memory:") as db:
            cid = db.save_calibration(cr)
            assert db.load_calibration(cid) == cr
