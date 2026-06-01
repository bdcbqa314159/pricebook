"""Tests for unified curve methods across all currencies."""

import pytest
import math
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.curves.curve_builder import build_curves, get_conventions

REF = date(2024, 11, 4)


def _make_inputs(rate=0.04):
    """Generic deposit + swap inputs for any currency."""
    deposits = [
        (REF + relativedelta(months=1), rate - 0.002),
        (REF + relativedelta(months=3), rate - 0.001),
        (REF + relativedelta(months=6), rate),
    ]
    swaps = [
        (REF + relativedelta(years=1), rate + 0.001),
        (REF + relativedelta(years=2), rate + 0.002),
        (REF + relativedelta(years=3), rate + 0.003),
        (REF + relativedelta(years=5), rate + 0.005),
        (REF + relativedelta(years=10), rate + 0.008),
    ]
    return deposits, swaps


class TestG10AllMethods:
    """G10 currencies should work with all 5 methods (was always the case)."""

    def test_usd_sequential(self):
        deps, swaps = _make_inputs(0.05)
        result = build_curves("USD", REF, deps, swaps, method="sequential")
        assert result.ois is not None

    def test_usd_nelson_siegel(self):
        deps, swaps = _make_inputs(0.05)
        result = build_curves("USD", REF, deps, swaps, method="nelson_siegel")
        assert result.ois is not None

    def test_usd_svensson(self):
        deps, swaps = _make_inputs(0.05)
        result = build_curves("USD", REF, deps, swaps, method="svensson")
        assert result.ois is not None

    def test_usd_smith_wilson(self):
        deps, swaps = _make_inputs(0.05)
        result = build_curves("USD", REF, deps, swaps, method="smith_wilson")
        assert result.ois is not None


class TestEMAllMethods:
    """EM currencies should NOW work with all 5 methods (was locked to sequential)."""

    def test_brl_conventions_found(self):
        """BRL conventions should be resolved via EM fallthrough."""
        conv = get_conventions("BRL")
        assert conv is not None

    def test_brl_sequential(self):
        deps, swaps = _make_inputs(0.11)
        result = build_curves("BRL", REF, deps, swaps, method="sequential")
        assert result.ois is not None

    def test_brl_nelson_siegel(self):
        deps, swaps = _make_inputs(0.11)
        result = build_curves("BRL", REF, deps, swaps, method="nelson_siegel")
        assert result.ois is not None

    def test_cny_svensson(self):
        deps, swaps = _make_inputs(0.018)
        result = build_curves("CNY", REF, deps, swaps, method="svensson")
        assert result.ois is not None

    def test_try_sequential(self):
        """TRY at 45% — sequential handles extreme rates."""
        deps, swaps = _make_inputs(0.45)
        result = build_curves("TRY", REF, deps, swaps, method="sequential")
        assert result.ois is not None

    def test_huf_smith_wilson(self):
        """HUF at moderate rate — Smith-Wilson works."""
        deps, swaps = _make_inputs(0.065)
        result = build_curves("HUF", REF, deps, swaps, method="smith_wilson")
        assert result.ois is not None

    def test_krw_nelson_siegel(self):
        deps, swaps = _make_inputs(0.035)
        result = build_curves("KRW", REF, deps, swaps, method="nelson_siegel")
        assert result.ois is not None

    def test_zar_sequential(self):
        deps, swaps = _make_inputs(0.08)
        result = build_curves("ZAR", REF, deps, swaps, method="sequential")
        assert result.ois is not None

    def test_pln_svensson(self):
        deps, swaps = _make_inputs(0.058)
        result = build_curves("PLN", REF, deps, swaps, method="svensson")
        assert result.ois is not None

    def test_inr_sequential(self):
        deps, swaps = _make_inputs(0.065)
        result = build_curves("INR", REF, deps, swaps, method="sequential")
        assert result.ois is not None

    def test_mxn_nelson_siegel(self):
        deps, swaps = _make_inputs(0.105)
        result = build_curves("MXN", REF, deps, swaps, method="nelson_siegel")
        assert result.ois is not None


class TestCrossMethodConsistency:
    def test_5y_rate_similar(self):
        """5Y zero rate should be roughly similar across methods for same inputs."""
        deps, swaps = _make_inputs(0.04)
        zeros = {}
        for method in ["sequential", "nelson_siegel", "svensson"]:
            result = build_curves("USD", REF, deps, swaps, method=method)
            zeros[method] = result.ois.zero_rate(REF + relativedelta(years=5))

        # All within 100bp of each other
        vals = list(zeros.values())
        assert max(vals) - min(vals) < 0.01

    def test_unknown_currency_raises(self):
        with pytest.raises(ValueError, match="Unknown currency"):
            get_conventions("XYZ")
