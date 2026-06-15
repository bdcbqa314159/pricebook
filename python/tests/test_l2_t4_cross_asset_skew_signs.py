"""Regression for L2 T4 audit of `options.skew_trading.cross_asset_skew_comparison`:

Pre-fix the function sorted assets by SIGNED skew and labelled
``steepest = entries[0]`` (lowest signed) and ``flattest = entries[-1]``
(highest signed).  This is correct only when all skews share a sign.

For the canonical cross-asset use case — equity put-skew (negative RR)
alongside commodity call-skew (positive RR) — the labels were
INVERTED: the strong call-skew (e.g. commodity with RR = +0.10) was
labelled "flattest", and a moderate put-skew (RR = -0.05) was
labelled "steepest" even though its magnitude is half.

Fix (T4-SK1): sort by ``|skew|`` descending so "steepest" always means
the largest absolute skew and "flattest" the closest to zero.
"""

from __future__ import annotations

import pytest

from pricebook.options.skew_trading import cross_asset_skew_comparison


class TestMixedSignSkew:
    def test_call_skew_dominates_steepest(self):
        """Commodity has the LARGEST |skew| (call-skew of +0.10).  It
        must be labelled steepest, not flattest."""
        r = cross_asset_skew_comparison({
            "equity": -0.05,
            "fx": 0.0,
            "commodity": +0.10,
        })
        assert r.steepest_skew == "commodity"
        assert r.flattest_skew == "fx"

    def test_zero_skew_is_flattest(self):
        """Whichever asset has skew = 0 should be flattest regardless
        of sign of the others."""
        r = cross_asset_skew_comparison({
            "a": -0.03,
            "b": 0.0,
            "c": +0.04,
        })
        assert r.flattest_skew == "b"
        # Steepest = c (|0.04| > |0.03|).
        assert r.steepest_skew == "c"


class TestSameSignBackwardCompatible:
    def test_all_negative_unchanged(self):
        """All-negative-skews case continues to work — pre-fix test
        ``test_ranking`` covers this and must keep passing."""
        r = cross_asset_skew_comparison({
            "equity": -0.04, "fx": -0.02, "rates": -0.01,
        })
        assert r.steepest_skew == "equity"   # |−0.04| largest
        assert r.flattest_skew == "rates"    # |−0.01| smallest

    def test_all_positive_works(self):
        r = cross_asset_skew_comparison({
            "a": 0.04, "b": 0.02, "c": 0.01,
        })
        assert r.steepest_skew == "a"
        assert r.flattest_skew == "c"
