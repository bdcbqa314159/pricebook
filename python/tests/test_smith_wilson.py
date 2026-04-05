"""Tests for Smith-Wilson curve extrapolation."""

import math
import pytest
from datetime import date

from pricebook.smith_wilson import (
    smith_wilson_calibrate,
    smith_wilson_df,
    smith_wilson_forward,
    smith_wilson_curve,
)


REF = date(2024, 1, 15)
UFR = 0.0345  # EIOPA UFR
ALPHA = 0.1

# Market data: flat 3% curve out to 20Y
MATURITIES = [1, 2, 3, 5, 7, 10, 15, 20]
MARKET_DFS = [math.exp(-0.03 * t) for t in MATURITIES]


class TestCalibration:
    def test_matches_market(self):
        """SW curve matches market DFs at liquid points."""
        zeta = smith_wilson_calibrate(MATURITIES, MARKET_DFS, UFR, ALPHA)
        for t, df_mkt in zip(MATURITIES, MARKET_DFS):
            df_sw = smith_wilson_df(t, MATURITIES, zeta, UFR, ALPHA)
            assert df_sw == pytest.approx(df_mkt, rel=1e-8)

    def test_zeta_vector_length(self):
        zeta = smith_wilson_calibrate(MATURITIES, MARKET_DFS, UFR, ALPHA)
        assert len(zeta) == len(MATURITIES)


class TestExtrapolation:
    def test_converges_to_ufr(self):
        """Forward rate converges to UFR at long maturities."""
        zeta = smith_wilson_calibrate(MATURITIES, MARKET_DFS, UFR, ALPHA)
        fwd_100y = smith_wilson_forward(100.0, MATURITIES, zeta, UFR, ALPHA)
        assert fwd_100y == pytest.approx(UFR, abs=0.005)

    def test_forward_at_ufr_asymptotically(self):
        """Forward rate should approach UFR as maturity increases."""
        zeta = smith_wilson_calibrate(MATURITIES, MARKET_DFS, UFR, ALPHA)
        fwd_30 = smith_wilson_forward(30.0, MATURITIES, zeta, UFR, ALPHA)
        fwd_60 = smith_wilson_forward(60.0, MATURITIES, zeta, UFR, ALPHA)
        fwd_100 = smith_wilson_forward(100.0, MATURITIES, zeta, UFR, ALPHA)
        # Should get closer to UFR
        assert abs(fwd_60 - UFR) < abs(fwd_30 - UFR)
        assert abs(fwd_100 - UFR) < abs(fwd_60 - UFR)

    def test_df_positive_at_long_maturity(self):
        zeta = smith_wilson_calibrate(MATURITIES, MARKET_DFS, UFR, ALPHA)
        df_100 = smith_wilson_df(100.0, MATURITIES, zeta, UFR, ALPHA)
        assert df_100 > 0

    def test_df_decreasing(self):
        zeta = smith_wilson_calibrate(MATURITIES, MARKET_DFS, UFR, ALPHA)
        dfs = [smith_wilson_df(t, MATURITIES, zeta, UFR, ALPHA) for t in [10, 30, 50, 100]]
        assert all(dfs[i] >= dfs[i+1] for i in range(len(dfs)-1))


class TestSmithWilsonCurve:
    def test_builds_curve(self):
        curve = smith_wilson_curve(REF, MATURITIES, MARKET_DFS, UFR, ALPHA)
        d10y = date.fromordinal(REF.toordinal() + int(10 * 365))
        df = curve.df(d10y)
        assert 0 < df < 1

    def test_matches_at_pillars(self):
        curve = smith_wilson_curve(REF, MATURITIES, MARKET_DFS, UFR, ALPHA)
        for t, df_mkt in zip(MATURITIES, MARKET_DFS):
            d = date.fromordinal(REF.toordinal() + int(t * 365))
            assert curve.df(d) == pytest.approx(df_mkt, rel=0.01)

    def test_long_end_positive(self):
        curve = smith_wilson_curve(REF, MATURITIES, MARKET_DFS, UFR, ALPHA)
        d100y = date.fromordinal(REF.toordinal() + int(100 * 365))
        assert curve.df(d100y) > 0

    def test_eiopa_defaults(self):
        """Default UFR=3.45%, alpha=0.1."""
        curve = smith_wilson_curve(REF, MATURITIES, MARKET_DFS)
        d50y = date.fromordinal(REF.toordinal() + int(50 * 365))
        df = curve.df(d50y)
        assert 0 < df < 0.5
