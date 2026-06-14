"""Regression for L2 T4 audit of `options.futures_options.FuturesOption`:

Pre-fix the Greeks block in ``FuturesOption.price`` ALWAYS used the
Black-76 analytical Greeks even when ``model="bachelier"`` — silently
returning lognormal Greeks for normal-model prices.  This is wrong for
products that *must* be priced under a normal model (e.g. short-rate
futures where rates can go negative), where lognormal-and-normal d1/d2
diverge and the analytical Greek forms differ.

Fix: pick the analytical Greek family that matches the pricing model.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from pricebook.options.futures_options import (
    FuturesOption, FuturesOptionSpec, FuturesAsset,
    ExerciseStyle, SettlementMethod,
)
from pricebook.models.black76 import (
    bachelier_delta, bachelier_gamma, bachelier_vega, bachelier_theta,
    black76_delta, OptionType,
)


REF = date(2026, 4, 28)
EXPIRY = REF + timedelta(days=180)


@pytest.fixture
def sr3_spec():
    # European-style for a clean Bachelier test (no BAW interference).
    return FuturesOptionSpec(
        "SR3_EU", FuturesAsset.INTEREST_RATE, 2500.0, 0.0025, 6.25,
        ExerciseStyle.EUROPEAN, SettlementMethod.CASH,
    )


class TestBachelierGreeks:
    def test_bachelier_delta_uses_normal_formula(self, sr3_spec):
        """Bachelier price + Black-76 delta is wrong; post-fix the delta
        must match ``bachelier_delta`` analytics, not ``black76_delta``."""
        opt = FuturesOption(
            spec=sr3_spec, futures_price=95.0, strike=95.5,
            expiry=EXPIRY, vol=0.50, option_type="call",
            model="bachelier",
        )
        r = opt.price(valuation_date=REF, rate=0.04)

        import math
        T = (EXPIRY - REF).days / 365.0
        df = math.exp(-0.04 * T)
        expected = bachelier_delta(95.0, 95.5, 0.50, T, df, OptionType.CALL)
        assert r.delta == pytest.approx(expected, rel=1e-12)

    def test_bachelier_greeks_differ_from_black76(self, sr3_spec):
        """The two model families must give numerically different deltas
        for a non-degenerate strike — proves we're not silently using
        Black-76 in the Bachelier branch."""
        opt_b = FuturesOption(
            spec=sr3_spec, futures_price=95.0, strike=95.5,
            expiry=EXPIRY, vol=0.50, option_type="call",
            model="bachelier",
        )
        opt_bs = FuturesOption(
            spec=sr3_spec, futures_price=95.0, strike=95.5,
            expiry=EXPIRY, vol=0.005, option_type="call",
            model="black76",
        )
        r_b = opt_b.price(valuation_date=REF, rate=0.04)
        r_bs = opt_bs.price(valuation_date=REF, rate=0.04)
        # Different models give different deltas (proves model dispatch).
        assert abs(r_b.delta - r_bs.delta) > 1e-3

    def test_bachelier_gamma_vega_theta_consistent(self, sr3_spec):
        """All four Greeks must come from the bachelier_* analytics."""
        import math
        opt = FuturesOption(
            spec=sr3_spec, futures_price=95.0, strike=95.5,
            expiry=EXPIRY, vol=0.50, option_type="put",
            model="bachelier",
        )
        r = opt.price(valuation_date=REF, rate=0.04)

        T = (EXPIRY - REF).days / 365.0
        df = math.exp(-0.04 * T)
        F, K, sigma = 95.0, 95.5, 0.50

        assert r.delta == pytest.approx(
            bachelier_delta(F, K, sigma, T, df, OptionType.PUT), rel=1e-12)
        assert r.gamma == pytest.approx(
            bachelier_gamma(F, K, sigma, T, df), rel=1e-12)
        assert r.vega == pytest.approx(
            bachelier_vega(F, K, sigma, T, df) * 0.01, rel=1e-12)
        assert r.theta == pytest.approx(
            bachelier_theta(F, K, sigma, T, df, OptionType.PUT) / 365.0, rel=1e-12)


class TestBlack76GreeksUnchanged:
    def test_black76_path_still_uses_black76_greeks(self, sr3_spec):
        """Default model="black76" path must still produce Black-76 Greeks."""
        import math
        opt = FuturesOption(
            spec=sr3_spec, futures_price=100.0, strike=100.0,
            expiry=EXPIRY, vol=0.20, option_type="call",
            model="black76",
        )
        r = opt.price(valuation_date=REF, rate=0.04)

        T = (EXPIRY - REF).days / 365.0
        df = math.exp(-0.04 * T)
        expected = black76_delta(100.0, 100.0, 0.20, T, df, OptionType.CALL)
        assert r.delta == pytest.approx(expected, rel=1e-12)
