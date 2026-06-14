"""Regression for L2 T4 audit of `desks.api.asian_option`:

Pre-fix:
1. ``n_observations`` was a silent no-op — declared in signature but
   the body always used the continuous-monitoring approximation
   ``σ_geom = σ/√3`` regardless of the value passed.
2. The spot was passed as forward to Black-76 without rate drift
   adjustment via df, so positive rates were ignored.
3. The geometric-average Jensen correction
   ``F · exp(-½·(σ²-σ²_g)·t_avg)`` was missing, so even at df=1 the
   pricing was biased by ~σ²T/12.

Fix: Kemna-Vorst discrete-monitoring formula
``σ²_geom = σ²·(n+1)(2n+1)/(6n²)`` honours n_observations; builds the
geometric-average forward correctly using F = spot/df.
"""

from __future__ import annotations

import math

import pytest

import pricebook.desks.api as pb


class TestAsianNObservations:
    def test_n_observations_no_longer_silent(self):
        """Pre-fix: n=12 and n=∞ (large) gave identical prices.
        Post-fix: discrete n=12 should give noticeably higher vol than
        continuous (σ²_geom is ~13% higher at n=12), so the price
        differs."""
        p_n12 = pb.asian_option(100.0, 100.0, 0.30, 1.0,
                                 n_observations=12, df=1.0)
        p_continuous_proxy = pb.asian_option(100.0, 100.0, 0.30, 1.0,
                                              n_observations=10_000, df=1.0)
        # Discrete n=12 has higher effective vol → higher ATM price.
        assert p_n12 > p_continuous_proxy
        # Difference should be 3-6% range for σ=0.30, T=1.
        ratio = p_n12 / p_continuous_proxy
        assert 1.02 < ratio < 1.10

    def test_continuous_limit(self):
        """Large n should match σ/√3 continuous formula (the prior
        behaviour) up to Jensen correction."""
        # Continuous approximation: σ_geom = σ/√3, F_G = F · exp(-σ²T/12).
        spot, K, vol, T = 100.0, 100.0, 0.20, 1.0
        n = 100_000
        adj_vol = vol / math.sqrt(3)
        F = spot  # df=1.0
        t_avg = T / 2.0
        F_geom = F * math.exp(-0.5 * (vol**2 - adj_vol**2) * t_avg)
        from pricebook.models.black76 import black76_price, OptionType
        expected = black76_price(F_geom, K, adj_vol, T, 1.0, OptionType.CALL)
        actual = pb.asian_option(spot, K, vol, T, n_observations=n, df=1.0)
        assert actual == pytest.approx(expected, rel=1e-3)


class TestAsianRateDrift:
    def test_positive_rates_handled_via_df(self):
        """With df < 1 (positive rates), the forward should be higher
        than spot.  Asian call should be more valuable than the
        pre-fix value (which incorrectly used spot)."""
        spot, K, vol, T = 100.0, 100.0, 0.20, 1.0
        # df = exp(-0.05 * 1) → F = spot / df ≈ 105.13.
        df = math.exp(-0.05 * T)
        # Same call with df=1 (no rates) for comparison.
        c_no_rate = pb.asian_option(spot, K, vol, T,
                                     n_observations=12, df=1.0)
        c_with_rate = pb.asian_option(spot, K, vol, T,
                                       n_observations=12, df=df)
        # Risk-neutral probability under positive rates is higher (forward up),
        # but the df multiplier compensates somewhat.  Most importantly the
        # difference between with-rate and no-rate must reflect a real shift:
        # the pre-fix `with_rate` ≈ `no_rate × df` (just discount, no forward).
        # Post-fix difference is larger than just `× df`.
        assert c_with_rate / c_no_rate > df  # better than just discount
