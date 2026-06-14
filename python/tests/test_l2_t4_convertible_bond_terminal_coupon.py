"""Regression for L2 T4 audit of `options.convertible_bond`:

Pre-fix the MC backward-induction terminal payoff was
``max(notional, conv_ratio·S_T)`` and the loop ran
``range(n_steps-1, -1, -1)``, never visiting ``step == n_steps``.  So
the final coupon paid at maturity was silently dropped, biasing the
MC PV downward by ~``coupon_amount · DF(T)``.

But the analytical ``bond_floor`` calculation DID include the maturity
coupon, making the two paths inconsistent.  A deep-OTM CB (conversion
never optimal) should price exactly at the bond floor under common
random numbers; pre-fix it priced systematically below.

Fix: terminal payoff is ``max(notional + coupon_amount, conv_ratio·S_T)``
in all four backward-induction sites: ``price``, ``_compute_delta``,
``convertible_soft_call``, plus the bumped ``V_up`` mirror.
"""

from __future__ import annotations

import math

import pytest

from pricebook.options.convertible_bond import ConvertibleBond


class TestTerminalCouponIncluded:
    def test_otm_converges_to_bond_floor_tight(self):
        """For deep-OTM equity (conversion never optimal), MC PV should
        agree with the analytical ``bond_floor`` to within MC stderr.
        Pre-fix the MC was ~coupon × DF below bond_floor."""
        cb = ConvertibleBond(notional=1000, coupon_rate=0.05,
                             maturity_years=3.0, conversion_ratio=10,
                             n_coupons_per_year=2)
        # spot = 1: conversion value ≈ 10 vs notional 1000 — never optimal.
        result = cb.price(spot=1.0, rate=0.03, equity_vol=0.20,
                          credit_spread=0.02, n_paths=20_000, seed=42)
        # Tight bound: within 0.5% of bond floor.
        assert result.price == pytest.approx(result.bond_floor, rel=5e-3)

    def test_pre_fix_value_below_bond_floor(self):
        """Sanity check the magnitude of the pre-fix bias.

        Final coupon at maturity for the above test:
            coupon_amount = 1000 × 0.05 / 2 = 25.
            DF(T=3) under disc_rate = 0.05 → e^{-0.15} ≈ 0.8607.
            Missing PV ≈ 25 × 0.8607 ≈ 21.5.
        Post-fix the MC should NOT undershoot by anywhere near that much.
        """
        cb = ConvertibleBond(notional=1000, coupon_rate=0.05,
                             maturity_years=3.0, conversion_ratio=10,
                             n_coupons_per_year=2)
        result = cb.price(spot=1.0, rate=0.03, equity_vol=0.20,
                          credit_spread=0.02, n_paths=20_000, seed=42)
        # If terminal coupon were missing, gap would be ≈ 21.5.
        # Post-fix, gap should be well under 5 (MC noise on PV ~750).
        gap = result.bond_floor - result.price
        assert gap < 5.0, f"gap={gap:.3f} suggests terminal coupon still missing"


class TestZeroCouponUnaffected:
    def test_zero_coupon_cb(self):
        """Zero-coupon CB: terminal coupon = 0, so behaviour unchanged."""
        cb = ConvertibleBond(notional=1000, coupon_rate=0.0,
                             maturity_years=2.0, conversion_ratio=10,
                             n_coupons_per_year=2)
        result = cb.price(spot=50, rate=0.03, equity_vol=0.25,
                          credit_spread=0.01, n_paths=10_000, seed=42)
        # With no coupons, bond floor ≈ notional × DF.
        disc = math.exp(-0.04 * 2)  # disc_rate = rate + cs
        expected_floor = 1000 * disc
        assert result.bond_floor == pytest.approx(expected_floor, rel=1e-6)
        # And price ≥ bond floor (conversion option ≥ 0).
        assert result.price >= result.bond_floor * 0.98
