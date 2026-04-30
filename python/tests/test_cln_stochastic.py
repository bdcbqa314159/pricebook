"""Tests for CLN stochastic recovery: single-name pricing, seniority, leverage."""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.cln import CreditLinkedNote
from pricebook.recovery_pricing import RecoverySpec
from tests.conftest import make_flat_curve, make_flat_survival


REF = date(2024, 1, 15)
END_5Y = REF + relativedelta(years=5)


def _make_cln(recovery=0.4, leverage=1.0):
    return CreditLinkedNote(
        REF, END_5Y, coupon_rate=0.06, notional=1_000_000,
        recovery=recovery, leverage=leverage,
    )


# ---- Stochastic recovery pricing ----

class TestStochasticRecoveryPricing:

    def test_matches_fixed_when_deterministic(self):
        """With fixed RecoverySpec, should match dirty_price()."""
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        cln = _make_cln()
        fixed_price = cln.dirty_price(dc, sc)
        spec = RecoverySpec.fixed(0.4)
        stoch_price = cln.price_stochastic_recovery(dc, sc, spec, n_sims=50_000).price
        assert stoch_price == pytest.approx(fixed_price, rel=0.02)

    def test_wrong_way_reduces_price(self):
        """With negative ρ_DR, stochastic price < fixed price."""
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.03)  # moderate hazard
        cln = _make_cln()
        fixed_price = cln.dirty_price(dc, sc)
        spec = RecoverySpec(mean=0.4, std=0.15, correlation_to_default=-0.4)
        stoch_price = cln.price_stochastic_recovery(dc, sc, spec, n_sims=50_000).price
        assert stoch_price < fixed_price

    def test_zero_correlation_matches(self):
        """Zero ρ_DR with stochastic R should be close to fixed R pricing."""
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        cln = _make_cln()
        fixed_price = cln.dirty_price(dc, sc)
        spec = RecoverySpec(mean=0.4, std=0.15, correlation_to_default=0.0)
        stoch_price = cln.price_stochastic_recovery(dc, sc, spec, n_sims=50_000).price
        assert stoch_price == pytest.approx(fixed_price, rel=0.03)


# ---- Seniority ----

class TestSeniority:

    def test_1l_higher_price(self):
        """1L CLN (higher recovery) should have higher price."""
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.03)
        cln_1l = CreditLinkedNote.from_seniority(REF, END_5Y, "1L", coupon_rate=0.05)
        cln_sub = CreditLinkedNote.from_seniority(REF, END_5Y, "sub", coupon_rate=0.05)
        p_1l = cln_1l.dirty_price(dc, sc)
        p_sub = cln_sub.dirty_price(dc, sc)
        assert p_1l > p_sub

    def test_recovery_from_seniority(self):
        cln = CreditLinkedNote.from_seniority(REF, END_5Y, "1L")
        assert cln.recovery == pytest.approx(0.77)

    def test_unknown_seniority(self):
        with pytest.raises(ValueError, match="Unknown seniority"):
            CreditLinkedNote.from_seniority(REF, END_5Y, "XXX")


# ---- Recovery vol sensitivity ----

class TestRecVol01:

    def test_negative(self):
        """Higher recovery vol reduces CLN PV (wrong-way risk)."""
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.03)
        cln = _make_cln()
        spec = RecoverySpec(mean=0.4, std=0.15, correlation_to_default=-0.3)
        rv01 = cln.rec_vol_01(dc, sc, spec, n_sims=30_000)
        assert rv01 < 0

    def test_zero_when_no_correlation(self):
        """Without correlation, recovery vol has minimal impact."""
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.03)
        cln = _make_cln()
        spec = RecoverySpec(mean=0.4, std=0.15, correlation_to_default=0.0)
        rv01 = cln.rec_vol_01(dc, sc, spec, n_sims=30_000)
        assert abs(rv01) < 500  # small relative to 1M notional


# ---- Leveraged CLN ----

class TestLeveragedStochastic:

    def test_leverage_amplifies_wrong_way(self):
        """Leveraged CLN should have larger wrong-way premium."""
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.03)
        spec = RecoverySpec(mean=0.4, std=0.15, correlation_to_default=-0.4)

        vanilla = _make_cln(leverage=1.0)
        leveraged = _make_cln(leverage=3.0)

        fixed_v = vanilla.dirty_price(dc, sc)
        stoch_v = vanilla.price_stochastic_recovery(dc, sc, spec, n_sims=30_000).price
        premium_v = fixed_v - stoch_v

        fixed_l = leveraged.dirty_price(dc, sc)
        stoch_l = leveraged.price_stochastic_recovery(dc, sc, spec, n_sims=30_000).price
        premium_l = fixed_l - stoch_l

        assert premium_l > premium_v
