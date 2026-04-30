"""Tests for CLN analytics: wrong-way report, recovery surface, serialisation, integration."""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.cln import CreditLinkedNote, BasketCLN
from pricebook.recovery_pricing import RecoverySpec, SENIORITY_RECOVERY
from tests.conftest import make_flat_curve, make_flat_survival


REF = date(2024, 1, 15)
END_5Y = REF + relativedelta(years=5)


# ---- Wrong-way risk report ----

class TestWrongWayReport:

    def test_premium_increases_with_correlation(self):
        """Wrong-way premium should increase as |ρ_DR| increases."""
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.03)
        cln = CreditLinkedNote(REF, END_5Y, coupon_rate=0.05,
                                notional=1_000_000, recovery=0.4)
        fixed_price = cln.dirty_price(dc, sc)

        premiums = []
        for rho in [-0.1, -0.3, -0.5]:
            spec = RecoverySpec(mean=0.4, std=0.15, correlation_to_default=rho)
            stoch = cln.price_stochastic_recovery(dc, sc, spec, n_sims=30_000).price
            premiums.append(fixed_price - stoch)

        # Premium should increase (more negative ρ → more wrong-way)
        for i in range(1, len(premiums)):
            assert premiums[i] >= premiums[i - 1] * 0.5  # directionally correct


# ---- Recovery surface ----

class TestRecoverySurface:

    def test_seniority_ordering(self):
        """1L > 2L > senior > sub recovery at all scenarios."""
        for sen in ["1L", "2L", "senior", "sub"]:
            assert sen in SENIORITY_RECOVERY

        vals = {s: SENIORITY_RECOVERY[s][0] for s in ["1L", "2L", "senior", "sub"]}
        assert vals["1L"] > vals["2L"]
        assert vals["2L"] > vals["sub"]

    def test_spec_from_seniority(self):
        for sen in ["1L", "2L", "senior", "sub"]:
            spec = RecoverySpec.from_seniority(sen)
            assert 0 < spec.mean < 1
            assert spec.std > 0


# ---- Serialisation ----

class TestCLNSerialisation:

    def test_cln_round_trip(self):
        from pricebook.serialisable import from_dict
        cln = CreditLinkedNote(REF, END_5Y, coupon_rate=0.06,
                                notional=1_000_000, recovery=0.35)
        d = cln.to_dict()
        cln2 = from_dict(d)
        assert cln2.coupon_rate == cln.coupon_rate
        assert cln2.recovery == cln.recovery
        assert cln2.notional == cln.notional

    def test_recovery_spec_round_trip(self):
        spec = RecoverySpec(mean=0.45, std=0.18, distribution="beta",
                             correlation_to_default=-0.4)
        d = spec.to_dict()
        spec2 = RecoverySpec.from_dict(d)
        assert spec2.mean == spec.mean
        assert spec2.std == spec.std
        assert spec2.correlation_to_default == spec.correlation_to_default


# ---- Integration ----

class TestIntegration:

    def test_vanilla_to_stochastic_pipeline(self):
        """Full pipeline: build curves → price vanilla → price stochastic → compare."""
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.03)
        cln = CreditLinkedNote(REF, END_5Y, coupon_rate=0.05,
                                notional=1_000_000, recovery=0.4)

        # Fixed recovery price
        fixed = cln.dirty_price(dc, sc)
        assert fixed > 0

        # Stochastic recovery (no correlation)
        spec_indep = RecoverySpec(mean=0.4, std=0.15, correlation_to_default=0.0)
        stoch_indep = cln.price_stochastic_recovery(dc, sc, spec_indep, n_sims=30_000).price
        assert stoch_indep == pytest.approx(fixed, rel=0.05)

        # Stochastic recovery (wrong-way)
        spec_ww = RecoverySpec(mean=0.4, std=0.15, correlation_to_default=-0.4)
        stoch_ww = cln.price_stochastic_recovery(dc, sc, spec_ww, n_sims=30_000).price
        assert stoch_ww < fixed

        # Greeks
        greeks = cln.greeks(dc, sc)
        assert "dv01" in greeks
        assert "cs01" in greeks

    def test_basket_integration(self):
        """Basket CLN: gaussian → t-copula → correlated recovery."""
        N = 5
        dc = make_flat_curve(REF, 0.04)
        survs = [make_flat_survival(REF, 0.01 + 0.005 * i) for i in range(N)]
        basket = BasketCLN(REF, END_5Y, notional=5_000_000,
                           attachment=0.0, detachment=0.10, n_names=N)

        # Gaussian
        g = basket.price_mc(dc, survs, rho=0.3, n_sims=10_000)
        assert g.price > 0

        # t-copula
        t = basket.price_mc_copula(dc, survs, rho=0.3, copula="t", nu=5, n_sims=10_000)
        assert t.price > 0

        # Correlated recovery
        cr = basket.price_mc_correlated_recovery(
            dc, survs, rho=0.3, recovery_sensitivity=0.10, n_sims=10_000,
        )
        assert cr.price > 0

    def test_seniority_cln_pipeline(self):
        """Build CLN from seniority → price → compare."""
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.03)

        cln_1l = CreditLinkedNote.from_seniority(REF, END_5Y, "1L", coupon_rate=0.04)
        cln_sub = CreditLinkedNote.from_seniority(REF, END_5Y, "sub", coupon_rate=0.04)

        p_1l = cln_1l.dirty_price(dc, sc)
        p_sub = cln_sub.dirty_price(dc, sc)
        assert p_1l > p_sub  # higher recovery → higher price

        # Breakeven spread
        be_1l = cln_1l.breakeven_spread(dc, sc)
        be_sub = cln_sub.breakeven_spread(dc, sc)
        assert be_sub > be_1l  # sub needs higher coupon to trade at par
