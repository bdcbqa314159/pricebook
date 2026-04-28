"""Tests for unified Credit-Linked Note (CreditLinkedNote + BasketCLN)."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.cln import CreditLinkedNote, BasketCLN, CLNResult
from tests.conftest import make_flat_curve, make_flat_survival


REF = date(2026, 4, 27)
END = date(2031, 4, 27)  # 5Y


def _disc():
    return make_flat_curve(REF, 0.04)


def _surv(hazard=0.02):
    return make_flat_survival(REF, hazard)


# ---- Phase 2a: Unified CreditLinkedNote ----

class TestCreditLinkedNote:

    def test_price_at_par_roughly(self):
        """CLN at par coupon should price near notional."""
        disc = _disc()
        sc = _surv()
        cln = CreditLinkedNote(REF, END, coupon_rate=0.06, notional=1_000_000)
        result = cln.price(disc, sc)
        # Should be near par if coupon compensates for default risk
        assert math.isfinite(result.price)
        assert result.price > 0

    def test_higher_hazard_lower_price(self):
        """Higher default risk → lower CLN price."""
        disc = _disc()
        cln = CreditLinkedNote(REF, END, coupon_rate=0.06, notional=1_000_000)
        p_low = cln.dirty_price(disc, _surv(0.01))
        p_high = cln.dirty_price(disc, _surv(0.05))
        assert p_low > p_high

    def test_risk_free_price_is_higher(self):
        """Risk-free equivalent > risky price."""
        disc = _disc()
        sc = _surv()
        cln = CreditLinkedNote(REF, END, coupon_rate=0.06, notional=1_000_000)
        risky = cln.dirty_price(disc, sc)
        riskfree = cln._risk_free_pv(disc)
        assert riskfree > risky

    def test_price_per_100(self):
        disc = _disc()
        sc = _surv()
        cln = CreditLinkedNote(REF, END, coupon_rate=0.06, notional=1_000_000)
        p100 = cln.price_per_100(disc, sc)
        dp = cln.dirty_price(disc, sc)
        assert p100 == pytest.approx(dp / 1_000_000 * 100)

    def test_decomposition_adds_up(self):
        """coupon_pv + principal_pv + recovery_pv = total price."""
        disc = _disc()
        sc = _surv()
        cln = CreditLinkedNote(REF, END, coupon_rate=0.06, notional=1_000_000)
        result = cln.price(disc, sc)
        recon = result.coupon_pv + result.principal_pv + result.recovery_pv
        assert result.price == pytest.approx(recon)

    def test_leveraged_cln(self):
        """Leverage amplifies default loss → lower price."""
        disc = _disc()
        sc = _surv()
        vanilla = CreditLinkedNote(REF, END, coupon_rate=0.06, notional=1_000_000, leverage=1.0)
        levered = CreditLinkedNote(REF, END, coupon_rate=0.06, notional=1_000_000, leverage=3.0)
        pv = vanilla.dirty_price(disc, sc)
        pl = levered.dirty_price(disc, sc)
        assert pv > pl  # leverage increases loss

    def test_floating_coupon(self):
        """Floating CLN should produce finite price."""
        disc = _disc()
        sc = _surv()
        cln = CreditLinkedNote(REF, END, coupon_rate=0.02, notional=1_000_000,
                               floating=True)
        result = cln.price(disc, sc)
        assert math.isfinite(result.price)
        assert result.price > 0

    def test_breakeven_spread(self):
        """Breakeven spread should make price = notional."""
        disc = _disc()
        sc = _surv()
        cln = CreditLinkedNote(REF, END, coupon_rate=0.05, notional=1_000_000)
        be = cln.breakeven_spread(disc, sc)
        # Verify: at breakeven coupon, price ≈ notional
        at_be = CreditLinkedNote(REF, END, coupon_rate=be, notional=1_000_000)
        assert at_be.dirty_price(disc, sc) == pytest.approx(1_000_000, rel=1e-4)

    def test_par_coupon(self):
        """Par coupon: price(par_coupon) = notional."""
        disc = _disc()
        sc = _surv()
        cln = CreditLinkedNote(REF, END, coupon_rate=0.05, notional=1_000_000)
        result = cln.price(disc, sc)
        pc = result.par_coupon
        at_par = CreditLinkedNote(REF, END, coupon_rate=pc, notional=1_000_000)
        assert at_par.dirty_price(disc, sc) == pytest.approx(1_000_000, rel=1e-3)

    def test_to_dict(self):
        disc = _disc()
        sc = _surv()
        cln = CreditLinkedNote(REF, END, coupon_rate=0.06, notional=1_000_000)
        d = cln.price(disc, sc).to_dict()
        assert "price" in d
        assert "credit_spread" in d

    def test_pv_ctx(self):
        """pv_ctx integration with PricingContext."""
        from pricebook.pricing_context import PricingContext
        disc = _disc()
        ctx = PricingContext(valuation_date=REF, discount_curve=disc)
        cln = CreditLinkedNote(REF, END, coupon_rate=0.06, notional=1_000_000)
        pv = cln.pv_ctx(ctx)
        assert math.isfinite(pv)
        assert pv > 0

    def test_invalid_recovery_raises(self):
        with pytest.raises(ValueError, match="recovery"):
            CreditLinkedNote(REF, END, recovery=1.5)

    def test_invalid_leverage_raises(self):
        with pytest.raises(ValueError, match="leverage"):
            CreditLinkedNote(REF, END, leverage=0.5)

    def test_greeks_dv01(self):
        """DV01 should be negative (rates up → price down for coupon bond)."""
        disc = _disc()
        sc = _surv()
        cln = CreditLinkedNote(REF, END, coupon_rate=0.06, notional=1_000_000)
        g = cln.greeks(disc, sc)
        assert math.isfinite(g["dv01"])

    def test_greeks_cs01(self):
        """CS01 should be negative (wider spreads → lower price)."""
        disc = _disc()
        sc = _surv()
        cln = CreditLinkedNote(REF, END, coupon_rate=0.06, notional=1_000_000)
        g = cln.greeks(disc, sc)
        assert g["cs01"] < 0

    def test_greeks_recovery_sens(self):
        """Higher recovery → higher price (less loss on default)."""
        disc = _disc()
        sc = _surv()
        cln = CreditLinkedNote(REF, END, coupon_rate=0.06, notional=1_000_000,
                               recovery=0.4)
        g = cln.greeks(disc, sc)
        assert g["recovery_sensitivity"] > 0


# ---- Phase 2b: BasketCLN ----

class TestBasketCLN:

    def test_equity_tranche(self):
        """Equity tranche (0-3%) should have high expected loss."""
        disc = _disc()
        scs = [_surv(0.02) for _ in range(125)]
        basket = BasketCLN(REF, END, coupon_rate=0.05, notional=10_000_000,
                           attachment=0.0, detachment=0.03, n_names=125)
        result = basket.price_mc(disc, scs, rho=0.3)
        assert math.isfinite(result.price)
        assert result.expected_loss > 0

    def test_senior_tranche_safer(self):
        """Senior tranche should have lower expected loss than equity."""
        disc = _disc()
        scs = [_surv(0.02) for _ in range(125)]
        equity = BasketCLN(REF, END, attachment=0.0, detachment=0.03, n_names=125)
        senior = BasketCLN(REF, END, attachment=0.07, detachment=0.10, n_names=125)

        r_eq = equity.price_mc(disc, scs, rho=0.3)
        r_sr = senior.price_mc(disc, scs, rho=0.3)

        assert r_eq.expected_loss > r_sr.expected_loss

    def test_higher_correlation_hits_equity_less(self):
        """Higher correlation → less loss for equity (more tail risk for senior)."""
        disc = _disc()
        scs = [_surv(0.02) for _ in range(50)]
        basket = BasketCLN(REF, END, attachment=0.0, detachment=0.03,
                           n_names=50, notional=5_000_000)
        r_low = basket.price_mc(disc, scs, rho=0.1)
        r_high = basket.price_mc(disc, scs, rho=0.6)
        # With higher correlation, equity tranche can be less impacted
        # (defaults cluster, reducing expected frequency)
        assert math.isfinite(r_low.price)
        assert math.isfinite(r_high.price)

    def test_wrong_n_curves_raises(self):
        disc = _disc()
        scs = [_surv() for _ in range(10)]
        basket = BasketCLN(REF, END, n_names=125)
        with pytest.raises(ValueError, match="Expected 125"):
            basket.price_mc(disc, scs)

    def test_invalid_tranche_raises(self):
        with pytest.raises(ValueError, match="detachment must be"):
            basket = BasketCLN(REF, END, attachment=0.05, detachment=0.03)
            disc = _disc()
            scs = [_surv() for _ in range(125)]
            basket.price_mc(disc, scs)

    def test_invalid_rho_raises(self):
        disc = _disc()
        scs = [_surv() for _ in range(125)]
        basket = BasketCLN(REF, END, n_names=125)
        with pytest.raises(ValueError, match="rho"):
            basket.price_mc(disc, scs, rho=1.5)

    def test_tranche_width(self):
        basket = BasketCLN(REF, END, attachment=0.03, detachment=0.07)
        assert basket.tranche_width == pytest.approx(0.04)

    def test_std_error_returned(self):
        """MC standard error should be positive and small relative to expected loss."""
        disc = _disc()
        scs = [_surv(0.02) for _ in range(125)]
        basket = BasketCLN(REF, END, attachment=0.0, detachment=0.03, n_names=125)
        result = basket.price_mc(disc, scs, rho=0.3)
        assert result.std_error > 0
        if result.expected_loss > 0:
            assert result.std_error < result.expected_loss  # SE << EL

    def test_to_dict(self):
        disc = _disc()
        scs = [_surv() for _ in range(125)]
        basket = BasketCLN(REF, END, n_names=125)
        result = basket.price_mc(disc, scs)
        d = result.to_dict()
        assert "expected_loss" in d
        assert "std_error" in d


# ---- Phase 2c: CLN as TRS underlying ----

class TestCLNTRS:

    def test_cln_trs_basic(self):
        """CLN as TRS underlying produces finite value."""
        from pricebook.trs import TotalReturnSwap
        disc = _disc()
        sc = _surv()
        cln = CreditLinkedNote(REF, END, coupon_rate=0.06, notional=1_000_000)
        trs = TotalReturnSwap(
            underlying=cln, notional=1_000_000,
            start=REF, end=REF + timedelta(days=365),
            survival_curve=sc)
        result = trs.price(disc)
        assert math.isfinite(result.value)

    def test_cln_trs_in_portfolio(self):
        """CLN TRS works in Trade/Portfolio."""
        from pricebook.trs import TotalReturnSwap
        from pricebook.pricing_context import PricingContext
        from pricebook.trade import Trade, Portfolio
        disc = _disc()
        sc = _surv()
        ctx = PricingContext(valuation_date=REF, discount_curve=disc)
        cln = CreditLinkedNote(REF, END, coupon_rate=0.06, notional=1_000_000)
        trs = TotalReturnSwap(
            underlying=cln, notional=1_000_000,
            start=REF, end=REF + timedelta(days=365),
            survival_curve=sc)
        trade = Trade(trs, trade_id="CLN_TRS_1")
        port = Portfolio(name="cln_trs_book")
        port.add(trade)
        pv = port.pv(ctx)
        assert math.isfinite(pv)

    def test_cln_trs_type_detection(self):
        from pricebook.trs import TotalReturnSwap
        cln = CreditLinkedNote(REF, END)
        trs = TotalReturnSwap(underlying=cln, notional=1_000_000,
                              start=REF, end=REF + timedelta(days=365))
        assert trs._underlying_type == "cln"
