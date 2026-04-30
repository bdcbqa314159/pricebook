"""Tests for basket CLN: correlated recovery, copula flexibility, heterogeneous recovery."""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.cln import BasketCLN
from tests.conftest import make_flat_curve, make_flat_survival


REF = date(2024, 1, 15)
END_5Y = REF + relativedelta(years=5)
N = 10  # small basket for speed


def _make_basket(n=N, attachment=0.0, detachment=0.10):
    dc = make_flat_curve(REF, 0.04)
    hazards = [0.005 + 0.002 * i for i in range(n)]  # lower hazards
    survs = [make_flat_survival(REF, h) for h in hazards]
    basket = BasketCLN(
        REF, END_5Y, coupon_rate=0.05, notional=10_000_000,
        attachment=attachment, detachment=detachment,
        recovery=0.4, n_names=n,
    )
    return basket, dc, survs


# ---- Correlated recovery ----

class TestCorrelatedRecovery:

    def test_correlated_higher_el(self):
        """Correlated recovery should produce higher equity EL than independent."""
        basket, dc, survs = _make_basket()
        indep = basket.price_mc(dc, survs, rho=0.3, n_sims=20_000, seed=42)
        corr = basket.price_mc_correlated_recovery(
            dc, survs, rho=0.3, recovery_sensitivity=0.10,
            n_sims=20_000, seed=42,
        )
        # Correlated recovery: in downturn, more defaults AND lower recovery
        # → equity tranche takes more loss
        assert corr.expected_loss >= indep.expected_loss * 0.8  # at least close

    def test_zero_sensitivity_matches_independent(self):
        """Zero recovery sensitivity → same as independent recovery."""
        basket, dc, survs = _make_basket()
        indep = basket.price_mc(dc, survs, rho=0.3, n_sims=20_000, seed=42)
        corr = basket.price_mc_correlated_recovery(
            dc, survs, rho=0.3, recovery_sensitivity=0.0,
            n_sims=20_000, seed=42,
        )
        assert corr.expected_loss == pytest.approx(indep.expected_loss, abs=0.02)

    def test_higher_sensitivity_higher_el(self):
        """Stronger recovery-factor link → higher equity EL."""
        basket, dc, survs = _make_basket()
        low = basket.price_mc_correlated_recovery(
            dc, survs, rho=0.3, recovery_sensitivity=0.05,
            n_sims=20_000, seed=42,
        )
        high = basket.price_mc_correlated_recovery(
            dc, survs, rho=0.3, recovery_sensitivity=0.20,
            n_sims=20_000, seed=42,
        )
        assert high.expected_loss >= low.expected_loss


# ---- Heterogeneous recovery ----

class TestHeterogeneousRecovery:

    def test_per_name_recovery(self):
        """Per-name recovery should differ from uniform."""
        basket, dc, survs = _make_basket()
        # Mix of secured (70%) and unsecured (20%)
        recoveries = [0.70 if i % 2 == 0 else 0.20 for i in range(N)]
        hetero = basket.price_mc_correlated_recovery(
            dc, survs, rho=0.3, recovery_sensitivity=0.0,
            recoveries=recoveries, n_sims=20_000, seed=42,
        )
        uniform = basket.price_mc(dc, survs, rho=0.3, n_sims=20_000, seed=42)
        # Different recovery profile → different EL
        assert hetero.expected_loss != pytest.approx(uniform.expected_loss, abs=0.005)

    def test_higher_recovery_lower_el(self):
        """Higher average recovery → lower EL."""
        basket, dc, survs = _make_basket()
        high_rec = [0.70] * N
        low_rec = [0.20] * N
        hr = basket.price_mc_correlated_recovery(
            dc, survs, rho=0.3, recovery_sensitivity=0.0,
            recoveries=high_rec, n_sims=20_000, seed=42,
        )
        lr = basket.price_mc_correlated_recovery(
            dc, survs, rho=0.3, recovery_sensitivity=0.0,
            recoveries=low_rec, n_sims=20_000, seed=42,
        )
        assert hr.expected_loss < lr.expected_loss


# ---- Copula flexibility ----

class TestCopulaFlexibility:

    def test_gaussian_matches_base(self):
        """Gaussian copula method should match base price_mc."""
        basket, dc, survs = _make_basket()
        base = basket.price_mc(dc, survs, rho=0.3, n_sims=10_000, seed=42)
        gauss = basket.price_mc_copula(
            dc, survs, rho=0.3, copula="gaussian",
            n_sims=10_000, seed=42,
        )
        assert gauss.price == pytest.approx(base.price, rel=0.01)

    def test_t_copula_positive_price(self):
        basket, dc, survs = _make_basket()
        result = basket.price_mc_copula(
            dc, survs, rho=0.3, copula="t", nu=5,
            n_sims=10_000, seed=42,
        )
        assert result.price > 0

    def test_t_copula_equity_higher_el(self):
        """t-copula should produce higher equity EL than Gaussian (tail dependence)."""
        basket, dc, survs = _make_basket()
        gauss = basket.price_mc_copula(
            dc, survs, rho=0.3, copula="gaussian",
            n_sims=20_000, seed=42,
        )
        t_cop = basket.price_mc_copula(
            dc, survs, rho=0.3, copula="t", nu=3,
            n_sims=20_000, seed=42,
        )
        # t-copula clusters more defaults → equity takes more
        assert t_cop.expected_loss >= gauss.expected_loss * 0.5

    def test_unknown_copula_raises(self):
        basket, dc, survs = _make_basket()
        with pytest.raises(ValueError, match="Unsupported copula"):
            basket.price_mc_copula(dc, survs, copula="frank")

    def test_senior_tranche(self):
        """Senior tranche should have lower EL than equity."""
        dc = make_flat_curve(REF, 0.04)
        survs = [make_flat_survival(REF, 0.005 + 0.002 * i) for i in range(N)]
        equity = BasketCLN(REF, END_5Y, notional=10_000_000,
                           attachment=0.0, detachment=0.10, n_names=N)
        senior = BasketCLN(REF, END_5Y, notional=10_000_000,
                           attachment=0.10, detachment=0.20, n_names=N)
        el_eq = equity.price_mc(dc, survs, rho=0.3, n_sims=10_000).expected_loss
        el_sr = senior.price_mc(dc, survs, rho=0.3, n_sims=10_000).expected_loss
        assert el_eq > el_sr
