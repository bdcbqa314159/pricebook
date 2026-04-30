"""Tests for loan credit: recovery, stochastic LGD, spread prepay, regulatory."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.loan_credit import (
    RecoveryModel, StochasticRecovery, SpreadPrepayModel,
    lgd_regulatory, capital_requirement,
    RECOVERY_BY_SENIORITY, REGULATORY_LGD,
)


# ---- Recovery by seniority ----

class TestRecoveryModel:

    def test_1l_higher_than_2l(self):
        r1 = RecoveryModel("1L").expected_recovery()
        r2 = RecoveryModel("2L").expected_recovery()
        assert r1 > r2

    def test_senior_higher_than_sub(self):
        rs = RecoveryModel("senior").expected_recovery()
        rb = RecoveryModel("sub").expected_recovery()
        assert rs > rb

    def test_industry_adjustment(self):
        r_util = RecoveryModel("1L", industry="utilities").expected_recovery()
        r_retail = RecoveryModel("1L", industry="retail").expected_recovery()
        assert r_util > r_retail

    def test_bounded(self):
        for sen in RECOVERY_BY_SENIORITY:
            r = RecoveryModel(sen).expected_recovery()
            assert 0 <= r <= 1

    def test_beta_distribution(self):
        model = RecoveryModel("1L")
        a, b = model.recovery_distribution()
        assert a > 0 and b > 0

    def test_sample(self):
        model = RecoveryModel("1L")
        samples = model.sample(n=10_000)
        assert samples.shape == (10_000,)
        assert 0.6 < samples.mean() < 0.9  # near 77%
        assert samples.min() >= 0 and samples.max() <= 1

    def test_unknown_seniority_raises(self):
        with pytest.raises(ValueError, match="Unknown seniority"):
            RecoveryModel("AAA")

    def test_round_trip(self):
        model = RecoveryModel("2L", industry="energy")
        d = model.to_dict()
        model2 = RecoveryModel.from_dict(d)
        assert model2.seniority == "2L"
        assert model2.expected_recovery() == model.expected_recovery()


# ---- Stochastic recovery ----

class TestStochasticRecovery:

    def test_negative_correlation(self):
        """Negative corr: when default_normal is high (bad), recovery is low."""
        model = RecoveryModel("1L")
        stoch = StochasticRecovery(model, default_correlation=-0.5)
        # High default normal → bad credit → low recovery
        high_default = np.full(10_000, 2.0)  # 2σ bad
        low_default = np.full(10_000, -2.0)  # 2σ good
        rec_bad = stoch.sample_correlated(high_default).mean()
        rec_good = stoch.sample_correlated(low_default).mean()
        assert rec_bad < rec_good

    def test_zero_correlation(self):
        """Zero corr: recovery independent of default."""
        model = RecoveryModel("1L")
        stoch = StochasticRecovery(model, default_correlation=0.0)
        high = stoch.sample_correlated(np.full(50_000, 2.0)).mean()
        low = stoch.sample_correlated(np.full(50_000, -2.0)).mean()
        # Should be similar (within sampling noise)
        assert abs(high - low) < 0.05

    def test_samples_in_range(self):
        model = RecoveryModel("2L")
        stoch = StochasticRecovery(model, default_correlation=-0.3)
        samples = stoch.sample_correlated(np.random.standard_normal(10_000))
        assert samples.min() >= 0
        assert samples.max() <= 1

    def test_downturn_lgd(self):
        assert StochasticRecovery.lgd_downturn(0.77) > 1 - 0.77
        assert StochasticRecovery.lgd_downturn(0.77, 0.75) == pytest.approx(1 - 0.77 * 0.75)

    def test_invalid_corr(self):
        with pytest.raises(ValueError, match="correlation"):
            StochasticRecovery(RecoveryModel("1L"), default_correlation=1.5)


# ---- Spread-dependent prepayment ----

class TestSpreadPrepay:

    def test_tight_market_high_cpr(self):
        """When market spread << loan spread: high CPR (refinancing)."""
        model = SpreadPrepayModel(loan_spread=0.04)
        cpr = model.conditional_cpr(market_spread=0.02)
        assert cpr > 0.30  # strong incentive

    def test_wide_market_low_cpr(self):
        """When market spread >> loan spread: low CPR (no incentive)."""
        model = SpreadPrepayModel(loan_spread=0.04)
        cpr = model.conditional_cpr(market_spread=0.06)
        assert cpr < 0.15  # near base rate

    def test_at_the_money(self):
        """At-the-money: CPR near base + some."""
        model = SpreadPrepayModel(loan_spread=0.04, call_premium=0.0)
        cpr = model.conditional_cpr(market_spread=0.04)
        assert 0.30 < cpr < 0.50  # near midpoint

    def test_s_curve_monotonic(self):
        """CPR should decrease as market spread increases."""
        model = SpreadPrepayModel(loan_spread=0.04)
        spreads = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
        cprs = model.s_curve(spreads)
        for i in range(1, len(cprs)):
            assert cprs[i] <= cprs[i-1] + 1e-10

    def test_call_premium_shifts(self):
        """Call premium shifts the S-curve: harder to refinance."""
        no_prem = SpreadPrepayModel(loan_spread=0.04, call_premium=0.0)
        with_prem = SpreadPrepayModel(loan_spread=0.04, call_premium=0.02)
        # At same market spread, call premium reduces incentive
        assert no_prem.conditional_cpr(0.03) > with_prem.conditional_cpr(0.03)

    def test_round_trip(self):
        model = SpreadPrepayModel(loan_spread=0.04, call_premium=0.01, base_cpr=0.12)
        d = model.to_dict()
        model2 = SpreadPrepayModel.from_dict(d)
        assert model2.conditional_cpr(0.03) == model.conditional_cpr(0.03)


# ---- Regulatory LGD ----

class TestRegulatoryLGD:

    def test_downturn_higher(self):
        lgd_ttc = lgd_regulatory("senior_unsecured", downturn=False)
        lgd_dt = lgd_regulatory("senior_unsecured", downturn=True)
        assert lgd_dt > lgd_ttc

    def test_sub_higher_than_senior(self):
        lgd_sr = lgd_regulatory("senior_unsecured")
        lgd_sub = lgd_regulatory("subordinated")
        assert lgd_sub > lgd_sr

    def test_capital_requirement_positive(self):
        k = capital_requirement(pd=0.02, lgd=0.45, maturity=5.0)
        assert k > 0

    def test_capital_increases_with_pd(self):
        k_low = capital_requirement(pd=0.01, lgd=0.45)
        k_high = capital_requirement(pd=0.05, lgd=0.45)
        assert k_high > k_low

    def test_capital_increases_with_lgd(self):
        k_low = capital_requirement(pd=0.02, lgd=0.30)
        k_high = capital_requirement(pd=0.02, lgd=0.60)
        assert k_high > k_low

    def test_zero_pd_zero_capital(self):
        assert capital_requirement(pd=0.0, lgd=0.45) == 0.0
