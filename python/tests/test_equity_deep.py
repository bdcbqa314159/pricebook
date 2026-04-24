"""Deep tests for equity — DD6 hardening.

Covers: put-call parity, Greeks signs, theta with dividends, forward with
discrete dividends, jump model convergence, TRS, borrow cost.
"""

import math
import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.equity_option import (
    equity_option_price, equity_delta, equity_gamma, equity_vega,
    equity_theta, equity_rho,
)
from pricebook.equity_forward import EquityForward, pv_dividends
from pricebook.equity_jumps import kou_equity_price, merton_equity_hybrid
from pricebook.black76 import OptionType
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)
S, K, r, vol, T = 100.0, 100.0, 0.05, 0.20, 1.0


class TestPutCallParity:

    def test_parity_no_dividends(self):
        call = equity_option_price(S, K, r, vol, T, OptionType.CALL)
        put = equity_option_price(S, K, r, vol, T, OptionType.PUT)
        parity = call - put - (S - K * math.exp(-r * T))
        assert abs(parity) < 1e-10

    def test_parity_with_dividends(self):
        q = 0.02
        call = equity_option_price(S, K, r, vol, T, OptionType.CALL, div_yield=q)
        put = equity_option_price(S, K, r, vol, T, OptionType.PUT, div_yield=q)
        parity = call - put - (S * math.exp(-q * T) - K * math.exp(-r * T))
        assert abs(parity) < 1e-10


class TestGreeksSigns:

    def test_call_delta_between_0_and_1(self):
        d = equity_delta(S, K, r, vol, T, OptionType.CALL)
        assert 0 < d < 1

    def test_put_delta_between_neg1_and_0(self):
        d = equity_delta(S, K, r, vol, T, OptionType.PUT)
        assert -1 < d < 0

    def test_gamma_positive(self):
        g = equity_gamma(S, K, r, vol, T)
        assert g > 0

    def test_vega_positive(self):
        v = equity_vega(S, K, r, vol, T)
        assert v > 0

    def test_call_theta_negative(self):
        th = equity_theta(S, K, r, vol, T, OptionType.CALL)
        assert th < 0

    def test_call_rho_positive(self):
        rh = equity_rho(S, K, r, vol, T, OptionType.CALL)
        assert rh > 0

    def test_put_rho_negative(self):
        rh = equity_rho(S, K, r, vol, T, OptionType.PUT)
        assert rh < 0


class TestThetaWithDividends:

    def test_theta_div_yield_bump_match(self):
        """Full theta matches bump-and-reprice when div_yield != 0."""
        q = 0.03
        dt = 1e-5
        p1 = equity_option_price(S, K, r, vol, T, OptionType.CALL, div_yield=q)
        p2 = equity_option_price(S, K, r, vol, T - dt, OptionType.CALL, div_yield=q)
        bump_theta = (p2 - p1) / dt
        th = equity_theta(S, K, r, vol, T, OptionType.CALL, div_yield=q)
        assert th == pytest.approx(bump_theta, rel=0.02)


class TestEquityForward:

    def test_forward_no_dividends(self):
        curve = make_flat_curve(REF, 0.05)
        mat = REF + relativedelta(years=1)
        fwd = EquityForward(100.0, mat, REF)
        f = fwd.forward_price(curve)
        expected = 100.0 * math.exp(0.05)
        assert f == pytest.approx(expected, rel=0.01)

    def test_borrow_cost_negative_applied(self):
        """Negative borrow cost (lending fee) should reduce forward."""
        curve = make_flat_curve(REF, 0.05)
        mat = REF + relativedelta(years=1)
        fwd_pos = EquityForward(100.0, mat, REF, borrow_cost=0.01)
        fwd_neg = EquityForward(100.0, mat, REF, borrow_cost=-0.01)
        assert fwd_neg.forward_price(curve) < fwd_pos.forward_price(curve)


class TestJumpModels:

    def test_kou_zero_jumps_matches_bs(self):
        """Kou with lambda=0 should match Black-Scholes."""
        kou = kou_equity_price(S, K, r, 0.0, vol, T, lambda_jump=0.0,
                         p=0.5, eta1=10, eta2=10, is_call=True)
        bs = equity_option_price(S, K, r, vol, T, OptionType.CALL)
        assert kou.price == pytest.approx(bs, rel=0.02)

    def test_merton_positive(self):
        result = merton_equity_hybrid(S, K, r, 0.0, vol, T,
                                       lambda_jump=1.0, jump_mean=0.0, jump_vol=0.1)
        assert result.price > 0

    def test_jumps_increase_otm_price(self):
        """Jumps increase OTM option prices (fat tails)."""
        bs = equity_option_price(S, 130, r, vol, T, OptionType.CALL)
        kou = kou_equity_price(S, 130, r, 0.0, vol, T, lambda_jump=2.0,
                         p=0.6, eta1=10, eta2=5, is_call=True)
        assert kou.price > bs
