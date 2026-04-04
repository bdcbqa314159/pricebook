"""Tests for Jarrow-Rudd and Leisen-Reimer binomial trees."""

import math
import pytest

from pricebook.binomial_jr_lr import (
    jr_european,
    jr_american,
    lr_european,
    lr_american,
    _peizer_pratt,
)
from pricebook.binomial_tree import binomial_european, binomial_american
from pricebook.black76 import OptionType
from pricebook.equity_option import equity_option_price


S, K, R, VOL, T = 100.0, 105.0, 0.05, 0.20, 1.0
BS_CALL = equity_option_price(S, K, R, VOL, T, OptionType.CALL)
BS_PUT = equity_option_price(S, K, R, VOL, T, OptionType.PUT)


# ---------------------------------------------------------------------------
# Step 1 — Jarrow-Rudd
# ---------------------------------------------------------------------------


class TestJR:
    def test_call_converges_to_bs(self):
        price = jr_european(S, K, R, VOL, T, n_steps=500)
        assert price == pytest.approx(BS_CALL, rel=0.01)

    def test_put_converges_to_bs(self):
        price = jr_european(S, K, R, VOL, T, n_steps=500, option_type=OptionType.PUT)
        assert price == pytest.approx(BS_PUT, rel=0.01)

    def test_call_positive(self):
        price = jr_european(S, K, R, VOL, T, n_steps=100)
        assert price > 0

    def test_put_positive(self):
        price = jr_european(S, K, R, VOL, T, n_steps=100, option_type=OptionType.PUT)
        assert price > 0

    def test_put_call_parity(self):
        call = jr_european(S, K, R, VOL, T, n_steps=300)
        put = jr_european(S, K, R, VOL, T, n_steps=300, option_type=OptionType.PUT)
        parity = call - put - (S - K * math.exp(-R * T))
        assert parity == pytest.approx(0.0, abs=0.05)

    def test_with_dividend(self):
        price = jr_european(S, K, R, VOL, T, n_steps=200, div_yield=0.02)
        price_no_div = jr_european(S, K, R, VOL, T, n_steps=200)
        assert price < price_no_div  # dividends reduce call value


# ---------------------------------------------------------------------------
# Step 2 — Leisen-Reimer
# ---------------------------------------------------------------------------


class TestLR:
    def test_call_high_accuracy(self):
        """LR with N=51 should match BS to 4+ significant figures."""
        price = lr_european(S, K, R, VOL, T, n_steps=51)
        assert price == pytest.approx(BS_CALL, rel=1e-4)

    def test_put_high_accuracy(self):
        price = lr_european(S, K, R, VOL, T, n_steps=51, option_type=OptionType.PUT)
        assert price == pytest.approx(BS_PUT, rel=1e-4)

    def test_atm_call(self):
        bs_atm = equity_option_price(100, 100, R, VOL, T)
        lr_atm = lr_european(100, 100, R, VOL, T, n_steps=51)
        assert lr_atm == pytest.approx(bs_atm, rel=1e-4)

    def test_deep_itm(self):
        bs = equity_option_price(100, 80, R, VOL, T)
        lr = lr_european(100, 80, R, VOL, T, n_steps=51)
        assert lr == pytest.approx(bs, rel=1e-3)

    def test_deep_otm(self):
        bs = equity_option_price(100, 130, R, VOL, T)
        lr = lr_european(100, 130, R, VOL, T, n_steps=51)
        assert lr == pytest.approx(bs, rel=1e-2)

    def test_put_call_parity(self):
        call = lr_european(S, K, R, VOL, T, n_steps=51)
        put = lr_european(S, K, R, VOL, T, n_steps=51, option_type=OptionType.PUT)
        parity = call - put - (S - K * math.exp(-R * T))
        assert parity == pytest.approx(0.0, abs=0.01)

    def test_with_dividend(self):
        price = lr_european(S, K, R, VOL, T, n_steps=51, div_yield=0.02)
        price_no_div = lr_european(S, K, R, VOL, T, n_steps=51)
        assert price < price_no_div


class TestPeizerPratt:
    def test_zero(self):
        assert _peizer_pratt(0.0, 51) == pytest.approx(0.5)

    def test_positive(self):
        assert _peizer_pratt(1.0, 51) > 0.5

    def test_negative(self):
        assert _peizer_pratt(-1.0, 51) < 0.5

    def test_symmetry(self):
        p_pos = _peizer_pratt(1.5, 101)
        p_neg = _peizer_pratt(-1.5, 101)
        assert p_pos + p_neg == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Step 3 — Convergence comparison
# ---------------------------------------------------------------------------


class TestConvergence:
    def test_lr_converges_faster_than_crr(self):
        """LR error at N=51 should be smaller than CRR error at N=51."""
        crr = binomial_european(S, K, R, VOL, T, n_steps=51)
        lr = lr_european(S, K, R, VOL, T, n_steps=51)
        err_crr = abs(crr - BS_CALL)
        err_lr = abs(lr - BS_CALL)
        assert err_lr < err_crr

    def test_all_converge_to_same_limit(self):
        """CRR, JR, LR all converge to Black-Scholes."""
        crr = binomial_european(S, K, R, VOL, T, n_steps=500)
        jr = jr_european(S, K, R, VOL, T, n_steps=500)
        lr = lr_european(S, K, R, VOL, T, n_steps=501)  # LR uses odd
        assert crr == pytest.approx(BS_CALL, rel=0.01)
        assert jr == pytest.approx(BS_CALL, rel=0.01)
        assert lr == pytest.approx(BS_CALL, rel=0.001)

    def test_lr_error_decreases(self):
        """LR error should decrease with more steps."""
        err_low = abs(lr_european(S, K, R, VOL, T, n_steps=21) - BS_CALL)
        err_high = abs(lr_european(S, K, R, VOL, T, n_steps=101) - BS_CALL)
        assert err_high < err_low


# ---------------------------------------------------------------------------
# Step 4 — American options
# ---------------------------------------------------------------------------


class TestAmerican:
    def test_jr_american_put_geq_european(self):
        eur = jr_european(S, K, R, VOL, T, n_steps=200, option_type=OptionType.PUT)
        amer = jr_american(S, K, R, VOL, T, n_steps=200, option_type=OptionType.PUT)
        assert amer >= eur - 0.01

    def test_lr_american_put_geq_european(self):
        eur = lr_european(S, K, R, VOL, T, n_steps=51, option_type=OptionType.PUT)
        amer = lr_american(S, K, R, VOL, T, n_steps=51, option_type=OptionType.PUT)
        assert amer >= eur - 0.01

    def test_crr_jr_lr_american_agree(self):
        """All three trees should give similar American put prices."""
        crr = binomial_american(S, K, R, VOL, T, n_steps=200, option_type=OptionType.PUT)
        jr = jr_american(S, K, R, VOL, T, n_steps=200, option_type=OptionType.PUT)
        lr = lr_american(S, K, R, VOL, T, n_steps=201, option_type=OptionType.PUT)
        assert jr == pytest.approx(crr, rel=0.02)
        assert lr == pytest.approx(crr, rel=0.02)

    def test_american_call_no_div_equals_european(self):
        """American call with no dividends = European call."""
        eur = lr_european(S, K, R, VOL, T, n_steps=51)
        amer = lr_american(S, K, R, VOL, T, n_steps=51)
        assert amer == pytest.approx(eur, rel=1e-4)
