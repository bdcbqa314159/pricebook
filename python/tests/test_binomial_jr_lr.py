"""Tests for Jarrow-Rudd and Leisen-Reimer binomial trees."""

import math
import pytest

from pricebook.numerical._trees import solve_tree, TreeMethod, ExerciseType, _peizer_pratt
from pricebook.models.black76 import OptionType
from pricebook.options.equity_option import equity_option_price


S, K, R, VOL, T = 100.0, 105.0, 0.05, 0.20, 1.0
BS_CALL = equity_option_price(S, K, R, VOL, T, OptionType.CALL)
BS_PUT = equity_option_price(S, K, R, VOL, T, OptionType.PUT)


def _tree(spot, strike, rate, vol, T, method, n, exercise=ExerciseType.EUROPEAN,
          opt_type=OptionType.CALL, q=0.0):
    is_call = str(getattr(opt_type, 'value', opt_type)).lower() != "put"
    return solve_tree(spot, strike, rate, vol, T, method, n, exercise,
                      is_call=is_call, div_yield=q).price


# ---------------------------------------------------------------------------
# Step 1 — Jarrow-Rudd
# ---------------------------------------------------------------------------


class TestJR:
    def test_call_converges_to_bs(self):
        price = _tree(S, K, R, VOL, T, TreeMethod.JR, 500)
        assert price == pytest.approx(BS_CALL, rel=0.01)

    def test_put_converges_to_bs(self):
        price = _tree(S, K, R, VOL, T, TreeMethod.JR, 500, opt_type=OptionType.PUT)
        assert price == pytest.approx(BS_PUT, rel=0.01)

    def test_call_positive(self):
        price = _tree(S, K, R, VOL, T, TreeMethod.JR, 100)
        assert price > 0

    def test_put_positive(self):
        price = _tree(S, K, R, VOL, T, TreeMethod.JR, 100, opt_type=OptionType.PUT)
        assert price > 0

    def test_put_call_parity(self):
        call = _tree(S, K, R, VOL, T, TreeMethod.JR, 300)
        put = _tree(S, K, R, VOL, T, TreeMethod.JR, 300, opt_type=OptionType.PUT)
        parity = call - put - (S - K * math.exp(-R * T))
        assert parity == pytest.approx(0.0, abs=0.05)

    def test_with_dividend(self):
        price = _tree(S, K, R, VOL, T, TreeMethod.JR, 200, q=0.02)
        price_no_div = _tree(S, K, R, VOL, T, TreeMethod.JR, 200)
        assert price < price_no_div  # dividends reduce call value


# ---------------------------------------------------------------------------
# Step 2 — Leisen-Reimer
# ---------------------------------------------------------------------------


class TestLR:
    def test_call_high_accuracy(self):
        """LR with N=51 should match BS to 4+ significant figures."""
        price = _tree(S, K, R, VOL, T, TreeMethod.LR, 51)
        assert price == pytest.approx(BS_CALL, rel=1e-4)

    def test_put_high_accuracy(self):
        price = _tree(S, K, R, VOL, T, TreeMethod.LR, 51, opt_type=OptionType.PUT)
        assert price == pytest.approx(BS_PUT, rel=1e-4)

    def test_atm_call(self):
        bs_atm = equity_option_price(100, 100, R, VOL, T)
        lr_atm = _tree(100, 100, R, VOL, T, TreeMethod.LR, 51)
        assert lr_atm == pytest.approx(bs_atm, rel=1e-4)

    def test_deep_itm(self):
        bs = equity_option_price(100, 80, R, VOL, T)
        lr = _tree(100, 80, R, VOL, T, TreeMethod.LR, 51)
        assert lr == pytest.approx(bs, rel=1e-3)

    def test_deep_otm(self):
        bs = equity_option_price(100, 130, R, VOL, T)
        lr = _tree(100, 130, R, VOL, T, TreeMethod.LR, 51)
        assert lr == pytest.approx(bs, rel=1e-2)

    def test_put_call_parity(self):
        call = _tree(S, K, R, VOL, T, TreeMethod.LR, 51)
        put = _tree(S, K, R, VOL, T, TreeMethod.LR, 51, opt_type=OptionType.PUT)
        parity = call - put - (S - K * math.exp(-R * T))
        assert parity == pytest.approx(0.0, abs=0.01)

    def test_with_dividend(self):
        price = _tree(S, K, R, VOL, T, TreeMethod.LR, 51, q=0.02)
        price_no_div = _tree(S, K, R, VOL, T, TreeMethod.LR, 51)
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
        crr = _tree(S, K, R, VOL, T, TreeMethod.CRR, 51)
        lr = _tree(S, K, R, VOL, T, TreeMethod.LR, 51)
        err_crr = abs(crr - BS_CALL)
        err_lr = abs(lr - BS_CALL)
        assert err_lr < err_crr

    def test_all_converge_to_same_limit(self):
        """CRR, JR, LR all converge to Black-Scholes."""
        crr = _tree(S, K, R, VOL, T, TreeMethod.CRR, 500)
        jr = _tree(S, K, R, VOL, T, TreeMethod.JR, 500)
        lr = _tree(S, K, R, VOL, T, TreeMethod.LR, 501)  # LR uses odd
        assert crr == pytest.approx(BS_CALL, rel=0.01)
        assert jr == pytest.approx(BS_CALL, rel=0.01)
        assert lr == pytest.approx(BS_CALL, rel=0.001)

    def test_lr_error_decreases(self):
        """LR error should decrease with more steps."""
        err_low = abs(_tree(S, K, R, VOL, T, TreeMethod.LR, 21) - BS_CALL)
        err_high = abs(_tree(S, K, R, VOL, T, TreeMethod.LR, 101) - BS_CALL)
        assert err_high < err_low


# ---------------------------------------------------------------------------
# Step 4 — American options
# ---------------------------------------------------------------------------


class TestAmerican:
    def test_jr_american_put_geq_european(self):
        eur = _tree(S, K, R, VOL, T, TreeMethod.JR, 200, opt_type=OptionType.PUT)
        amer = _tree(S, K, R, VOL, T, TreeMethod.JR, 200, ExerciseType.AMERICAN, OptionType.PUT)
        assert amer >= eur - 0.01

    def test_lr_american_put_geq_european(self):
        eur = _tree(S, K, R, VOL, T, TreeMethod.LR, 51, opt_type=OptionType.PUT)
        amer = _tree(S, K, R, VOL, T, TreeMethod.LR, 51, ExerciseType.AMERICAN, OptionType.PUT)
        assert amer >= eur - 0.01

    def test_crr_jr_lr_american_agree(self):
        """All three trees should give similar American put prices."""
        crr = _tree(S, K, R, VOL, T, TreeMethod.CRR, 200, ExerciseType.AMERICAN, OptionType.PUT)
        jr = _tree(S, K, R, VOL, T, TreeMethod.JR, 200, ExerciseType.AMERICAN, OptionType.PUT)
        lr = _tree(S, K, R, VOL, T, TreeMethod.LR, 201, ExerciseType.AMERICAN, OptionType.PUT)
        assert jr == pytest.approx(crr, rel=0.02)
        assert lr == pytest.approx(crr, rel=0.02)

    def test_american_call_no_div_equals_european(self):
        """American call with no dividends = European call."""
        eur = _tree(S, K, R, VOL, T, TreeMethod.LR, 51)
        amer = _tree(S, K, R, VOL, T, TreeMethod.LR, 51, ExerciseType.AMERICAN)
        assert amer == pytest.approx(eur, rel=1e-4)
