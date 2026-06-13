"""Regression for L2 Wave-2 audit — single-pass sweep of `T<=0 or vol<=0`
degenerate-branch defects across FX, equity, and inflation modules.

All seven defects share one of three patterns:

(a) Spot-vs-forward indicator — at vol=0/T>0 the deterministic terminal
    equals the forward, not the spot.  An OTM-spot but ITM-forward
    position is misclassified by a spot-only indicator.

(b) Missing discount factor — even when the indicator is right, the
    payoff is at maturity and must be present-valued.  Pre-fix code
    returned undiscounted intrinsic.

(c) Missing ATM half-limit — at ``forward == strike`` the indicator
    sits exactly on the boundary; the one-sided limit is half the ITM
    value, not 0.

Defects fixed in this sweep:

| # | Site                                           | Pattern  |
|---|------------------------------------------------|----------|
| 1 | ``fx.fx_option.fx_spot_delta``                  | (a)+(c)  |
| 2 | ``fx.fx_option.fx_forward_delta``               | (a)+(c)  |
| 3 | ``fx.fx_american._gk_european``                 | (a)+(b)  |
| 4 | ``equity.equity_exotic.equity_digital_cash``    | (a)+(b)+(c) |
| 5 | ``equity.equity_exotic.equity_digital_asset``   | (a)+(b)+(c) |
| 6 | ``fx.fx_exotic_extensions.fx_digital_option``   | (b)+(c)  |
| 7 | ``fixed_income.inflation_bond_advanced.deflation_floor_value`` | (b) linearised intrinsic |

Each test verifies the exact closed-form deterministic limit and confirms
the pre-fix value was wrong.
"""

from __future__ import annotations

import math

import pytest

from pricebook.models.black76 import OptionType


# ═══════════════════════════════════════════════════════════════
# 1+2. FX spot/forward delta
# ═══════════════════════════════════════════════════════════════

class TestFxSpotDeltaDegenerate:
    def test_vol_zero_otm_spot_itm_forward(self):
        """Spot=95, K=100, r_d=10%, T=1 → fwd≈105>K so ITM call.
        Pre-fix returned 0 (since spot<K); should return exp(-r_f·T)=1."""
        from pricebook.fx.fx_option import fx_spot_delta
        delta = fx_spot_delta(spot=95.0, strike=100.0, r_d=0.10, r_f=0.0,
                              vol=0.0, T=1.0, option_type=OptionType.CALL)
        assert delta == pytest.approx(1.0, abs=1e-12)

    def test_vol_zero_atm_forward_half_limit(self):
        """spot=100, K=100, r_d=r_f → fwd=100=K (ATM forward).
        Pre-fix had no ATM branch → returned 0; should return 0.5·exp(-r_f·T)."""
        from pricebook.fx.fx_option import fx_spot_delta
        delta = fx_spot_delta(spot=100.0, strike=100.0, r_d=0.05, r_f=0.05,
                              vol=0.0, T=1.0, option_type=OptionType.CALL)
        expected = 0.5 * math.exp(-0.05)
        assert delta == pytest.approx(expected, abs=1e-12)

    def test_T_zero_immediate_indicator(self):
        from pricebook.fx.fx_option import fx_spot_delta
        # T=0: indicator on spot, factor = 1.
        delta = fx_spot_delta(spot=110.0, strike=100.0, r_d=0.05, r_f=0.02,
                              vol=0.2, T=0.0, option_type=OptionType.CALL)
        assert delta == pytest.approx(1.0, abs=1e-12)

    def test_interior_unchanged(self):
        from pricebook.fx.fx_option import fx_spot_delta
        # Interior (vol>0, T>0): formula unchanged.
        delta = fx_spot_delta(spot=100.0, strike=100.0, r_d=0.05, r_f=0.02,
                              vol=0.15, T=1.0, option_type=OptionType.CALL)
        assert 0.4 < delta < 0.7


class TestFxForwardDeltaDegenerate:
    def test_vol_zero_itm_forward_returns_one(self):
        """Pre-fix used spot indicator; fix uses forward indicator."""
        from pricebook.fx.fx_option import fx_forward_delta
        delta = fx_forward_delta(spot=95.0, strike=100.0, r_d=0.10, r_f=0.0,
                                 vol=0.0, T=1.0, option_type=OptionType.CALL)
        assert delta == pytest.approx(1.0, abs=1e-12)

    def test_vol_zero_atm_half(self):
        from pricebook.fx.fx_option import fx_forward_delta
        delta = fx_forward_delta(spot=100.0, strike=100.0, r_d=0.05, r_f=0.05,
                                 vol=0.0, T=1.0, option_type=OptionType.CALL)
        assert delta == pytest.approx(0.5, abs=1e-12)

    def test_vol_zero_put_itm(self):
        from pricebook.fx.fx_option import fx_forward_delta
        delta = fx_forward_delta(spot=120.0, strike=100.0, r_d=0.0, r_f=0.10,
                                 vol=0.0, T=1.0, option_type=OptionType.PUT)
        # forward = 120·exp(-0.10) ≈ 108.6 > 100 → put is OTM → 0.
        assert delta == 0.0


# ═══════════════════════════════════════════════════════════════
# 3. _gk_european (FX American European fallback)
# ═══════════════════════════════════════════════════════════════

class TestGkEuropeanDegenerate:
    def test_vol_zero_itm_call_uses_forward_intrinsic_with_df(self):
        """Pre-fix returned max(spot-K, 0) = 0 (spot<K).
        Fix: forward = 95·exp(0.10) > 100 → df_d · (F − K).
        """
        from pricebook.fx.fx_american import _gk_european
        price = _gk_european(spot=95.0, strike=100.0, r_d=0.10, r_f=0.0,
                             vol=0.0, T=1.0, is_call=True)
        fwd = 95.0 * math.exp(0.10)
        df_d = math.exp(-0.10)
        expected = df_d * (fwd - 100.0)
        assert price == pytest.approx(expected, abs=1e-12)
        assert price > 0  # pre-fix returned 0

    def test_T_zero_uses_spot_intrinsic(self):
        from pricebook.fx.fx_american import _gk_european
        price = _gk_european(spot=110.0, strike=100.0, r_d=0.05, r_f=0.02,
                             vol=0.2, T=0.0, is_call=True)
        assert price == pytest.approx(10.0, abs=1e-12)


# ═══════════════════════════════════════════════════════════════
# 4. equity_digital_cash
# ═══════════════════════════════════════════════════════════════

class TestEquityDigitalCashDegenerate:
    def test_vol_zero_itm_forward_includes_df(self):
        """Pre-fix returned payout undiscounted (= 1.0).
        Fix: payout · df.
        """
        from pricebook.equity.equity_exotic import equity_digital_cash
        r = 0.05
        res = equity_digital_cash(spot=120.0, strike=100.0, rate=r,
                                  dividend_yield=0.0, vol=0.0, T=1.0,
                                  payout=1.0, is_call=True)
        assert res.price == pytest.approx(math.exp(-r), abs=1e-12)

    def test_vol_zero_otm_spot_itm_forward_pays_df(self):
        """spot=95 < K=100 but forward = 95·exp(0.10) > K → ITM fwd.
        Pre-fix returned 0 (spot indicator); fix returns df·payout.
        """
        from pricebook.equity.equity_exotic import equity_digital_cash
        res = equity_digital_cash(spot=95.0, strike=100.0, rate=0.10,
                                  dividend_yield=0.0, vol=0.0, T=1.0,
                                  payout=1.0, is_call=True)
        assert res.price == pytest.approx(math.exp(-0.10), abs=1e-12)

    def test_vol_zero_atm_half_df(self):
        from pricebook.equity.equity_exotic import equity_digital_cash
        # rate==div_yield → forward = spot = K → ATM forward.
        res = equity_digital_cash(spot=100.0, strike=100.0, rate=0.05,
                                  dividend_yield=0.05, vol=0.0, T=1.0,
                                  payout=2.0, is_call=True)
        assert res.price == pytest.approx(0.5 * 2.0 * math.exp(-0.05), abs=1e-12)


# ═══════════════════════════════════════════════════════════════
# 5. equity_digital_asset
# ═══════════════════════════════════════════════════════════════

class TestEquityDigitalAssetDegenerate:
    def test_vol_zero_itm_returns_dividend_discounted_spot(self):
        from pricebook.equity.equity_exotic import equity_digital_asset
        q = 0.02
        res = equity_digital_asset(spot=120.0, strike=100.0, rate=0.05,
                                   dividend_yield=q, vol=0.0, T=1.0,
                                   is_call=True)
        # forward = 120·exp(0.03) > 100 → ITM.  Asset-or-nothing PV = spot·exp(-qT).
        assert res.price == pytest.approx(120.0 * math.exp(-q), abs=1e-12)

    def test_vol_zero_otm_spot_itm_forward_pays(self):
        from pricebook.equity.equity_exotic import equity_digital_asset
        # spot < K but forward > K (high rate, no div).
        res = equity_digital_asset(spot=95.0, strike=100.0, rate=0.10,
                                   dividend_yield=0.0, vol=0.0, T=1.0,
                                   is_call=True)
        # Pre-fix returned 0 (spot < K).  Fix: spot · exp(-0·T) = 95.
        assert res.price == pytest.approx(95.0, abs=1e-12)


# ═══════════════════════════════════════════════════════════════
# 6. fx_digital_option
# ═══════════════════════════════════════════════════════════════

class TestFxDigitalOptionDegenerate:
    def test_vol_zero_domestic_payout_discounts_by_df_d(self):
        from pricebook.fx.fx_exotic_extensions import fx_digital_option
        # forward = 1.10 · exp(0.05) ≈ 1.156 > strike=1.10 → ITM call.
        res = fx_digital_option(spot=1.10, strike=1.10, r_d=0.05, r_f=0.0,
                                vol=0.0, T=1.0, payout=1000.0,
                                option_type="call", payout_currency="domestic")
        # Pre-fix returned undiscounted 1000.  Fix: 1000·exp(-0.05·1).
        assert res.price == pytest.approx(1000.0 * math.exp(-0.05), abs=1e-9)

    def test_vol_zero_foreign_payout_discounts_by_df_f(self):
        from pricebook.fx.fx_exotic_extensions import fx_digital_option
        res = fx_digital_option(spot=1.10, strike=1.10, r_d=0.05, r_f=0.02,
                                vol=0.0, T=1.0, payout=1000.0,
                                option_type="call", payout_currency="foreign")
        # Forward > strike (rate diff is +3%) → ITM, foreign payout → df_f.
        assert res.price == pytest.approx(1000.0 * math.exp(-0.02), abs=1e-9)


# ═══════════════════════════════════════════════════════════════
# 7. inflation deflation floor
# ═══════════════════════════════════════════════════════════════

class TestDeflationFloorDegenerate:
    def test_vol_zero_uses_exact_intrinsic(self):
        """Pre-fix used linearised ``-breakeven·T`` which over-states for large |breakeven·T|.
        breakeven=-5%, T=30 → pre-fix: 0.05·30=1.5 (capped to 1.0 only by max(...,0)?).
        Actually pre-fix returned 1.5 since the original max(...) only floored at 0.
        Fix: ``max(1 - exp(-1.5), 0) = 1 - 0.2231 ≈ 0.7769``.
        """
        from pricebook.fixed_income.inflation_bond_advanced import (
            deflation_floor_value,
        )
        res = deflation_floor_value(breakeven=-0.05,
                                        inflation_vol=0.0, T=30.0,
                                        discount_factor=1.0)
        expected_intrinsic = 1.0 - math.exp(-0.05 * 30.0)
        assert res.floor_value == pytest.approx(expected_intrinsic, abs=1e-12)
        # Pre-fix would have given 1.5.
        assert res.floor_value < 1.0

    def test_vol_zero_no_deflation_when_positive_breakeven(self):
        from pricebook.fixed_income.inflation_bond_advanced import (
            deflation_floor_value,
        )
        res = deflation_floor_value(breakeven=0.025,
                                        inflation_vol=0.0, T=10.0,
                                        discount_factor=0.9)
        assert res.floor_value == 0.0

    def test_vol_zero_small_breakeven_matches_linearisation(self):
        """Sanity: for tiny breakeven·T, exact and linearised agree."""
        from pricebook.fixed_income.inflation_bond_advanced import (
            deflation_floor_value,
        )
        be = -0.001  # tiny
        T = 1.0
        res = deflation_floor_value(breakeven=be, inflation_vol=0.0, T=T,
                                        discount_factor=1.0)
        exact = 1.0 - math.exp(be * T)
        linear = -be * T
        # Should match both to leading order.
        assert res.floor_value == pytest.approx(exact, abs=1e-12)
        assert abs(res.floor_value - linear) < 1e-5
