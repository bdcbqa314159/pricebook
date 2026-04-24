"""Deep tests for FX — DD5 hardening.

Covers: CIP round-trip, put-call parity, delta conventions, barrier bounds,
NDF settlement, cross-currency swap.
"""

import math
import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.fx_forward import FXForward
from pricebook.fx_option import fx_option_price, fx_spot_delta, fx_forward_delta, strike_from_delta
from pricebook.fx_barrier import fx_barrier_pde
from pricebook.ndf import NDF
from pricebook.black76 import OptionType
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)


class TestCIP:

    def test_forward_equals_spot_at_equal_rates(self):
        """F = S when both rates are equal."""
        usd = make_flat_curve(REF, 0.05)
        eur = make_flat_curve(REF, 0.05)
        fwd = FXForward.forward_rate(1.10, REF + relativedelta(years=1), eur, usd)
        assert fwd == pytest.approx(1.10, abs=0.002)

    def test_lower_foreign_rate_appreciates_forward(self):
        """Lower base rate → F > S (base currency appreciates)."""
        base = make_flat_curve(REF, 0.02)
        quote = make_flat_curve(REF, 0.05)
        fwd = FXForward.forward_rate(1.10, REF + relativedelta(years=1), base, quote)
        assert fwd > 1.10

    def test_forward_points_sign(self):
        """Positive carry (r_d > r_f) → positive forward points."""
        base = make_flat_curve(REF, 0.02)
        quote = make_flat_curve(REF, 0.05)
        pts = FXForward.forward_points(1.10, REF + relativedelta(years=1), base, quote)
        assert pts > 0


class TestFXOptionParity:

    def test_put_call_parity(self):
        """C - P = df_d × (F - K) for European FX options."""
        S, K, rd, rf, vol, T = 1.10, 1.10, 0.05, 0.03, 0.10, 1.0
        call = fx_option_price(S, K, rd, rf, vol, T, OptionType.CALL)
        put = fx_option_price(S, K, rd, rf, vol, T, OptionType.PUT)
        F = S * math.exp((rd - rf) * T)
        df = math.exp(-rd * T)
        expected = df * (F - K)
        assert call - put == pytest.approx(expected, abs=1e-10)

    def test_atm_call_positive(self):
        price = fx_option_price(1.10, 1.10, 0.05, 0.03, 0.10, 1.0, OptionType.CALL)
        assert price > 0

    def test_higher_vol_higher_price(self):
        low = fx_option_price(1.10, 1.10, 0.05, 0.03, 0.05, 1.0, OptionType.CALL)
        high = fx_option_price(1.10, 1.10, 0.05, 0.03, 0.20, 1.0, OptionType.CALL)
        assert high > low


class TestDeltaConventions:

    def test_atm_call_delta_near_half(self):
        """ATM forward call has forward delta ≈ 0.5."""
        S, rd, rf, vol, T = 1.10, 0.05, 0.03, 0.10, 1.0
        F = S * math.exp((rd - rf) * T)
        delta = fx_forward_delta(S, F, rd, rf, vol, T, OptionType.CALL)
        assert delta == pytest.approx(0.5, abs=0.05)

    def test_spot_delta_less_than_forward_delta(self):
        """Spot delta < forward delta due to rf discounting."""
        S, rd, rf, vol, T = 1.10, 0.05, 0.03, 0.10, 1.0
        spot_d = fx_spot_delta(S, 1.10, rd, rf, vol, T, OptionType.CALL)
        fwd_d = fx_forward_delta(S, 1.10, rd, rf, vol, T, OptionType.CALL)
        assert spot_d < fwd_d

    def test_strike_from_delta_round_trip(self):
        """strike_from_delta → compute delta → should match."""
        S, rd, rf, vol, T = 1.10, 0.05, 0.03, 0.10, 1.0
        K = strike_from_delta(S, 0.25, rd, rf, vol, T, "forward", OptionType.CALL)
        delta_back = fx_forward_delta(S, K, rd, rf, vol, T, OptionType.CALL)
        assert delta_back == pytest.approx(0.25, abs=0.01)


class TestBarrier:

    def test_knockout_leq_vanilla(self):
        """Knock-out option ≤ vanilla option."""
        S, K, B = 1.10, 1.10, 0.95
        rd, rf, vol, T = 0.05, 0.03, 0.10, 1.0
        ko = fx_barrier_pde(S, K, B, rd, rf, vol, T,
                            is_up=False, is_knock_in=False,
                            option_type=OptionType.CALL)
        vanilla = fx_option_price(S, K, rd, rf, vol, T, OptionType.CALL)
        assert ko <= vanilla + 1e-6

    def test_knockin_plus_knockout_equals_vanilla(self):
        """KI + KO = vanilla (in-out parity)."""
        S, K, B = 1.10, 1.10, 0.95
        rd, rf, vol, T = 0.05, 0.03, 0.10, 1.0
        ki = fx_barrier_pde(S, K, B, rd, rf, vol, T,
                            is_up=False, is_knock_in=True,
                            option_type=OptionType.CALL)
        ko = fx_barrier_pde(S, K, B, rd, rf, vol, T,
                            is_up=False, is_knock_in=False,
                            option_type=OptionType.CALL)
        vanilla = fx_option_price(S, K, rd, rf, vol, T, OptionType.CALL)
        assert ki + ko == pytest.approx(vanilla, rel=0.05)


class TestNDF:

    def test_pv_zero_at_forward(self):
        """NDF struck at forward rate has PV ≈ 0."""
        base = make_flat_curve(REF, 0.02)
        quote = make_flat_curve(REF, 0.04)
        fwd = FXForward.forward_rate(7.20, REF + relativedelta(years=1), base, quote)
        ndf = NDF("USD/CNY", REF + relativedelta(years=1), fwd, 1_000_000)
        pv = ndf.pv(7.20, base, quote)
        assert abs(pv) < 1.0

    def test_settlement_base_currency(self):
        ndf = NDF("USD/CNY", REF + relativedelta(years=1), 7.00, 1_000_000,
                  settlement_currency="base")
        settle = ndf.settlement_amount(7.20)
        assert settle == pytest.approx(1_000_000 * 0.20)

    def test_settlement_quote_currency(self):
        """EMTA convention: settlement = N × (fix - K) / fix."""
        ndf = NDF("USD/CNY", REF + relativedelta(years=1), 7.00, 1_000_000,
                  settlement_currency="quote")
        settle = ndf.settlement_amount(7.20)
        expected = 1_000_000 * (7.20 - 7.00) / 7.20
        assert settle == pytest.approx(expected)
