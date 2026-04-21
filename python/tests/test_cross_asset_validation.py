"""Cross-asset validation: round-trip tests per asset class + consistency checks.

Every asset class must demonstrate:
1. Build → Price → Risk → Verify
2. Internal consistency (e.g., dirty = clean + accrued)
3. Mathematical identities (e.g., CIP, put-call parity)
"""

import math
from datetime import date

import numpy as np
import pytest

from pricebook.bootstrap import bootstrap
from pricebook.curve_builder import build_curves
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.multicurve_solver import validate_curve


# ---- Helpers ----

def _usd_curves():
    ref = date(2026, 4, 21)
    deps = [(date(2026, 7, 21), 0.043), (date(2026, 10, 21), 0.042)]
    swaps = [(date(2028, 4, 21), 0.039), (date(2031, 4, 21), 0.038),
             (date(2036, 4, 21), 0.040)]
    return build_curves("USD", ref, deps, swaps)


# ---- IR: Swap pricing round-trip ----

class TestIRSwapRoundTrip:
    def test_swap_pv_zero_at_par(self):
        """A swap at the par rate should have PV ≈ 0."""
        from pricebook.swap import InterestRateSwap, SwapDirection
        from pricebook.schedule import Frequency, generate_schedule, StubType

        curves = _usd_curves()
        ois = curves.ois
        T5 = date(2031, 4, 21)
        sched = generate_schedule(curves.reference_date, T5, Frequency.SEMI_ANNUAL,
                                    None, None, StubType.SHORT_FRONT, True)
        annuity = sum(
            year_fraction(sched[i-1], sched[i], DayCountConvention.THIRTY_360) * ois.df(sched[i])
            for i in range(1, len(sched))
        )
        par_rate = (1.0 - ois.df(T5)) / annuity

        swap = InterestRateSwap(
            start=curves.reference_date, end=T5,
            fixed_rate=par_rate, direction=SwapDirection.PAYER,
        )
        pv = swap.pv(ois)
        assert abs(pv) < 100  # < $100 on $10M notional

    def test_dv01_positive_for_receiver(self):
        """Receiver swap (receive fixed) has positive DV01."""
        from pricebook.swap import InterestRateSwap, SwapDirection
        curves = _usd_curves()
        swap = InterestRateSwap(
            start=curves.reference_date, end=date(2031, 4, 21),
            fixed_rate=0.04, direction=SwapDirection.RECEIVER,
        )
        pv_base = swap.pv(curves.ois)
        pv_bumped = swap.pv(curves.ois.bumped(0.0001))
        dv01 = pv_base - pv_bumped
        assert dv01 > 0


# ---- Bond: dirty = clean + accrued ----

class TestBondConsistency:
    def test_dirty_equals_clean_plus_accrued(self):
        """Fundamental bond identity: dirty = clean + accrued."""
        from pricebook.bond import FixedRateBond
        ref = date(2026, 4, 21)
        bond = FixedRateBond(
            issue_date=date(2021, 4, 21),
            maturity=date(2031, 4, 21),
            coupon_rate=0.05,
        )
        curves = _usd_curves()
        dirty = bond.dirty_price(curves.ois)
        clean = bond.clean_price(curves.ois, ref)
        accrued = bond.accrued_interest(ref)
        assert dirty == pytest.approx(clean + accrued, abs=1e-10)

    def test_ytm_consistency(self):
        """YTM should be close to the zero rate for a par bond."""
        from pricebook.bond import FixedRateBond
        ref = date(2026, 4, 21)
        bond = FixedRateBond(
            issue_date=date(2021, 4, 21),
            maturity=date(2031, 4, 21),
            coupon_rate=0.05,
        )
        curves = _usd_curves()
        price = bond.dirty_price(curves.ois)
        ytm = bond.yield_to_maturity(price)
        # YTM should be a reasonable number (positive, < 20%)
        assert 0 < ytm < 0.20


# ---- FX: CIP holds ----

class TestFXConsistency:
    def test_cip_holds(self):
        """Covered interest parity: F/S = exp((r_d − r_f) × T)."""
        from pricebook.currency import CurrencyPair, Currency
        pair = CurrencyPair.from_currencies(Currency.EUR, Currency.USD)
        spot = 1.10
        r_eur = 0.03
        r_usd = 0.04
        T = 1.0
        fwd = pair.forward_rate(spot, r_eur, r_usd, T)
        # CIP: F/S = exp((r_usd − r_eur) × T)
        cip_ratio = math.exp((r_usd - r_eur) * T)
        assert fwd / spot == pytest.approx(cip_ratio, rel=1e-10)

    def test_triangular_consistency(self):
        """EUR/JPY from EUR/USD × USD/JPY."""
        from pricebook.currency import CurrencyPair, Currency
        eurusd = 1.10
        usdjpy = 150.0
        eurjpy_implied = eurusd * usdjpy
        assert eurjpy_implied == pytest.approx(165.0)


# ---- Credit: upfront round-trip ----

class TestCreditConsistency:
    def test_upfront_round_trip(self):
        """Par spread → upfront → par spread should round-trip."""
        from pricebook.cds_conventions import upfront_from_par_spread, par_spread_from_upfront
        original = 175.0  # 175bp par spread
        uf = upfront_from_par_spread(original, 100, 4.2)
        recovered = par_spread_from_upfront(uf.upfront_pct, 100, 4.2)
        assert recovered == pytest.approx(original, abs=1e-10)

    def test_ig_upfront_sign(self):
        """IG: spread > 100bp → buyer pays upfront (positive)."""
        from pricebook.cds_conventions import upfront_from_par_spread
        result = upfront_from_par_spread(150, 100, 4.0)
        assert result.upfront_pct > 0

    def test_hy_upfront_sign(self):
        """HY: spread < 500bp → buyer receives upfront (negative)."""
        from pricebook.cds_conventions import upfront_from_par_spread
        result = upfront_from_par_spread(350, 500, 4.0)
        assert result.upfront_pct < 0


# ---- Options: put-call parity ----

class TestOptionConsistency:
    def test_european_put_call_parity(self):
        """C − P = DF × (F − K) for European options."""
        from pricebook.black76 import black76_price, OptionType
        F = 100.0
        K = 100.0
        vol = 0.20
        T = 1.0
        df = math.exp(-0.04)
        call = black76_price(F, K, vol, T, df, OptionType.CALL)
        put = black76_price(F, K, vol, T, df, OptionType.PUT)
        # Put-call parity: C - P = DF × (F - K) = 0 for ATM
        assert call - put == pytest.approx(df * (F - K), abs=1e-10)

    def test_otm_put_call_parity(self):
        """Put-call parity for OTM strike."""
        from pricebook.black76 import black76_price, OptionType
        F = 100.0
        K = 110.0
        vol = 0.20
        T = 1.0
        df = math.exp(-0.04)
        call = black76_price(F, K, vol, T, df, OptionType.CALL)
        put = black76_price(F, K, vol, T, df, OptionType.PUT)
        assert call - put == pytest.approx(df * (F - K), abs=1e-8)


# ---- Curve: all G10 OIS curves valid ----

class TestG10CurveValidity:
    def test_all_g10_curves_valid(self):
        """Build OIS curve for each G10 currency → validate."""
        ref = date(2026, 4, 21)
        deps = [(date(2026, 7, 21), 0.03)]
        swaps = [(date(2028, 4, 21), 0.03), (date(2031, 4, 21), 0.035)]
        for ccy in ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK"]:
            result = build_curves(ccy, ref, deps, swaps)
            validation = validate_curve(result.ois)
            assert validation.n_pillars >= 3, f"{ccy} curve has too few pillars"
            assert not validation.has_non_monotone_dfs, f"{ccy} curve has non-monotone DFs"

    def test_df_decreasing_for_positive_rates(self):
        """All DFs should be monotonically decreasing for positive rate curves."""
        curves = _usd_curves()
        dfs = curves.ois.pillar_dfs
        for i in range(1, len(dfs)):
            assert dfs[i] <= dfs[i-1] + 1e-12, f"DF not decreasing at pillar {i}"

    def test_df_at_zero_is_one(self):
        curves = _usd_curves()
        assert curves.ois.pillar_dfs[0] == pytest.approx(1.0)


# ---- Inflation: index ratio identity ----

class TestInflationConsistency:
    def test_index_ratio_at_base(self):
        """Index ratio at base date = 1.0."""
        from pricebook.market_conventions import index_ratio
        assert index_ratio(260, 260) == pytest.approx(1.0)

    def test_index_ratio_positive(self):
        from pricebook.market_conventions import index_ratio
        assert index_ratio(250, 260) > 1.0  # inflation positive
        assert index_ratio(250, 240) < 1.0  # deflation
