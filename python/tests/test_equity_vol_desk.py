"""Tests for equity vol desk: surface, RV strategies, vega ladder, cross-Greeks."""

import math

import pytest
from datetime import date

from pricebook.black76 import OptionType
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.equity_option import equity_option_price, equity_vega
from pricebook.equity_vol_desk import (
    CalendarSpread,
    EquityVolSurface,
    RiskReversal,
    VarianceSwap,
    VegaBucket,
    VolPillar,
    total_vega,
    vanna,
    vega_ladder,
    volga,
)


REF = date(2024, 1, 15)
EXP_3M = date(2024, 4, 15)
EXP_6M = date(2024, 7, 15)
EXP_1Y = date(2025, 1, 15)
ATM = 100.0
SPOT = 100.0
RATE = 0.05


def _flat_surface(vol: float = 0.20) -> EquityVolSurface:
    """Flat 20% vol across all strikes and expiries."""
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    pillars = [
        VolPillar(EXP_3M, strikes, [vol] * 5),
        VolPillar(EXP_6M, strikes, [vol] * 5),
        VolPillar(EXP_1Y, strikes, [vol] * 5),
    ]
    return EquityVolSurface(REF, ATM, pillars)


def _smile_surface() -> EquityVolSurface:
    """Surface with smile + term structure (to exercise interp/bumps)."""
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    pillars = [
        VolPillar(EXP_3M, strikes, [0.30, 0.25, 0.22, 0.24, 0.28]),
        VolPillar(EXP_6M, strikes, [0.28, 0.24, 0.21, 0.23, 0.27]),
        VolPillar(EXP_1Y, strikes, [0.26, 0.23, 0.20, 0.22, 0.25]),
    ]
    return EquityVolSurface(REF, ATM, pillars)


# ---- Step 1: surface management ----

class TestEquityVolSurface:
    def test_basic_construction(self):
        s = _flat_surface()
        assert s.atm_strike == 100.0
        assert s.n_pillars == 3
        assert s.expiries == [EXP_3M, EXP_6M, EXP_1Y]

    def test_vol_lookup(self):
        s = _flat_surface(0.25)
        assert s.vol(EXP_3M, 100.0) == pytest.approx(0.25)
        assert s.vol(EXP_6M, 100.0) == pytest.approx(0.25)
        # Default to ATM strike
        assert s.vol(EXP_3M) == pytest.approx(0.25)

    def test_atm_term(self):
        s = _smile_surface()
        term = s.atm_term()
        assert len(term) == 3
        assert term[0][0] == EXP_3M
        # ATM vol declines with maturity in our test surface
        assert term[0][1] == pytest.approx(0.22)
        assert term[2][1] == pytest.approx(0.20)

    def test_skew_at_expiry(self):
        s = _smile_surface()
        skew = s.skew_at_expiry(EXP_6M)
        assert len(skew) == 5
        # Smile shape: down vol > atm vol
        downside = next(v for k, v in skew if k == 80.0)
        atm_vol = next(v for k, v in skew if k == 100.0)
        assert downside > atm_vol

    def test_bump_parallel_immutable(self):
        s = _flat_surface(0.20)
        bumped = s.bump_parallel(0.01)
        # Original unchanged
        assert s.vol(EXP_3M, 100.0) == pytest.approx(0.20)
        # Bumped surface up by 1 vol point everywhere
        assert bumped.vol(EXP_3M, 100.0) == pytest.approx(0.21)
        assert bumped.vol(EXP_1Y, 110.0) == pytest.approx(0.21)

    def test_bump_term_targets_one_pillar(self):
        s = _flat_surface(0.20)
        bumped = s.bump_term(EXP_6M, 0.02)
        # Other expiries untouched
        assert bumped.vol(EXP_3M, 100.0) == pytest.approx(0.20)
        assert bumped.vol(EXP_1Y, 100.0) == pytest.approx(0.20)
        # Targeted expiry shifted
        assert bumped.vol(EXP_6M, 100.0) == pytest.approx(0.22)

    def test_bump_skew_tilt(self):
        s = _flat_surface(0.20)
        bumped = s.bump_skew(0.05)
        # ATM strike unchanged (k = atm => 0 tilt)
        assert bumped.vol(EXP_3M, 100.0) == pytest.approx(0.20)
        # Upside vol higher
        assert bumped.vol(EXP_3M, 120.0) > 0.20
        # Downside vol lower
        assert bumped.vol(EXP_3M, 80.0) < 0.20

    def test_bump_curvature_quadratic(self):
        s = _flat_surface(0.20)
        bumped = s.bump_curvature(0.03)
        # ATM unchanged
        assert bumped.vol(EXP_3M, 100.0) == pytest.approx(0.20)
        # Both wings up
        assert bumped.vol(EXP_3M, 80.0) > 0.20
        assert bumped.vol(EXP_3M, 120.0) > 0.20
        # Symmetric
        assert bumped.vol(EXP_3M, 80.0) == pytest.approx(bumped.vol(EXP_3M, 120.0))

    def test_bumped_surface_reprices_vanilla(self):
        """Step 1 test: bumped surface should reprice vanillas correctly.

        Vega is positive, so bumping vol up must lift the call price by
        approximately Δσ × vega.
        """
        s = _flat_surface(0.20)
        T = year_fraction(REF, EXP_1Y, DayCountConvention.ACT_365_FIXED)
        K = 100.0
        p_base = equity_option_price(SPOT, K, RATE, s.vol(EXP_1Y, K), T, OptionType.CALL)

        dvol = 0.01
        bumped = s.bump_parallel(dvol)
        p_bumped = equity_option_price(
            SPOT, K, RATE, bumped.vol(EXP_1Y, K), T, OptionType.CALL,
        )

        # Predicted move via analytic vega
        predicted = equity_vega(SPOT, K, RATE, 0.20, T) * dvol
        actual = p_bumped - p_base
        # Tight: vega first-order should be very accurate for 1 vol point
        assert actual == pytest.approx(predicted, rel=0.01)


# ---- Step 2: RV strategies ----

class TestCalendarSpread:
    def test_pv_long_front(self):
        s = _flat_surface(0.20)
        spread = CalendarSpread(strike=100.0, short_expiry=EXP_3M,
                                long_expiry=EXP_1Y, direction=1)
        pv = spread.pv(s, SPOT, RATE)
        # Long short-dated, short long-dated → negative for ATM (long-dated worth more)
        assert pv < 0.0

    def test_pv_direction_flips(self):
        s = _flat_surface(0.20)
        long_front = CalendarSpread(100.0, EXP_3M, EXP_1Y, direction=1)
        short_front = CalendarSpread(100.0, EXP_3M, EXP_1Y, direction=-1)
        assert long_front.pv(s, SPOT, RATE) == pytest.approx(
            -short_front.pv(s, SPOT, RATE)
        )

    def test_quantity_scales_pv(self):
        s = _flat_surface(0.20)
        single = CalendarSpread(100.0, EXP_3M, EXP_1Y, quantity=1)
        triple = CalendarSpread(100.0, EXP_3M, EXP_1Y, quantity=3)
        assert triple.pv(s, SPOT, RATE) == pytest.approx(
            3.0 * single.pv(s, SPOT, RATE)
        )


class TestRiskReversal:
    def test_pv_zero_at_atm_flat_vol(self):
        """For symmetric strikes around forward and flat vol, RR ≈ 0."""
        s = _flat_surface(0.20)
        rr = RiskReversal(EXP_1Y, call_strike=110.0, put_strike=90.0)
        # Not exactly zero because forward != spot (due to rate)
        # but should be small in magnitude vs the option premiums
        T = year_fraction(REF, EXP_1Y, DayCountConvention.ACT_365_FIXED)
        c = equity_option_price(SPOT, 110.0, RATE, 0.20, T, OptionType.CALL)
        p = equity_option_price(SPOT, 90.0, RATE, 0.20, T, OptionType.PUT)
        assert rr.pv(s, SPOT, RATE) == pytest.approx(c - p)

    def test_pv_picks_up_skew(self):
        """With downside vol > upside vol the put leg is richer and the
        call leg is cheaper, so ``long_call - short_put`` is *lower* than
        under flat vol."""
        s_smile = _smile_surface()       # downside vol > upside vol
        s_flat = _flat_surface(0.20)
        rr = RiskReversal(EXP_1Y, call_strike=110.0, put_strike=90.0, direction=1)
        assert rr.pv(s_smile, SPOT, RATE) < rr.pv(s_flat, SPOT, RATE)

    def test_direction_flips(self):
        s = _smile_surface()
        long = RiskReversal(EXP_1Y, 110.0, 90.0, direction=1)
        short = RiskReversal(EXP_1Y, 110.0, 90.0, direction=-1)
        assert long.pv(s, SPOT, RATE) == pytest.approx(-short.pv(s, SPOT, RATE))


class TestVarianceSwap:
    def test_fair_variance_flat_vol(self):
        """Strip replication with flat 20% vol should give variance ≈ 0.04."""
        s = _flat_surface(0.20)
        vs = VarianceSwap(EXP_1Y, var_strike=0.04)
        fair = vs.fair_variance(s, SPOT, RATE)
        # Replication is approximate; allow 10% tolerance
        assert fair == pytest.approx(0.04, rel=0.10)

    def test_fair_variance_increases_with_vol(self):
        s_low = _flat_surface(0.15)
        s_high = _flat_surface(0.30)
        vs = VarianceSwap(EXP_1Y, var_strike=0.04)
        fair_low = vs.fair_variance(s_low, SPOT, RATE)
        fair_high = vs.fair_variance(s_high, SPOT, RATE)
        assert fair_high > fair_low
        # Roughly the squared ratio
        assert fair_high == pytest.approx(0.09, rel=0.15)
        assert fair_low == pytest.approx(0.0225, rel=0.15)

    def test_pv_zero_when_strike_equals_fair(self):
        s = _flat_surface(0.20)
        vs = VarianceSwap(EXP_1Y, var_strike=0.04)  # roughly fair
        # Set strike exactly to the computed fair value
        fair = vs.fair_variance(s, SPOT, RATE)
        vs_at_fair = VarianceSwap(EXP_1Y, var_strike=fair, notional=1_000_000)
        assert vs_at_fair.pv(s, SPOT, RATE) == pytest.approx(0.0, abs=1e-6)

    def test_long_variance_pays_when_realised_higher(self):
        """If fair variance > strike, long position has positive PV."""
        s = _flat_surface(0.30)
        vs = VarianceSwap(EXP_1Y, var_strike=0.04, direction=1, notional=1_000_000)
        # fair ≈ 0.09 > 0.04 strike → long PV positive
        assert vs.pv(s, SPOT, RATE) > 0.0

    def test_direction_flips_pv(self):
        s = _flat_surface(0.30)
        long = VarianceSwap(EXP_1Y, 0.04, notional=1_000_000, direction=1)
        short = VarianceSwap(EXP_1Y, 0.04, notional=1_000_000, direction=-1)
        assert long.pv(s, SPOT, RATE) == pytest.approx(-short.pv(s, SPOT, RATE))


# ---- Step 3: vega ladder + cross-Greeks ----

class TestVegaLadder:
    def test_aggregates_same_bucket(self):
        positions = [
            (EXP_3M, 100.0, 1500.0),
            (EXP_3M, 100.0, -500.0),
            (EXP_3M, 110.0, 800.0),
        ]
        ladder = vega_ladder(positions)
        # 2 buckets: (3M, 100), (3M, 110)
        assert len(ladder) == 2
        atm_bucket = next(b for b in ladder if b.strike == 100.0)
        assert atm_bucket.vega == pytest.approx(1000.0)

    def test_separate_expiries(self):
        positions = [
            (EXP_3M, 100.0, 1000.0),
            (EXP_6M, 100.0, 1500.0),
            (EXP_1Y, 100.0, 2000.0),
        ]
        ladder = vega_ladder(positions)
        assert len(ladder) == 3
        # Total vega
        assert total_vega(ladder) == pytest.approx(4500.0)

    def test_sorted_output(self):
        positions = [
            (EXP_1Y, 110.0, 100.0),
            (EXP_3M, 90.0, 100.0),
            (EXP_3M, 110.0, 100.0),
            (EXP_6M, 100.0, 100.0),
        ]
        ladder = vega_ladder(positions)
        # Sorted by (expiry, strike)
        assert ladder[0].expiry == EXP_3M
        assert ladder[0].strike == 90.0
        assert ladder[-1].expiry == EXP_1Y

    def test_total_equals_sum_of_position_vegas(self):
        positions = [
            (EXP_3M, 100.0, 1234.5),
            (EXP_6M, 90.0, -567.8),
            (EXP_1Y, 110.0, 234.6),
        ]
        ladder = vega_ladder(positions)
        assert total_vega(ladder) == pytest.approx(
            sum(v for _, _, v in positions)
        )


class TestCrossGreeks:
    def test_volga_positive_otm(self):
        """Volga is positive for out-of-the-money options (vol-of-vol convex)."""
        T = 1.0
        v = volga(SPOT, 120.0, RATE, 0.20, T)
        assert v > 0.0

    def test_volga_near_zero_atm(self):
        """At-the-money volga is small (vega is near maximum, so flat in σ)."""
        T = 1.0
        v_atm = volga(SPOT, 100.0, RATE, 0.20, T)
        v_otm = volga(SPOT, 130.0, RATE, 0.20, T)
        assert abs(v_atm) < abs(v_otm)

    def test_vanna_sign_matches_moneyness(self):
        """Vanna for OTM call (k > spot) is positive: as spot rises, vega rises."""
        T = 1.0
        v = vanna(SPOT, 120.0, RATE, 0.20, T)
        assert v > 0.0

    def test_vanna_atm_near_zero(self):
        """Vanna at ATM is near zero (vega is maxed at the forward)."""
        T = 1.0
        v = vanna(SPOT, 100.0, RATE, 0.20, T)
        # Roughly zero compared to OTM vanna
        v_otm = vanna(SPOT, 130.0, RATE, 0.20, T)
        assert abs(v) < abs(v_otm)
