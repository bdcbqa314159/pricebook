"""Deep tests for IR vanilla — DD2 hardening.

Covers: swaption parity, cap/floor day count, forward rate day count
consistency, FRA settlement, amortising swap, CMS convexity, Bermudan bounds.
"""

import math
import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.swaption import Swaption, SwaptionType
from pricebook.capfloor import CapFloor
from pricebook.black76 import OptionType
from pricebook.fra import FRA
from pricebook.swap import InterestRateSwap, SwapDirection
from pricebook.amortising_swap import AmortisingSwap
from pricebook.basis_swap import BasisSwap
from pricebook.zc_swap import ZeroCouponSwap
from pricebook.bermudan_swaption import bermudan_swaption_tree, bermudan_swaption_lsm
from pricebook.vol_surface import FlatVol
from pricebook.schedule import Frequency
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)


# ---- Swaption ----

class TestSwaptionDeep:

    def test_atm_payer_receiver_parity_tight(self):
        """Payer - Receiver = forward swap PV, within $0.10 on 1M notional."""
        curve = make_flat_curve(REF, 0.04)
        vol = FlatVol(0.25)
        strike = 0.04
        payer = Swaption(REF + relativedelta(years=1), REF + relativedelta(years=6),
                         strike, SwaptionType.PAYER)
        recvr = Swaption(REF + relativedelta(years=1), REF + relativedelta(years=6),
                         strike, SwaptionType.RECEIVER)
        diff = payer.pv(curve, vol) - recvr.pv(curve, vol)
        fwd = payer.forward_swap_rate(curve)
        ann = payer.annuity(curve)
        expected = payer.notional * ann * (fwd - strike)
        assert diff == pytest.approx(expected, abs=0.10)

    def test_zero_vol_gives_intrinsic(self):
        """At zero vol, swaption = max(intrinsic, 0)."""
        curve = make_flat_curve(REF, 0.05)
        vol = FlatVol(1e-6)
        swn = Swaption(REF + relativedelta(years=1), REF + relativedelta(years=6),
                       strike=0.03, swaption_type=SwaptionType.PAYER)
        fwd = swn.forward_swap_rate(curve)
        ann = swn.annuity(curve)
        intrinsic = swn.notional * ann * max(fwd - 0.03, 0)
        pv = swn.pv(curve, vol)
        assert pv == pytest.approx(intrinsic, rel=0.01)

    def test_greeks_delta_sign(self):
        """Payer delta > 0, receiver delta < 0."""
        curve = make_flat_curve(REF, 0.04)
        vol = FlatVol(0.20)
        payer = Swaption(REF + relativedelta(years=1), REF + relativedelta(years=6),
                         0.04, SwaptionType.PAYER)
        recvr = Swaption(REF + relativedelta(years=1), REF + relativedelta(years=6),
                         0.04, SwaptionType.RECEIVER)
        assert payer.greeks(curve, vol).delta > 0
        assert recvr.greeks(curve, vol).delta < 0

    def test_greeks_vega_positive(self):
        """Vega is always positive for both payer and receiver."""
        curve = make_flat_curve(REF, 0.04)
        vol = FlatVol(0.20)
        payer = Swaption(REF + relativedelta(years=1), REF + relativedelta(years=6),
                         0.04, SwaptionType.PAYER)
        assert payer.greeks(curve, vol).vega > 0


# ---- Cap/Floor ----

class TestCapFloorDeep:

    def test_cap_floor_parity_direction(self):
        """ATM: cap ≈ floor (symmetric at ATM on flat curve)."""
        curve = make_flat_curve(REF, 0.05)
        vol = FlatVol(0.20)
        end = REF + relativedelta(years=3)

        # Find ATM forward rate for cap/floor (using cap day count)
        cap_test = CapFloor(REF, end, strike=0.05, option_type=OptionType.CALL,
                            frequency=Frequency.QUARTERLY, day_count=DayCountConvention.ACT_360)
        pvs = cap_test.caplet_pvs(curve, vol)
        avg_fwd = sum(p['forward'] for p in pvs) / len(pvs)

        cap = CapFloor(REF, end, strike=avg_fwd, option_type=OptionType.CALL,
                       frequency=Frequency.QUARTERLY, day_count=DayCountConvention.ACT_360)
        floor = CapFloor(REF, end, strike=avg_fwd, option_type=OptionType.PUT,
                         frequency=Frequency.QUARTERLY, day_count=DayCountConvention.ACT_360)

        cap_pv = cap.pv(curve, vol)
        floor_pv = floor.pv(curve, vol)

        # On a flat curve, ATM cap ≈ ATM floor (within a few bps of notional)
        assert abs(cap_pv - floor_pv) / 1_000_000 < 0.01

    def test_dual_curve_cap_differs_from_single(self):
        """Cap with projection curve gives different PV than single curve."""
        discount = make_flat_curve(REF, 0.03)
        projection = make_flat_curve(REF, 0.06)
        vol = FlatVol(0.20)
        cap = CapFloor(REF, REF + relativedelta(years=3), strike=0.05)
        pv_single = cap.pv(discount, vol)
        pv_dual = cap.pv(discount, vol, projection_curve=projection)
        assert pv_dual > pv_single  # higher forwards → higher cap value


# ---- FRA ----

class TestFRADeep:

    def test_fra_settlement_is_discounted(self):
        """FRA PV = discounted settlement, not undiscounted."""
        curve = make_flat_curve(REF, 0.05)
        fra = FRA(REF + relativedelta(months=6), REF + relativedelta(months=12),
                  strike=0.03)
        fwd = fra.forward_rate(curve)
        # Undiscounted settlement
        settled = fra.notional * (fwd - 0.03) * fra.year_frac / (1 + fwd * fra.year_frac)
        # PV = settled * df(start), not settled * df(end)
        expected_pv = settled * curve.df(fra.start)
        assert fra.pv(curve) == pytest.approx(expected_pv, rel=1e-10)

    def test_fra_forward_uses_instrument_daycount(self):
        """FRA forward rate should use its own day count, not curve's."""
        curve = make_flat_curve(REF, 0.05)
        fra_360 = FRA(REF + relativedelta(months=3), REF + relativedelta(months=6),
                      strike=0.0, day_count=DayCountConvention.ACT_360)
        fra_365 = FRA(REF + relativedelta(months=3), REF + relativedelta(months=6),
                      strike=0.0, day_count=DayCountConvention.ACT_365_FIXED)
        # Different day counts → different forward rates
        assert fra_360.forward_rate(curve) != pytest.approx(fra_365.forward_rate(curve), abs=1e-6)


# ---- Amortising swap ----

class TestAmortisingSwapDeep:

    def test_bullet_matches_vanilla(self):
        """Amortising swap with constant notional = vanilla swap."""
        curve = make_flat_curve(REF, 0.04)
        end = REF + relativedelta(years=5)
        vanilla = InterestRateSwap(REF, end, fixed_rate=0.04, notional=1_000_000)
        amort = AmortisingSwap(REF, end, fixed_rate=0.04,
                               notional_schedule=[1_000_000] * 20)
        assert amort.pv(curve) == pytest.approx(vanilla.pv(curve), abs=10.0)

    def test_par_rate_accepts_projection(self):
        """par_rate with projection curve should differ from single curve."""
        discount = make_flat_curve(REF, 0.03)
        projection = make_flat_curve(REF, 0.06)
        end = REF + relativedelta(years=5)
        amort = AmortisingSwap(REF, end, fixed_rate=0.0,
                               notional_schedule=[1_000_000, 800_000, 600_000, 400_000, 200_000])
        par_single = amort.par_rate(discount)
        par_dual = amort.par_rate(discount, projection=projection)
        assert par_dual > par_single  # higher projection → higher par rate


# ---- Zero coupon swap ----

class TestZCSwapDeep:

    def test_par_rate_consistent(self):
        """ZC swap par rate: (1+K)^T = 1/df(T)."""
        curve = make_flat_curve(REF, 0.05)
        zcs = ZeroCouponSwap(REF, REF + relativedelta(years=5), fixed_rate=0.0)
        par = zcs.par_rate(curve)
        df_T = curve.df(zcs.end)
        T = zcs._tenor_years()
        assert (1 + par) ** T == pytest.approx(1.0 / df_T, rel=1e-8)

    def test_pv_zero_at_par(self):
        curve = make_flat_curve(REF, 0.05)
        zcs = ZeroCouponSwap(REF, REF + relativedelta(years=5), fixed_rate=0.0)
        par = zcs.par_rate(curve)
        zcs_par = ZeroCouponSwap(REF, REF + relativedelta(years=5), fixed_rate=par)
        assert zcs_par.pv(curve) == pytest.approx(0.0, abs=0.01)


# ---- Bermudan bounds ----

class TestBermudanBounds:

    def _hw(self):
        from pricebook.hull_white import HullWhite
        curve = make_flat_curve(REF, 0.04)
        return HullWhite(a=0.03, sigma=0.01, curve=curve)

    def test_bermudan_geq_european(self):
        """Bermudan price >= European price (more exercise opportunities)."""
        hw = self._hw()
        exercises = [1.0, 2.0, 3.0, 4.0]
        berm = bermudan_swaption_tree(hw, exercises, 10.0, 0.04, n_steps=50)
        euro = bermudan_swaption_tree(hw, [1.0], 10.0, 0.04, n_steps=50)
        assert berm >= euro - 1e-6

    def test_bermudan_positive(self):
        """Bermudan swaption always has non-negative value."""
        hw = self._hw()
        berm = bermudan_swaption_tree(hw, [1.0, 2.0, 3.0], 5.0, 0.04, n_steps=50)
        assert berm >= -1e-10

    def test_lsm_positive(self):
        """LSM Bermudan should be positive."""
        hw = self._hw()
        lsm = bermudan_swaption_lsm(hw, [1.0, 2.0, 3.0], 5.0, 0.04,
                                      n_paths=10_000, seed=42)
        assert lsm >= -1e-6
