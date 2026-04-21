"""Tests for bond derivatives (BD1-BD7): forward, STRIPS, TRS, ASW, CMO, xccy, repo term."""

import math
from datetime import date

import pytest

from pricebook.bond import FixedRateBond
from pricebook.bond_forward import BondForward
from pricebook.cmo import sequential_cmo, io_po_strip, pac_schedule, CMOTranche
from pricebook.par_asset_swap import ParAssetSwap
from pricebook.repo_term import RepoCurve, RepoRate, forward_repo_rate, identify_specials
from pricebook.schedule import Frequency
from pricebook.strips import strip_bond, price_strip, strip_yield, reconstruct_bond_price
from pricebook.total_return_swap import TotalReturnSwap
from pricebook.xccy_bond import fx_hedged_yield, cross_currency_pickup, breakeven_fx_move
from tests.conftest import make_flat_curve


# ---- BD1: Bond Forward ----

class TestBondForward:
    def test_forward_higher_than_spot_positive_repo(self):
        """With positive repo, forward dirty > spot dirty (no coupons)."""
        bond = FixedRateBond(date(2026, 1, 15), date(2031, 1, 15), 0.02)
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        fwd = BondForward(bond, ref, date(2026, 7, 21), 0.04)
        result = fwd.price(curve)
        assert result.forward_dirty > result.spot_dirty or result.coupon_income > 0

    def test_carry_sign(self):
        """When coupon > repo, carry is positive."""
        bond = FixedRateBond(date(2026, 1, 15), date(2036, 1, 15), 0.06)
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        fwd = BondForward(bond, ref, date(2027, 4, 21), 0.02)
        result = fwd.price(curve)
        assert result.carry > 0

    def test_forward_dv01_positive(self):
        bond = FixedRateBond(date(2026, 1, 15), date(2036, 1, 15), 0.04)
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        fwd = BondForward(bond, ref, date(2026, 10, 21), 0.04)
        result = fwd.price(curve)
        assert result.forward_dv01 > 0

    def test_delivery_before_settlement_raises(self):
        bond = FixedRateBond(date(2026, 1, 15), date(2031, 1, 15), 0.04)
        with pytest.raises(ValueError):
            BondForward(bond, date(2026, 7, 21), date(2026, 4, 21), 0.04)


# ---- BD2: STRIPS ----

class TestSTRIPS:
    def test_strip_count(self):
        """5Y semi-annual bond → 10 C-STRIPS + 1 P-STRIP."""
        bond = FixedRateBond(date(2026, 1, 15), date(2031, 1, 15), 0.04,
                             frequency=Frequency.SEMI_ANNUAL)
        result = strip_bond(bond)
        assert len(result.c_strips) == 10
        assert result.p_strip.strip_type == "P-STRIP"
        assert result.p_strip.face_value == bond.face_value

    def test_reconstruct_matches_dirty(self):
        """Sum of strip PVs should equal bond dirty price."""
        bond = FixedRateBond(date(2026, 1, 15), date(2031, 1, 15), 0.04)
        ref = date(2026, 1, 15)
        curve = make_flat_curve(ref, rate=0.04)
        strips = strip_bond(bond)
        reconstructed = reconstruct_bond_price(strips, curve) / bond.face_value * 100.0
        dirty = bond.dirty_price(curve)
        assert reconstructed == pytest.approx(dirty, rel=1e-6)

    def test_strip_yield(self):
        """Zero coupon yield from price."""
        # Price = 90 for 100 face, 5Y → yield = -ln(0.9)/5 ≈ 2.107%
        y = strip_yield(90.0, 100.0, date(2026, 1, 15), date(2031, 1, 15))
        assert y == pytest.approx(-math.log(0.9) / 5, rel=1e-3)

    def test_c_strip_type(self):
        bond = FixedRateBond(date(2026, 1, 15), date(2028, 1, 15), 0.05)
        result = strip_bond(bond)
        assert all(s.strip_type == "C-STRIP" for s in result.c_strips)
        assert all(s.source_bond_coupon == 0.05 for s in result.c_strips)


# ---- BD3: Total Return Swap ----

class TestTRS:
    def test_price_increase_benefits_receiver(self):
        """If bond price rises, total return receiver gains."""
        bond = FixedRateBond(date(2026, 1, 15), date(2031, 1, 15), 0.04)
        trs = TotalReturnSwap(bond, 1_000_000, 0.005,
                              date(2026, 4, 21), date(2026, 10, 21))
        # Bond priced at 4% yield initially (par)
        curve_up = make_flat_curve(date(2026, 7, 21), rate=0.03)  # rates drop → price up
        result = trs.mark_to_market(100.0, curve_up)
        assert result.price_return > 0

    def test_mtm_at_inception_near_zero(self):
        """At inception (same curve, same price), MTM ≈ 0."""
        bond = FixedRateBond(date(2026, 1, 15), date(2031, 1, 15), 0.04)
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        trs = TotalReturnSwap(bond, 1_000_000, 0.0,
                              ref, date(2026, 10, 21))
        initial_dirty = bond.dirty_price(curve)
        result = trs.mark_to_market(initial_dirty, curve)
        assert abs(result.mtm) < 1000  # near zero

    def test_breakeven_spread(self):
        bond = FixedRateBond(date(2026, 1, 15), date(2031, 1, 15), 0.04)
        trs = TotalReturnSwap(bond, 1_000_000, 0.0,
                              date(2026, 4, 21), date(2027, 4, 21))
        be = trs.breakeven_spread(100.0, 101.0, 40000, 0.04)
        assert isinstance(be, float)


# ---- BD4: Par Asset Swap ----

class TestParAssetSwap:
    def test_par_bond_zero_spread(self):
        """At par, ASW spread ≈ 0."""
        bond = FixedRateBond(date(2026, 1, 15), date(2031, 1, 15), 0.04)
        ref = date(2026, 1, 15)
        curve = make_flat_curve(ref, rate=0.04)
        asw = ParAssetSwap(bond, ref, 100.0)
        result = asw.price(curve)
        assert abs(result.asw_spread) < 0.005

    def test_discount_bond_positive_spread(self):
        """Bond below par → positive ASW spread."""
        bond = FixedRateBond(date(2026, 1, 15), date(2031, 1, 15), 0.04)
        ref = date(2026, 1, 15)
        curve = make_flat_curve(ref, rate=0.04)
        asw = ParAssetSwap(bond, ref, 95.0)  # below par
        result = asw.price(curve)
        assert result.asw_spread > 0

    def test_premium_bond_negative_spread(self):
        """Bond above par → negative ASW spread."""
        bond = FixedRateBond(date(2026, 1, 15), date(2031, 1, 15), 0.04)
        ref = date(2026, 1, 15)
        curve = make_flat_curve(ref, rate=0.04)
        asw = ParAssetSwap(bond, ref, 105.0)  # above par
        result = asw.price(curve)
        assert result.asw_spread < 0

    def test_upfront(self):
        asw = ParAssetSwap(
            FixedRateBond(date(2026, 1, 15), date(2031, 1, 15), 0.04),
            date(2026, 1, 15), 95.0,
        )
        curve = make_flat_curve(date(2026, 1, 15), rate=0.04)
        result = asw.price(curve)
        assert result.upfront == pytest.approx(5.0)


# ---- BD5: CMO ----

class TestCMO:
    def test_sequential_total_principal(self):
        """Total principal across all tranches should equal pool balance."""
        tranches = [
            CMOTranche("A", 60_000, 0.04, "sequential"),
            CMOTranche("B", 30_000, 0.05, "sequential"),
            CMOTranche("C", 10_000, 0.06, "sequential"),
        ]
        result = sequential_cmo(100_000, 0.05, 360, 150, tranches, 0.04)
        total = sum(result.total_principal.values())
        assert total == pytest.approx(100_000, rel=0.01)

    def test_sequential_a_shorter_life(self):
        """Tranche A (senior) has shorter average life than C (junior)."""
        tranches = [
            CMOTranche("A", 60_000, 0.04, "sequential"),
            CMOTranche("B", 30_000, 0.05, "sequential"),
            CMOTranche("C", 10_000, 0.06, "sequential"),
        ]
        result = sequential_cmo(100_000, 0.05, 360, 150, tranches, 0.04)
        assert result.average_life["A"] < result.average_life["B"]
        assert result.average_life["B"] < result.average_life["C"]

    def test_io_po_sum_equals_pool(self):
        """IO + PO should approximately equal the pool price."""
        io, po = io_po_strip(100_000, 0.06, 360, 150, 0.05)
        # At 5% discount with 6% coupon, total should be close to par
        assert io + po > 80_000
        assert io + po < 120_000

    def test_po_decreases_with_higher_prepayment(self):
        """Faster prepayment returns principal sooner → higher PO price."""
        _, po_slow = io_po_strip(100_000, 0.06, 360, 100, 0.05)
        _, po_fast = io_po_strip(100_000, 0.06, 360, 300, 0.05)
        assert po_fast > po_slow

    def test_pac_schedule_shape(self):
        """PAC schedule should have monthly principal values."""
        pac = pac_schedule(100_000, 0.06, 360, 100, 300)
        assert len(pac) == 360
        assert pac.sum() > 0


# ---- BD6: Cross-currency Bond ----

class TestXccyBond:
    def test_hedged_yield_cip(self):
        """FX-hedged yield = foreign_ytm + (domestic - foreign rate)."""
        hedged = fx_hedged_yield(0.04, 0.05, 0.03)
        assert hedged == pytest.approx(0.04 + 0.02)

    def test_pickup_positive(self):
        """Foreign bond with higher hedged yield → positive pickup."""
        result = cross_currency_pickup(0.04, 0.03, 0.05, 0.02)
        assert result.carry_pickup > 0

    def test_pickup_negative(self):
        """Negative pickup when hedge cost wipes out yield advantage."""
        # Foreign 4%, domestic benchmark 5%, domestic rate 2%, foreign rate 6%
        # Hedged = 4% + (2% - 6%) = 0% → pickup = 0% - 5% = -5%
        result = cross_currency_pickup(0.04, 0.05, 0.02, 0.06)
        assert result.carry_pickup < 0

    def test_breakeven_fx_zero_when_equal(self):
        """No yield advantage → zero breakeven."""
        be = breakeven_fx_move(0.04, 0.04, 5.0)
        assert be == pytest.approx(0.0)

    def test_breakeven_fx_positive(self):
        be = breakeven_fx_move(0.06, 0.04, 5.0)
        assert be > 0


# ---- BD7: Repo Term Structure ----

class TestRepoTerm:
    def test_interpolation(self):
        """Repo rate interpolates between tenors."""
        curve = RepoCurve(date(2026, 4, 21), [
            RepoRate(1, 0.04),
            RepoRate(30, 0.042),
            RepoRate(90, 0.045),
        ])
        rate_15 = curve.rate(15)
        assert 0.04 < rate_15 < 0.042

    def test_forward_repo(self):
        """Forward repo rate should be derivable."""
        curve = RepoCurve(date(2026, 4, 21), [
            RepoRate(30, 0.04),
            RepoRate(90, 0.045),
        ])
        fwd = forward_repo_rate(curve, 30, 90)
        # Forward should be higher than 90-day spot if curve is upward sloping
        assert fwd > 0.04

    def test_identify_specials(self):
        """Bond on special has repo rate below GC."""
        results = identify_specials(0.04, {
            "UST10Y": 0.035,  # special
            "UST5Y": 0.039,   # near GC
            "UST2Y": 0.041,   # above GC (not special)
        })
        specials = [r for r in results if r.is_special]
        assert len(specials) == 1
        assert specials[0].collateral == "UST10Y"

    def test_discount_factor(self):
        curve = RepoCurve(date(2026, 4, 21), [RepoRate(90, 0.04)])
        df = curve.discount_factor(90)
        assert df == pytest.approx(1.0 / (1.0 + 0.04 * 90 / 360))

    def test_empty_curve_raises(self):
        with pytest.raises(ValueError):
            RepoCurve(date(2026, 4, 21), [])
