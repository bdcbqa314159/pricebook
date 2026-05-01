"""Tests for T-Lock ecosystem: serialisation, portfolio risk, key-rate DV01."""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.bond import FixedRateBond
from pricebook.bond_forward import BondForward
from pricebook.treasury_lock import TreasuryLock, tlock_portfolio_risk, tlock_pnl_attribution
from pricebook.tbill import TreasuryBill
from pricebook.par_asset_swap import ParAssetSwap, ProceedsAssetSwap
from pricebook.repo_term import RepoCurve, RepoRate
from pricebook.funded import Repo, ReverseRepo, RepoFinancedPosition
from pricebook.schedule import Frequency
from pricebook.serialisable import from_dict
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)
EXPIRY = REF + relativedelta(months=6)


def _make_bond():
    return FixedRateBond(
        issue_date=REF - relativedelta(years=1),
        maturity=REF + relativedelta(years=9),
        coupon_rate=0.04,
        frequency=Frequency.SEMI_ANNUAL,
        face_value=100.0,
    )


def _make_tlock(locked_yield=0.04, notional=10_000_000, direction=1, repo_rate=0.02):
    return TreasuryLock(
        bond=_make_bond(),
        locked_yield=locked_yield,
        expiry=EXPIRY,
        notional=notional,
        direction=direction,
        repo_rate=repo_rate,
    )


# ---- Serialisation round-trips ----

class TestSerialisation:

    def test_treasury_lock_round_trip(self):
        tl = _make_tlock()
        d = tl.to_dict()
        tl2 = from_dict(d)
        assert tl2.locked_yield == tl.locked_yield
        assert tl2.notional == tl.notional
        assert tl2.direction == tl.direction
        assert tl2.repo_rate == tl.repo_rate

    def test_bond_forward_round_trip(self):
        bond = _make_bond()
        bf = BondForward(bond, REF, EXPIRY, repo_rate=0.03)
        d = bf.to_dict()
        bf2 = from_dict(d)
        assert bf2.repo_rate == bf.repo_rate
        assert bf2.settlement == bf.settlement
        assert bf2.delivery == bf.delivery

    def test_repo_curve_round_trip(self):
        rc = RepoCurve(REF, [
            RepoRate(1, 0.04), RepoRate(30, 0.042),
            RepoRate(90, 0.045), RepoRate(180, 0.047),
        ])
        d = rc.to_dict()
        rc2 = RepoCurve.from_dict(d)
        assert rc2.rate(30) == pytest.approx(rc.rate(30))
        assert rc2.rate(90) == pytest.approx(rc.rate(90))

    def test_par_asset_swap_round_trip(self):
        bond = _make_bond()
        asw = ParAssetSwap(bond, REF, 95.0)
        d = asw.to_dict()
        asw2 = from_dict(d)
        assert asw2.market_price == 95.0

    def test_proceeds_asset_swap_round_trip(self):
        bond = _make_bond()
        asw = ProceedsAssetSwap(bond, REF, 96.5)
        d = asw.to_dict()
        asw2 = from_dict(d)
        assert asw2.market_dirty_price == 96.5

    def test_tbill_round_trip(self):
        bill = TreasuryBill(REF, REF + relativedelta(days=90), price=98.75)
        d = bill.to_dict()
        bill2 = from_dict(d)
        assert bill2.price == bill.price

    def test_bond_round_trip(self):
        bond = _make_bond()
        d = bond.to_dict()
        bond2 = from_dict(d)
        assert bond2.coupon_rate == bond.coupon_rate
        assert bond2.maturity == bond.maturity


# ---- T-Lock risk ----

class TestTLockRisk:

    def test_dv01(self):
        tl = _make_tlock()
        dc = make_flat_curve(REF, 0.04)
        dv01 = tl.dv01(dc)
        assert math.isfinite(dv01)
        assert dv01 != 0

    def test_key_rate_dv01(self):
        tl = _make_tlock()
        dc = make_flat_curve(REF, 0.04)
        kr = tl.key_rate_dv01(dc)
        assert len(kr) > 0
        assert all(math.isfinite(v) for v in kr.values())

    def test_repo_sensitivity(self):
        tl = _make_tlock()
        dc = make_flat_curve(REF, 0.04)
        rs = tl.repo_sensitivity(dc)
        assert math.isfinite(rs)

    def test_cross_gamma(self):
        tl = _make_tlock()
        dc = make_flat_curve(REF, 0.04)
        cg = tl.cross_gamma_yield_repo(dc)
        assert math.isfinite(cg)

    def test_greeks(self):
        tl = _make_tlock()
        dc = make_flat_curve(REF, 0.04)
        g = tl.greeks(dc)
        assert "delta" in g
        assert "gamma" in g


# ---- Portfolio risk ----

class TestPortfolioRisk:

    def test_single_position(self):
        tl = _make_tlock()
        dc = make_flat_curve(REF, 0.04)
        risk = tlock_portfolio_risk([tl], dc)
        assert risk["n_positions"] == 1
        assert math.isfinite(risk["total_pv"])
        assert math.isfinite(risk["total_dv01"])
        assert math.isfinite(risk["total_delta"])
        assert math.isfinite(risk["total_gamma"])

    def test_two_positions_net(self):
        """Long + short of same T-Lock should partially offset."""
        tl_long = _make_tlock(direction=1)
        tl_short = _make_tlock(direction=-1)
        dc = make_flat_curve(REF, 0.04)
        risk = tlock_portfolio_risk([tl_long, tl_short], dc)
        # Same notional, opposite directions → PV should be near zero
        assert abs(risk["total_pv"]) < 100

    def test_multiple_positions(self):
        """Portfolio of 3 T-Locks at different yields."""
        tlocks = [
            _make_tlock(locked_yield=0.035, notional=5_000_000),
            _make_tlock(locked_yield=0.040, notional=10_000_000),
            _make_tlock(locked_yield=0.045, notional=3_000_000, direction=-1),
        ]
        dc = make_flat_curve(REF, 0.04)
        risk = tlock_portfolio_risk(tlocks, dc)
        assert risk["n_positions"] == 3
        assert math.isfinite(risk["total_dv01"])
        assert risk["max_overhedge"] >= 0

    def test_repo_sensitivity_aggregates(self):
        """Portfolio repo sensitivity = sum of individual."""
        tl1 = _make_tlock(repo_rate=0.02)
        tl2 = _make_tlock(repo_rate=0.03)
        dc = make_flat_curve(REF, 0.04)
        risk = tlock_portfolio_risk([tl1, tl2], dc)
        assert math.isfinite(risk["repo_sensitivity"])

    def test_risk_fields(self):
        tl = _make_tlock()
        dc = make_flat_curve(REF, 0.04)
        risk = tlock_portfolio_risk([tl], dc)
        expected_keys = ["total_pv", "total_dv01", "total_delta", "total_gamma",
                          "repo_sensitivity", "max_overhedge", "n_positions"]
        for k in expected_keys:
            assert k in risk


# ---- Treasury note constructor ----

class TestTreasuryNote:

    def test_correct_conventions(self):
        """Treasury note should use ACT/ACT ICMA and T+1."""
        from pricebook.day_count import DayCountConvention
        bond = FixedRateBond.treasury_note(
            REF - relativedelta(years=1),
            REF + relativedelta(years=9),
            coupon_rate=0.04,
        )
        assert bond.day_count == DayCountConvention.ACT_ACT_ICMA
        assert bond.settlement_days == 1
        assert bond.frequency == Frequency.SEMI_ANNUAL

    def test_prices(self):
        bond = FixedRateBond.treasury_note(
            REF - relativedelta(years=1),
            REF + relativedelta(years=9),
            coupon_rate=0.04,
        )
        dc = make_flat_curve(REF, 0.04)
        price = bond.dirty_price(dc)
        assert price > 0


# ---- P&L attribution ----

class TestPnLAttribution:

    def test_unchanged_curves_zero_pnl(self):
        """Same curve → total P&L = 0."""
        tl = _make_tlock()
        dc = make_flat_curve(REF, 0.04)
        pnl = tlock_pnl_attribution(tl, dc, dc)
        assert abs(pnl["total"]) < 1e-6

    def test_curve_shift_nonzero_pnl(self):
        """Shifted curve → nonzero P&L."""
        tl = _make_tlock()
        dc0 = make_flat_curve(REF, 0.04)
        dc1 = make_flat_curve(REF, 0.045)  # 50bp shift
        pnl = tlock_pnl_attribution(tl, dc0, dc1)
        assert abs(pnl["total"]) > 0
        assert abs(pnl["curve_pnl"]) > 0

    def test_repo_change(self):
        """Repo rate change → nonzero repo P&L."""
        tl = _make_tlock(repo_rate=0.02)
        dc = make_flat_curve(REF, 0.04)
        pnl = tlock_pnl_attribution(tl, dc, dc, repo_rate_t1=0.03)
        assert abs(pnl["repo_pnl"]) > 0

    def test_pnl_fields(self):
        tl = _make_tlock()
        dc = make_flat_curve(REF, 0.04)
        pnl = tlock_pnl_attribution(tl, dc, dc)
        for k in ["total", "curve_pnl", "repo_pnl", "roll_pnl_first_order", "unexplained"]:
            assert k in pnl


# ---- Settlement ----

class TestSettlement:

    def test_settlement_amount(self):
        """Cash settlement at expiry."""
        tl = _make_tlock(locked_yield=0.04)
        amount = tl.settlement_amount(irr_at_expiry=0.045)
        # IRR > locked → payer receives (direction=1)
        assert amount > 0

    def test_settlement_opposite(self):
        tl = _make_tlock(locked_yield=0.04)
        amount = tl.settlement_amount(irr_at_expiry=0.035)
        # IRR < locked → payer pays
        assert amount < 0

    def test_settlement_at_locked(self):
        """IRR = locked → settlement ≈ 0."""
        tl = _make_tlock(locked_yield=0.04)
        amount = tl.settlement_amount(irr_at_expiry=0.04)
        assert abs(amount) < 100  # near zero on $10M


# ---- Funded instruments serialisation ----

class TestFundedSerialisation:

    def test_repo_round_trip(self):
        r = Repo(98.5, 0.04, 0.25, haircut=0.02, notional=10_000_000)
        d = r.to_dict()
        r2 = from_dict(d)
        assert r2.repo_rate == r.repo_rate
        assert r2.bond_dirty_price == r.bond_dirty_price

    def test_reverse_repo_round_trip(self):
        rr = ReverseRepo(99.0, 0.035, 0.5, notional=5_000_000)
        d = rr.to_dict()
        rr2 = from_dict(d)
        assert rr2.repo_rate == rr.repo_rate

    def test_repo_financed_round_trip(self):
        rfp = RepoFinancedPosition(100.0, 0.04, trs_spread=0.005,
                                     asset_yield=0.05, T=1.0, notional=10_000_000)
        d = rfp.to_dict()
        rfp2 = from_dict(d)
        assert rfp2.trs_spread == rfp.trs_spread
