"""Tests for credit risk measures and credit book."""

import pytest
from datetime import date

from pricebook.credit_risk import (
    cs01, spread_dv01, jump_to_default,
    CreditBook, CreditPosition,
    _bump_survival_curve,
)
from pricebook.cds import CDS, bootstrap_credit_curve
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.trade import Trade


REF = date(2024, 1, 15)


def _dc(rate=0.05):
    return DiscountCurve.flat(REF, rate)


def _sc(hazard=0.02):
    return SurvivalCurve.flat(REF, hazard)


def _cds(spread=0.01, notional=10_000_000, recovery=0.4):
    return CDS(REF, date(2029, 1, 15), spread, notional=notional, recovery=recovery)


# ---- Survival curve bumping ----

class TestBumpSurvival:
    def test_bumped_survives_less(self):
        sc = _sc(0.02)
        bumped = _bump_survival_curve(sc, 0.01)  # +100bp hazard
        d = date(2029, 1, 15)
        assert bumped.survival(d) < sc.survival(d)

    def test_zero_bump_unchanged(self):
        sc = _sc(0.02)
        bumped = _bump_survival_curve(sc, 0.0)
        d = date(2029, 1, 15)
        assert bumped.survival(d) == pytest.approx(sc.survival(d), rel=1e-6)


# ---- CS01 ----

class TestCS01:
    def test_protection_buyer_positive_cs01(self):
        """Protection buyer gains from wider spreads → CS01 > 0."""
        dc = _dc()
        sc = _sc()
        cds = _cds()
        result = cs01(cds, dc, sc)
        # When hazard rates go up: protection leg gains (more default),
        # premium leg shrinks (fewer payments) → net PV rises
        assert result > 0

    def test_cs01_scales_with_notional(self):
        dc = _dc()
        sc = _sc()
        c1 = cs01(_cds(notional=10_000_000), dc, sc)
        c2 = cs01(_cds(notional=20_000_000), dc, sc)
        assert c2 == pytest.approx(2 * c1, rel=0.01)

    def test_cs01_matches_finite_difference(self):
        """CS01 should match a manual finite-difference bump."""
        dc = _dc()
        sc = _sc(0.02)
        cds = _cds()
        # Manual bump
        shift = 0.0001  # 1bp
        sc_up = _bump_survival_curve(sc, shift)
        pv_base = cds.pv(dc, sc)
        pv_up = cds.pv(dc, sc_up)
        manual_cs01 = (pv_up - pv_base) / 1.0  # per bp
        computed = cs01(cds, dc, sc)
        assert computed == pytest.approx(manual_cs01, rel=0.01)


# ---- Spread DV01 ----

class TestSpreadDV01:
    def test_returns_per_pillar(self):
        dc = _dc()
        sc = _sc()
        cds = _cds()
        result = spread_dv01(cds, dc, sc)
        assert len(result) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in result)

    def test_sum_approx_cs01(self):
        """Sum of key-rate CS01s should approximate parallel CS01."""
        dc = _dc()
        sc = _sc()
        cds = _cds()
        total = sum(v for _, v in spread_dv01(cds, dc, sc))
        parallel = cs01(cds, dc, sc)
        # Won't be exact due to cross-effects, but should be close
        assert total == pytest.approx(parallel, rel=0.3)


# ---- Jump-to-default ----

class TestJTD:
    def test_protection_buyer_positive_jtd(self):
        """Protection buyer receives LGD on default."""
        dc = _dc()
        sc = _sc()
        cds = _cds(recovery=0.4)
        jtd = jump_to_default(cds, dc, sc)
        # JTD = (1-R) * notional - PV. For near-par CDS, PV ≈ 0
        # so JTD ≈ 6M
        assert jtd > 0
        assert jtd == pytest.approx(0.6 * 10_000_000, rel=0.1)

    def test_jtd_scales_with_lgd(self):
        dc = _dc()
        sc = _sc()
        jtd_40 = jump_to_default(_cds(recovery=0.4), dc, sc)
        jtd_60 = jump_to_default(_cds(recovery=0.6), dc, sc)
        assert jtd_40 > jtd_60  # lower recovery → higher JTD


# ---- Credit Book ----

class TestCreditBook:
    def test_create_empty(self):
        book = CreditBook("credit")
        assert len(book) == 0

    def test_add_trades(self):
        book = CreditBook("credit")
        sc = _sc()
        book.add(Trade(_cds(), trade_id="t1"), "ACME", sc)
        book.add(Trade(_cds(), trade_id="t2"), "BETA", sc)
        assert len(book) == 2

    def test_pv(self):
        dc = _dc()
        sc = _sc()
        book = CreditBook("credit")
        cds = _cds()
        book.add(Trade(cds, trade_id="t1"), "ACME", sc)
        pv = book.pv(dc)
        assert pv == pytest.approx(cds.pv(dc, sc))

    def test_total_cs01(self):
        dc = _dc()
        sc = _sc()
        book = CreditBook("credit")
        cds = _cds()
        book.add(Trade(cds, trade_id="t1"), "ACME", sc)
        total = book.total_cs01(dc)
        single = cs01(cds, dc, sc)
        assert total == pytest.approx(single)

    def test_cs01_sum_of_trades(self):
        dc = _dc()
        sc1 = _sc(0.02)
        sc2 = _sc(0.03)
        cds1 = _cds(spread=0.01)
        cds2 = _cds(spread=0.02)
        book = CreditBook("credit")
        book.add(Trade(cds1, trade_id="t1"), "ACME", sc1)
        book.add(Trade(cds2, trade_id="t2"), "BETA", sc2)
        total = book.total_cs01(dc)
        individual = cs01(cds1, dc, sc1) + cs01(cds2, dc, sc2)
        assert total == pytest.approx(individual)

    def test_cs01_by_name(self):
        dc = _dc()
        sc = _sc()
        book = CreditBook("credit")
        book.add(Trade(_cds(), trade_id="t1"), "ACME", sc)
        book.add(Trade(_cds(), trade_id="t2"), "BETA", sc)
        by_name = book.cs01_by_name(dc)
        assert "ACME" in by_name
        assert "BETA" in by_name

    def test_jtd_by_name(self):
        dc = _dc()
        sc = _sc()
        book = CreditBook("credit")
        book.add(Trade(_cds(), trade_id="t1"), "ACME", sc)
        jtd = book.jtd_by_name(dc)
        assert "ACME" in jtd
        assert jtd["ACME"] > 0

    def test_positions(self):
        dc = _dc()
        sc = _sc()
        book = CreditBook("credit")
        book.add(Trade(_cds(notional=10_000_000), trade_id="t1"), "ACME", sc)
        book.add(Trade(_cds(notional=5_000_000), direction=-1, trade_id="t2"), "ACME", sc)
        positions = book.positions(dc)
        assert len(positions) == 1
        assert positions[0].name == "ACME"
        assert positions[0].net_notional == pytest.approx(5_000_000)
        assert positions[0].trade_count == 2

    def test_long_short_offset(self):
        dc = _dc()
        sc = _sc()
        book = CreditBook("credit")
        book.add(Trade(_cds(), direction=1, trade_id="long"), "ACME", sc)
        book.add(Trade(_cds(), direction=-1, trade_id="short"), "ACME", sc)
        total = book.total_cs01(dc)
        assert total == pytest.approx(0.0, abs=1e-6)
