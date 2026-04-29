"""Tests for CDS strategies: curve trades, basis, recovery lock."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta

import pytest

from pricebook.cds import CDS
from pricebook.cds_strategies import (
    CDSCurveTrade, flatten, steepen,
    cds_bond_basis, basis_trade,
    recovery_lock_pv, digital_cds_spread,
)
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.serialisable import from_dict

REF = date(2026, 4, 28)


def _disc():
    return DiscountCurve.flat(REF, 0.03)


def _surv():
    return SurvivalCurve.flat(REF, 0.02)


class TestCurveTrade:

    def test_flatten(self):
        trade = flatten(REF, _disc(), _surv())
        pnl = trade.pnl(_disc(), _surv())
        assert math.isfinite(pnl.total_pv)

    def test_dv01_neutral(self):
        """DV01-neutral flattener should have near-zero parallel CS01."""
        trade = flatten(REF, _disc(), _surv(), dv01_neutral=True)
        pnl = trade.pnl(_disc(), _surv())
        assert abs(pnl.parallel_cs01) < abs(pnl.short_cs01) * 0.1

    def test_steepen_opposite(self):
        flat = flatten(REF, _disc(), _surv())
        steep = steepen(REF, _disc(), _surv())
        pnl_flat = flat.pnl(_disc(), _surv())
        pnl_steep = steep.pnl(_disc(), _surv())
        # Opposite positions
        assert pnl_flat.total_pv == pytest.approx(-pnl_steep.total_pv, abs=1.0)

    def test_serialisation(self):
        trade = flatten(REF, _disc(), _surv())
        d = trade.to_dict()
        assert d["type"] == "cds_curve_trade"
        trade2 = from_dict(d)
        assert trade2.short_cds.spread == trade.short_cds.spread

    def test_json(self):
        trade = flatten(REF, _disc(), _surv())
        s = json.dumps(trade.to_dict())
        trade2 = from_dict(json.loads(s))
        assert math.isfinite(trade2.short_cds.notional)


class TestBasis:

    def test_negative_basis(self):
        b = cds_bond_basis(cds_spread=0.004, asw_spread=0.006)
        assert b < 0

    def test_positive_basis(self):
        b = cds_bond_basis(cds_spread=0.008, asw_spread=0.005)
        assert b > 0

    def test_basis_trade_negative(self):
        r = basis_trade(cds_spread=0.004, asw_spread=0.006)
        assert r.is_negative
        assert r.pv_if_no_default > 0  # negative basis → positive carry

    def test_basis_trade_positive(self):
        r = basis_trade(cds_spread=0.008, asw_spread=0.005)
        assert not r.is_negative
        assert r.pv_if_no_default < 0


class TestRecoveryLock:

    def test_lock_at_market_recovery(self):
        cds = CDS(REF, REF + timedelta(days=1825), spread=0.01, recovery=0.4)
        pv = recovery_lock_pv(cds, lock_recovery=0.4, discount_curve=_disc(),
                               survival_curve=_surv())
        assert pv == pytest.approx(0.0, abs=1.0)

    def test_lock_above_market(self):
        cds = CDS(REF, REF + timedelta(days=1825), spread=0.01, recovery=0.4)
        pv = recovery_lock_pv(cds, lock_recovery=0.6, discount_curve=_disc(),
                               survival_curve=_surv())
        assert pv > 0  # higher lock → seller profits

    def test_digital_cds_spread(self):
        cds = CDS(REF, REF + timedelta(days=1825), spread=0.01, recovery=0.4)
        ds = digital_cds_spread(cds, _disc(), _surv())
        par = cds.par_spread(_disc(), _surv())
        assert ds == pytest.approx(par / 0.6, abs=0.001)
