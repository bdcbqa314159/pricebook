"""Tests for cross-asset desk aggregation."""

from __future__ import annotations

import math
from datetime import date

from dateutil.relativedelta import relativedelta

from pricebook.desks.cross_asset_desk import (
    CrossAssetDesk, CrossAssetDashboard, DeskRiskSummary,
)
from pricebook.fixed_income.swap import InterestRateSwap, SwapDirection
from pricebook.desks.swap_desk import SwapBook, SwapBookEntry
from pricebook.cds import CDS
from pricebook.desks.cds_desk import CDSBook, CDSBookEntry, CDSProductType
from pricebook.survival_curve import SurvivalCurve
from tests.conftest import make_flat_curve, make_flat_survival


REF = date(2024, 7, 15)


class TestCrossAssetDesk:

    def _swap_book(self):
        book = SwapBook()
        swap = InterestRateSwap(REF, REF + relativedelta(years=5),
                                fixed_rate=0.04, notional=50_000_000)
        book.add(SwapBookEntry("SW1", swap, "JPM"))
        return book

    def _cds_book(self):
        book = CDSBook()
        surv = make_flat_survival(REF, 0.02)
        cds = CDS(REF, REF + relativedelta(years=5), spread=0.01, notional=10_000_000)
        book.add(CDSBookEntry("CD1", cds, surv, reference_name="AAPL", sector="tech"))
        return book

    def test_add_desks(self):
        desk = CrossAssetDesk()
        desk.add("swap", self._swap_book())
        desk.add("cds", self._cds_book())
        assert desk.n_desks() == 2
        assert "swap" in desk.desk_names

    def test_aggregate(self):
        desk = CrossAssetDesk()
        desk.add("swap", self._swap_book())
        desk.add("cds", self._cds_book())
        curve = make_flat_curve(REF, 0.04)
        summaries = desk.aggregate(curve)
        assert len(summaries) == 2
        assert all(isinstance(s, DeskRiskSummary) for s in summaries)

    def test_dashboard(self):
        desk = CrossAssetDesk()
        desk.add("swap", self._swap_book())
        desk.add("cds", self._cds_book())
        curve = make_flat_curve(REF, 0.04)
        db = desk.dashboard(REF, curve)
        assert db.n_desks == 2
        assert db.n_positions == 2
        assert math.isfinite(db.total_pv)
        assert len(db.by_desk) == 2

    def test_total_pv(self):
        desk = CrossAssetDesk()
        desk.add("swap", self._swap_book())
        curve = make_flat_curve(REF, 0.04)
        pv = desk.total_pv(curve)
        assert math.isfinite(pv)

    def test_to_dict(self):
        desk = CrossAssetDesk()
        desk.add("swap", self._swap_book())
        curve = make_flat_curve(REF, 0.04)
        d = desk.dashboard(REF, curve).to_dict()
        assert "by_desk" in d
        assert "total_pv" in d
        assert "n_desks" in d

    def test_empty_desk(self):
        desk = CrossAssetDesk()
        curve = make_flat_curve(REF, 0.04)
        db = desk.dashboard(REF, curve)
        assert db.n_desks == 0
        assert db.total_pv == 0.0
