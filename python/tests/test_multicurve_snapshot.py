"""MarketSnapshot linkage on multicurve_newton + bond hazard bootstrap (G1 P2 Slice 3)."""

from __future__ import annotations

import inspect
from datetime import date

import pytest

from pricebook.calibration import CalibrationResult
from pricebook.core.discount_curve import DiscountCurve
from pricebook.credit.bond_hazard_bootstrap import (
    BondInput,
    _price_risky_bond,
    bootstrap_hazard_from_bonds,
)
from pricebook.core.survival_curve import SurvivalCurve
from pricebook.curves.multicurve_solver import multicurve_newton
from pricebook.market_data import MarketSnapshot, Quote, QuoteId, QuoteKind


# ============================================================
# Shared fixtures
# ============================================================

@pytest.fixture
def ref_date() -> date:
    return date(2026, 6, 11)


@pytest.fixture
def snapshot() -> MarketSnapshot:
    return MarketSnapshot.new(
        quotes=[
            Quote(QuoteId(QuoteKind.SWAP_RATE, "1Y", "USD"), value=0.030),
            Quote(QuoteId(QuoteKind.SWAP_RATE, "2Y", "USD"), value=0.032),
        ],
        label="EOD-test",
    )


# ============================================================
# multicurve_newton
# ============================================================

class TestMulticurveSnapshot:
    def _instruments(self, ref_date):
        ois_insts = [
            {"type": "swap", "maturity": date(ref_date.year + 1, 6, 11), "rate": 0.030},
            {"type": "swap", "maturity": date(ref_date.year + 2, 6, 11), "rate": 0.032},
        ]
        proj_insts = [
            {"type": "swap", "maturity": date(ref_date.year + 1, 6, 11), "rate": 0.032},
            {"type": "swap", "maturity": date(ref_date.year + 2, 6, 11), "rate": 0.034},
        ]
        ois_dates = [d["maturity"] for d in ois_insts]
        proj_dates = [d["maturity"] for d in proj_insts]
        return ois_insts, proj_insts, ois_dates, proj_dates

    def test_snapshot_is_keyword_only(self):
        sig = inspect.signature(multicurve_newton)
        param = sig.parameters["market_snapshot"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is None

    def test_without_snapshot_id_is_none(self, ref_date):
        ois_insts, proj_insts, ois_dates, proj_dates = self._instruments(ref_date)
        result = multicurve_newton(
            ref_date, ois_insts, proj_insts, ois_dates, proj_dates,
        )
        cr = result.calibration_result
        assert isinstance(cr, CalibrationResult)
        assert cr.provenance.market_snapshot_id is None

    def test_with_snapshot_links_id(self, ref_date, snapshot):
        ois_insts, proj_insts, ois_dates, proj_dates = self._instruments(ref_date)
        result = multicurve_newton(
            ref_date, ois_insts, proj_insts, ois_dates, proj_dates,
            market_snapshot=snapshot,
        )
        cr = result.calibration_result
        assert cr is not None
        assert cr.provenance.market_snapshot_id == snapshot.id
        # Both curves carry the linked CR (same instance).
        assert result.ois_curve.calibration_result is cr
        assert result.projection_curve.calibration_result is cr

    def test_with_snapshot_non_converged_path(self, ref_date, snapshot):
        ois_insts, proj_insts, ois_dates, proj_dates = self._instruments(ref_date)
        # Force the non-converged path: max_iter=1 is too few.
        with pytest.warns(RuntimeWarning):
            result = multicurve_newton(
                ref_date, ois_insts, proj_insts, ois_dates, proj_dates,
                tol=1e-20, max_iter=1, market_snapshot=snapshot,
            )
        # Even on non-convergence, the snapshot id is recorded.
        assert result.calibration_result.provenance.market_snapshot_id == snapshot.id


# ============================================================
# bond_hazard_bootstrap
# ============================================================

def _make_bond_at_spread(ref_date, maturity_years, coupon, spread_bp, recovery, flat_rate):
    mat = date(ref_date.year + maturity_years, ref_date.month, ref_date.day)
    dc = DiscountCurve.flat(ref_date, flat_rate)
    hazard = spread_bp / 10_000 / (1 - recovery)
    sc = SurvivalCurve.flat(
        ref_date, hazard, tenors=list(range(1, maturity_years + 2)),
    )
    price = _price_risky_bond(ref_date, mat, coupon, 2, recovery, dc, sc)
    return BondInput(
        maturity=mat, coupon=coupon, market_price=price,
        frequency=2, recovery=recovery,
    )


class TestBondHazardSnapshot:
    @pytest.fixture
    def bonds(self, ref_date):
        return [
            _make_bond_at_spread(ref_date, 2, 0.04, 100, 0.40, 0.04),
            _make_bond_at_spread(ref_date, 5, 0.05, 150, 0.40, 0.04),
            _make_bond_at_spread(ref_date, 10, 0.06, 200, 0.40, 0.04),
        ]

    @pytest.fixture
    def cds_snapshot(self) -> MarketSnapshot:
        # Bond/CDS-style snapshot — quote contents don't have to match the
        # internal bond list (snapshot is a provenance pointer in this slice).
        return MarketSnapshot.new(
            quotes=[
                Quote(QuoteId(QuoteKind.CDS_SPREAD, "5Y", "USD"), value=0.012),
            ],
            label="EOD-credit",
        )

    def test_snapshot_is_keyword_only(self):
        sig = inspect.signature(bootstrap_hazard_from_bonds)
        param = sig.parameters["market_snapshot"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is None

    def test_sequential_without_snapshot(self, ref_date, bonds):
        flat = DiscountCurve.flat(ref_date, 0.04)
        result = bootstrap_hazard_from_bonds(
            ref_date, bonds, flat, method="sequential",
        )
        cr = result.calibration_result
        assert cr is not None
        assert cr.provenance.market_snapshot_id is None

    def test_sequential_with_snapshot(self, ref_date, bonds, cds_snapshot):
        flat = DiscountCurve.flat(ref_date, 0.04)
        result = bootstrap_hazard_from_bonds(
            ref_date, bonds, flat, method="sequential",
            market_snapshot=cds_snapshot,
        )
        assert result.calibration_result.provenance.market_snapshot_id == cds_snapshot.id

    def test_global_with_snapshot(self, ref_date, bonds, cds_snapshot):
        flat = DiscountCurve.flat(ref_date, 0.04)
        result = bootstrap_hazard_from_bonds(
            ref_date, bonds, flat, method="global", n_pillars=2,
            market_snapshot=cds_snapshot,
        )
        assert result.calibration_result.provenance.market_snapshot_id == cds_snapshot.id

    def test_snapshot_does_not_alter_numerics(self, ref_date, bonds, cds_snapshot):
        flat = DiscountCurve.flat(ref_date, 0.04)
        r_a = bootstrap_hazard_from_bonds(
            ref_date, bonds, flat, method="sequential",
        )
        r_b = bootstrap_hazard_from_bonds(
            ref_date, bonds, flat, method="sequential",
            market_snapshot=cds_snapshot,
        )
        assert r_a.pillar_hazards == pytest.approx(r_b.pillar_hazards, rel=1e-12)
        assert r_a.fitted_prices == pytest.approx(r_b.fitted_prices, rel=1e-12)
