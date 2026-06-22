"""MarketSnapshot linkage on HW + G2++ + SABR + LMM calibrators (G1 P2 Slice 4)."""

from __future__ import annotations

import inspect
import math
from datetime import date

import pytest

from pricebook.calibration import CalibrationResult
from pricebook.core.discount_curve import DiscountCurve
from pricebook.market_data import MarketSnapshot, Quote, QuoteId, QuoteKind
from pricebook.models.g2pp_calibration import calibrate_g2pp
from pricebook.models.hw_calibration import calibrate_hull_white
from pricebook.models.lmm_calibration import calibrate_lmm_vols
from pricebook.options.sabr import sabr_calibrate


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def ref_date() -> date:
    return date(2026, 6, 11)


@pytest.fixture
def flat_curve(ref_date) -> DiscountCurve:
    return DiscountCurve.flat(ref_date, 0.04)


@pytest.fixture
def swaption_vols() -> dict[tuple[float, float], float]:
    # Small grid — keeps the calibration tests fast.
    return {
        (1, 5): 0.0065,
        (5, 5): 0.0055,
    }


@pytest.fixture
def vol_snapshot() -> MarketSnapshot:
    return MarketSnapshot.new(
        quotes=[
            Quote(QuoteId(QuoteKind.SWAPTION_VOL, "1Yx5Y", "USD"), value=0.0065),
            Quote(QuoteId(QuoteKind.SWAPTION_VOL, "5Yx5Y", "USD"), value=0.0055),
        ],
        label="EOD-vols",
    )


# ============================================================
# Hull-White
# ============================================================

class TestHullWhiteSnapshot:
    def test_snapshot_is_keyword_only(self):
        param = inspect.signature(calibrate_hull_white).parameters["market_snapshot"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is None

    def test_without_snapshot(self, flat_curve, swaption_vols):
        r = calibrate_hull_white(flat_curve, swaption_vols, n_steps=20)
        assert isinstance(r.calibration_result, CalibrationResult)
        assert r.calibration_result.provenance.market_snapshot_id is None

    def test_with_snapshot_links_id(self, flat_curve, swaption_vols, vol_snapshot):
        r = calibrate_hull_white(
            flat_curve, swaption_vols, n_steps=20, market_snapshot=vol_snapshot,
        )
        assert r.calibration_result.provenance.market_snapshot_id == vol_snapshot.id


# ============================================================
# G2++
# ============================================================

class TestG2ppSnapshot:
    def test_snapshot_is_keyword_only(self):
        param = inspect.signature(calibrate_g2pp).parameters["market_snapshot"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is None

    def test_with_snapshot_links_id(self, flat_curve, swaption_vols, vol_snapshot):
        # Use minimize (fast) instead of DE so this stays under a second.
        r = calibrate_g2pp(
            flat_curve, swaption_vols, method="minimize", market_snapshot=vol_snapshot,
        )
        assert r.calibration_result.provenance.market_snapshot_id == vol_snapshot.id

    def test_without_snapshot(self, flat_curve, swaption_vols):
        r = calibrate_g2pp(flat_curve, swaption_vols, method="minimize")
        assert r.calibration_result.provenance.market_snapshot_id is None


# ============================================================
# SABR
# ============================================================

class TestSabrSnapshot:
    @pytest.fixture
    def smile_snapshot(self) -> MarketSnapshot:
        return MarketSnapshot.new(
            quotes=[
                Quote(QuoteId(QuoteKind.VOL_POINT, "smile-1Y", "USD", "K=80"), value=0.25),
                Quote(QuoteId(QuoteKind.VOL_POINT, "smile-1Y", "USD", "K=100"), value=0.22),
                Quote(QuoteId(QuoteKind.VOL_POINT, "smile-1Y", "USD", "K=120"), value=0.24),
            ],
            label="EOD-smile",
        )

    def test_snapshot_is_keyword_only(self):
        param = inspect.signature(sabr_calibrate).parameters["market_snapshot"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is None

    def test_without_snapshot(self):
        d = sabr_calibrate(
            forward=100.0, strikes=[80.0, 100.0, 120.0],
            market_vols=[0.25, 0.22, 0.24], T=1.0, beta=0.5,
        )
        cr = d.calibration_result
        assert cr.provenance.market_snapshot_id is None

    def test_with_snapshot_links_id(self, smile_snapshot):
        d = sabr_calibrate(
            forward=100.0, strikes=[80.0, 100.0, 120.0],
            market_vols=[0.25, 0.22, 0.24], T=1.0, beta=0.5,
            market_snapshot=smile_snapshot,
        )
        assert d.calibration_result.provenance.market_snapshot_id == smile_snapshot.id


# ============================================================
# LMM
# ============================================================

class TestLmmSnapshot:
    def test_snapshot_is_keyword_only(self):
        param = inspect.signature(calibrate_lmm_vols).parameters["market_snapshot"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is None

    def test_without_snapshot(self):
        forwards = [0.04, 0.045, 0.05, 0.055]
        targets = {(0, 2): 0.20, (1, 2): 0.18}
        r = calibrate_lmm_vols(forwards, targets, max_iter=20)
        assert r.calibration_result.provenance.market_snapshot_id is None

    def test_with_snapshot_links_id(self, vol_snapshot):
        forwards = [0.04, 0.045, 0.05, 0.055]
        targets = {(0, 2): 0.20, (1, 2): 0.18}
        r = calibrate_lmm_vols(
            forwards, targets, max_iter=20, market_snapshot=vol_snapshot,
        )
        assert r.calibration_result.provenance.market_snapshot_id == vol_snapshot.id
