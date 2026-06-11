"""MarketSnapshot linkage on jump-model calibrators (G1 P2 Slice 5, closes G1 P2)."""

from __future__ import annotations

import inspect

import pytest

from pricebook.calibration import CalibrationResult
from pricebook.market_data import MarketSnapshot, Quote, QuoteId, QuoteKind
from pricebook.models.jump_calibration import (
    calibrate_jump_model,
    calibrate_jump_surface,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def smile_snapshot() -> MarketSnapshot:
    return MarketSnapshot.new(
        quotes=[
            Quote(QuoteId(QuoteKind.VOL_POINT, "smile-1Y", "USD", "K=80"), value=0.25),
            Quote(QuoteId(QuoteKind.VOL_POINT, "smile-1Y", "USD", "K=100"), value=0.22),
            Quote(QuoteId(QuoteKind.VOL_POINT, "smile-1Y", "USD", "K=120"), value=0.24),
        ],
        label="EOD-smile",
    )


# ============================================================
# calibrate_jump_model
# ============================================================

class TestJumpModelSnapshot:
    def _calibrate(self, **kw):
        # Small problem, low maxiter — keeps tests fast.
        return calibrate_jump_model(
            model_type="merton",
            strikes=[80.0, 100.0, 120.0],
            market_vols=[0.25, 0.22, 0.24],
            spot=100.0, rate=0.04, T=1.0, maxiter=30,
            **kw,
        )

    def test_snapshot_is_keyword_only(self):
        param = inspect.signature(calibrate_jump_model).parameters["market_snapshot"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is None

    def test_without_snapshot(self):
        r = self._calibrate()
        assert isinstance(r.calibration_result, CalibrationResult)
        assert r.calibration_result.market_snapshot_id is None

    def test_with_snapshot_links_id(self, smile_snapshot):
        r = self._calibrate(market_snapshot=smile_snapshot)
        assert r.calibration_result.market_snapshot_id == smile_snapshot.id


# ============================================================
# calibrate_jump_surface
# ============================================================

class TestJumpSurfaceSnapshot:
    def _market_data(self):
        return [
            {"T": 0.5, "strikes": [90.0, 100.0, 110.0],
             "vols": [0.24, 0.21, 0.23]},
            {"T": 1.0, "strikes": [90.0, 100.0, 110.0],
             "vols": [0.23, 0.20, 0.22]},
        ]

    def test_snapshot_is_keyword_only(self):
        param = inspect.signature(calibrate_jump_surface).parameters["market_snapshot"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is None

    def test_with_snapshot_stamps_every_result(self, smile_snapshot):
        results = calibrate_jump_surface(
            model_type="merton",
            market_data=self._market_data(),
            spot=100.0, rate=0.04, maxiter=30,
            market_snapshot=smile_snapshot,
        )
        assert len(results) == 2
        for r in results:
            assert r.calibration_result.market_snapshot_id == smile_snapshot.id
        # Distinct calibration_result ids per expiry (one snapshot can underlie
        # many independent fits — the snapshot is shared, the calibrations aren't).
        assert results[0].calibration_result.id != results[1].calibration_result.id

    def test_without_snapshot_all_none(self):
        results = calibrate_jump_surface(
            model_type="merton",
            market_data=self._market_data(),
            spot=100.0, rate=0.04, maxiter=30,
        )
        for r in results:
            assert r.calibration_result.market_snapshot_id is None
