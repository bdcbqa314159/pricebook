"""Tests for Phase 5 credit modules: ML PD, CDS-bond basis, joint equity-credit."""

import pytest
import numpy as np

from pricebook.credit.ml_pd import (
    LogisticPD, FinancialRatios, PDModelResult, predict_pd,
)
from pricebook.credit.cds_bond_basis import (
    compute_basis, BasisResult, basis_z_score, negative_basis_pnl,
)
from pricebook.credit.joint_equity_credit import (
    joint_calibrate, JointCalibrationResult,
)


# ═══════════════════════════════════════════════════════════════
# A2: ML-based PD
# ═══════════════════════════════════════════════════════════════


class TestMLPD:
    def test_ig_company(self):
        """Low leverage, high coverage → low PD."""
        r = FinancialRatios(
            leverage=0.30, interest_coverage=8.0, profitability=0.08,
            liquidity=2.0, size_log_assets=10.0, market_to_book=2.0,
            retained_earnings_ta=0.20, sales_ta=0.80, equity_vol=0.20,
        )
        result = predict_pd(r)
        assert result.pd_1y < 0.01  # < 1% PD
        assert result.implied_rating in ("AAA", "AA", "A", "BBB")

    def test_hy_company(self):
        """High leverage, low coverage → high PD."""
        r = FinancialRatios(
            leverage=0.75, interest_coverage=1.5, profitability=0.02,
            liquidity=0.8, size_log_assets=7.0, market_to_book=0.5,
            retained_earnings_ta=-0.05, sales_ta=0.50, equity_vol=0.50,
        )
        result = predict_pd(r)
        assert result.pd_1y > 0.02  # > 2% PD
        assert result.implied_rating in ("B", "CCC", "CC", "C")

    def test_pd_ordering(self):
        """Higher leverage → higher PD, all else equal."""
        base = FinancialRatios(0.30, 5.0, 0.05, 1.5, 9.0)
        high_lev = FinancialRatios(0.70, 5.0, 0.05, 1.5, 9.0)
        assert predict_pd(high_lev).pd_1y > predict_pd(base).pd_1y

    def test_pd_5y_greater(self):
        r = FinancialRatios(0.50, 3.0, 0.03, 1.2, 8.0)
        result = predict_pd(r)
        assert result.pd_5y > result.pd_1y

    def test_hazard_rate_positive(self):
        r = FinancialRatios(0.40, 4.0, 0.05, 1.5, 9.0)
        assert predict_pd(r).hazard_rate > 0

    def test_z_score(self):
        r = FinancialRatios(0.30, 6.0, 0.08, 2.0, 10.0, retained_earnings_ta=0.15, sales_ta=0.90)
        result = predict_pd(r)
        assert result.z_score > 0  # healthy company → positive Z

    def test_batch_predict(self):
        model = LogisticPD()
        ratios = [
            FinancialRatios(0.30, 6.0, 0.06, 2.0, 10.0),
            FinancialRatios(0.70, 1.5, 0.01, 0.8, 7.0),
        ]
        results = model.predict_batch(ratios)
        assert len(results) == 2
        assert results[0].pd_1y < results[1].pd_1y

    def test_to_dict(self):
        r = FinancialRatios(0.40, 4.0, 0.05, 1.5, 9.0)
        d = predict_pd(r).to_dict()
        assert "pd_1y" in d
        assert "implied_rating" in d


# ═══════════════════════════════════════════════════════════════
# A3: Sovereign CDS-bond basis
# ═══════════════════════════════════════════════════════════════


class TestCDSBondBasis:
    def test_positive_basis(self):
        r = compute_basis(200, 170)
        assert r.basis_bp == 30
        assert r.signal == "POSITIVE_BASIS"

    def test_negative_basis(self):
        r = compute_basis(170, 220)
        assert r.basis_bp == -50
        assert r.signal == "NEGATIVE_BASIS"

    def test_neutral(self):
        r = compute_basis(200, 195)
        assert r.signal == "NEUTRAL"

    def test_funding_component(self):
        r = compute_basis(200, 170, funding_spread_bp=20)
        assert r.funding_component_bp == 20

    def test_restructuring(self):
        r_with = compute_basis(200, 170, has_restructuring=True)
        r_without = compute_basis(200, 170, has_restructuring=False)
        assert r_with.restructuring_bp > r_without.restructuring_bp

    def test_components_sum(self):
        r = compute_basis(200, 170, funding_spread_bp=10)
        total = r.funding_component_bp + r.delivery_option_bp + r.restructuring_bp + r.residual_bp
        # Components + repo should roughly explain the basis
        # (repo effect is internal, absorbed in residual)
        assert abs(total - r.basis_bp) < 1.0

    def test_z_score(self):
        z = basis_z_score(50, 20, 15)
        assert z == pytest.approx(2.0)

    def test_negative_basis_pnl(self):
        """Negative basis trade: profit when basis becomes MORE negative (tightens further)."""
        # Entry at -30, exit at -50: basis tightened (more negative) → profit
        pnl = negative_basis_pnl(-30, -50, 10e6, 1.0, carry_bp=20)
        assert pnl["mtm_pnl"] > 0
        assert pnl["carry_pnl"] > 0
        assert pnl["total_pnl"] > 0

    def test_to_dict(self):
        d = compute_basis(200, 170).to_dict()
        assert "basis_bp" in d
        assert "signal" in d


# ═══════════════════════════════════════════════════════════════
# A5: Joint equity-credit calibration
# ═══════════════════════════════════════════════════════════════


class TestJointEquityCredit:
    def test_basic_calibration(self):
        result = joint_calibrate(equity_vol=0.30, cds_spread_bp=150)
        assert isinstance(result, JointCalibrationResult)
        assert result.asset_vol > 0
        assert 0 < result.leverage < 1

    def test_fit_quality(self):
        """Should fit both targets reasonably well."""
        result = joint_calibrate(equity_vol=0.30, cds_spread_bp=150)
        assert result.equity_vol_error_pct < 30  # within 30%
        assert result.cds_spread_error_bp < 100  # within 100bp

    def test_higher_vol_higher_asset_vol(self):
        r1 = joint_calibrate(equity_vol=0.20, cds_spread_bp=100)
        r2 = joint_calibrate(equity_vol=0.50, cds_spread_bp=100)
        assert r2.asset_vol > r1.asset_vol

    def test_higher_spread_higher_leverage(self):
        r1 = joint_calibrate(equity_vol=0.30, cds_spread_bp=50)
        r2 = joint_calibrate(equity_vol=0.30, cds_spread_bp=500)
        assert r2.leverage > r1.leverage

    def test_to_dict(self):
        d = joint_calibrate(0.30, 150).to_dict()
        assert "asset_vol" in d
        assert "fit_quality" in d
        assert "calibration_id" in d

    def test_canonical_record(self):
        result = joint_calibrate(equity_vol=0.30, cds_spread_bp=150)
        cr = result.to_calibration_result()
        assert cr is result.calibration_result
        assert cr.fit.model_class == "joint_equity_credit"
        assert set(cr.fit.parameters) == {"asset_vol", "leverage"}
        assert cr.fit.objective.value == "weighted_sse"
        assert len(cr.fit.residuals) == 2

    def test_canonical_on_demand_rebuild(self):
        from pricebook.credit.joint_equity_credit import JointCalibrationResult
        r = JointCalibrationResult(
            asset_vol=0.2, leverage=0.4, recovery_mean=0.4, recovery_vol=0.25,
            equity_vol_model=0.33, equity_vol_market=0.30,
            cds_spread_model_bp=152.0, cds_spread_market_bp=150.0,
            equity_vol_error_pct=10.0, cds_spread_error_bp=2.0, fit_quality=0.01,
        )
        cr = r.to_calibration_result()
        assert cr.fit.model_class == "joint_equity_credit"
        assert len(cr.fit.residuals) == 2

    def test_persists_via_db(self):
        from pricebook.db.db import PricebookDB
        result = joint_calibrate(equity_vol=0.30, cds_spread_bp=150)
        with PricebookDB(":memory:") as db:
            cid = db.save_calibration(result)
            assert db.load_calibration(cid) == result.to_calibration_result()
            assert db.list_calibrations(model_class="joint_equity_credit")[0]["calibration_id"] == cid
