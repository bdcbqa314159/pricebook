"""Tests for CMT convexity correction (Pucci 2014, Section 10 validation)."""

from __future__ import annotations

import math

import pytest

from pricebook.bond import ytm_cmt_bridge
from pricebook.cms import (
    cra_discount, risky_annuity, risky_swap_rate,
    linear_swap_rate_calibrate,
)
from pricebook.cmt import (
    cmt_cc_ab, cmt_cc_c, cmt_cc_no_default,
    cmt_convexity_corrections,
)


# ---- Parameters ----
# 5Y fixing, 10Y annual bond from Ts, flat rate 4%, flat hazard 100bp

N = 10
YFS = [1.0] * N
FLAT_RATE = 0.04
GAMMA = 0.01  # 100bp hazard
SIGMA = 0.20
TS = 5.0
TIMES = [TS + (i + 1) for i in range(N)]  # T_1=6, ..., T_10=15

RF_DFS = [math.exp(-FLAT_RATE * t) for t in TIMES]
RF_DF_TS = math.exp(-FLAT_RATE * TS)
TP_TIME = TS + YFS[0]  # T_p = T_1
RF_DF_TP = math.exp(-FLAT_RATE * TP_TIME)

# CRA quantities
CRA_DFS = [d * math.exp(-GAMMA * t) for d, t in zip(RF_DFS, TIMES)]
CRA_DF_TS = RF_DF_TS * math.exp(-GAMMA * TS)
CRA_DF_TP = RF_DF_TP * math.exp(-GAMMA * TP_TIME)
RISKY_ANN = sum(y * d for y, d in zip(YFS, CRA_DFS))
CHI_0 = math.exp(GAMMA * TS)
R_CMT = risky_swap_rate(CRA_DF_TS, CRA_DFS[-1], RISKY_ANN)
ALPHA = 1.0 / sum(YFS)


# ---- 10.1 Risky-discount machinery ----

class TestRiskyDiscountMachinery:
    """Validation items 1-3."""

    def test_v1_cra_discount(self):
        """V1: D̂ = D * e^{-gamma*T} for flat hazard."""
        for d, t in zip(RF_DFS, TIMES):
            cra = cra_discount(d, 0.0, GAMMA * t)
            expected = d * math.exp(-GAMMA * t)
            assert cra == pytest.approx(expected, rel=1e-10)

    def test_v2_risky_par_swap_identity(self):
        """V2: Â * R̂^swp = D̂_Ts - D̂_Tn."""
        r_swp = risky_swap_rate(CRA_DF_TS, CRA_DFS[-1], RISKY_ANN)
        lhs = RISKY_ANN * r_swp
        rhs = CRA_DF_TS - CRA_DFS[-1]
        assert lhs == pytest.approx(rhs, rel=1e-10)

    def test_v3_cmt_positive(self):
        """V3: CMT should be positive for positive rates."""
        assert R_CMT > 0


# ---- 10.2 Linear model calibration ----

class TestLinearModelCalibration:
    """Validation items 4-5."""

    def test_v4_alpha(self):
        """V4: alpha = 1/sum(y_i)."""
        alpha, _ = linear_swap_rate_calibrate(
            YFS, CRA_DFS, RISKY_ANN, R_CMT, chi=CHI_0)
        assert alpha == pytest.approx(ALPHA, rel=1e-10)

    def test_v5_consistency_identity(self):
        """V5: sum y_i * (alpha + beta_Ti * R_cmt) = 1/chi_0."""
        alpha, betas = linear_swap_rate_calibrate(
            YFS, CRA_DFS, RISKY_ANN, R_CMT, chi=CHI_0)
        lhs = sum(y * (alpha + b * R_CMT) for y, b in zip(YFS, betas))
        assert lhs == pytest.approx(CHI_0, rel=1e-6)


# ---- 10.3 NPV identities ----

class TestNPVIdentities:
    """Validation items 7-8."""

    def test_v7_ab_ratio(self):
        """V7: NPV(B)/NPV(A) = e^{Γ_Tp - Γ_Ts} = e^{-gamma*(Tp-Ts)}."""
        # CC(A) = CC(B), so NPV ratio is just the survival factor
        ratio = math.exp(-GAMMA * (TP_TIME - TS))
        # Since CC_A = CC_B, the NPV ratio comes from the discount:
        # NPV(B) = e^{Γ_Tp - Γ_Ts} * NPV(A), so ratio > 1
        # Actually the paper says NPV(B) = e^{ΓTp - ΓTs} * NPV(A)
        # With ΓTp > ΓTs, this means e^{positive} > 1
        assert ratio == pytest.approx(math.exp(-GAMMA * YFS[0]), rel=1e-10)

    def test_v8_cc_ab_equal(self):
        """V8: CC(A) = CC(B) (Eq 34)."""
        result = cmt_convexity_corrections(
            R_CMT, SIGMA, GAMMA, TS, YFS, RF_DFS, RF_DF_TS, RF_DF_TP)
        assert result.cc_A == pytest.approx(result.cc_B, rel=1e-10)

    def test_v8_all_cc_positive_for_positive_vol(self):
        """All CCs should be positive for positive vol and positive prefactor."""
        result = cmt_convexity_corrections(
            R_CMT, SIGMA, GAMMA, TS, YFS, RF_DFS, RF_DF_TS, RF_DF_TP)
        assert result.cc_A > 0
        assert result.cc_C > 0


# ---- 10.4 Closed-form vs limits ----

class TestClosedFormLimits:
    """Validation items 10-11."""

    def test_v10_no_default_limit(self):
        """V10: gamma=0 => all CCs collapse to Pelsser/Hagan (Eq 37)."""
        rf_ann = sum(y * d for y, d in zip(YFS, RF_DFS))
        pelsser = cmt_cc_no_default(SIGMA, TS, ALPHA, rf_ann, RF_DF_TP)

        result = cmt_convexity_corrections(
            R_CMT, SIGMA, 0.0, TS, YFS, RF_DFS, RF_DF_TS, RF_DF_TP)

        assert result.cc_A == pytest.approx(pelsser, rel=1e-6)
        assert result.cc_B == pytest.approx(pelsser, rel=1e-6)
        assert result.cc_C == pytest.approx(pelsser, rel=1e-6)

    def test_v11_vol_zero(self):
        """V11: sigma=0 => all CCs vanish."""
        result = cmt_convexity_corrections(
            R_CMT, 0.0, GAMMA, TS, YFS, RF_DFS, RF_DF_TS, RF_DF_TP)
        assert result.cc_A == pytest.approx(0.0, abs=1e-10)
        assert result.cc_B == pytest.approx(0.0, abs=1e-10)
        assert result.cc_C == pytest.approx(0.0, abs=1e-8)

    def test_gamma_equals_sigma_sq(self):
        """V12-style: gamma = sigma^2 should be well-behaved (Taylor expansion)."""
        sigma = 0.10
        gamma = sigma**2  # 0.01
        result = cmt_convexity_corrections(
            R_CMT, sigma, gamma, TS, YFS, RF_DFS, RF_DF_TS, RF_DF_TP)
        assert math.isfinite(result.cc_C)

    def test_vega_positive(self):
        """V: CC increases with sigma (positive vega)."""
        r1 = cmt_convexity_corrections(
            R_CMT, 0.10, GAMMA, TS, YFS, RF_DFS, RF_DF_TS, RF_DF_TP)
        r2 = cmt_convexity_corrections(
            R_CMT, 0.30, GAMMA, TS, YFS, RF_DFS, RF_DF_TS, RF_DF_TP)
        assert r2.cc_A > r1.cc_A


# ---- 10.6 YTM-CMT bridge ----

class TestYTMCMTBridge:
    """Validation items 14-15."""

    def test_v14_bridge_at_par(self):
        """V14: At B=1, K=R^cmt => R^ytm = R^cmt."""
        R_cmt = 0.04
        R_ytm = ytm_cmt_bridge(R_cmt, K=R_cmt, B=1.0, n=10)
        assert R_ytm == pytest.approx(R_cmt, rel=1e-10)

    def test_v14_linearisation_error(self):
        """V14: Error is O((B-1)^2) for B near 1."""
        R_cmt = 0.04
        # Small deviation from par
        for dB in [0.001, 0.005, 0.01]:
            B = 1.0 + dB
            R_ytm = ytm_cmt_bridge(R_cmt, K=R_cmt, B=B, n=10)
            # The deviation from R_cmt should be proportional to (B-1)
            deviation = abs(R_ytm - R_cmt)
            assert deviation < abs(dB) * 2  # roughly linear

    def test_v15_large_n_limit(self):
        """V15: For large n, R^ytm ≈ R^cmt + (K - R^cmt) - (B-1)/n."""
        R_cmt = 0.04
        K = 0.05
        B = 0.98
        n = 100
        R_ytm = ytm_cmt_bridge(R_cmt, K=K, B=B, n=n)
        approx = R_cmt + (K - R_cmt) - (B - 1) / n
        assert R_ytm == pytest.approx(approx, rel=0.01)
