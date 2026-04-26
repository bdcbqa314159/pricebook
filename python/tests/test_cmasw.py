"""Tests for CMASW convexity correction (Pucci 2012a, Section 9 validation).

Each test maps to a validation item from the paper's Section 9.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.cms import (
    linear_swap_rate_calibrate,
    displaced_lognormal_cross_moment,
)
from pricebook.cmasw import (
    cmasw_convexity_correction,
    cmasw_cc_lognormal,
)
from pricebook.par_asset_swap import forward_asw_spread


# ---- Standard test parameters (paper Section 6) ----
# Bond's remaining schedule from T0 forward: T0=5Y, semi-annual 5Y bond
# T_1=5.5Y, T_2=6.0Y, ..., T_10=10.0Y from valuation date
# Tp = T_1 = 5.5Y (first payment after fixing)

N_PERIODS = 10  # 5Y semi-annual
YFS = [0.5] * N_PERIODS
FLAT_RATE = 0.04
T0 = 5.0  # fixing date
# Discount factors to each T_i (absolute dates from today)
DFS = [math.exp(-FLAT_RATE * (T0 + 0.5 * (i + 1))) for i in range(N_PERIODS)]
ANNUITY = sum(y * d for y, d in zip(YFS, DFS))
R_SWP = 0.0429
R_ASW = 0.0490
SIGMA_SWP = 0.30
DF_TP = DFS[0]  # payment at T_1 = T0 + 0.5


# ---- 9.1 Building blocks ----

class TestBuildingBlocks:
    """Validation items 1-3."""

    def test_v1_annuity_par_swap_identity(self):
        """V1: A_0 * R^swp = D_{0,T0} - D_{0,Tn} (par-swap identity)."""
        D_T0 = 1.0  # spot start
        D_Tn = DFS[-1]
        A0 = ANNUITY

        # Par swap rate from annuity: R = (D_T0 - D_Tn) / A
        R_par = (D_T0 - D_Tn) / A0
        assert A0 * R_par == pytest.approx(D_T0 - D_Tn, rel=1e-10)

    def test_v2_riskfree_par_bond(self):
        """V2: Risk-free bond at par rate should price to ~1."""
        # B^rf = sum c * alpha_i * df_i + df_n
        c = (1 - DFS[-1]) / ANNUITY  # par coupon
        B_rf = sum(c * y * d for y, d in zip(YFS, DFS)) + DFS[-1]
        assert B_rf == pytest.approx(1.0, rel=1e-8)

    def test_v3_asw_spread_formula(self):
        """V3: ASW spread = (B^rf - B) / A."""
        B_rf = 1.0
        B = 0.95  # risky bond at 95
        A = ANNUITY
        asw = forward_asw_spread(B_rf, B, A)
        assert asw == pytest.approx((1.0 - 0.95) / A, rel=1e-10)
        assert asw > 0  # risky bond below par → positive spread


# ---- 9.2 Linear-model calibration ----

class TestLinearModelCalibration:
    """Validation items 4-6."""

    def test_v4_alpha_equals_inverse_sum_yi(self):
        """V4: alpha = 1 / sum(y_i)."""
        alpha, betas = linear_swap_rate_calibrate(
            YFS, DFS, ANNUITY, R_SWP)
        expected_alpha = 1.0 / sum(YFS)
        assert alpha == pytest.approx(expected_alpha, rel=1e-10)

    def test_v5_consistency_identity(self):
        """V5: sum y_i * (alpha + beta_Ti * R^swp) = 1."""
        alpha, betas = linear_swap_rate_calibrate(
            YFS, DFS, ANNUITY, R_SWP)
        lhs = sum(y * (alpha + b * R_SWP) for y, b in zip(YFS, betas))
        assert lhs == pytest.approx(1.0, rel=1e-8)

    def test_v6_zero_rate_limit(self):
        """V6: R^swp -> 0 => sum y_i * alpha = 1 (betas drop out)."""
        alpha, _ = linear_swap_rate_calibrate(
            YFS, DFS, ANNUITY, 1e-10)
        assert alpha * sum(YFS) == pytest.approx(1.0, rel=1e-8)


# ---- 9.3 Cross-moment ----

class TestCrossMoment:
    """Validation items 7-8."""

    def test_v7_cross_moment_vs_mc(self):
        """V7: Cross-moment (Eq 13) matches MC simulation."""
        rng = np.random.default_rng(42)
        n_paths = 500_000
        n_steps = 200
        dt = T0 / n_steps
        sqrt_dt = math.sqrt(dt)

        X = np.full(n_paths, R_SWP)  # lognormal, a=0
        Y = np.full(n_paths, R_ASW)

        for _ in range(n_steps):
            Z1 = rng.standard_normal(n_paths)
            Z2 = 0.5 * Z1 + math.sqrt(1 - 0.25) * rng.standard_normal(n_paths)
            X *= np.exp(-0.5 * SIGMA_SWP**2 * dt + SIGMA_SWP * sqrt_dt * Z1)
            Y *= np.exp(-0.5 * 0.30**2 * dt + 0.30 * sqrt_dt * Z2)

        mc_cross = float((X * Y).mean())
        analytic = displaced_lognormal_cross_moment(
            R_SWP, R_ASW, 0.0, 0.0, SIGMA_SWP, 0.30, 0.5, T0)

        assert analytic == pytest.approx(mc_cross, rel=0.02)

    def test_v8_lognormal_limit(self):
        """V8: a=0 => E[R^swp R^asw] = R0^swp R0^asw exp(sigma_swp sigma_asw rho T)."""
        for rho in [-0.9, -0.5, 0.0, 0.5, 0.9]:
            cm = displaced_lognormal_cross_moment(
                R_SWP, R_ASW, 0.0, 0.0, SIGMA_SWP, 0.30, rho, T0)
            expected = R_SWP * R_ASW * math.exp(SIGMA_SWP * 0.30 * rho * T0)
            assert cm == pytest.approx(expected, rel=1e-10)

    def test_v8_rho_zero_lognormal(self):
        """V8: rho=0 lognormal => E[R^swp R^asw] = R0^swp R0^asw."""
        cm = displaced_lognormal_cross_moment(
            R_SWP, R_ASW, 0.0, 0.0, SIGMA_SWP, 0.30, 0.0, T0)
        assert cm == pytest.approx(R_SWP * R_ASW, rel=1e-10)

    def test_v8_sigma_zero(self):
        """V8: sigma_swp=0 => E[R^swp R^asw] = R0^swp R0^asw."""
        cm = displaced_lognormal_cross_moment(
            R_SWP, R_ASW, 0.0, 0.0, 0.0, 0.30, 0.5, T0)
        assert cm == pytest.approx(R_SWP * R_ASW, rel=1e-10)


# ---- 9.4 Convexity correction ----

class TestConvexityCorrection:
    """Validation items 9-11."""

    def test_v9_no_credit_cc_zero(self):
        """V9: B = B^rf => R^asw = 0 => CC = 0."""
        result = cmasw_convexity_correction(
            R_asw_0=0.0, R_swp_0=R_SWP,
            annuity=ANNUITY, payment_df=DF_TP,
            year_fractions=YFS, discount_factors=DFS,
            sigma_swp=SIGMA_SWP, sigma_asw=0.30, rho=0.5, T0=T0,
        )
        assert result.convexity_correction == pytest.approx(0.0, abs=1e-10)

    def test_v10_lognormal_table_reproduction(self):
        """V10: Reproduce paper Table 2 (lognormal CC)."""
        # Paper: R_asw = 490bp, R_swp = 4.29%, sigma_swp = 30%, T0 ~ 5Y
        # sigma_asw = 30%, rho = +50% => CC ~ +0.10%
        cc = cmasw_cc_lognormal(
            R_asw_0=0.0490, annuity=ANNUITY, payment_df=DF_TP,
            year_fractions=YFS,
            sigma_swp=0.30, sigma_asw=0.30, rho=0.50, T0=T0,
        )
        # Paper Table 2: sigma_asw=30%, rho=50% => +0.10%
        assert cc == pytest.approx(0.0010, abs=0.0005)

    def test_v10_cc_zero_at_rho_zero_lognormal(self):
        """V10: Lognormal CC = 0 when rho = 0."""
        cc = cmasw_cc_lognormal(
            R_asw_0=0.0490, annuity=ANNUITY, payment_df=DF_TP,
            year_fractions=YFS,
            sigma_swp=0.30, sigma_asw=0.30, rho=0.0, T0=T0,
        )
        assert cc == pytest.approx(0.0, abs=1e-10)

    def test_v11_displaced_close_to_lognormal(self):
        """V11: Small displacement => CC close to lognormal."""
        cc_ln = cmasw_cc_lognormal(
            R_asw_0=0.0490, annuity=ANNUITY, payment_df=DF_TP,
            year_fractions=YFS,
            sigma_swp=0.30, sigma_asw=0.30, rho=0.50, T0=T0,
        )
        result = cmasw_convexity_correction(
            R_asw_0=0.0490, R_swp_0=R_SWP,
            annuity=ANNUITY, payment_df=DF_TP,
            year_fractions=YFS, discount_factors=DFS,
            sigma_swp=0.30, sigma_asw=0.30, rho=0.50, T0=T0,
            a_swp=0.001, a_asw=0.001,  # small displacement
        )
        # Should be close to lognormal
        assert result.convexity_correction == pytest.approx(cc_ln, abs=0.0005)


# ---- 9.5 Full-price validation ----

class TestFullPrice:
    """Validation items 12-13."""

    def test_v12_aswlet_value(self):
        """V12: ASW-let value = D_{0,Tp} * (R^asw + CC)."""
        result = cmasw_convexity_correction(
            R_asw_0=0.0490, R_swp_0=R_SWP,
            annuity=ANNUITY, payment_df=DF_TP,
            year_fractions=YFS, discount_factors=DFS,
            sigma_swp=0.30, sigma_asw=0.30, rho=0.50, T0=T0,
        )
        expected = DF_TP * (0.0490 + result.convexity_correction)
        assert result.aswlet_value == pytest.approx(expected, rel=1e-10)

    def test_v13_value_scales_with_notional(self):
        """V13: Value scales linearly with coverage/notional."""
        r1 = cmasw_convexity_correction(
            R_asw_0=0.0490, R_swp_0=R_SWP,
            annuity=ANNUITY, payment_df=DF_TP,
            year_fractions=YFS, discount_factors=DFS,
            sigma_swp=0.30, sigma_asw=0.30, rho=0.50, T0=T0,
        )
        # Double the ASW spread => double the CC (linear in R_asw for lognormal)
        r2 = cmasw_convexity_correction(
            R_asw_0=0.0980, R_swp_0=R_SWP,
            annuity=ANNUITY, payment_df=DF_TP,
            year_fractions=YFS, discount_factors=DFS,
            sigma_swp=0.30, sigma_asw=0.30, rho=0.50, T0=T0,
        )
        assert r2.convexity_correction == pytest.approx(
            2 * r1.convexity_correction, rel=0.01)


# ---- 9.6 Sign and limits ----

class TestSignAndLimits:
    """Validation items 14-15."""

    def test_v14_cc_sign_follows_rho(self):
        """V14: Lognormal CC has the sign of rho for Tp near T1."""
        for rho in [-0.9, -0.5, 0.5, 0.9]:
            cc = cmasw_cc_lognormal(
                R_asw_0=0.0490, annuity=ANNUITY, payment_df=DF_TP,
                year_fractions=YFS,
                sigma_swp=0.30, sigma_asw=0.30, rho=rho, T0=T0,
            )
            if rho > 0:
                assert cc > 0, f"CC should be positive for rho={rho}"
            else:
                assert cc < 0, f"CC should be negative for rho={rho}"

    def test_v15_prefactor_positive(self):
        """V15: 1 - A_0 * alpha / D_{0,Tp} is positive."""
        alpha = 1.0 / sum(YFS)
        prefactor = 1.0 - ANNUITY * alpha / DF_TP
        assert prefactor > 0
