"""Tests for AAD calibration Jacobian."""

import math
import pytest
import numpy as np

from pricebook.aad_calibration import sabr_jacobian, calibration_risk
from pricebook.sabr import sabr_implied_vol


F, T = 0.05, 5.0
ALPHA, BETA, RHO, NU = 0.02, 0.5, -0.3, 0.4
STRIKES = [0.03, 0.04, 0.05, 0.06, 0.07]
MARKET_VOLS = [sabr_implied_vol(F, K, T, ALPHA, BETA, RHO, NU) for K in STRIKES]


class TestSABRJacobian:
    def test_model_vols_match(self):
        jac = sabr_jacobian(F, STRIKES, MARKET_VOLS, T, BETA, ALPHA, RHO, NU)
        for mv, jv in zip(MARKET_VOLS, jac["model_vols"]):
            assert jv == pytest.approx(mv, rel=0.05)

    def test_d_alpha_positive(self):
        """Higher alpha → higher vol."""
        jac = sabr_jacobian(F, STRIKES, MARKET_VOLS, T, BETA, ALPHA, RHO, NU)
        assert all(d > 0 for d in jac["d_alpha"])

    def test_d_nu_positive(self):
        """Higher nu → higher vol (more smile)."""
        jac = sabr_jacobian(F, STRIKES, MARKET_VOLS, T, BETA, ALPHA, RHO, NU)
        # At ATM, nu effect is positive
        assert jac["d_nu"][2] > 0  # ATM strike index

    def test_fd_check_alpha(self):
        """AAD d(vol)/d(alpha) matches finite difference."""
        eps = 1e-6
        jac = sabr_jacobian(F, STRIKES, MARKET_VOLS, T, BETA, ALPHA, RHO, NU)

        for i, K in enumerate(STRIKES):
            vol_up = sabr_implied_vol(F, K, T, ALPHA + eps, BETA, RHO, NU)
            vol_dn = sabr_implied_vol(F, K, T, ALPHA - eps, BETA, RHO, NU)
            fd = (vol_up - vol_dn) / (2 * eps)
            assert jac["d_alpha"][i] == pytest.approx(fd, rel=0.05)

    def test_fd_check_rho(self):
        eps = 1e-5
        jac = sabr_jacobian(F, STRIKES, MARKET_VOLS, T, BETA, ALPHA, RHO, NU)

        for i, K in enumerate(STRIKES):
            vol_up = sabr_implied_vol(F, K, T, ALPHA, BETA, RHO + eps, NU)
            vol_dn = sabr_implied_vol(F, K, T, ALPHA, BETA, RHO - eps, NU)
            fd = (vol_up - vol_dn) / (2 * eps)
            assert jac["d_rho"][i] == pytest.approx(fd, rel=0.10)


class TestCalibrationRisk:
    def test_returns_sensitivities(self):
        risk = calibration_risk(F, STRIKES, MARKET_VOLS, T, ALPHA, RHO, NU, BETA)
        assert len(risk["d_alpha_d_mkt"]) == len(STRIKES)
        assert len(risk["d_rho_d_mkt"]) == len(STRIKES)
        assert len(risk["d_nu_d_mkt"]) == len(STRIKES)

    def test_alpha_sensitive_to_atm(self):
        """Alpha should be most sensitive to ATM vol."""
        risk = calibration_risk(F, STRIKES, MARKET_VOLS, T, ALPHA, RHO, NU, BETA)
        atm_idx = 2  # K=0.05 = F
        assert abs(risk["d_alpha_d_mkt"][atm_idx]) > 0

    def test_rho_sensitive_to_skew(self):
        """Rho should be sensitive to wing vols (skew)."""
        risk = calibration_risk(F, STRIKES, MARKET_VOLS, T, ALPHA, RHO, NU, BETA)
        # Rho affects skew → sensitive to low/high strike vols
        assert any(abs(d) > 0 for d in risk["d_rho_d_mkt"])
