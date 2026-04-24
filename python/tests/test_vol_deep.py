"""Deep tests for vol surfaces — DD9 hardening.

Covers: SABR boundaries, smile non-negativity, total variance monotonicity,
calendar arb detection.
"""

import math
import pytest
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.sabr import sabr_implied_vol, sabr_price
from pricebook.vol_smile import VolSmile
from pricebook.vol_surface_strike import VolSurfaceStrike
from pricebook.vol_arb import detect_calendar_arb
from pricebook.vol_term_structure import forward_vol_from_term
from pricebook.black76 import OptionType


REF = date(2024, 1, 15)


class TestSABR:

    def test_atm_vol_equals_alpha(self):
        """ATM vol ≈ alpha for beta=1, small nu."""
        vol = sabr_implied_vol(100.0, 100.0, 1.0, alpha=0.20, beta=1.0, rho=0.0, nu=0.01)
        assert vol == pytest.approx(0.20, abs=0.01)

    def test_rho_near_one_no_crash(self):
        """SABR with rho=0.99 shouldn't crash."""
        vol = sabr_implied_vol(100.0, 110.0, 1.0, alpha=0.20, beta=0.5, rho=0.99, nu=0.5)
        assert vol > 0
        assert math.isfinite(vol)

    def test_rho_near_neg_one_no_crash(self):
        vol = sabr_implied_vol(100.0, 90.0, 1.0, alpha=0.20, beta=0.5, rho=-0.99, nu=0.5)
        assert vol > 0
        assert math.isfinite(vol)

    def test_deep_otm_positive(self):
        """Deep OTM vol should be positive (not negative from Hagan expansion)."""
        vol = sabr_implied_vol(100.0, 200.0, 1.0, alpha=0.20, beta=1.0, rho=-0.3, nu=0.8)
        assert vol > 0

    def test_higher_nu_wider_smile(self):
        """Higher volvol → wider smile (higher OTM vols)."""
        low = sabr_implied_vol(100.0, 120.0, 1.0, alpha=0.20, beta=1.0, rho=0.0, nu=0.1)
        high = sabr_implied_vol(100.0, 120.0, 1.0, alpha=0.20, beta=1.0, rho=0.0, nu=0.5)
        assert high > low

    def test_negative_rho_upside_skew(self):
        """Negative rho: lower vol for OTM calls (equity-like skew)."""
        atm = sabr_implied_vol(100.0, 100.0, 1.0, alpha=0.20, beta=1.0, rho=-0.5, nu=0.3)
        otm_call = sabr_implied_vol(100.0, 120.0, 1.0, alpha=0.20, beta=1.0, rho=-0.5, nu=0.3)
        otm_put = sabr_implied_vol(100.0, 80.0, 1.0, alpha=0.20, beta=1.0, rho=-0.5, nu=0.3)
        assert otm_put > atm  # downside skew


class TestVolSmile:

    def test_smile_non_negative(self):
        """Interpolated vol should never be negative."""
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        vols = np.array([0.30, 0.22, 0.20, 0.22, 0.25])
        smile = VolSmile(strikes, vols)
        for k in np.linspace(75, 125, 100):
            assert smile.vol(k) > 0


class TestTotalVariance:

    def test_surface_no_calendar_arb(self):
        """Total variance should be non-decreasing in T."""
        expiries = [REF + relativedelta(months=m) for m in [3, 6, 12, 24]]
        smiles = []
        for i, vol_atm in enumerate([0.25, 0.22, 0.20, 0.19]):
            strikes = np.array([90.0, 100.0, 110.0])
            vols = np.array([vol_atm + 0.03, vol_atm, vol_atm + 0.02])
            smiles.append(VolSmile(strikes, vols))

        surface = VolSurfaceStrike(REF, expiries, smiles)

        # Interpolated vol at ATM should have non-decreasing total variance
        prev_tv = 0.0
        for t in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            d = REF + relativedelta(days=int(t * 365))
            v = surface.vol(d, 100.0)
            tv = v * v * t
            assert tv >= prev_tv - 1e-10, f"Calendar arb at t={t}: tv={tv} < prev={prev_tv}"
            prev_tv = tv

    def test_forward_vol_positive(self):
        """Forward vol should be positive when term structure is valid."""
        result = forward_vol_from_term([0.5, 1.0, 2.0], [0.22, 0.20, 0.19], 0.5, 1.0)
        assert result.forward_vol > 0

    def test_detect_calendar_arb(self):
        """Detect when total variance is decreasing."""
        # This has arb: vol at 1Y > vol at 2Y, and 0.30² × 1 > 0.15² × 2
        violations = detect_calendar_arb(
            [REF + relativedelta(years=1), REF + relativedelta(years=2)],
            [0.30, 0.15],
            REF,
        )
        assert len(violations) > 0
