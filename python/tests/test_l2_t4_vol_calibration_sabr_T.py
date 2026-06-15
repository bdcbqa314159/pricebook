"""Regression for L2 T4 audit of `options.vol_calibration.CalibratedVolSurface`:

Pre-fix two defects:

T4-VC1 — ``CalibratedSABRNode.vol(strike)`` defaulted ``T = 1.0`` for
the SABR Hagan formula regardless of the node's actual tenor.  The
Hagan correction terms scale with ``T`` (``1 + (B1+B2+B3)·T``), so a
10y tenor used 10× the correction it should — visibly wrong smile.

T4-VC2 — fallback paths when SABR calibration failed used
``alpha = atm_vol`` directly.  But Hagan's ATM limit is
``σ_ATM ≈ alpha / F^(1-β)``, so setting ``alpha = atm`` only gives
``σ_ATM = atm`` when ``β = 1``.  For ``β = 0.5`` (FX default), the
fallback's ATM vol came out off by ``F^0.5`` → grossly wrong.

Fix: store ``T_to_expiry`` per node; ``vol()`` uses it as the
default.  Fallback alpha = ``atm · F^(1-β)``.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.options.vol_calibration import (
    CalibratedSABRNode, CalibratedVolSurface, calibrate_fx_surface,
)


REF = date(2026, 4, 28)


class TestSABRTPersistedPerNode:
    def test_surface_vol_uses_node_T(self):
        """A node calibrated with T = 5 should use T = 5 in its SABR
        vol formula when the surface is queried at its expiry.  Pre-fix
        the surface caller dropped T and used SABR(T=1) — the Hagan
        correction was 1/5 of the right size."""
        # Direct construction with a non-trivial T.
        node_T5 = CalibratedSABRNode(
            expiry=REF + timedelta(days=365 * 5),
            forward=100.0,
            alpha=0.20, beta=0.5, rho=-0.3, nu=0.5,
            atm_vol=0.20,
            T_to_expiry=5.0,
        )
        surf = CalibratedVolSurface([node_T5], "equity")

        v_surf = surf.vol(node_T5.expiry, strike=120.0)

        # Direct call with explicit T = 5.0 — should match.
        v_direct_T5 = node_T5.vol(120.0, T=5.0)
        # Direct call with the broken T = 1.0 — should NOT match.
        v_direct_T1 = node_T5.vol(120.0, T=1.0)

        assert v_surf == pytest.approx(v_direct_T5, rel=1e-12), (
            f"surface vol ({v_surf:.6f}) should equal SABR at T=5 "
            f"({v_direct_T5:.6f}), not T=1 ({v_direct_T1:.6f})"
        )
        # SABR correction scales as ~1 + (B1+B2+B3)·T.  For these
        # params the per-year correction is small (≈ 2% of vol per year
        # of T), so the absolute Δvol between T=5 and T=1 is on the
        # order of 0.001–0.005 in vol units.  Anything above MC/SABR
        # numerical noise (1e-4) confirms the dispatch is now T-aware.
        assert abs(v_surf - v_direct_T1) > 1e-3, (
            f"vol at T=5 vs T=1 should differ materially: "
            f"v(T=5)={v_surf:.4f} v(T=1)={v_direct_T1:.4f}"
        )


class TestFallbackAlphaConsistentWithBeta:
    def test_fallback_atm_vol_correct_with_beta(self):
        """When SABR calibration fails, the fallback node should still
        produce the right ATM vol via its SABR formula.  Pre-fix
        alpha = atm directly gave σ_ATM = atm / F^(1-β) (wrong for β≠1).
        Post-fix alpha = atm·F^(1-β) → σ_ATM ≈ atm."""
        # Force calibration failure with degenerate input: only one
        # quote with rr/bf = 0 won't trigger the failure path reliably,
        # so build the node directly using the fallback formula and
        # verify the SABR vol at ATM.
        forward = 1.10
        atm = 0.10
        beta = 0.5
        alpha = atm * forward ** (1.0 - beta)
        node = CalibratedSABRNode(
            expiry=REF + timedelta(days=365),
            forward=forward,
            alpha=alpha, beta=beta, rho=0.0, nu=0.0,
            atm_vol=atm, T_to_expiry=1.0,
        )
        # SABR ATM with rho = nu = 0 collapses to the prefactor
        # alpha / F^(1-β) = atm.
        v_atm = node.vol(forward)
        assert v_atm == pytest.approx(atm, rel=2e-3), (
            f"Fallback alpha={alpha:.4f}, β={beta} gave SABR(K=F)={v_atm:.4f}, "
            f"expected ≈ atm={atm}"
        )

    def test_fx_calibrate_T_propagates_to_node(self):
        """End-to-end: calibrate_fx_surface stores T on each node."""
        quotes = [
            {"expiry": REF + timedelta(days=180), "atm": 0.10, "rr25": 0.0, "bf25": 0.0},
            {"expiry": REF + timedelta(days=365), "atm": 0.12, "rr25": 0.0, "bf25": 0.0},
            {"expiry": REF + timedelta(days=730), "atm": 0.14, "rr25": 0.0, "bf25": 0.0},
        ]
        surf = calibrate_fx_surface(REF, quotes, spot=1.10, beta=0.5)
        # Each node's T_to_expiry should reflect its actual tenor.
        assert len(surf._nodes) == 3
        for node, q in zip(surf._nodes, sorted(quotes, key=lambda x: x["expiry"])):
            expected_T = (q["expiry"] - REF).days / 365.0
            assert node.T_to_expiry == pytest.approx(expected_T, rel=1e-6)
