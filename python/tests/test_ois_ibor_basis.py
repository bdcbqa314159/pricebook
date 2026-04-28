"""Tests for OIS-IBOR basis: upgraded bootstrap, decomposition."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.discount_curve import DiscountCurve
from pricebook.ibor_curve import (
    bootstrap_ibor, EURIBOR_3M_CONVENTIONS,
)
from pricebook.ois_ibor_basis import OISIBORBasis
from pricebook.rfr import SpreadCurve, bootstrap_spread_curve, IBORProjection
from pricebook.schedule import Frequency

REF = date(2026, 4, 27)


def _ois():
    return DiscountCurve.flat(REF, 0.03)


def _ibor_swap_rates():
    return [
        (REF + timedelta(days=365), 0.032),
        (REF + timedelta(days=730), 0.033),
        (REF + timedelta(days=1825), 0.035),
        (REF + timedelta(days=3650), 0.037),
    ]


# ---- Upgraded bootstrap_spread_curve ----

class TestBootstrapSpreadCurve:

    def test_spread_positive(self):
        """IBOR > OIS → positive spread."""
        ois = _ois()
        sc = bootstrap_spread_curve(REF, _ibor_swap_rates(), ois)
        for d in sc.dates:
            assert sc.spread(d) > 0

    def test_spread_reprices_via_ibor_projection(self):
        """IBORProjection with bootstrapped spread should give IBOR forwards above OIS."""
        ois = _ois()
        rates = _ibor_swap_rates()
        sc = bootstrap_spread_curve(REF, rates, ois)
        proj = IBORProjection(ois, sc)

        # At each swap maturity, the IBOR projection forward should be above OIS
        for mat, par_rate in rates:
            d1 = REF + timedelta(days=30)
            ibor_fwd = proj.forward_rate(d1, mat)
            ois_fwd = ois.forward_rate(d1, mat)
            assert ibor_fwd > ois_fwd

    def test_spread_increases_with_rate(self):
        """Higher IBOR swap rate → larger spread."""
        ois = _ois()
        sc_low = bootstrap_spread_curve(REF, [(REF + timedelta(days=1825), 0.032)], ois)
        sc_high = bootstrap_spread_curve(REF, [(REF + timedelta(days=1825), 0.040)], ois)
        d = REF + timedelta(days=1825)
        assert sc_high.spread(d) > sc_low.spread(d)

    def test_backward_compat(self):
        """Old signature still works (uses new defaults)."""
        ois = _ois()
        sc = bootstrap_spread_curve(REF, _ibor_swap_rates(), ois)
        assert len(sc.dates) == 4


# ---- OISIBORBasis ----

class TestOISIBORBasis:

    def test_from_curves(self):
        """Extract basis from calibrated IBOR vs OIS curves."""
        ois = _ois()
        ibor = bootstrap_ibor(REF, EURIBOR_3M_CONVENTIONS, ois,
                              swaps=_ibor_swap_rates())
        pillars = [REF + timedelta(days=d) for d in [365, 730, 1825]]
        basis = OISIBORBasis.from_curves(ibor, ois, pillars)
        for d in pillars:
            assert basis.basis(d) > 0  # IBOR > OIS

    def test_from_swap_rates(self):
        """Bootstrap basis from swap rates."""
        ois = _ois()
        basis = OISIBORBasis.from_swap_rates(
            REF, _ibor_swap_rates(), ois,
            conventions=EURIBOR_3M_CONVENTIONS,
        )
        assert basis.basis(REF + timedelta(days=365)) > 0
        assert basis.tenor == "3M"

    def test_forward_basis(self):
        ois = _ois()
        ibor = bootstrap_ibor(REF, EURIBOR_3M_CONVENTIONS, ois,
                              swaps=_ibor_swap_rates())
        pillars = [REF + timedelta(days=365), REF + timedelta(days=3650)]
        basis = OISIBORBasis.from_curves(ibor, ois, pillars)
        fb = basis.forward_basis(REF + timedelta(days=365), REF + timedelta(days=730))
        assert math.isfinite(fb)

    def test_decompose(self):
        """Credit + liquidity = total."""
        ois = _ois()
        ibor = bootstrap_ibor(REF, EURIBOR_3M_CONVENTIONS, ois,
                              swaps=_ibor_swap_rates())
        pillars = [REF + timedelta(days=365), REF + timedelta(days=1825)]
        basis = OISIBORBasis.from_curves(ibor, ois, pillars)

        basis.decompose(cds_spreads={
            "BankA": 0.003, "BankB": 0.005, "BankC": 0.004,
        })

        assert basis.credit_component is not None
        assert basis.liquidity_component is not None

        for d in pillars:
            total = basis.basis(d)
            credit = basis.credit_component.spread(d)
            liq = basis.liquidity_component.spread(d)
            assert credit + liq == pytest.approx(total, abs=1e-10)
            assert credit >= 0
            assert liq >= 0

    def test_decompose_credit_capped(self):
        """Credit component should not exceed total basis."""
        ois = _ois()
        ibor = bootstrap_ibor(REF, EURIBOR_3M_CONVENTIONS, ois,
                              swaps=_ibor_swap_rates())
        pillars = [REF + timedelta(days=365)]
        basis = OISIBORBasis.from_curves(ibor, ois, pillars)

        # Very high CDS → credit capped at total basis
        basis.decompose(cds_spreads={"BigBank": 0.10})
        d = pillars[0]
        assert basis.credit_component.spread(d) <= basis.basis(d) + 1e-10

    def test_ibor_projection_with_basis(self):
        """IBORProjection still works with upgraded SpreadCurve."""
        ois = _ois()
        sc = bootstrap_spread_curve(REF, _ibor_swap_rates(), ois)
        proj = IBORProjection(ois, sc)
        d1 = REF + timedelta(days=365)
        d2 = REF + timedelta(days=456)
        fwd = proj.forward_rate(d1, d2)
        ois_fwd = ois.forward_rate(d1, d2)
        assert fwd > ois_fwd  # projection includes spread
