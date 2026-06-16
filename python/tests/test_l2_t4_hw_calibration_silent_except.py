"""Regression for L2 T4 audit of `models.hw_calibration`:

Mirror of T2.11 (which removed the same blanket ``except Exception``
from ``g2pp_swaption_price``) — ``_hw_swaption_price`` and
``_hw_implied_vol`` both wrapped their entire bodies in
``except Exception: return 0.0``, masking every error mode (bracketing
failure, brentq divergence, overflow in tree formulas, calibration
bugs) as a silent zero.  Callers couldn't distinguish "swaption ≈ 0"
from "pricer crashed".

Fix (T4-HW1): let real exceptions propagate.  For ``_hw_implied_vol``
the only legitimate recoverable case is ``ValueError`` from
``implied_vol_black76`` at arbitrage-violating prices — catch that
narrowly, not ``Exception``.

This test pins that a *correct* call returns a sensible non-zero value
on a vanilla setup, and a *broken* call (HullWhite raising) is no
longer silently masked.
"""

from __future__ import annotations

from datetime import date
from unittest import mock

import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.models.hw_calibration import _hw_swaption_price


REF = date(2026, 1, 15)


class TestSwaptionReturnsRealPrice:
    """Vanilla HW swaption price should be positive — not silenced to 0."""

    def test_atm_swaption_positive(self):
        curve = DiscountCurve.flat(REF, 0.04)
        price = _hw_swaption_price(
            a=0.05, sigma=0.01, curve=curve,
            expiry_years=2.0, tenor_years=5.0, strike=0.04,
            is_payer=True, n_steps=40,
        )
        assert price > 0.0


class TestExceptionsPropagate:
    """If the underlying ``HullWhite.tree_european_swaption`` raises (e.g.
    construction failure, OOM, or a numerical assertion), the wrapper
    must surface it rather than return 0.  Pre-fix this would silently
    return 0 and feed the calibration optimiser garbage."""

    def test_hullwhite_failure_propagates(self):
        curve = DiscountCurve.flat(REF, 0.04)
        with mock.patch(
            "pricebook.models.hw_calibration.HullWhite",
            side_effect=RuntimeError("synthetic failure"),
        ):
            with pytest.raises(RuntimeError, match="synthetic failure"):
                _hw_swaption_price(
                    a=0.05, sigma=0.01, curve=curve,
                    expiry_years=2.0, tenor_years=5.0, strike=0.04,
                )
