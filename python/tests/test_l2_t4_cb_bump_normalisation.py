"""Regression for L2 T4 audit of `desks.convertible_bond_desk.cb_risk_metrics`:

Pre-fix the ``cs01`` and ``dv01`` outputs returned the raw PV change for
whatever ``bump_spread``/``bump_rate`` the caller supplied — only
producing a "per 1bp" value when the caller used the default 0.0001
bump.  ``delta``/``gamma``/``vega`` were correctly normalised by their
bump sizes, so the asymmetry was a clear correctness defect.

Fix: scale the differences by ``0.0001 / bump`` so the outputs are
always "per 1bp" regardless of the bump tuning.
"""

from __future__ import annotations

from datetime import date

import pytest


def _make_cb():
    from pricebook.options.convertible_bond import ConvertibleBond
    return ConvertibleBond(
        notional=100.0, coupon_rate=0.03,
        maturity_years=5.0, conversion_ratio=1.0,
    )


class TestCS01BumpNormalisation:
    def test_cs01_invariant_under_bump_scaling(self):
        """For a smooth pricing function, cs01 should be insensitive to
        the bump size — varying bump_spread by 10× must produce the same
        cs01 to within numerical noise."""
        from pricebook.desks.convertible_bond_desk import cb_risk_metrics

        cb = _make_cb()
        spot, rate, vol, spread = 100.0, 0.04, 0.25, 0.02
        rm_default = cb_risk_metrics(
            cb, spot, rate, vol, spread,
            bump_spread=0.0001, n_paths=5_000, seed=42,
        )
        rm_5bp = cb_risk_metrics(
            cb, spot, rate, vol, spread,
            bump_spread=0.0005, n_paths=5_000, seed=42,
        )
        # Pre-fix: rm_5bp.credit_cs01 would be ~5× rm_default.credit_cs01 (raw PV change).
        # Post-fix: both should be ~equal (per-1bp normalised).
        # Allow generous tolerance for MC noise + linearisation error
        # over 5bp vs 1bp.
        assert rm_default.credit_cs01 == pytest.approx(rm_5bp.credit_cs01, rel=0.20, abs=0.5)


class TestDV01BumpNormalisation:
    def test_dv01_invariant_under_bump_scaling(self):
        """Same property for rate bump."""
        from pricebook.desks.convertible_bond_desk import cb_risk_metrics

        cb = _make_cb()
        spot, rate, vol, spread = 100.0, 0.04, 0.25, 0.02
        rm_default = cb_risk_metrics(
            cb, spot, rate, vol, spread,
            bump_rate=0.0001, n_paths=5_000, seed=42,
        )
        rm_5bp = cb_risk_metrics(
            cb, spot, rate, vol, spread,
            bump_rate=0.0005, n_paths=5_000, seed=42,
        )
        assert rm_default.rate_dv01 == pytest.approx(rm_5bp.rate_dv01, rel=0.20, abs=0.5)
