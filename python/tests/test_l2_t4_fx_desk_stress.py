"""Regression for L2 T4 audit of `desks.fx_desk` stress and dashboard.

Pre-fix issues:

1. ``fx_stress_suite`` "combined" scenario was labelled "Spot -5%,
   rates +100bp" but the PnL formula was identical to the bare
   "spot_dn_5" scenario — the rates +100bp shock was silently
   dropped.  The function signature (pair, notional, spot) has no
   rates data so the only correct fix is to remove the misleading
   label.  A proper rate+spot reprice is provided by
   ``fx_scenario_stress`` (PricingContext-based).

2. ``fx_dashboard.total_delta`` used ``sum(abs(net_notional))`` which
   reports gross-notional, not net delta — a long EUR/USD + short
   EUR/USD of equal size would show large total_delta instead of 0.
"""

from __future__ import annotations

import pytest


class TestStressNoLyingCombined:
    def test_combined_scenario_removed(self):
        """The misleadingly-labelled 'combined' scenario must be gone."""
        from pricebook.desks.fx_desk import fx_stress_suite

        positions = [("EURUSD", 1_000_000.0, 1.08)]
        results = fx_stress_suite(positions)
        names = [r.scenario for r in results]
        # 4 spot scenarios only — no false "combined".
        assert names == ["spot_dn_5", "spot_up_5", "spot_dn_10", "spot_up_10"]
        for r in results:
            # No scenario description should claim rates+spot.
            assert "rates" not in r.description.lower()
            assert "+100bp" not in r.description

    def test_spot_stress_linearity(self):
        """+5% should equal -(-5%)."""
        from pricebook.desks.fx_desk import fx_stress_suite

        positions = [("EURUSD", 1_000_000.0, 1.08)]
        results = {r.scenario: r.pnl for r in fx_stress_suite(positions)}
        assert results["spot_up_5"] == pytest.approx(-results["spot_dn_5"], rel=1e-9)
        # +10% should be 2× +5%.
        assert results["spot_up_10"] == pytest.approx(2 * results["spot_up_5"], rel=1e-9)
