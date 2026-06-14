"""Regression for L2 T4 audit of desk risk-metrics bump normalisation.

Same bug pattern as v1.028 (cb_risk_metrics cs01/dv01) found in
desk functions that expose a ``bump`` parameter and compute centred
differences without dividing by the bump size:

- `desks.inflation_desk.inflation_risk_metrics` — ie01, real_dv01, nominal_dv01
- `desks.risk_participation_desk.rp_risk_metrics` — cs01, dv01
- `desks.structured_credit_desk.sc_risk_metrics` — dv01

Pre-fix the centred difference `(pv_up - pv_dn) / 2` returned the PV
change for whatever ``bump`` the caller supplied — only "per 1bp" when
the default 0.0001 was used.  Now normalised by ``0.0001 / bump`` so
the outputs are always "PV per 1bp".
"""

from __future__ import annotations

from datetime import date

import pytest


class TestRiskParticipationBumpNormalisation:
    def test_cs01_invariant_under_bump_scaling(self):
        """RP cs01 from 1bp and 5bp bumps must agree within numerical noise."""
        from pricebook.credit.risk_participation import RiskParticipation
        from pricebook.desks.risk_participation_desk import rp_risk_metrics
        from pricebook.core.discount_curve import DiscountCurve
        from pricebook.core.survival_curve import SurvivalCurve
        from pricebook.core.day_count import DayCountConvention
        import math

        ref = date(2026, 1, 15)
        pillars = [date(ref.year + y, ref.month, ref.day) for y in (1, 3, 5, 10)]
        disc = DiscountCurve(ref, pillars, [math.exp(-0.05 * y) for y in (1, 3, 5, 10)],
                              DayCountConvention.ACT_365_FIXED)
        surv = SurvivalCurve(ref, pillars, [math.exp(-0.02 * y) for y in (1, 3, 5, 10)],
                              DayCountConvention.ACT_365_FIXED)
        rp = RiskParticipation(
            start=ref, end=pillars[-1],
            loan_notional=1_000_000,
            participation_rate=0.50,
            spread=0.005,
            recovery=0.40,
        )
        rm_default = rp_risk_metrics(rp, disc, surv, bump=0.0001)
        rm_5bp = rp_risk_metrics(rp, disc, surv, bump=0.0005)
        # Pre-fix: rm_5bp.cs01 would be ~5× rm_default.cs01.
        # Post-fix: both per-1bp, agree within MC/linearisation noise.
        assert rm_default.cs01 == pytest.approx(rm_5bp.cs01, rel=0.10, abs=1.0)
        assert rm_default.dv01 == pytest.approx(rm_5bp.dv01, rel=0.10, abs=1.0)


class TestInflationBumpNormalisation:
    def test_ie01_invariant_under_bump_scaling(self):
        """Inflation ie01/real_dv01/nominal_dv01 must be per-1bp regardless of bump."""
        from pricebook.fixed_income.inflation import ZCInflationSwap, CPICurve
        from pricebook.desks.inflation_desk import inflation_risk_metrics
        from pricebook.core.discount_curve import DiscountCurve
        from pricebook.core.day_count import DayCountConvention
        import math

        ref = date(2026, 1, 15)
        pillars = [date(ref.year + y, ref.month, ref.day) for y in (1, 3, 5, 10)]
        disc = DiscountCurve(ref, pillars, [math.exp(-0.05 * y) for y in (1, 3, 5, 10)],
                              DayCountConvention.ACT_365_FIXED)
        cpi = CPICurve(ref, 100.0, pillars,
                       [100.0 * (1.02 ** y) for y in (1, 3, 5, 10)])
        zcis = ZCInflationSwap(
            notional=1_000_000.0,
            start=ref, end=pillars[-2],  # 5y
            fixed_rate=0.02,
        )
        rm_default = inflation_risk_metrics(zcis, disc, cpi, bump=0.0001)
        rm_5bp = inflation_risk_metrics(zcis, disc, cpi, bump=0.0005)
        # All three "per-1bp" metrics must agree.
        assert rm_default.ie01 == pytest.approx(rm_5bp.ie01, rel=0.10, abs=1.0)
        assert rm_default.real_dv01 == pytest.approx(rm_5bp.real_dv01, rel=0.10, abs=1.0)
        assert rm_default.nominal_dv01 == pytest.approx(rm_5bp.nominal_dv01, rel=0.10, abs=1.0)
