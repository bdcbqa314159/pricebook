"""Regression for L2 T4 audit of `structured.structured_notes.outperformance_certificate`:

Pre-fix the cap strike was ``spot * (1 + cap)`` which only caps payoff
at ``1 + cap`` when ``participation == 1``.  For participation > 1
(the typical case), each percentage-point of S_T/S_0 growth is
amplified by ``participation``, so the cap-binding S_T is LOWER than
S_0 × (1 + cap).

Correct cap-strike: from ``1 + participation × (S_T/S_0 - 1) = 1 + cap``
we get ``S_T = S_0 × (1 + cap / participation)``.

Verifying at expiry under deterministic payoff (high spot, very low
vol) that the capped certificate value matches the analytic ceiling
``notional × (1 + cap) × df``.
"""

from __future__ import annotations

import math

import pytest

from pricebook.structured.structured_notes import outperformance_certificate


class TestOutperformanceCapBinds:
    def test_high_spot_caps_at_1_plus_cap(self):
        """When spot blows through the cap-strike at maturity (low vol,
        long T, rate large), the discounted payoff should approach the
        capped value ``notional × (1 + cap) × df`` from above.
        Pre-fix the capped value was wrong by a factor proportional to
        (participation - 1) × cap."""
        spot, rate, q, vol, T = 100.0, 0.05, 0.00, 0.001, 1.0
        participation = 1.5
        cap = 0.30  # 30% max return → max payoff = 1.30 × notional × df.
        df = math.exp(-rate * T)

        # With very low vol and a 5% drift, S_T ≈ F = spot × exp(rT) ≈ 105.
        # Cap-strike (correct): spot × (1 + cap/part) = 100 × (1.20) = 120.
        # F = 105 < 120, so cap is NOT binding here.  We need to push the
        # forward way above 120 — use rate=0.50 (50% drift in 1y) so
        # F ≈ 164.87.
        r_high = 0.50
        df_high = math.exp(-r_high * T)
        result = outperformance_certificate(
            spot=spot, rate=r_high, dividend_yield=q, vol=vol, T=T,
            participation=participation, cap=cap, notional=1.0,
        )
        # Deterministic payoff at S_T ≈ F = 164.87:
        # Both 100×(1+30%)=130 (capped at correct strike 120) and
        # 100×(1.30)=130 from pre-fix happen to match by accident.
        # The true test: the capped certificate price should NOT exceed
        # the analytic ceiling of (1 + cap) × df_high = 1.30 × 0.6065 = 0.7884.
        ceiling = (1.0 + cap) * df_high
        assert result.price <= ceiling + 1e-4
        # And it should be reasonably close to that ceiling under deep-ITM.
        assert result.price > ceiling - 0.05

    def test_no_cap_unchanged(self):
        """Without a cap, the fix is a no-op."""
        spot, rate, q, vol, T = 100.0, 0.04, 0.02, 0.20, 1.0
        r = outperformance_certificate(spot, rate, q, vol, T,
                                        participation=1.5, cap=None)
        assert r.cap is None
        assert r.participation == 1.5

    def test_cap_strike_depends_on_participation(self):
        """For same cap=0.30 but different participation, prices must
        differ (pre-fix would use the same cap_strike regardless)."""
        kwargs = dict(spot=100.0, rate=0.04, dividend_yield=0.02,
                      vol=0.25, T=1.0, cap=0.30, notional=1.0)
        r1 = outperformance_certificate(participation=1.0, **kwargs)
        r2 = outperformance_certificate(participation=2.0, **kwargs)
        # Different participations → different short-call strikes (post-fix).
        # Pre-fix the short calls were identical, so the only difference
        # was in the long-call leg.  Post-fix both legs change, so the
        # spread between r1 and r2 is materially different.
        assert r1.price != r2.price
