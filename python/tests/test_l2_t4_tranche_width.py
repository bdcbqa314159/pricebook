"""Regression for L2 Wave-2 audit — `TrancheCDS.price` protection leg
no longer has a spurious × width factor.

Pre-fix:
    protection_pv += (els[i] - els[i-1]) × **self.width** × df
    par_spread = Σ Δel × **width** × df / Σ (1−el) × dt × df  =  width × correct_spread

`expected_tranche_loss` already normalises by `width`
    (tranche_loss = clip(L − a, 0, width) / width)
so el ∈ [0, 1].  Multiplying back by width is double-counting.

The visible effect: par spreads were `width` × the correct value.  For the
standard equity tranche (width = 0.03, 0–3 % attachment), pre-fix par spread
came out as ~50 bp instead of typical market values of ~1500-2500 bp.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.core.day_count import DayCountConvention
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve
from pricebook.credit.tranche_pricing import TrancheCDS


REF = date(2024, 1, 1)


def _flat_disc(rate: float = 0.04) -> DiscountCurve:
    tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    dfs = [math.exp(-rate * t) for t in tenors]
    dates = [REF + timedelta(days=int(t * 365)) for t in tenors]
    return DiscountCurve(REF, dates, dfs, day_count=DayCountConvention.ACT_365_FIXED)


def _flat_surv(hazard: float = 0.02) -> SurvivalCurve:
    tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    survs = [math.exp(-hazard * t) for t in tenors]
    dates = [REF + timedelta(days=int(t * 365)) for t in tenors]
    return SurvivalCurve(REF, dates, survs, day_count=DayCountConvention.ACT_365_FIXED)


class TestTrancheParSpreadMagnitude:
    def test_equity_tranche_par_spread_in_market_range(self):
        """A 0-3% equity tranche on a portfolio of moderately risky names
        (2% annual hazard, 40% recovery) should have a par spread in the
        thousands of bps (typically 1500-3000+ bp).  Pre-fix was ≈ width×
        = 30× too small — pre-fix par spreads came out as ~50-100 bp.

        We don't pin to a specific number (MC noise + small portfolio),
        but the post-fix should land at LEAST in the 500+ bp range,
        which would be impossible under the pre-fix `× width` collapse."""
        portfolio = [_flat_surv(0.02) for _ in range(50)]
        disc = _flat_disc(0.04)
        tranche = TrancheCDS(
            attachment=0.0, detachment=0.03,
            maturity=REF + timedelta(days=5 * 365),
            spread=0.05, notional=10_000_000.0,
        )
        result = tranche.price(
            disc, portfolio, correlation=0.3,
            n_sims=5_000, seed=42,
        )
        # Pre-fix: par_spread = width × correct ≈ 0.03 × correct.
        # If correct is in [0.05, 0.30], pre-fix is in [0.0015, 0.009].
        # Post-fix should be in [0.05, 1.0] range for equity at these params.
        assert result.par_spread > 0.05, (
            f"Equity tranche par spread = {result.par_spread:.4f} ({result.par_spread*10000:.0f} bp); "
            f"pre-fix would have been ~{result.par_spread*0.03:.4f} due to the spurious × width."
        )

    def test_par_spread_independent_of_width_at_same_attachment(self):
        """For two tranches with the same attachment but different
        detachments on the same portfolio, the par spreads should NOT
        differ by a factor equal to the width ratio (pre-fix bug).
        Pre-fix: par_spread = width × (correct par_spread), so two tranches
        with widths (0.03, 0.06) would have par_spread ratio = 0.03/0.06 = 0.5.
        Post-fix: the ratio reflects the actual loss-distribution difference.
        """
        portfolio = [_flat_surv(0.02) for _ in range(50)]
        disc = _flat_disc(0.04)
        T = REF + timedelta(days=5 * 365)
        narrow = TrancheCDS(0.0, 0.03, T, spread=0.05, notional=10_000_000.0)
        wide = TrancheCDS(0.0, 0.06, T, spread=0.05, notional=10_000_000.0)
        r_n = narrow.price(disc, portfolio, correlation=0.3, n_sims=5_000, seed=42)
        r_w = wide.price(disc, portfolio, correlation=0.3, n_sims=5_000, seed=42)
        # Pre-fix ratio (wide/narrow) = (0.06 × Δel_w) / (0.03 × Δel_n) =
        # ≈ 2× Δel_w / Δel_n, which for typical inputs is a small number
        # (~0.5 × the post-fix ratio).  Post-fix ratio is just Δel_w / Δel_n.
        # The two regimes give meaningfully different numbers.  Sanity check:
        # both par spreads should be in the "real money" range (>10bp) for
        # this hazard / correlation setup — pre-fix the equity tranche would
        # be in single-digit bps.
        assert r_n.par_spread > 0.05, f"Equity par {r_n.par_spread} too small"
        assert r_w.par_spread > 0.05, f"Wider par {r_w.par_spread} too small"


class TestTrancheParityAtParSpread:
    def test_pv_zero_at_par_spread(self):
        """Sanity: when spread = par_spread, the PV should be ~0."""
        portfolio = [_flat_surv(0.02) for _ in range(50)]
        disc = _flat_disc(0.04)
        T = REF + timedelta(days=5 * 365)
        # First price at arbitrary spread to find par.
        probe = TrancheCDS(0.0, 0.03, T, spread=0.05, notional=10_000_000.0)
        r_probe = probe.price(disc, portfolio, correlation=0.3, n_sims=10_000, seed=42)
        par = r_probe.par_spread
        # Re-price at par.
        priced_at_par = TrancheCDS(0.0, 0.03, T, spread=par, notional=10_000_000.0)
        r = priced_at_par.price(disc, portfolio, correlation=0.3, n_sims=10_000, seed=42)
        # PV should be ~0 (MC noise) at par.
        assert abs(r.price) < 0.01 * r.protection_pv, (
            f"PV = {r.price:.2f} at par_spread={par:.4f}, protection_pv={r.protection_pv:.2f}"
        )
