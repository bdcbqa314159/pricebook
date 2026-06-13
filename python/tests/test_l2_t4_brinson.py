"""Regression for L2 phase-2 audit of `risk.brinson_attribution`:

Pre-fix `brinson_multi_period` claimed "geometric linking" but used
the ad-hoc scaling ``cum_alloc += effect_t × cum_bench_before_t``
which does NOT preserve the geometric active-return identity:

    Σ F_t · effect_t == Π(1+r_p_t) − Π(1+r_b_t)

For two periods of port=1%/bench=0% (Brinson effect = 1%/period),
pre-fix accumulates 0.01 + 0.01·1 = 0.02 while the geometric active
is 1.0201 − 1 = 0.0201.  Off by 1bp — small for the example but
quadratic-in-T over long horizons.

Fix: Frongello (2002) recursive linking
    cum_t = cum_{t-1} · (1+r_b_t) + effect_t · cum_port_{t-1}

which is equivalent to the closed-form linking coefficient
    F_t = Π_{s<t}(1+r_p_s) · Π_{s>t}(1+r_b_s).
"""

from __future__ import annotations

import pytest

from pricebook.risk.brinson_attribution import (
    BrinsonResult, BrinsonSectorResult, brinson_attribution, brinson_multi_period,
)


def _one_sector(port_w, bench_w, port_r, bench_r):
    """Helper: build a 1-sector BrinsonResult from raw inputs."""
    return brinson_attribution(
        portfolio_weights=[port_w], benchmark_weights=[bench_w],
        portfolio_returns=[port_r], benchmark_returns=[bench_r],
    )


class TestFrongelloIdentity:
    def test_two_period_equal_returns(self):
        """port=1%, bench=0% for two periods → geometric active = 1.0201 − 1 = 0.0201."""
        # Single-sector portfolio with weight 1.0 in both portfolio and benchmark
        # — degenerate case but valid for the identity check.
        p1 = _one_sector(1.0, 1.0, 0.01, 0.0)
        p2 = _one_sector(1.0, 1.0, 0.01, 0.0)
        m = brinson_multi_period([p1, p2])
        expected_active = (1.01 * 1.01) - (1.0 * 1.0)
        assert m["cumulative_active_return"] == pytest.approx(expected_active, abs=1e-12)
        # The identity: cumulative_allocation + selection + interaction == active.
        total_effects = (m["cumulative_allocation"]
                         + m["cumulative_selection"]
                         + m["cumulative_interaction"])
        assert total_effects == pytest.approx(expected_active, abs=1e-12)

    def test_three_period_mixed_returns(self):
        """3 periods with different port/bench returns; identity must hold."""
        results = [
            _one_sector(1.0, 1.0, 0.05, 0.03),
            _one_sector(1.0, 1.0, -0.02, 0.01),
            _one_sector(1.0, 1.0, 0.04, 0.02),
        ]
        m = brinson_multi_period(results)
        total_effects = (m["cumulative_allocation"]
                         + m["cumulative_selection"]
                         + m["cumulative_interaction"])
        assert total_effects == pytest.approx(m["cumulative_active_return"], abs=1e-12)

    def test_ten_period_compounding(self):
        """Long horizon: pre-fix accumulates ~T·1% but geometric is larger by O(T²·1%²)."""
        results = [_one_sector(1.0, 1.0, 0.01, 0.0) for _ in range(10)]
        m = brinson_multi_period(results)
        # Geometric active = 1.01^10 − 1 ≈ 0.10462.
        # Pre-fix sum = 10·1% = 0.10.  Off by ~46bp.
        expected = (1.01 ** 10) - 1.0
        assert m["cumulative_active_return"] == pytest.approx(expected, abs=1e-12)
        total = (m["cumulative_allocation"]
                 + m["cumulative_selection"]
                 + m["cumulative_interaction"])
        assert total == pytest.approx(expected, abs=1e-10)


class TestBrinsonSinglePeriodIdentity:
    """Per-period: alloc + sel + inter = active (when weights sum to 1). Unchanged."""

    def test_two_sector_weights_sum_to_one(self):
        r = brinson_attribution(
            portfolio_weights=[0.4, 0.6],
            benchmark_weights=[0.5, 0.5],
            portfolio_returns=[0.10, 0.05],
            benchmark_returns=[0.08, 0.06],
        )
        total = r.total_allocation + r.total_selection + r.total_interaction
        active = r.portfolio_return - r.benchmark_return
        assert total == pytest.approx(active, abs=1e-12)

    def test_three_sector_weights_sum_to_one(self):
        r = brinson_attribution(
            portfolio_weights=[0.3, 0.4, 0.3],
            benchmark_weights=[0.4, 0.4, 0.2],
            portfolio_returns=[0.05, 0.10, 0.08],
            benchmark_returns=[0.04, 0.09, 0.06],
        )
        total = r.total_allocation + r.total_selection + r.total_interaction
        active = r.portfolio_return - r.benchmark_return
        assert total == pytest.approx(active, abs=1e-12)
