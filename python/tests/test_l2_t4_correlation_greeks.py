"""Regression for L2 phase-2 audit of `risk.correlation_greeks`:

(a) ``correlation_pnl_attribution`` made 8 ``price_fn`` calls when only
    4 were needed.  ``correlation_delta`` (3 calls) and
    ``correlation_gamma`` (3 calls) both re-compute price at rho_old,
    rho_old+bump, rho_old-bump — plus 2 direct calls at the end.
    Now computed inline with 4 calls — halves cost for expensive
    pricers (basket MC, multi-asset PDE).

(b) ``CorrelationLadder`` exposed only ``total_rho_delta`` /
    ``total_rho_gamma`` which sum ``abs(...)`` (gross magnitudes).
    For portfolio risk reporting, the signed totals matter too —
    they represent the actual portfolio sensitivity if correlations
    drift together.  Added ``net_rho_delta`` / ``net_rho_gamma``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pricebook.risk.correlation_greeks import (
    correlation_pnl_attribution,
    correlation_sensitivity_ladder,
)


class TestCorrelationPnlAttributionCallCount:
    def test_only_four_unique_pricer_calls(self):
        """4 unique evaluations: rho_old, rho_old+bump, rho_old-bump, rho_new."""
        calls: list[float] = []

        def counting_pricer(rho):
            calls.append(rho)
            return 1.0 + 0.5 * rho - 0.3 * rho**2  # smooth quadratic

        correlation_pnl_attribution(counting_pricer, rho_old=0.3, rho_new=0.5)
        # Pre-fix: 8 calls.  Post-fix: 4.
        assert len(calls) == 4

    def test_taylor_attribution_matches_quadratic(self):
        """For a quadratic price_fn, gamma_pnl should fully explain total."""
        # V(ρ) = a + bρ + cρ²
        a, b, c = 1.0, 0.5, -0.3
        rho_old, rho_new = 0.2, 0.4
        # Analytical: dV/drho = b + 2c·ρ. d²V/drho² = 2c. So:
        #   delta_old = b + 2c·rho_old = 0.5 - 0.12 = 0.38
        #   gamma = 2c = -0.6
        #   drho = 0.2
        #   delta_pnl = 0.38 · 0.2 = 0.076
        #   gamma_pnl = 0.5 · -0.6 · 0.04 = -0.012
        #   total = (a + b·0.4 + c·0.16) - (a + b·0.2 + c·0.04) = 0.5·0.2 - 0.3·0.12 = 0.064
        result = correlation_pnl_attribution(
            lambda rho: a + b*rho + c*rho**2,
            rho_old=rho_old, rho_new=rho_new,
        )
        assert result.delta_pnl == pytest.approx(0.076, abs=1e-9)
        assert result.gamma_pnl == pytest.approx(-0.012, abs=1e-9)
        assert result.total_pnl == pytest.approx(0.064, abs=1e-9)
        # For quadratic, Taylor explains exactly → unexplained = 0.
        assert result.unexplained == pytest.approx(0.0, abs=1e-9)


class TestCorrelationLadderNet:
    def test_net_and_gross_sums_correctly(self):
        """Net is signed sum, gross is sum of abs."""
        n = 3
        names = ["A", "B", "C"]
        corr = 0.3 * np.ones((n, n))
        np.fill_diagonal(corr, 1.0)

        # Construct a price function that depends on each pair's correlation
        # with different signs — so signed sum != absolute sum.
        signs = {(0, 1): +1.0, (0, 2): -1.0, (1, 2): +1.0}

        def price_fn(c):
            total = 0.0
            for (i, j), sign in signs.items():
                total += sign * c[i, j]  # linear in each rho
            return total

        ladder = correlation_sensitivity_ladder(names, corr, price_fn)
        # Each pair has delta = +/-1 (linear). Net = +1 + (-1) + 1 = +1. Gross = 3.
        assert ladder.net_rho_delta == pytest.approx(1.0, abs=1e-6)
        assert ladder.total_rho_delta == pytest.approx(3.0, abs=1e-6)

    def test_zero_gamma_for_linear_pricer(self):
        """A linear-in-rho pricer has zero second derivative."""
        n = 2
        names = ["X", "Y"]
        corr = 0.5 * np.ones((n, n))
        np.fill_diagonal(corr, 1.0)

        ladder = correlation_sensitivity_ladder(
            names, corr,
            lambda c: 2.5 * c[0, 1],
        )
        assert abs(ladder.net_rho_gamma) < 1e-6
        assert abs(ladder.total_rho_gamma) < 1e-6
