"""Regression for L2 T4 audit of `regulatory.capital_allocation.euler_allocation`:

Pre-fix the correlation-matrix branch mixed ``w = s/total_standalone``
with ``cov = outer(s, s) * corr`` and then formed ``RC_i = w_i × (cov @ w)_i``,
which collapses to ``s_i² × Σ_j s_j² ρ_ij / total²``. Under uncorrelated
desks this gives ``s_i⁴`` fractions instead of the standard ``s_i²``
(variance-proportional) Euler split — a strong over-concentration on
the largest desk.

Fix: implement Tasche Euler std-dev decomposition correctly,
    σ_p = √(s' corr s),   RC_i = s_i (corr s)_i / σ_p,
which sums to σ_p and gives s_i² fractions when corr is identity.

References:
    Tasche (2008). Capital Allocation to Business Units and Sub-Portfolios.
    McNeil, Frey & Embrechts (2015) Ch. 8.
"""

from __future__ import annotations

import numpy as np
import pytest

from pricebook.regulatory.capital_allocation import (
    DeskCapitalInput, euler_allocation,
)


class TestEulerUncorrelated:
    def test_variance_proportional_for_identity_corr(self):
        """For uncorrelated desks, Euler allocates capital ∝ s_i²."""
        desks = [
            DeskCapitalInput("A", standalone_capital=10.0, pnl_annual=0),
            DeskCapitalInput("B", standalone_capital=20.0, pnl_annual=0),
            DeskCapitalInput("C", standalone_capital=10.0, pnl_annual=0),
        ]
        corr = np.eye(3)
        alloc = euler_allocation(desks, correlation_matrix=corr)
        total = sum(alloc)
        fracs = [a / total for a in alloc]
        # Expected: s_i² / Σ s_j² = [100, 400, 100] / 600.
        expected = [100 / 600, 400 / 600, 100 / 600]
        for f, e in zip(fracs, expected):
            assert f == pytest.approx(e, rel=1e-6)

    def test_sums_to_sigma_p_when_capital_none(self):
        """portfolio_capital=None → allocated sums to σ_p = √(s' corr s)."""
        s = np.array([10.0, 20.0, 10.0])
        desks = [DeskCapitalInput(f"D{i}", standalone_capital=float(s[i]),
                                   pnl_annual=0) for i in range(3)]
        corr = np.eye(3)
        sigma_p = float(np.sqrt(s @ corr @ s))
        alloc = euler_allocation(desks, correlation_matrix=corr)
        assert sum(alloc) == pytest.approx(sigma_p, rel=1e-9)


class TestEulerWithCorrelation:
    def test_perfectly_correlated_sums_to_sum_s(self):
        """ρ=1 ⇒ σ_p = Σ s_i and RC_i = s_i."""
        s = np.array([30.0, 50.0, 20.0])
        desks = [DeskCapitalInput(f"D{i}", standalone_capital=float(s[i]),
                                   pnl_annual=0) for i in range(3)]
        corr = np.ones((3, 3))
        alloc = euler_allocation(desks, correlation_matrix=corr)
        # σ_p = sqrt((Σs)²) = Σ s.
        assert sum(alloc) == pytest.approx(100.0, rel=1e-9)
        # RC_i = s_i × (corr s)_i / σ_p = s_i × Σs / Σs = s_i.
        for a, s_i in zip(alloc, s):
            assert a == pytest.approx(s_i, rel=1e-9)

    def test_negative_correlation_reduces_capital(self):
        """Negatively correlated desks → portfolio capital < sum of standalones."""
        s = np.array([50.0, 50.0])
        desks = [DeskCapitalInput("A", 50.0, 0), DeskCapitalInput("B", 50.0, 0)]
        corr = np.array([[1.0, -0.5], [-0.5, 1.0]])
        alloc = euler_allocation(desks, correlation_matrix=corr)
        sigma_p = float(np.sqrt(s @ corr @ s))
        assert sum(alloc) == pytest.approx(sigma_p, rel=1e-9)
        # Symmetric correlation with symmetric s ⇒ equal allocation.
        assert alloc[0] == pytest.approx(alloc[1], rel=1e-9)


class TestEulerScalesWithPortfolioCapital:
    def test_explicit_portfolio_capital_overrides_sigma_p(self):
        """When portfolio_capital is given, allocated sums to that value
        (Euler ratios preserved)."""
        s = np.array([10.0, 30.0])
        desks = [DeskCapitalInput("A", 10.0, 0), DeskCapitalInput("B", 30.0, 0)]
        corr = np.eye(2)
        alloc = euler_allocation(desks, portfolio_capital=1000.0,
                                  correlation_matrix=corr)
        assert sum(alloc) == pytest.approx(1000.0, rel=1e-9)
        # Ratios match s_i² / Σ s_j² = [100, 900] / 1000.
        assert alloc[0] / sum(alloc) == pytest.approx(100 / 1000, rel=1e-9)
        assert alloc[1] / sum(alloc) == pytest.approx(900 / 1000, rel=1e-9)
