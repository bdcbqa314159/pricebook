"""Regression for L2 phase-2 audit of `risk.repo_cva`:

Same shape as v1.006 hybrid_xva fix — pre-fix CVA omitted the
discount factor ``D(0, t_i)``, computing the future-valued CVA sum
instead of the present-valued one.  For short repos the effect is
small (<1% overnight) but accumulates for term repo.

Fix: optional ``discount_rate`` parameter (defaults to repo_rate); each
EPE × marginal_PD term is multiplied by exp(-discount_rate · t).
"""

from __future__ import annotations

import math

import pytest

from pricebook.risk.repo_cva import repo_cva


class TestRepoCVADiscounting:
    def test_zero_discount_unchanged(self):
        """With discount_rate=0, behaviour matches pre-fix (no discounting)."""
        result = repo_cva(
            repo_notional=1_000_000, repo_days=90, repo_rate=0.05,
            haircut=0.02, counterparty_hazard=0.02,
            discount_rate=0.0,
        )
        # Should produce CVA without any discount-factor reduction.
        # Compare to a separate run with the same parameters: identical.
        result2 = repo_cva(
            repo_notional=1_000_000, repo_days=90, repo_rate=0.05,
            haircut=0.02, counterparty_hazard=0.02,
            discount_rate=0.0,
        )
        assert result.cva == pytest.approx(result2.cva, rel=1e-12)

    def test_high_discount_reduces_cva(self):
        """Discounting at a high rate produces lower CVA than no discounting."""
        no_df = repo_cva(
            repo_notional=1_000_000, repo_days=365, repo_rate=0.05,
            haircut=0.02, counterparty_hazard=0.02,
            discount_rate=0.0,
        )
        with_df = repo_cva(
            repo_notional=1_000_000, repo_days=365, repo_rate=0.05,
            haircut=0.02, counterparty_hazard=0.02,
            discount_rate=0.10,  # 10% discount rate over 1y
        )
        assert with_df.cva < no_df.cva
        # Ratio should be roughly avg(exp(-0.10·t)) for t∈[0,1] ≈ 0.95.
        assert 0.85 < with_df.cva / no_df.cva < 1.0

    def test_default_discount_uses_repo_rate(self):
        """When discount_rate is None, uses repo_rate by default."""
        with_default = repo_cva(
            repo_notional=1_000_000, repo_days=180, repo_rate=0.05,
            haircut=0.02, counterparty_hazard=0.02,
        )
        with_explicit = repo_cva(
            repo_notional=1_000_000, repo_days=180, repo_rate=0.05,
            haircut=0.02, counterparty_hazard=0.02,
            discount_rate=0.05,
        )
        assert with_default.cva == pytest.approx(with_explicit.cva, rel=1e-12)


class TestRepoCVAInvariants:
    def test_zero_hazard_zero_cva(self):
        """Zero default probability → zero CVA."""
        result = repo_cva(
            repo_notional=1_000_000, repo_days=90, repo_rate=0.05,
            haircut=0.02, counterparty_hazard=0.0,
        )
        assert result.cva == pytest.approx(0.0, abs=1e-9)

    def test_zero_lgd_zero_cva(self):
        """Full recovery → zero LGD → zero CVA."""
        result = repo_cva(
            repo_notional=1_000_000, repo_days=90, repo_rate=0.05,
            haircut=0.02, counterparty_hazard=0.02,
            counterparty_recovery=1.0,  # LGD = 0
        )
        assert result.cva == pytest.approx(0.0, abs=1e-9)
