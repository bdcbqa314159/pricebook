"""Tests for heterogeneous portfolios in basket CDS."""

import pytest
import numpy as np

from pricebook.credit.basket_cds import bespoke_tranche


class TestHeterogeneousPortfolio:
    def test_uniform_notionals_matches_default(self):
        """Equal notionals should match default (no notionals)."""
        pds = [0.03] * 10
        flat = bespoke_tranche(pds, 0.03, 0.07, lgd=0.6, seed=42)
        with_notionals = bespoke_tranche(pds, 0.03, 0.07, lgd=0.6, seed=42,
                                          notionals=[1.0] * 10)
        assert with_notionals.expected_loss == pytest.approx(flat.expected_loss, rel=0.01)

    def test_concentrated_portfolio(self):
        """One name with 50% notional should dominate."""
        pds = [0.05] * 10
        notionals = [5.0] + [1.0] * 9  # first name = 5x weight
        result = bespoke_tranche(pds, 0.0, 0.05, seed=42, notionals=notionals)
        assert result.expected_loss > 0

    def test_per_name_lgds(self):
        """Per-name LGDs should work."""
        pds = [0.03] * 10
        lgds_list = [0.8] * 5 + [0.3] * 5  # mix of high and low LGD
        result = bespoke_tranche(pds, 0.03, 0.07, seed=42, lgds=lgds_list)
        assert result.expected_loss > 0

    def test_per_name_lgd_vs_uniform(self):
        """Uniform per-name LGD should match flat lgd."""
        pds = [0.03] * 10
        flat = bespoke_tranche(pds, 0.03, 0.07, lgd=0.6, seed=42)
        per_name = bespoke_tranche(pds, 0.03, 0.07, seed=42, lgds=[0.6] * 10)
        assert per_name.expected_loss == pytest.approx(flat.expected_loss, rel=0.01)

    def test_high_lgd_increases_loss(self):
        """Higher LGD → higher expected loss."""
        pds = [0.03] * 10
        low = bespoke_tranche(pds, 0.0, 0.05, lgd=0.3, seed=42)
        high = bespoke_tranche(pds, 0.0, 0.05, lgd=0.8, seed=42)
        assert high.expected_loss > low.expected_loss

    def test_notionals_plus_lgds(self):
        """Both notionals and per-name LGDs together."""
        pds = [0.03] * 5
        notionals = [2.0, 1.0, 1.0, 1.0, 0.5]
        lgds_list = [0.8, 0.6, 0.6, 0.4, 0.3]
        result = bespoke_tranche(pds, 0.0, 0.05, seed=42,
                                  notionals=notionals, lgds=lgds_list)
        assert result.expected_loss > 0
        assert result.tranche_spread > 0
