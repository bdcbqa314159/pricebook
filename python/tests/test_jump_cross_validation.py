"""Tests for jump model cross-validation framework."""

import pytest
import math

from pricebook.models.jump_cross_validation import (
    cross_validate_model, cross_validate_all,
)


class TestCrossValidateModel:
    def test_merton_cos_vs_mc(self):
        r = cross_validate_model("merton", n_mc_paths=100_000)
        assert r.max_cos_mc_pct < 10.0  # within 10%
        assert r.mean_cos_mc_pct < 5.0   # within 5% on average

    def test_vg_cos_vs_mc(self):
        r = cross_validate_model("vg", n_mc_paths=100_000)
        assert r.max_cos_mc_pct < 10.0
        assert r.mean_cos_mc_pct < 5.0

    def test_nig_cos_vs_mc(self):
        r = cross_validate_model("nig", n_mc_paths=100_000)
        assert r.max_cos_mc_pct < 10.0

    def test_kou_cos_only(self):
        """Kou has no MC terminal — should still produce COS prices."""
        r = cross_validate_model("kou")
        assert len(r.strikes) == 7
        assert all(s.cos_price > 0 for s in r.strikes if s.moneyness <= 1.0)

    def test_bates_cos_only(self):
        """Bates has no simple MC terminal — COS prices should be reasonable."""
        r = cross_validate_model("bates")
        atm = [s for s in r.strikes if abs(s.moneyness - 1.0) < 0.01]
        assert len(atm) == 1
        assert 5 < atm[0].cos_price < 20  # reasonable ATM call

    def test_cgmy_cos_only(self):
        r = cross_validate_model("cgmy")
        assert len(r.strikes) == 7

    def test_custom_moneyness(self):
        r = cross_validate_model("merton", moneyness_range=[0.90, 1.00, 1.10],
                                  n_mc_paths=50_000)
        assert len(r.strikes) == 3

    def test_to_dict(self):
        r = cross_validate_model("merton", n_mc_paths=50_000)
        d = r.to_dict()
        assert d["model"] == "merton"
        assert "max_cos_mc_pct" in d


class TestCrossValidateAll:
    def test_all_models(self):
        results = cross_validate_all(models=["merton", "vg", "nig"],
                                      n_mc_paths=50_000)
        assert len(results) == 3
        # Results should be sorted by mean_cos_mc_pct
        pcts = [r.mean_cos_mc_pct for r in results]
        assert pcts == sorted(pcts)

    def test_all_positive_prices(self):
        results = cross_validate_all(models=["merton", "kou", "bates"],
                                      n_mc_paths=50_000)
        for r in results:
            atm = [s for s in r.strikes if abs(s.moneyness - 1.0) < 0.01]
            for s in atm:
                assert s.cos_price > 0
