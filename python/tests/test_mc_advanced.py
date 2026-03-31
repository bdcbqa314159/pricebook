"""Tests for advanced MC variance reduction."""

import pytest
import math

from pricebook.mc_advanced import mc_stratified, mc_importance, mc_mlmc
from pricebook.mc_pricer import mc_european
from pricebook.equity_option import equity_option_price
from pricebook.black76 import OptionType


SPOT, STRIKE, RATE, VOL, T = 100.0, 100.0, 0.05, 0.20, 1.0
BS = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)


class TestStratified:
    def test_matches_bs(self):
        r = mc_stratified(SPOT, STRIKE, RATE, VOL, T, n_paths=100_000)
        assert r.price == pytest.approx(BS, rel=0.01)

    def test_within_ci(self):
        r = mc_stratified(SPOT, STRIKE, RATE, VOL, T, n_paths=200_000)
        assert abs(r.price - BS) < 3 * r.std_error

    def test_lower_error_than_plain(self):
        plain = mc_european(SPOT, STRIKE, RATE, VOL, T, n_paths=50_000, seed=42)
        strat = mc_stratified(SPOT, STRIKE, RATE, VOL, T, n_paths=50_000, seed=42)
        assert strat.std_error < plain.std_error

    def test_put(self):
        bs_put = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT)
        r = mc_stratified(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT, n_paths=100_000)
        assert r.price == pytest.approx(bs_put, rel=0.02)


class TestImportanceSampling:
    def test_matches_bs(self):
        r = mc_importance(SPOT, STRIKE, RATE, VOL, T, n_paths=100_000)
        assert r.price == pytest.approx(BS, rel=0.02)

    def test_within_ci(self):
        r = mc_importance(SPOT, STRIKE, RATE, VOL, T, n_paths=200_000)
        assert abs(r.price - BS) < 3 * r.std_error

    def test_otm_variance_reduction(self):
        """Importance sampling shines for deep OTM options."""
        K_otm = 140.0
        bs_otm = equity_option_price(SPOT, K_otm, RATE, VOL, T, OptionType.CALL)
        plain = mc_european(SPOT, K_otm, RATE, VOL, T, OptionType.CALL, n_paths=50_000)
        imp = mc_importance(SPOT, K_otm, RATE, VOL, T, OptionType.CALL, n_paths=50_000)
        # Importance should be closer to analytical
        assert abs(imp.price - bs_otm) < abs(plain.price - bs_otm) + 0.5

    def test_put(self):
        bs_put = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT)
        r = mc_importance(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT, n_paths=100_000)
        assert r.price == pytest.approx(bs_put, rel=0.03)


class TestMLMC:
    def test_matches_bs(self):
        r = mc_mlmc(SPOT, STRIKE, RATE, VOL, T, n_levels=4, n_paths_base=10_000)
        assert r.price == pytest.approx(BS, rel=0.02)

    def test_within_ci(self):
        r = mc_mlmc(SPOT, STRIKE, RATE, VOL, T, n_levels=5, n_paths_base=20_000)
        assert abs(r.price - BS) < 3 * r.std_error

    def test_put(self):
        bs_put = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT)
        r = mc_mlmc(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT,
                    n_levels=4, n_paths_base=10_000)
        assert r.price == pytest.approx(bs_put, rel=0.02)

    def test_more_levels_stable(self):
        """Adding levels shouldn't change the price much."""
        r3 = mc_mlmc(SPOT, STRIKE, RATE, VOL, T, n_levels=3, n_paths_base=10_000)
        r5 = mc_mlmc(SPOT, STRIKE, RATE, VOL, T, n_levels=5, n_paths_base=10_000)
        assert r3.price == pytest.approx(r5.price, rel=0.05)


class TestAllTechniquesAgree:
    def test_call_prices_agree(self):
        """All techniques give the same price (within noise)."""
        plain = mc_european(SPOT, STRIKE, RATE, VOL, T, n_paths=200_000)
        strat = mc_stratified(SPOT, STRIKE, RATE, VOL, T, n_paths=200_000)
        imp = mc_importance(SPOT, STRIKE, RATE, VOL, T, n_paths=200_000)
        mlmc = mc_mlmc(SPOT, STRIKE, RATE, VOL, T, n_levels=4, n_paths_base=20_000)

        for r in [plain, strat, imp, mlmc]:
            assert r.price == pytest.approx(BS, rel=0.02)
