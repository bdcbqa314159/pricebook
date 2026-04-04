"""Tests for unified pricer registry: COS, Greek engines, Heston PDE."""

import pytest

from pricebook.registry import (
    get_pricer,
    list_pricers,
    get_greek_engine,
    list_greek_engines,
    get_pde_pricer,
)
from pricebook.equity_option import equity_option_price


class TestCOSRegistry:
    def test_cos_bs_exists(self):
        pricer = get_pricer("cos_bs")
        assert callable(pricer)

    def test_cos_bs_prices(self):
        price = get_pricer("cos_bs")(
            spot=100, strike=100, rate=0.05, vol=0.20, T=1.0,
        )
        bs = equity_option_price(100, 100, 0.05, 0.20, 1.0)
        assert price == pytest.approx(bs, rel=0.01)

    def test_cos_heston_exists(self):
        pricer = get_pricer("cos_heston")
        assert callable(pricer)

    def test_cos_heston_prices(self):
        price = get_pricer("cos_heston")(
            spot=100, strike=100, rate=0.05, T=1.0,
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
        )
        assert price > 0

    def test_list_pricers(self):
        pricers = list_pricers()
        assert "cos_bs" in pricers
        assert "cos_heston" in pricers

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Available"):
            get_pricer("nonexistent")


class TestGreekEngineRegistry:
    def test_aad_engine(self):
        engine = get_greek_engine("aad")
        assert "black_scholes" in engine
        assert "swap" in engine
        assert "cds" in engine

    def test_bump_engine(self):
        engine = get_greek_engine("bump")
        assert "dv01" in engine
        assert "key_rate" in engine

    def test_list(self):
        engines = list_greek_engines()
        assert "aad" in engines
        assert "bump" in engines

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Available"):
            get_greek_engine("jax")

    def test_aad_bs_callable(self):
        engine = get_greek_engine("aad")
        assert callable(engine["black_scholes"])


class TestHestonPDERegistry:
    def test_heston_registered(self):
        pricer = get_pde_pricer("heston")
        assert callable(pricer)

    def test_heston_prices(self):
        pricer = get_pde_pricer("heston")
        price = pricer(
            spot=100, strike=100, rate=0.05, T=1.0,
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
        )
        assert price > 0
