"""Tests for pricebook.options.american_analytical."""

import pytest

from pricebook.options.american_analytical import (
    ju_zhong,
    kim_integral,
    medvedev_scaillet,
    american_comparison,
)


S, K, r, vol, T, q = 100.0, 100.0, 0.05, 0.20, 1.0, 0.03


class TestJuZhong:
    def test_price_positive_call(self):
        res = ju_zhong(S, K, r, vol, T, q, "call")
        assert res.price > 0

    def test_price_positive_put(self):
        res = ju_zhong(S, K, r, vol, T, q, "put")
        assert res.price > 0

    def test_american_call_ge_european(self):
        res = ju_zhong(S, K, r, vol, T, q, "call")
        assert res.price >= res.european_price - 1e-8

    def test_american_put_ge_european(self):
        res = ju_zhong(S, K, r, vol, T, q, "put")
        assert res.price >= res.european_price - 1e-8

    def test_put_with_positive_rate_has_eep(self):
        """Put with r > q: American value should exceed European (early exercise)."""
        res = ju_zhong(S, K, rate=0.08, vol=vol, T=T, q=0.0, option_type="put")
        assert res.price >= res.european_price - 1e-6

    def test_method_name(self):
        res = ju_zhong(S, K, r, vol, T, q)
        assert res.method == "ju_zhong"


class TestKimIntegral:
    def test_price_positive(self):
        res = kim_integral(S, K, r, vol, T, q, "put", n_steps=20)
        assert res.price > 0

    def test_boundary_length(self):
        n = 20
        res = kim_integral(S, K, r, vol, T, q, "put", n_steps=n)
        assert len(res.exercise_boundary) == n + 1

    def test_close_to_ju_zhong_put(self):
        n = 30
        jz = ju_zhong(S, K, r, vol, T, q, "put")
        ki = kim_integral(S, K, r, vol, T, q, "put", n_steps=n)
        # Within 10% of each other
        ref = jz.price
        assert abs(ki.price - ref) / ref < 0.10

    def test_price_ge_european(self):
        res = kim_integral(S, K, r, vol, T, q, "put", n_steps=20)
        assert res.price >= res.european_price - 1e-6

    def test_method_name(self):
        res = kim_integral(S, K, r, vol, T, q, n_steps=10)
        assert res.method == "kim_integral"


class TestMedvedevScaillet:
    def test_price_positive(self):
        res = medvedev_scaillet(S, K, r, vol, T, q, "put")
        assert res.price > 0

    def test_price_ge_european(self):
        res = medvedev_scaillet(S, K, r, vol, T, q, "put")
        assert res.price >= res.european_price - 1e-6

    def test_method_name(self):
        res = medvedev_scaillet(S, K, r, vol, T, q)
        assert res.method == "medvedev_scaillet"


class TestAmericanComparison:
    def test_returns_three_methods(self):
        result = american_comparison(S, K, r, vol, T, q, "put")
        for key in ("ju_zhong", "kim_integral", "medvedev_scaillet"):
            assert key in result

    def test_all_prices_positive(self):
        result = american_comparison(S, K, r, vol, T, q, "put")
        for key in ("ju_zhong", "kim_integral", "medvedev_scaillet"):
            assert result[key]["price"] > 0

    def test_european_price_present(self):
        result = american_comparison(S, K, r, vol, T, q, "put")
        assert "european_price" in result
        assert result["european_price"] > 0
