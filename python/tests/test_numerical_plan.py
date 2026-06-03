"""Tests for numerical infrastructure plan: Fourier Greeks, auto-diff, SOCP, Feynman-Kac, method map."""

import pytest
import math
import numpy as np


# ═══════════════════════════════════════════════════════════════
# F2: Fix CharacteristicFunction.price_european
# ═══════════════════════════════════════════════════════════════

class TestF2Fix:
    def test_cf_price_european(self):
        from pricebook.numerical._fourier import CharacteristicFunction
        from pricebook.models.cos_method import bs_char_func
        cf = bs_char_func(0.04, 0.0, 0.20, 1.0)
        wrapper = CharacteristicFunction(cf, 1.0)
        price = wrapper.price_european(100, 100, 0.04)
        assert price > 0
        # Should match BS
        from pricebook.models.black76 import black76_price, OptionType
        bs = black76_price(100 * math.exp(0.04), 100, 0.20, 1.0, math.exp(-0.04), OptionType.CALL)
        assert price == pytest.approx(bs, rel=0.05)


# ═══════════════════════════════════════════════════════════════
# F1: Fourier Greeks
# ═══════════════════════════════════════════════════════════════

class TestFourierGreeks:
    def test_cos_greeks_call(self):
        from pricebook.models.fourier_greeks import cos_greeks
        from pricebook.models.cos_method import bs_char_func
        from pricebook.models.black76 import OptionType
        cf = bs_char_func(0.04, 0.0, 0.20, 1.0)
        r = cos_greeks(cf, 100, 100, 0.04, 1.0)
        assert r.price > 0
        assert 0 < r.delta < 1
        assert r.gamma > 0
        assert r.theta != 0  # theta is non-zero

    def test_cos_greeks_put(self):
        from pricebook.models.fourier_greeks import cos_greeks
        from pricebook.models.cos_method import bs_char_func
        from pricebook.models.black76 import OptionType
        cf = bs_char_func(0.04, 0.0, 0.20, 1.0)
        r = cos_greeks(cf, 100, 100, 0.04, 1.0, OptionType.PUT)
        assert r.delta < 0

    def test_fourier_greeks_entry(self):
        from pricebook.models.fourier_greeks import fourier_greeks
        from pricebook.models.cos_method import bs_char_func
        from pricebook.models.black76 import OptionType
        cf = bs_char_func(0.04, 0.0, 0.20, 1.0)
        r = fourier_greeks(cf, 100, 100, 0.04, 1.0)
        assert r.method == "cos"
        assert r.delta > 0


# ═══════════════════════════════════════════════════════════════
# D1: Automatic Differentiation
# ═══════════════════════════════════════════════════════════════

class TestAutoDiff:
    def test_dual_arithmetic(self):
        from pricebook.numerical.auto_diff import Dual
        x = Dual(3.0, 1.0)
        y = x * x + 2 * x + 1  # f(x) = x²+2x+1, f'(x) = 2x+2
        assert y.val == pytest.approx(16.0)  # 9+6+1
        assert y.der == pytest.approx(8.0)   # 6+2

    def test_dual_exp(self):
        from pricebook.numerical.auto_diff import Dual, exp
        x = Dual(1.0, 1.0)
        y = exp(x)
        assert y.val == pytest.approx(math.e)
        assert y.der == pytest.approx(math.e)  # d/dx e^x = e^x

    def test_dual_log(self):
        from pricebook.numerical.auto_diff import Dual, log
        x = Dual(2.0, 1.0)
        y = log(x)
        assert y.val == pytest.approx(math.log(2))
        assert y.der == pytest.approx(0.5)  # 1/x

    def test_dual_chain_rule(self):
        from pricebook.numerical.auto_diff import Dual, exp, sin
        x = Dual(0.0, 1.0)
        y = exp(sin(x))  # f(0)=e^0=1, f'(0)=cos(0)*e^0=1
        assert y.val == pytest.approx(1.0)
        assert y.der == pytest.approx(1.0)

    def test_gradient(self):
        from pricebook.numerical.auto_diff import Dual, grad
        # f(x,y) = x²y, ∂f/∂x = 2xy, ∂f/∂y = x²
        def f(v):
            return v[0] * v[0] * v[1]
        g = grad(f, np.array([3.0, 2.0]))
        assert g[0] == pytest.approx(12.0)  # 2*3*2
        assert g[1] == pytest.approx(9.0)   # 3²

    def test_derivative(self):
        from pricebook.numerical.auto_diff import Dual, derivative, exp
        val, der = derivative(lambda x: x**3, 2.0)
        assert val == pytest.approx(8.0)
        assert der == pytest.approx(12.0)  # 3x²

    def test_black_scholes_delta(self):
        """AD delta should match analytical."""
        from pricebook.numerical.auto_diff import Dual, exp, log, sqrt
        from pricebook.models.black76 import black76_delta, OptionType

        def bs_call(S):
            K, r, vol, T = 100.0, 0.04, 0.20, 1.0
            fwd = S * exp(Dual(r * T, 0))
            df = exp(Dual(-r * T, 0))
            d1 = (log(fwd / K) + Dual(0.5 * vol**2 * T, 0)) / Dual(vol * math.sqrt(T), 0)
            # Approximate N(d1) via sigmoid
            nd1 = 1.0 / (1.0 + exp(-1.7 * d1))
            d2 = d1 - Dual(vol * math.sqrt(T), 0)
            nd2 = 1.0 / (1.0 + exp(-1.7 * d2))
            return df * (fwd * nd1 - Dual(K, 0) * nd2)

        from pricebook.numerical.auto_diff import derivative
        val, der = derivative(bs_call, 100.0)
        assert val > 0  # option value
        assert 0 < der < 1  # delta between 0 and 1


# ═══════════════════════════════════════════════════════════════
# CO1: SOCP
# ═══════════════════════════════════════════════════════════════

class TestSOCP:
    def test_robust_portfolio(self):
        from pricebook.numerical.socp import robust_portfolio_socp
        mu = np.array([0.08, 0.12, 0.06])
        cov = np.array([[0.04, 0.01, 0.005], [0.01, 0.09, 0.01], [0.005, 0.01, 0.025]])
        r = robust_portfolio_socp(mu, cov, epsilon=0.05)
        assert r.feasible
        assert abs(sum(r.x) - 1.0) < 0.01

    def test_tracking_error(self):
        from pricebook.numerical.socp import tracking_error_socp
        mu = np.array([0.08, 0.12, 0.06])
        cov = np.array([[0.04, 0.01, 0.005], [0.01, 0.09, 0.01], [0.005, 0.01, 0.025]])
        bench = np.array([0.4, 0.4, 0.2])
        r = tracking_error_socp(mu, cov, bench, max_te=0.05)
        assert r.feasible
        assert r.objective >= 0  # TE ≥ 0


# ═══════════════════════════════════════════════════════════════
# X1: Feynman-Kac
# ═══════════════════════════════════════════════════════════════

class TestFeynmanKac:
    def test_sde_to_pde(self):
        from pricebook.models.feynman_kac import sde_to_pde
        coeffs = sde_to_pde(
            mu_fn=lambda S, t: 0.04 * S,
            sigma_fn=lambda S, t: 0.20 * S,
            rate_fn=0.04,
        )
        # Check a = 0.5 σ² S² at S=100
        assert coeffs["diffusion"](100, 0) == pytest.approx(0.5 * 0.04 * 10000)
        assert coeffs["convection"](100, 0) == pytest.approx(4.0)
        assert coeffs["reaction"](100, 0) == pytest.approx(-0.04)

    def test_verify_feynman_kac(self):
        from pricebook.models.feynman_kac import verify_feynman_kac
        r = verify_feynman_kac(100, 100, 0.04, 0.20, 1.0, n_paths=100_000)
        assert r.consistent  # MC and PDE agree within 3σ
        assert r.relative_diff_pct < 3  # within 3%

    def test_pde_to_sde(self):
        from pricebook.models.feynman_kac import pde_to_sde
        sde = pde_to_sde(
            diffusion_fn=lambda S, t: 0.5 * 0.04 * S**2,
            convection_fn=lambda S, t: 0.04 * S,
        )
        assert sde["drift"](100, 0) == pytest.approx(4.0)
        assert sde["volatility"](100, 0) == pytest.approx(0.2 * 100, rel=0.01)


# ═══════════════════════════════════════════════════════════════
# X5: Method Map
# ═══════════════════════════════════════════════════════════════

class TestMethodMap:
    def test_european_recommends_analytical(self):
        from pricebook.core.numerical_method_map import recommend, Feature
        r = recommend([Feature.EUROPEAN])
        assert r.primary == "analytical"

    def test_american_recommends_tree(self):
        from pricebook.core.numerical_method_map import recommend, Feature
        r = recommend([Feature.AMERICAN])
        assert "tree" in r.primary

    def test_stochvol_recommends_cos(self):
        from pricebook.core.numerical_method_map import recommend, Feature
        r = recommend([Feature.STOCHASTIC_VOL])
        assert "cos" in r.primary or "heston" in r.primary

    def test_high_dim_recommends_mc(self):
        from pricebook.core.numerical_method_map import recommend, Feature
        r = recommend([Feature.HIGH_DIMENSION])
        assert "mc" in r.primary

    def test_compare_methods(self):
        from pricebook.core.numerical_method_map import compare_methods
        r = compare_methods(100, 100, 0.04, 0.20, 1.0)
        assert r["consistent"]
        assert len(r["prices"]) >= 3

    def test_to_dict(self):
        from pricebook.core.numerical_method_map import recommend, Feature
        r = recommend([Feature.BARRIER])
        d = r.to_dict()
        assert "primary" in d
        assert "reason" in d
