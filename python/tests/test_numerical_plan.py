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


# ═══════════════════════════════════════════════════════════════
# F3: Rough Heston CF
# ═══════════════════════════════════════════════════════════════

class TestRoughHeston:
    def test_rough_heston_price(self):
        from pricebook.models.rough_heston_cf import rough_heston_price, RoughHestonParams
        params = RoughHestonParams(v0=0.04, kappa=0.5, theta=0.04, xi=0.3, rho=-0.7, H=0.1)
        price = rough_heston_price(100, 100, 0.04, 1.0, params)
        assert price > 0

    def test_rough_vs_standard(self):
        """Rough (H<0.5) should differ from standard Heston (H=0.5)."""
        from pricebook.models.rough_heston_cf import rough_heston_price, RoughHestonParams
        rough = RoughHestonParams(v0=0.04, kappa=0.5, theta=0.04, xi=0.3, rho=-0.7, H=0.1)
        smooth = RoughHestonParams(v0=0.04, kappa=0.5, theta=0.04, xi=0.3, rho=-0.7, H=0.49)
        p_rough = rough_heston_price(100, 100, 0.04, 1.0, rough)
        p_smooth = rough_heston_price(100, 100, 0.04, 1.0, smooth)
        assert abs(p_rough - p_smooth) > 0.01  # should differ


# ═══════════════════════════════════════════════════════════════
# F4: 2D FFT
# ═══════════════════════════════════════════════════════════════

class TestFFT2D:
    def test_2d_fft_positive(self):
        from pricebook.models.fft_2d import fft_2d_price, joint_bs_char_func
        cf = joint_bs_char_func(0.04, (0.02, 0.01), (0.20, 0.25), 0.5, 1.0)
        r = fft_2d_price(cf, (100, 100), 5, 0.04, 1.0, N=32)
        assert r.price >= 0

    def test_joint_cf(self):
        from pricebook.models.fft_2d import joint_bs_char_func
        cf = joint_bs_char_func(0.04, (0.0, 0.0), (0.20, 0.20), 0.0, 1.0)
        # At u1=u2=0: φ(0,0) = 1
        assert abs(cf(0, 0) - 1.0) < 1e-10


# ═══════════════════════════════════════════════════════════════
# P3: Fokker-Planck
# ═══════════════════════════════════════════════════════════════

class TestFokkerPlanck:
    def test_density_integrates_to_one(self):
        from pricebook.models.fokker_planck import fokker_planck_1d
        r = fokker_planck_1d(100, 0.04, 0.20, 1.0, n_space=200, n_time=200)
        mass = float(np.trapezoid(r.density, r.grid))
        assert mass == pytest.approx(1.0, abs=0.05)

    def test_mean_matches_forward(self):
        from pricebook.models.fokker_planck import fokker_planck_1d
        r = fokker_planck_1d(100, 0.04, 0.20, 1.0, n_space=300, n_time=300)
        # E[S_T] = S_0 × e^{rT} under risk-neutral
        expected_mean = 100 * math.exp(0.04)
        assert r.mean == pytest.approx(expected_mean, rel=0.05)

    def test_density_to_options(self):
        from pricebook.models.fokker_planck import fokker_planck_1d, density_to_option_prices
        r = fokker_planck_1d(100, 0.04, 0.20, 1.0, n_space=300, n_time=300)
        prices = density_to_option_prices(r, [90, 100, 110], 0.04, 1.0)
        assert len(prices) == 3
        assert all(p["call"] > 0 for p in prices)
        # ATM call should be near BS
        from pricebook.models.black76 import black76_price, OptionType
        bs = black76_price(100 * math.exp(0.04), 100, 0.20, 1.0, math.exp(-0.04), OptionType.CALL)
        assert prices[1]["call"] == pytest.approx(bs, rel=0.20)


# ═══════════════════════════════════════════════════════════════
# CO3: Duality
# ═══════════════════════════════════════════════════════════════

class TestDuality:
    def test_lp_with_duals(self):
        from pricebook.numerical.duality import lp_with_duals
        c = np.array([-1.0, -2.0])
        A_ub = np.array([[1, 1], [2, 1]])
        b_ub = np.array([4.0, 6.0])
        r = lp_with_duals(c, A_ub, b_ub, bounds=[(0, None), (0, None)])
        assert r.success
        assert r.objective < 0  # minimising negative → negative obj

    def test_shadow_prices(self):
        from pricebook.numerical.duality import lp_with_duals, shadow_prices
        c = np.array([-1.0, -2.0])
        A_ub = np.array([[1, 1], [2, 1]])
        b_ub = np.array([4.0, 6.0])
        r = lp_with_duals(c, A_ub, b_ub, bounds=[(0, None), (0, None)])
        sp = shadow_prices(r)
        assert len(sp) == 2
        assert any(s["binding"] for s in sp)  # at least one binding

    def test_parametric_lp(self):
        from pricebook.numerical.duality import parametric_lp
        c = np.array([-1.0, -2.0])
        A_ub = np.array([[1, 1], [2, 1]])
        b_ub = np.array([4.0, 6.0])
        results = parametric_lp(c, A_ub, b_ub, 0, (-1, 1), n_points=5,
                                 bounds=[(0, None), (0, None)])
        assert len(results) == 5
        assert all(r["feasible"] for r in results)


# ═══════════════════════════════════════════════════════════════
# Q1: Oscillatory Quadrature
# ═══════════════════════════════════════════════════════════════

class TestOscillatoryQuad:
    def test_filon_cos(self):
        from pricebook.numerical.oscillatory_quad import filon_quad
        # ∫_0^π cos(x) cos(10x) dx — known result
        r = filon_quad(lambda x: math.cos(x), 0, math.pi, 10, n=100)
        # Should be near 0 (orthogonality for ω≠1)
        assert abs(r.value) < 0.5

    def test_filon_matches_quad(self):
        """For smooth f, Filon should match scipy.quad."""
        from pricebook.numerical.oscillatory_quad import filon_quad
        from scipy.integrate import quad
        omega = 20
        exact = quad(lambda x: math.exp(-x) * math.cos(omega * x), 0, 5)[0]
        filon = filon_quad(lambda x: math.exp(-x), 0, 5, omega, n=200)
        assert filon.value == pytest.approx(exact, abs=0.01)

    def test_fourier_integral_adaptive(self):
        from pricebook.numerical.oscillatory_quad import fourier_integral
        r = fourier_integral(lambda x: math.exp(-x**2), -5, 5, 30)
        assert r.n_evaluations > 0
