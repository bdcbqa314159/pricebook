"""Monte Carlo engine tests — validate against analytical prices."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from pricebook.models.mc_engine import MCEngine, TimeGrid, MCResult, ProcessSpec
from pricebook.models.mc_processes import (
    GBMProcess, BlackScholesProcess, HestonProcess,
    OUProcess, CIRProcess, CorrelatedGBMProcess,
)
from pricebook.models.mc_payoffs import (
    european_call, european_put, digital_call,
    asian_arithmetic, asian_geometric, lookback_call,
    barrier_knockout, american_put,
    basket_call, worst_of_put,
)


def black_scholes_call(s0, k, r, sigma, T):
    """Analytical Black-Scholes call price."""
    d1 = (math.log(s0 / k) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return s0 * norm.cdf(d1) - k * math.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(s0, k, r, sigma, T):
    return black_scholes_call(s0, k, r, sigma, T) - s0 + k * math.exp(-r * T)


# ── TimeGrid ──

class TestTimeGrid:

    def test_uniform(self):
        grid = TimeGrid.uniform(1.0, 100)
        assert grid.n_steps == 100
        assert grid.T == pytest.approx(1.0)
        assert len(grid.dt) == 100

    def test_from_array(self):
        grid = TimeGrid(np.array([0, 0.25, 0.5, 1.0]))
        assert grid.n_steps == 3
        assert grid.dt[0] == pytest.approx(0.25)


# ── GBM European vs Black-Scholes ──

class TestGBMEuropean:

    def test_call_matches_bs(self):
        s0, k, r, sigma, T = 100, 100, 0.05, 0.20, 1.0
        bs_price = black_scholes_call(s0, k, r, sigma, T)
        df = math.exp(-r * T)

        engine = MCEngine(
            process=BlackScholesProcess(s0, r, sigma),
            time_grid=TimeGrid.uniform(T, 1),  # 1 step = exact
            n_paths=200_000, seed=42, antithetic=True,
        )
        result = engine.price(european_call(k), discount_factor=df)
        assert result.price == pytest.approx(bs_price, rel=0.02)

    def test_put_matches_bs(self):
        s0, k, r, sigma, T = 100, 100, 0.05, 0.20, 1.0
        bs_price = black_scholes_put(s0, k, r, sigma, T)
        df = math.exp(-r * T)

        engine = MCEngine(
            process=BlackScholesProcess(s0, r, sigma),
            time_grid=TimeGrid.uniform(T, 1),
            n_paths=200_000, seed=42, antithetic=True,
        )
        result = engine.price(european_put(k), discount_factor=df)
        assert result.price == pytest.approx(bs_price, rel=0.02)

    def test_otm_call(self):
        s0, k, r, sigma, T = 100, 130, 0.05, 0.20, 1.0
        bs_price = black_scholes_call(s0, k, r, sigma, T)
        df = math.exp(-r * T)

        engine = MCEngine(
            process=BlackScholesProcess(s0, r, sigma),
            time_grid=TimeGrid.uniform(T, 1),
            n_paths=200_000, seed=42, antithetic=True,
        )
        result = engine.price(european_call(k), discount_factor=df)
        assert result.price == pytest.approx(bs_price, rel=0.05)

    def test_stderr_decreases_with_paths(self):
        engine_small = MCEngine(
            process=BlackScholesProcess(100, 0.05, 0.20),
            time_grid=TimeGrid.uniform(1.0, 1),
            n_paths=10_000, seed=42,
        )
        engine_large = MCEngine(
            process=BlackScholesProcess(100, 0.05, 0.20),
            time_grid=TimeGrid.uniform(1.0, 1),
            n_paths=100_000, seed=42,
        )
        r_small = engine_small.price(european_call(100))
        r_large = engine_large.price(european_call(100))
        assert r_large.stderr < r_small.stderr

    def test_antithetic_reduces_variance(self):
        engine_no = MCEngine(
            process=BlackScholesProcess(100, 0.05, 0.20),
            time_grid=TimeGrid.uniform(1.0, 1),
            n_paths=50_000, seed=42, antithetic=False,
        )
        engine_yes = MCEngine(
            process=BlackScholesProcess(100, 0.05, 0.20),
            time_grid=TimeGrid.uniform(1.0, 1),
            n_paths=50_000, seed=42, antithetic=True,
        )
        r_no = engine_no.price(european_call(100))
        r_yes = engine_yes.price(european_call(100))
        assert r_yes.stderr < r_no.stderr


# ── Path-dependent ──

class TestPathDependent:

    def test_asian_call_less_than_european(self):
        """Asian option worth less than European (averaging reduces vol)."""
        engine = MCEngine(
            process=BlackScholesProcess(100, 0.05, 0.20),
            time_grid=TimeGrid.uniform(1.0, 252),
            n_paths=100_000, seed=42,
        )
        df = math.exp(-0.05)
        euro = engine.price(european_call(100), df).price
        asian = engine.price(asian_arithmetic(100), df).price
        assert asian < euro

    def test_lookback_call_positive(self):
        engine = MCEngine(
            process=BlackScholesProcess(100, 0.05, 0.20),
            time_grid=TimeGrid.uniform(1.0, 252),
            n_paths=50_000, seed=42,
        )
        result = engine.price(lookback_call(), math.exp(-0.05))
        assert result.price > 0


# ── Barrier ──

class TestBarrier:

    def test_knockout_less_than_vanilla(self):
        """Up-and-out call worth less than vanilla call."""
        engine = MCEngine(
            process=BlackScholesProcess(100, 0.05, 0.20),
            time_grid=TimeGrid.uniform(1.0, 252),
            n_paths=100_000, seed=42,
        )
        df = math.exp(-0.05)
        vanilla = engine.price(european_call(100), df).price
        ko = engine.price(barrier_knockout(100, 150, "up-and-out"), df).price
        assert ko < vanilla
        assert ko > 0


# ── American ──

class TestAmerican:

    def test_american_put_geq_european(self):
        """American put ≥ European put (early exercise premium)."""
        engine = MCEngine(
            process=BlackScholesProcess(100, 0.05, 0.20),
            time_grid=TimeGrid.uniform(1.0, 50),
            n_paths=50_000, seed=42,
        )
        df = math.exp(-0.05)
        euro = engine.price(european_put(100), df).price
        amer = engine.price(american_put(100), df).price
        assert amer >= euro * 0.95  # allow small MC noise


# ── Heston ──

class TestHeston:

    def test_heston_call_finite(self):
        engine = MCEngine(
            process=HestonProcess(
                s0=100, v0=0.04, mu=0.05,
                kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
            ),
            time_grid=TimeGrid.uniform(1.0, 100),
            n_paths=50_000, seed=42,
        )
        # Payoff on first factor (log-spot)
        result = engine.price(european_call(100, log_space=True), math.exp(-0.05))
        assert result.price > 0
        assert math.isfinite(result.price)


# ── Multi-asset ──

class TestMultiAsset:

    def test_basket_call(self):
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        engine = MCEngine(
            process=CorrelatedGBMProcess([100, 100], [0.05, 0.05], [0.20, 0.25], corr),
            time_grid=TimeGrid.uniform(1.0, 1),
            n_paths=100_000, seed=42,
        )
        result = engine.price(basket_call(100), math.exp(-0.05))
        assert result.price > 0

    def test_worst_of_put(self):
        corr = np.array([[1.0, 0.3], [0.3, 1.0]])
        engine = MCEngine(
            process=CorrelatedGBMProcess([100, 100], [0.05, 0.05], [0.20, 0.25], corr),
            time_grid=TimeGrid.uniform(1.0, 1),
            n_paths=100_000, seed=42,
        )
        result = engine.price(worst_of_put(100), math.exp(-0.05))
        assert result.price > 0


# ── Mean-reverting ──

class TestMeanReverting:

    def test_ou_mean_reverts(self):
        engine = MCEngine(
            process=OUProcess(x0=0.10, kappa=5.0, theta=0.05, sigma=0.01),
            time_grid=TimeGrid.uniform(2.0, 200),
            n_paths=10_000, seed=42,
        )
        paths = engine.paths
        terminal_mean = float(np.mean(paths[:, -1]))
        assert abs(terminal_mean - 0.05) < 0.01  # should revert to theta

    def test_cir_stays_positive(self):
        engine = MCEngine(
            process=CIRProcess(x0=0.04, kappa=2.0, theta=0.04, sigma=0.1),
            time_grid=TimeGrid.uniform(5.0, 500),
            n_paths=10_000, seed=42,
        )
        paths = engine.paths
        assert np.all(paths >= 0)  # CIR is non-negative


# ── MCResult ──

class TestMCResult:

    def test_to_dict(self):
        r = MCResult(10.5, 0.1, 100_000, 100, (10.3, 10.7))
        d = r.to_dict()
        assert "price" in d
        assert "stderr" in d
        assert "ci_95" in d

    def test_relative_error(self):
        r = MCResult(10.0, 0.1, 100_000, 100, (9.8, 10.2))
        assert r.relative_error == pytest.approx(1.0)
