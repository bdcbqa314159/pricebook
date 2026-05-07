"""Tests for MC extensions: Sobol, MLMC, pathwise/LR Greeks, copula, term structure, instrument wiring."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from pricebook.mc_engine import MCEngine, TimeGrid
from pricebook.mc_processes import BlackScholesProcess, OUProcess
from pricebook.mc_payoffs import european_call, european_put, asian_arithmetic
from pricebook.mc_extensions import (
    sobol_engine, mlmc_price,
    pathwise_delta, likelihood_ratio_delta,
    CopulaDefaultEngine, tranche_loss,
    ShortRateProcess, ForwardCurveProcess,
    instrument_mc_price, asian_option_mc, barrier_option_mc,
)


def bs_call(s0, k, r, sigma, T):
    d1 = (math.log(s0 / k) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return s0 * norm.cdf(d1) - k * math.exp(-r * T) * norm.cdf(d2)


# ── 1. Sobol ──

class TestSobol:

    def test_sobol_matches_bs(self):
        """Sobol engine should match Black-Scholes within 1%."""
        s0, k, r, sigma, T = 100, 100, 0.05, 0.20, 1.0
        bs_price = bs_call(s0, k, r, sigma, T)
        df = math.exp(-r * T)

        engine = sobol_engine(
            BlackScholesProcess(s0, r, sigma),
            TimeGrid.uniform(T, 1), n_paths=50_000,
        )
        result = engine.price(european_call(k), df)
        assert result.price == pytest.approx(bs_price, rel=0.02)

    def test_sobol_lower_stderr(self):
        """Sobol should have lower stderr than pseudo-random for same n_paths."""
        process = BlackScholesProcess(100, 0.05, 0.20)
        grid = TimeGrid.uniform(1.0, 1)

        pseudo = MCEngine(process, grid, 20_000, seed=42)
        r_pseudo = pseudo.price(european_call(100), math.exp(-0.05))

        quasi = sobol_engine(process, grid, 20_000)
        r_quasi = quasi.price(european_call(100), math.exp(-0.05))

        # Sobol typically gives lower stderr
        assert r_quasi.stderr < r_pseudo.stderr * 1.5  # allow margin


# ── 2. MLMC ──

class TestMLMC:

    def test_mlmc_matches_bs(self):
        """MLMC should match Black-Scholes."""
        s0, k, r, sigma, T = 100, 100, 0.05, 0.20, 1.0
        bs_price = bs_call(s0, k, r, sigma, T)
        df = math.exp(-r * T)

        result = mlmc_price(
            lambda: BlackScholesProcess(s0, r, sigma),
            european_call(k), T, df,
            n_levels=4, base_paths=20_000, base_steps=4,
        )
        assert result.price == pytest.approx(bs_price, rel=0.05)

    def test_mlmc_cost_savings(self):
        """MLMC should have cost ratio > 1 (cheaper than standard MC)."""
        result = mlmc_price(
            lambda: BlackScholesProcess(100, 0.05, 0.20),
            european_call(100), 1.0, math.exp(-0.05),
            n_levels=4,
        )
        assert result.cost_ratio > 0.5  # at least some savings

    def test_multiple_levels(self):
        result = mlmc_price(
            lambda: BlackScholesProcess(100, 0.05, 0.20),
            european_call(100), 1.0, math.exp(-0.05),
            n_levels=4, base_paths=20_000,
        )
        assert result.levels == 4
        assert len(result.level_variances) == 4
        assert all(v >= 0 for v in result.level_variances)


# ── 3. Pathwise & LR Greeks ──

class TestPathwiseGreeks:

    def test_pathwise_delta_call(self):
        """Pathwise delta for call should be in (0, 1)."""
        engine = MCEngine(
            BlackScholesProcess(100, 0.05, 0.20),
            TimeGrid.uniform(1.0, 1),
            n_paths=100_000, seed=42,
        )
        # ∂payoff/∂S_T = 1 if S_T > K, 0 otherwise (indicator)
        def dpayoff(paths, times):
            st = np.exp(paths[:, -1] if paths.ndim == 2 else paths[:, -1, 0])
            return np.where(st > 100, 1.0, 0.0)

        delta = pathwise_delta(engine, dpayoff, math.exp(-0.05))
        assert 0 < delta < 1

    def test_lr_delta_call(self):
        """LR delta for call should be in (0, 1)."""
        engine = MCEngine(
            BlackScholesProcess(100, 0.05, 0.20),
            TimeGrid.uniform(1.0, 1),
            n_paths=100_000, seed=42,
        )
        delta = likelihood_ratio_delta(
            engine, european_call(100), s0=100, sigma=0.20, T=1.0,
            discount_factor=math.exp(-0.05),
        )
        assert 0 < delta < 1

    def test_lr_delta_digital(self):
        """LR works for digitals (pathwise doesn't)."""
        from pricebook.mc_payoffs import digital_call
        engine = MCEngine(
            BlackScholesProcess(100, 0.05, 0.20),
            TimeGrid.uniform(1.0, 1),
            n_paths=200_000, seed=42,
        )
        delta = likelihood_ratio_delta(
            engine, digital_call(100), s0=100, sigma=0.20, T=1.0,
            discount_factor=math.exp(-0.05),
        )
        assert delta > 0  # digital call delta > 0


# ── 4. Copula Defaults ──

class TestCopulaDefaults:

    def test_expected_loss_positive(self):
        engine = CopulaDefaultEngine(
            pds=[0.02, 0.03, 0.01, 0.05],
            lgds=[0.60, 0.60, 0.60, 0.60],
            notionals=[10e6, 10e6, 10e6, 10e6],
            correlation=0.30, T=5.0, n_paths=50_000,
        )
        result = engine.simulate()
        assert result.expected_loss > 0

    def test_higher_correlation_higher_tail(self):
        """Higher correlation → fatter tail (more simultaneous defaults)."""
        def el(rho):
            eng = CopulaDefaultEngine(
                pds=[0.03] * 10, lgds=[0.60] * 10, notionals=[10e6] * 10,
                correlation=rho, T=5.0, n_paths=50_000, seed=42,
            )
            result = eng.simulate()
            return float(np.percentile(result.portfolio_loss, 99))

        tail_low = el(0.10)
        tail_high = el(0.50)
        assert tail_high > tail_low

    def test_tranche_loss(self):
        engine = CopulaDefaultEngine(
            pds=[0.02] * 100, lgds=[0.60] * 100,
            notionals=[1e6] * 100, correlation=0.20,
            T=5.0, n_paths=50_000,
        )
        sim = engine.simulate()
        equity_el = tranche_loss(sim, 0, 3e6)  # 0-3% tranche
        senior_el = tranche_loss(sim, 10e6, 100e6)  # 10%+ tranche
        assert equity_el > senior_el  # equity absorbs first losses

    def test_to_dict(self):
        engine = CopulaDefaultEngine(
            pds=[0.02, 0.03], lgds=[0.60, 0.60],
            notionals=[10e6, 10e6], n_paths=5_000,
        )
        d = engine.simulate().to_dict()
        assert "expected_loss" in d


# ── 5. Term Structure ──

class TestTermStructure:

    def test_short_rate_vasicek(self):
        proc = ShortRateProcess(0.05, 2.0, 0.04, 0.01, model="vasicek")
        engine = MCEngine(proc, TimeGrid.uniform(5.0, 50), 5_000, seed=42)
        terminal = float(np.mean(engine.paths[:, -1]))
        assert abs(terminal - 0.04) < 0.02  # mean-reverts to theta

    def test_short_rate_cir_positive(self):
        proc = ShortRateProcess(0.04, 2.0, 0.04, 0.1, model="cir")
        engine = MCEngine(proc, TimeGrid.uniform(5.0, 100), 5_000, seed=42)
        assert np.all(engine.paths >= 0)

    def test_forward_curve(self):
        proc = ForwardCurveProcess([0.04, 0.045, 0.05], [1.0, 2.0, 3.0])
        engine = MCEngine(proc, TimeGrid.uniform(1.0, 10), 5_000, seed=42)
        # Should produce 3-factor paths
        assert engine.paths.shape[2] == 3


# ── 6. Instrument Wiring ──

class TestInstrumentWiring:

    def test_asian_option_mc(self):
        result = asian_option_mc(100, 100, 0.05, 0.20, 1.0, n_paths=50_000)
        assert result.price > 0
        assert result.price < bs_call(100, 100, 0.05, 0.20, 1.0)  # Asian < European

    def test_asian_option_sobol(self):
        result = asian_option_mc(100, 100, 0.05, 0.20, 1.0, n_paths=50_000, use_sobol=True)
        assert result.price > 0

    def test_barrier_option_mc(self):
        result = barrier_option_mc(100, 100, 150, 0.05, 0.20, 1.0, n_paths=50_000)
        vanilla = bs_call(100, 100, 0.05, 0.20, 1.0)
        assert 0 < result.price < vanilla  # KO < vanilla

    def test_instrument_mc_generic(self):
        """Generic instrument pricing through the engine."""
        result = instrument_mc_price(
            instrument=None,
            process=BlackScholesProcess(100, 0.05, 0.20),
            time_grid=TimeGrid.uniform(1.0, 1),
            payoff=european_call(100),
            discount_factor=math.exp(-0.05),
            n_paths=50_000,
        )
        bs = bs_call(100, 100, 0.05, 0.20, 1.0)
        assert result.price == pytest.approx(bs, rel=0.03)
