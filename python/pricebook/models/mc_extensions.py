"""MC engine extensions: Sobol, MLMC, pathwise Greeks, copula defaults,
term structure processes, and instrument wiring.

Extends the core mc_engine with:
1. Quasi-random (Sobol) path generation
2. Multi-Level Monte Carlo (MLMC)
3. Pathwise (IPA) and likelihood ratio Greeks
4. Copula-based default simulation
5. Term structure processes (short-rate, LMM-style)
6. Instrument adapters (wire existing instruments into the engine)

    from pricebook.models.mc_extensions import (
        sobol_engine, mlmc_price,
        pathwise_delta, likelihood_ratio_delta,
        CopulaDefaultEngine, default_simulation,
        ShortRateProcess, ForwardCurveProcess,
        instrument_mc_price,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from pricebook.models.mc_engine import MCEngine, TimeGrid, ProcessSpec, MCResult


# ===========================================================================
# 1. Quasi-random (Sobol) integration
# ===========================================================================

def sobol_engine(
    process: ProcessSpec,
    time_grid: TimeGrid,
    n_paths: int = 100_000,
    seed: int = 42,
) -> MCEngine:
    """Create an MCEngine with Sobol (quasi-random) path generation.

    Replaces pseudo-random normals with Sobol low-discrepancy sequences
    for faster convergence (O(1/N) vs O(1/√N)).

    Uses the existing rng.py QuasiRandom class internally.
    """
    from pricebook.statistics.rng import QuasiRandom

    # Generate Sobol normals for all steps
    qrng = QuasiRandom(dimension=time_grid.n_steps, seed=seed)
    z = qrng.normals(n_paths)  # (n_paths, n_steps)

    # Build engine and inject pre-generated paths
    engine = MCEngine(process, time_grid, n_paths, seed)

    # Override path generation with Sobol-driven paths
    nf = process.n_factors
    if nf == 1:
        paths = np.zeros((n_paths, time_grid.n_steps + 1))
        paths[:, 0] = process.x0[0]
        for i in range(time_grid.n_steps):
            dw = z[:, i] * np.sqrt(time_grid.dt[i])
            if process.exact_step is not None:
                paths[:, i + 1] = process.exact_step(
                    paths[:, i], time_grid.times[i], time_grid.dt[i], dw)
            else:
                from pricebook.models.mc_engine import euler_step
                paths[:, i + 1] = euler_step(
                    paths[:, i], time_grid.times[i], time_grid.dt[i], dw, process)
    else:
        # Multi-factor: generate n_factors × n_steps Sobol dimensions
        qrng_mf = QuasiRandom(dimension=time_grid.n_steps * nf, seed=seed)
        z_flat = qrng_mf.normals(n_paths)  # (n_paths, n_steps * n_factors)
        z_mf = z_flat.reshape(n_paths, time_grid.n_steps, nf)
        # Apply Cholesky
        z_corr = np.einsum('...j,kj->...k', z_mf, process.cholesky)

        paths = np.zeros((n_paths, time_grid.n_steps + 1, nf))
        paths[:, 0, :] = process.x0
        for i in range(time_grid.n_steps):
            dw = z_corr[:, i, :] * np.sqrt(time_grid.dt[i])
            if process.exact_step is not None:
                paths[:, i + 1, :] = process.exact_step(
                    paths[:, i, :], time_grid.times[i], time_grid.dt[i], dw)
            else:
                from pricebook.models.mc_engine import euler_step
                paths[:, i + 1, :] = euler_step(
                    paths[:, i, :], time_grid.times[i], time_grid.dt[i], dw, process)

    engine._paths = paths
    return engine


# ===========================================================================
# 2. Multi-Level Monte Carlo (MLMC)
# ===========================================================================

@dataclass
class MLMCResult:
    """Multi-Level MC result."""
    price: float
    stderr: float
    levels: int
    cost_ratio: float    # cost savings vs standard MC
    level_variances: list[float]

    def to_dict(self) -> dict:
        return {"price": self.price, "stderr": self.stderr,
                "levels": self.levels, "cost_ratio": self.cost_ratio}


def mlmc_price(
    process_factory,
    payoff,
    T: float,
    discount_factor: float = 1.0,
    n_levels: int = 4,
    base_paths: int = 50_000,
    base_steps: int = 4,
    seed: int = 42,
) -> MLMCResult:
    """Multi-Level Monte Carlo estimator (Giles 2008).

    E[P_L] = E[P_0] + Σ_{l=1}^{L} E[P_l - P_{l-1}]

    Each level doubles the number of time steps. The correction terms
    P_l - P_{l-1} have decreasing variance, so fewer paths are needed
    at finer levels.

    Args:
        process_factory: callable() → ProcessSpec.
        payoff: payoff callable.
        T: maturity.
        discount_factor: risk-neutral DF.
        n_levels: number of refinement levels.
        base_paths: paths at coarsest level.
        base_steps: time steps at coarsest level.
        seed: random seed.
    """
    total_price = 0.0
    total_var = 0.0
    total_cost = 0
    level_vars = []

    for level in range(n_levels):
        n_steps = base_steps * (2 ** level)
        # Fewer paths at finer levels (variance decreases)
        n_paths = max(base_paths // (2 ** level), 1000)

        rng = np.random.default_rng(seed + level)

        if level == 0:
            # Coarsest level: just price directly
            engine = MCEngine(process_factory(), TimeGrid.uniform(T, n_steps),
                              n_paths, seed + level)
            values = payoff(engine.paths, engine.time_grid.times) * discount_factor
            level_mean = float(np.mean(values))
            level_var = float(np.var(values, ddof=1))
        else:
            # Correction: P_fine - P_coarse on SAME Brownian path
            n_fine = n_steps
            n_coarse = n_steps // 2

            # Generate fine paths
            engine_fine = MCEngine(process_factory(), TimeGrid.uniform(T, n_fine),
                                   n_paths, seed + level)
            fine_values = payoff(engine_fine.paths, engine_fine.time_grid.times) * discount_factor

            # Generate coarse paths (same seed = same Brownian motion, subsampled)
            engine_coarse = MCEngine(process_factory(), TimeGrid.uniform(T, n_coarse),
                                     n_paths, seed + level)
            coarse_values = payoff(engine_coarse.paths, engine_coarse.time_grid.times) * discount_factor

            correction = fine_values - coarse_values
            level_mean = float(np.mean(correction))
            level_var = float(np.var(correction, ddof=1))

        total_price += level_mean
        total_var += level_var / n_paths
        total_cost += n_paths * n_steps
        level_vars.append(level_var)

    stderr = math.sqrt(total_var)

    # Cost ratio: standard MC at finest level vs MLMC
    finest_steps = base_steps * (2 ** (n_levels - 1))
    standard_cost = base_paths * finest_steps
    cost_ratio = standard_cost / max(total_cost, 1) if total_cost > 0 else 1.0

    return MLMCResult(
        price=total_price, stderr=stderr,
        levels=n_levels, cost_ratio=cost_ratio,
        level_variances=level_vars,
    )


# ===========================================================================
# 3. Pathwise (IPA) and Likelihood Ratio Greeks
# ===========================================================================

def pathwise_delta(
    engine: MCEngine,
    payoff_derivative,
    discount_factor: float = 1.0,
) -> float:
    """Pathwise (IPA) delta: E[df × ∂payoff/∂S₀ × ∂S_T/∂S₀].

    For GBM in log-space: ∂S_T/∂S₀ = S_T/S₀.
    For smooth payoffs only (not barriers/digitals).

    Args:
        engine: MC engine with paths generated.
        payoff_derivative: callable(paths, times) → ∂payoff/∂S_T array.
        discount_factor: risk-neutral DF.
    """
    paths = engine.paths
    if paths.ndim == 3:
        s_T = np.exp(paths[:, -1, 0])
        s_0 = np.exp(paths[:, 0, 0])
    else:
        s_T = np.exp(paths[:, -1])
        s_0 = np.exp(paths[:, 0])

    # Chain rule: ∂V/∂S₀ = E[df × ∂payoff/∂S_T × S_T/S₀]
    dpayoff_ds = payoff_derivative(paths, engine.time_grid.times)
    pathwise = dpayoff_ds * (s_T / s_0) * discount_factor

    return float(np.mean(pathwise))


def likelihood_ratio_delta(
    engine: MCEngine,
    payoff,
    s0: float,
    sigma: float,
    T: float,
    discount_factor: float = 1.0,
) -> float:
    """Likelihood ratio delta for GBM.

    Δ = E[df × payoff × score], where score = (log(S_T/S₀) - (μ-σ²/2)T) / (S₀ σ² T).

    Works for ALL payoffs including digitals and barriers.

    Args:
        engine: MC engine with paths.
        payoff: payoff callable.
        s0: initial spot.
        sigma: volatility.
        T: maturity.
        discount_factor: DF.
    """
    paths = engine.paths
    times = engine.time_grid.times

    if paths.ndim == 3:
        log_return = paths[:, -1, 0] - paths[:, 0, 0]
    else:
        log_return = paths[:, -1] - paths[:, 0]

    # Score function for GBM
    score = log_return / (s0 * sigma ** 2 * T)

    values = payoff(paths, times) * discount_factor
    lr_delta = values * score

    return float(np.mean(lr_delta))


# ===========================================================================
# 4. Copula-based default simulation
# ===========================================================================

@dataclass
class DefaultSimResult:
    """Copula default simulation result."""
    default_times: np.ndarray    # (n_paths, n_names) — default time per name
    n_defaults: np.ndarray       # (n_paths,) — number of defaults per path
    portfolio_loss: np.ndarray   # (n_paths,) — total loss per path
    expected_loss: float
    loss_std: float

    def to_dict(self) -> dict:
        return {"expected_loss": self.expected_loss, "loss_std": self.loss_std,
                "avg_defaults": float(np.mean(self.n_defaults))}


class CopulaDefaultEngine:
    """Gaussian copula default simulation for portfolio credit.

    Simulates correlated defaults using the 1-factor Gaussian copula:
        X_i = √ρ M + √(1-ρ) ε_i
    where M = systematic factor, ε_i = idiosyncratic.
    Default when Φ(X_i) < PD_i.

    Args:
        pds: list of per-name default probabilities (annual).
        lgds: list of per-name LGDs.
        notionals: list of per-name notionals.
        correlation: uniform pairwise correlation.
        T: horizon (years).
        n_paths: simulation paths.
        seed: random seed.
    """

    def __init__(
        self,
        pds: list[float],
        lgds: list[float],
        notionals: list[float],
        correlation: float = 0.30,
        T: float = 5.0,
        n_paths: int = 100_000,
        seed: int = 42,
    ):
        self.pds = np.array(pds)
        self.lgds = np.array(lgds)
        self.notionals = np.array(notionals)
        self.correlation = correlation
        self.T = T
        self.n_paths = n_paths
        self.seed = seed
        self.n_names = len(pds)

    def simulate(self) -> DefaultSimResult:
        """Run copula default simulation."""
        rng = np.random.default_rng(self.seed)
        rho = self.correlation
        sqrt_rho = math.sqrt(max(rho, 0))
        sqrt_1_rho = math.sqrt(max(1 - rho, 0))

        # Cumulative PD over horizon
        cum_pd = 1 - (1 - self.pds) ** self.T
        # Default threshold in normal space
        thresholds = norm.ppf(np.maximum(cum_pd, 1e-10))

        # Systematic factor
        M = rng.standard_normal(self.n_paths)

        # Per-name latent variable
        eps = rng.standard_normal((self.n_paths, self.n_names))
        X = sqrt_rho * M[:, np.newaxis] + sqrt_1_rho * eps

        # Default indicator
        defaults = X < thresholds[np.newaxis, :]  # (n_paths, n_names)

        # Default times (approximate: uniform within [0, T])
        default_times = np.where(defaults, rng.uniform(0, self.T, defaults.shape), np.inf)

        # Loss
        loss_per_name = defaults.astype(float) * self.lgds[np.newaxis, :] * self.notionals[np.newaxis, :]
        portfolio_loss = loss_per_name.sum(axis=1)
        n_defaults = defaults.sum(axis=1)

        return DefaultSimResult(
            default_times=default_times,
            n_defaults=n_defaults,
            portfolio_loss=portfolio_loss,
            expected_loss=float(np.mean(portfolio_loss)),
            loss_std=float(np.std(portfolio_loss, ddof=1)),
        )


def tranche_loss(
    sim: DefaultSimResult,
    attachment: float,
    detachment: float,
) -> float:
    """Tranche expected loss from default simulation.

    Loss to tranche [A, D] = E[min(max(L - A, 0), D - A)] / (D - A).
    """
    width = detachment - attachment
    if width <= 0:
        return 0.0
    tranche_losses = np.minimum(
        np.maximum(sim.portfolio_loss - attachment, 0.0), width
    )
    return float(np.mean(tranche_losses)) / width


# ===========================================================================
# 5. Term structure processes
# ===========================================================================

def ShortRateProcess(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    model: str = "vasicek",
    **kwargs,
) -> ProcessSpec:
    """Short-rate model for interest rate simulation.

    Supports:
    - "vasicek": dR = κ(θ-R) dt + σ dW (Gaussian, can go negative)
    - "cir": dR = κ(θ-R) dt + σ√R dW (non-negative)
    - "hull_white": dR = (θ(t)-κR) dt + σ dW, with time-dependent θ(t).
      Pass ``theta_func=callable(t)`` to supply θ(t); if omitted, falls back
      to constant θ (equivalent to Vasicek — not curve-fitted).

    For bond pricing: P(t,T) = E[exp(-∫r ds)].
    """
    from pricebook.models.mc_processes import OUProcess, CIRProcess

    if model == "vasicek":
        return OUProcess(r0, kappa, theta, sigma)
    elif model == "hull_white":
        theta_func = kwargs.get("theta_func", None)
        if theta_func is not None:
            from pricebook.models.mc_processes import HullWhiteProcess
            return HullWhiteProcess(r0, a=kappa, sigma=sigma, theta_func=theta_func)
        # Fallback: constant theta — calibration to term structure not applied.
        return OUProcess(r0, kappa, theta, sigma)
    elif model == "cir":
        return CIRProcess(r0, kappa, theta, sigma)
    else:
        raise ValueError(f"Unknown short-rate model: {model}")


def ForwardCurveProcess(
    initial_forwards: list[float],
    tenors: list[float],
    vol: float = 0.01,
    correlation: float = 0.95,
) -> ProcessSpec:
    """Simplified forward rate curve evolution (LMM-style).

    Each forward rate evolves as: dF_i = σ_i F_i dW_i
    with uniform correlation between adjacent forwards.

    For full LMM, use lmm.py directly. This is a simplified
    version for exposure simulation.
    """
    n = len(initial_forwards)
    vols = [vol] * n

    # Build correlation matrix (exponentially decaying)
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr[i, j] = correlation ** abs(i - j)

    from pricebook.models.mc_processes import CorrelatedGBMProcess
    return CorrelatedGBMProcess(
        s0=initial_forwards,
        mu=[0.0] * n,  # risk-neutral drift = 0 for forward rates
        sigma=vols,
        correlation=corr,
    )


# ===========================================================================
# 6. Instrument adapters
# ===========================================================================

def instrument_mc_price(
    instrument,
    process: ProcessSpec,
    time_grid: TimeGrid,
    payoff,
    discount_factor: float = 1.0,
    n_paths: int = 100_000,
    seed: int = 42,
    antithetic: bool = True,
    use_sobol: bool = False,
) -> MCResult:
    """Price any instrument through the unified MC engine.

    This is the bridge between existing instruments and the MC engine.
    Define the process and payoff for the instrument, get a price.

    Args:
        instrument: any pricebook instrument (for reference only — payoff is separate).
        process: SDE for the underlying factor.
        time_grid: simulation time grid.
        payoff: payoff callable(paths, times) → values.
        discount_factor: risk-neutral DF.
        n_paths: number of paths.
        seed: random seed.
        antithetic: use antithetic variates.
        use_sobol: use Sobol quasi-random instead of pseudo-random.
    """
    if use_sobol:
        engine = sobol_engine(process, time_grid, n_paths, seed)
    else:
        engine = MCEngine(process, time_grid, n_paths, seed, antithetic=antithetic)

    return engine.price(payoff, discount_factor)


def asian_option_mc(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    T: float,
    n_fixings: int = 252,
    n_paths: int = 100_000,
    seed: int = 42,
    use_sobol: bool = False,
) -> MCResult:
    """Price an Asian option through the unified engine.

    Example of wiring an existing instrument type through MCEngine.
    Replaces the standalone asian.py mc_asian_arithmetic().
    """
    from pricebook.models.mc_processes import BlackScholesProcess
    from pricebook.models.mc_payoffs import asian_arithmetic

    process = BlackScholesProcess(spot, rate, sigma)
    grid = TimeGrid.uniform(T, n_fixings)
    df = math.exp(-rate * T)

    return instrument_mc_price(
        instrument=None,  # no instrument object needed
        process=process,
        time_grid=grid,
        payoff=asian_arithmetic(strike),
        discount_factor=df,
        n_paths=n_paths,
        seed=seed,
        use_sobol=use_sobol,
    )


def barrier_option_mc(
    spot: float,
    strike: float,
    barrier: float,
    rate: float,
    sigma: float,
    T: float,
    barrier_type: str = "up-and-out",
    n_steps: int = 252,
    n_paths: int = 100_000,
    seed: int = 42,
) -> MCResult:
    """Price a barrier option through the unified engine."""
    from pricebook.models.mc_processes import BlackScholesProcess
    from pricebook.models.mc_payoffs import barrier_knockout

    process = BlackScholesProcess(spot, rate, sigma)
    grid = TimeGrid.uniform(T, n_steps)
    df = math.exp(-rate * T)

    return instrument_mc_price(
        instrument=None, process=process, time_grid=grid,
        payoff=barrier_knockout(strike, barrier, barrier_type),
        discount_factor=df, n_paths=n_paths, seed=seed,
    )
