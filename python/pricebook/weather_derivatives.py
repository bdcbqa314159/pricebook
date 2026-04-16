"""Weather derivatives: HDD/CDD, temperature model, rainfall/wind.

* :func:`hdd_index` / :func:`cdd_index` — heating/cooling degree days.
* :func:`hdd_future_price` / :func:`hdd_option_price` — CME weather contracts.
* :class:`SeasonalOUTemperature` — Alaton-Djehiche-Stillberger OU temperature.
* :func:`rainfall_derivative_price` — Poisson-Gamma rainfall.
* :func:`wind_index_option` — wind speed index derivatives.

References:
    Alaton, Djehiche & Stillberger, *On Modelling and Pricing Weather
    Derivatives*, Applied Math. Finance, 2002.
    Brody, Syroka & Zervos, *Dynamical Pricing of Weather Derivatives*, QF, 2002.
    Jewson & Brix, *Weather Derivative Valuation*, Cambridge UP, 2005.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---- HDD / CDD ----

REFERENCE_TEMP_F = 65.0      # standard US reference (Fahrenheit)


@dataclass
class DegreeDayIndex:
    """Degree days index result."""
    total: float            # cumulative degree days over period
    n_days: int
    index_type: str         # "HDD" or "CDD"
    mean_per_day: float


def hdd_index(
    daily_temps: list[float],
    reference: float = REFERENCE_TEMP_F,
) -> DegreeDayIndex:
    """Heating Degree Days: Σ max(reference − T_mean, 0).

    Measures heating demand; higher when colder.

    Args:
        daily_temps: daily mean temperatures (°F typical).
        reference: reference temperature (default 65°F).
    """
    T = np.array(daily_temps)
    daily_hdd = np.maximum(reference - T, 0.0)
    total = float(daily_hdd.sum())
    return DegreeDayIndex(total, len(T), "HDD", total / max(len(T), 1))


def cdd_index(
    daily_temps: list[float],
    reference: float = REFERENCE_TEMP_F,
) -> DegreeDayIndex:
    """Cooling Degree Days: Σ max(T_mean − reference, 0)."""
    T = np.array(daily_temps)
    daily_cdd = np.maximum(T - reference, 0.0)
    total = float(daily_cdd.sum())
    return DegreeDayIndex(total, len(T), "CDD", total / max(len(T), 1))


# ---- HDD/CDD pricing ----

@dataclass
class WeatherFutureResult:
    """HDD/CDD futures pricing result."""
    price: float
    expected_index: float
    tick_value: float           # $ per degree day
    index_type: str


def hdd_future_price(
    expected_hdd: float,
    tick_value: float = 20.0,
    discount_factor: float = 1.0,
) -> WeatherFutureResult:
    """CME HDD futures price.

    Contract pays tick_value × HDD at expiry. Fair price = DF × E[HDD] × tick.

    Args:
        expected_hdd: forecast HDD over the period.
        tick_value: $ per degree day (typically $20 for CME).
        discount_factor: DF to maturity.
    """
    price = discount_factor * expected_hdd * tick_value
    return WeatherFutureResult(
        price=float(price),
        expected_index=expected_hdd,
        tick_value=tick_value,
        index_type="HDD",
    )


@dataclass
class WeatherOptionResult:
    """HDD/CDD option pricing result."""
    price: float
    expected_index: float
    strike: float
    payoff_cap: float           # maximum payoff
    tick_value: float
    is_call: bool


def hdd_option_price(
    simulated_hdd: np.ndarray,      # (n_paths,) simulated HDD values
    strike: float,
    cap_level: float | None = None,
    tick_value: float = 20.0,
    discount_factor: float = 1.0,
    is_call: bool = True,
) -> WeatherOptionResult:
    """HDD call/put option (with optional cap).

    Call payoff = tick × min(cap, max(HDD − strike, 0))
    Put payoff = tick × min(cap, max(strike − HDD, 0))

    Args:
        simulated_hdd: MC draws of the HDD index.
        strike: HDD strike.
        cap_level: maximum payoff in HDD units; None = uncapped.
        tick_value: $ per degree day.
    """
    if is_call:
        payoff = np.maximum(simulated_hdd - strike, 0.0)
    else:
        payoff = np.maximum(strike - simulated_hdd, 0.0)

    if cap_level is not None:
        payoff = np.minimum(payoff, cap_level)

    price = float(discount_factor * tick_value * payoff.mean())

    return WeatherOptionResult(
        price=price,
        expected_index=float(simulated_hdd.mean()),
        strike=strike,
        payoff_cap=cap_level if cap_level is not None else float("inf"),
        tick_value=tick_value,
        is_call=is_call,
    )


# ---- Seasonal OU temperature ----

@dataclass
class TemperaturePaths:
    """Simulated daily temperature paths."""
    temperatures: np.ndarray    # (n_paths, n_days)
    times: np.ndarray           # (n_days,) in years
    n_paths: int


class SeasonalOUTemperature:
    """Alaton-Djehiche-Stillberger seasonal OU temperature model.

    T_t = A(t) + X_t
    dX = κ(−X) dt + σ(t) dW
    A(t) = a + b × t + c × sin(2π(t − t₀) / 365)

    A(t) captures seasonality + linear trend; X is mean-zero OU noise.

    Args:
        a, b, c: seasonal function parameters (level, trend, amplitude).
        t_shift: phase shift in days.
        kappa: mean reversion speed of noise.
        sigma: noise vol (can be seasonally varying).
    """

    def __init__(
        self,
        a: float = 50.0,            # long-run mean (°F)
        b: float = 0.0,              # trend per year
        c: float = 20.0,             # seasonal amplitude
        t_shift: float = 180.0,      # phase shift (days; 180 ≈ July)
        kappa: float = 0.3,
        sigma: float = 3.0,
    ):
        self.a = a
        self.b = b
        self.c = c
        self.t_shift = t_shift
        self.kappa = kappa
        self.sigma = sigma

    def seasonal_mean(self, t_days: float) -> float:
        """Expected temperature at day t_days."""
        t_years = t_days / 365.0
        return (self.a + self.b * t_years
                + self.c * math.sin(2 * math.pi * (t_days - self.t_shift) / 365))

    def simulate(
        self,
        n_days: int,
        n_paths: int = 1000,
        start_day_of_year: int = 1,
        initial_residual: float = 0.0,
        seed: int | None = 42,
    ) -> TemperaturePaths:
        rng = np.random.default_rng(seed)
        dt = 1.0 / 365.0    # one day
        sqrt_dt = math.sqrt(dt)

        days = np.arange(n_days) + start_day_of_year
        means = np.array([self.seasonal_mean(d) for d in days])

        X = np.zeros((n_paths, n_days))
        X[:, 0] = initial_residual

        for i in range(1, n_days):
            dW = rng.standard_normal(n_paths) * sqrt_dt
            X[:, i] = X[:, i - 1] + self.kappa * (-X[:, i - 1]) * dt \
                + self.sigma * dW

        T = means[np.newaxis, :] + X

        return TemperaturePaths(
            temperatures=T,
            times=np.arange(n_days) * dt,
            n_paths=n_paths,
        )


# ---- Rainfall ----

@dataclass
class RainfallResult:
    """Rainfall derivative pricing result."""
    price: float
    expected_total: float
    prob_below_threshold: float
    contract_type: str


def rainfall_derivative_price(
    n_days: int,
    mean_rainy_days: float,
    mean_rainfall_per_rainy_day: float,
    strike: float,
    payout: float = 1.0,
    discount_factor: float = 1.0,
    is_drought_option: bool = True,
    n_paths: int = 20_000,
    seed: int | None = 42,
) -> RainfallResult:
    """Simple rainfall derivative via Poisson-Gamma model.

    Daily rainfall = Bernoulli(p) × Gamma(k, θ).
    Option pays payout if total rainfall < strike (drought) or > strike (flood).

    Args:
        n_days: period length.
        mean_rainy_days: expected # rainy days.
        mean_rainfall_per_rainy_day: mean rainfall on a wet day.
        strike: threshold (total rainfall).
        is_drought_option: True = pays when below strike; False = above.
    """
    rng = np.random.default_rng(seed)
    p_rain = mean_rainy_days / n_days

    # Simulate
    total_rainfall = np.zeros(n_paths)
    for _ in range(n_days):
        is_rainy = rng.random(n_paths) < p_rain
        rainfall = np.where(is_rainy,
                             rng.gamma(shape=2.0, scale=mean_rainfall_per_rainy_day / 2.0,
                                        size=n_paths),
                             0.0)
        total_rainfall += rainfall

    if is_drought_option:
        triggered = total_rainfall < strike
        contract = "drought"
    else:
        triggered = total_rainfall > strike
        contract = "flood"

    prob = float(triggered.mean())
    price = float(discount_factor * payout * prob)

    return RainfallResult(
        price=price,
        expected_total=float(total_rainfall.mean()),
        prob_below_threshold=prob,
        contract_type=contract,
    )


# ---- Wind ----

@dataclass
class WindOptionResult:
    """Wind index option result."""
    price: float
    expected_wind_index: float
    strike: float


def wind_index_option(
    simulated_wind_index: np.ndarray,
    strike: float,
    payout_per_unit: float = 1.0,
    discount_factor: float = 1.0,
    is_call: bool = True,
) -> WindOptionResult:
    """Wind index option (e.g. average wind speed over a period).

    Call pays payout × max(wind − strike, 0).

    Args:
        simulated_wind_index: (n_paths,) MC draws.
    """
    if is_call:
        payoff = np.maximum(simulated_wind_index - strike, 0.0)
    else:
        payoff = np.maximum(strike - simulated_wind_index, 0.0)

    price = float(discount_factor * payout_per_unit * payoff.mean())

    return WindOptionResult(
        price=price,
        expected_wind_index=float(simulated_wind_index.mean()),
        strike=strike,
    )
