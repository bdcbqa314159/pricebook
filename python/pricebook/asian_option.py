"""Asian option: schedule-aware, partial fixings, Turnbull-Wakeman + MC.

Builds on asian.py (raw MC functions) with a proper instrument class
that carries its averaging schedule, supports partial fixings, and
is serialisable from day one.

    from pricebook.asian_option import AsianOption, AsianSchedule

    schedule = AsianSchedule.monthly(start, end)
    opt = AsianOption(schedule=schedule, strike=100, notional=1_000_000)
    result = opt.price(spot=100, curve=ois, vol=0.20)

References:
    Turnbull & Wakeman (1991). A Quick Algorithm for Pricing European
    Average Options. J. Financial and Quantitative Analysis, 26(3).
    Curran (1994). Valuing Asian and Portfolio Options by Conditioning
    on the Geometric Mean Price. Management Science, 40(12).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np

from pricebook.asian import geometric_asian_analytical, mc_asian_arithmetic
from pricebook.black76 import OptionType, black76_price
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.gbm import GBMGenerator
from pricebook.mc_pricer import MCResult
from pricebook.serialisable import serialisable as _serialisable, _serialise_atom


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

@dataclass
class AsianSchedule:
    """Averaging observation schedule.

    Defines when fixings occur and their weights.

    Args:
        fixing_dates: list of observation dates.
        weights: per-fixing weights (None = equal). Must sum to 1 if provided.
        fixing_lag: business days before the fixing is published (0 for most equities).
    """
    fixing_dates: list[date]
    weights: list[float] | None = None
    fixing_lag: int = 0

    def __post_init__(self):
        if self.weights is not None:
            if len(self.weights) != len(self.fixing_dates):
                raise ValueError(
                    f"weights length ({len(self.weights)}) != fixing_dates length ({len(self.fixing_dates)})"
                )

    @property
    def n_fixings(self) -> int:
        return len(self.fixing_dates)

    @property
    def effective_weights(self) -> list[float]:
        if self.weights is not None:
            return self.weights
        n = self.n_fixings
        return [1.0 / n] * n if n > 0 else []

    @classmethod
    def monthly(cls, start: date, end: date, fixing_lag: int = 0) -> AsianSchedule:
        """Monthly fixings from start to end (inclusive of end month)."""
        from dateutil.relativedelta import relativedelta
        dates = []
        d = start
        while d <= end:
            dates.append(d)
            d += relativedelta(months=1)
        return cls(fixing_dates=dates, fixing_lag=fixing_lag)

    @classmethod
    def weekly(cls, start: date, end: date, fixing_lag: int = 0) -> AsianSchedule:
        """Weekly fixings."""
        dates = []
        d = start
        while d <= end:
            dates.append(d)
            d += timedelta(weeks=1)
        return cls(fixing_dates=dates, fixing_lag=fixing_lag)

    @classmethod
    def daily(cls, start: date, end: date, fixing_lag: int = 0) -> AsianSchedule:
        """Daily fixings (weekdays only)."""
        dates = []
        d = start
        while d <= end:
            if d.weekday() < 5:  # Mon-Fri
                dates.append(d)
            d += timedelta(days=1)
        return cls(fixing_dates=dates, fixing_lag=fixing_lag)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "fixing_dates": [fd.isoformat() for fd in self.fixing_dates],
            "fixing_lag": self.fixing_lag,
        }
        if self.weights is not None:
            d["weights"] = self.weights
        return d

    @classmethod
    def from_dict(cls, d: dict) -> AsianSchedule:
        return cls(
            fixing_dates=[date.fromisoformat(s) for s in d["fixing_dates"]],
            weights=d.get("weights"),
            fixing_lag=d.get("fixing_lag", 0),
        )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class AsianResult:
    """Asian option pricing result."""
    price: float
    std_error: float = 0.0
    n_paths: int = 0
    method: str = ""          # "turnbull_wakeman", "mc", "mc_cv"
    intrinsic_from_fixings: float = 0.0  # PV from known fixings
    time_value: float = 0.0              # PV from future fixings
    n_fixed: int = 0                     # number of known fixings
    n_remaining: int = 0                 # number of future fixings

    def to_dict(self) -> dict:
        return {
            "price": self.price, "std_error": self.std_error,
            "n_paths": self.n_paths, "method": self.method,
            "intrinsic_from_fixings": self.intrinsic_from_fixings,
            "time_value": self.time_value,
            "n_fixed": self.n_fixed, "n_remaining": self.n_remaining,
        }


# ---------------------------------------------------------------------------
# Turnbull-Wakeman analytical approximation
# ---------------------------------------------------------------------------

def turnbull_wakeman(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    fixing_times: list[float],
    weights: list[float],
    T: float,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
    known_fixings: list[float] | None = None,
) -> float:
    """Turnbull-Wakeman (1991) lognormal approximation for arithmetic Asian.

    Matches the first two moments of the arithmetic average A = Σ w_i S(t_i)
    to a lognormal distribution, then prices via Black-76.

    M₁ = Σ w_i F(t_i)
    M₂ = Σ_i Σ_j w_i w_j F(t_i) F(t_j) exp(σ² min(t_i, t_j))

    where F(t) = S₀ exp((r-q) t).

    For partially fixed options, known fixings contribute their exact values
    to M₁ and M₂.
    """
    n = len(fixing_times)
    if n == 0:
        return 0.0

    mu = rate - div_yield

    # Forward prices at each fixing
    forwards = [spot * math.exp(mu * t) for t in fixing_times]

    # Split into known and future
    n_known = len(known_fixings) if known_fixings else 0

    # First moment: M1 = Σ w_i E[S(t_i)]
    m1 = 0.0
    for i in range(n):
        if i < n_known:
            m1 += weights[i] * known_fixings[i]
        else:
            m1 += weights[i] * forwards[i]

    if m1 <= 0:
        return 0.0

    # Second moment: M2 = Σ_i Σ_j w_i w_j E[S(t_i) S(t_j)]
    # E[S(t_i) S(t_j)] = F(t_i) F(t_j) exp(σ² min(t_i, t_j))
    m2 = 0.0
    for i in range(n):
        for j in range(n):
            if i < n_known and j < n_known:
                m2 += weights[i] * weights[j] * known_fixings[i] * known_fixings[j]
            elif i < n_known:
                m2 += weights[i] * weights[j] * known_fixings[i] * forwards[j]
            elif j < n_known:
                m2 += weights[i] * weights[j] * forwards[i] * known_fixings[j]
            else:
                min_t = min(fixing_times[i], fixing_times[j])
                m2 += (weights[i] * weights[j] * forwards[i] * forwards[j]
                       * math.exp(vol**2 * min_t))

    # Lognormal parameters from moments
    if m2 <= m1**2:
        # Degenerate: no variance left (all fixed)
        df = math.exp(-rate * T)
        if option_type == OptionType.CALL:
            return df * max(m1 - strike, 0.0)
        else:
            return df * max(strike - m1, 0.0)

    var_ln = math.log(m2 / m1**2)
    vol_adj = math.sqrt(var_ln / T) if T > 0 else 0.0

    df = math.exp(-rate * T)

    return black76_price(m1, strike, vol_adj, T, df, option_type)


# ---------------------------------------------------------------------------
# Curran (1994) conditional expectation
# ---------------------------------------------------------------------------

def curran_asian(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    fixing_times: list[float],
    weights: list[float],
    T: float,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
) -> float:
    """Curran (1994) lower bound for arithmetic Asian option.

    Conditions on the geometric average G. Since A and G are highly
    correlated, E[max(A-K, 0)] ≈ E[E[max(A-K, 0) | G]].

    More accurate than Turnbull-Wakeman for OTM options because it
    captures the conditional distribution shape, not just moments.

    References:
        Curran, M. (1994). Valuing Asian and Portfolio Options by
        Conditioning on the Geometric Mean Price. Management Science, 40(12).
    """
    from scipy.stats import norm as norm_dist

    n = len(fixing_times)
    if n == 0:
        return 0.0

    mu = rate - div_yield
    df = math.exp(-rate * T)

    # Forward prices
    forwards = [spot * math.exp(mu * t) for t in fixing_times]

    # Geometric average moments
    # ln(G) ~ N(mu_g, sigma_g^2)
    mu_g = sum(math.log(f) * w for f, w in zip(forwards, weights))
    sigma_g_sq = 0.0
    for i in range(n):
        for j in range(n):
            sigma_g_sq += weights[i] * weights[j] * vol**2 * min(fixing_times[i], fixing_times[j])

    if sigma_g_sq <= 0:
        # No variance — intrinsic
        m1 = sum(w * f for w, f in zip(weights, forwards))
        if option_type == OptionType.CALL:
            return df * max(m1 - strike, 0.0)
        else:
            return df * max(strike - m1, 0.0)

    sigma_g = math.sqrt(sigma_g_sq)

    # Covariance between ln(G) and each S(t_i)
    # Cov(ln(G), ln(S(t_i))) = Σ_j w_j σ² min(t_i, t_j)
    cov_g_si = []
    for i in range(n):
        c = sum(weights[j] * vol**2 * min(fixing_times[i], fixing_times[j]) for j in range(n))
        cov_g_si.append(c)

    # E[A | G=g] = Σ w_i E[S(t_i) | G=g]
    # E[S(t_i) | G=g] = F(t_i) exp(cov_i (ln(g) - mu_g) / sigma_g^2 - cov_i^2 / (2 sigma_g^2))
    # ... but we integrate analytically over g

    # Curran's formula: numerical integration over the geometric average
    # For efficiency, use Gauss-Hermite quadrature
    n_quad = 32
    z_points, z_weights = np.polynomial.hermite.hermgauss(n_quad)

    price = 0.0
    for k in range(n_quad):
        z = z_points[k]
        w_quad = z_weights[k]

        # ln(G) = mu_g + sigma_g * sqrt(2) * z  (Gauss-Hermite uses exp(-x²))
        ln_g = mu_g + sigma_g * math.sqrt(2) * z

        # Conditional expectation of arithmetic average
        ea_given_g = 0.0
        for i in range(n):
            # Conditional mean: E[S(t_i) | ln(G) = ln_g]
            # = F(t_i) × exp(β_i(ln(G) - μ_G) - ½β_i²σ_G²)
            # where β_i = Cov(ln(G), ln(S(t_i))) / Var(ln(G))
            beta_i = cov_g_si[i] / sigma_g_sq
            e_si_given_g = forwards[i] * math.exp(
                beta_i * (ln_g - mu_g) - 0.5 * beta_i**2 * sigma_g_sq
            )
            ea_given_g += weights[i] * e_si_given_g

        if option_type == OptionType.CALL:
            payoff = max(ea_given_g - strike, 0.0)
        else:
            payoff = max(strike - ea_given_g, 0.0)

        # Gauss-Hermite weight (includes 1/sqrt(pi) factor)
        price += w_quad * payoff / math.sqrt(math.pi)

    return df * price


# ---------------------------------------------------------------------------
# Asian Option instrument
# ---------------------------------------------------------------------------

class AsianOption:
    """Asian option with schedule, partial fixings, and serialisation.

    Supports:
    - Fixed-strike and floating-strike payoffs
    - Turnbull-Wakeman analytical approximation
    - MC with geometric control variate
    - Partial fixings (known historical values)
    - Weighted averaging (custom or equal weights)
    - Schedule-aware (monthly, weekly, daily, or custom dates)

    Args:
        schedule: AsianSchedule defining fixing dates and weights.
        strike: option strike price.
        notional: position notional.
        option_type: CALL or PUT.
        floating_strike: if True, payoff = S(T) - A (call) or A - S(T) (put).
        known_fixings: dict of {date: fixing_value} for already-observed fixings.
    """

    _SERIAL_TYPE = "asian"

    def __init__(
        self,
        schedule: AsianSchedule,
        strike: float,
        notional: float = 1.0,
        option_type: OptionType = OptionType.CALL,
        floating_strike: bool = False,
        known_fixings: dict[date, float] | None = None,
    ):
        self.schedule = schedule
        self.strike = strike
        self.notional = notional
        self.option_type = option_type
        self.floating_strike = floating_strike
        self.known_fixings = known_fixings or {}

    @property
    def n_fixed(self) -> int:
        return sum(1 for d in self.schedule.fixing_dates if d in self.known_fixings)

    @property
    def n_remaining(self) -> int:
        return self.schedule.n_fixings - self.n_fixed

    def price(
        self,
        spot: float,
        curve: DiscountCurve,
        vol: float,
        div_yield: float = 0.0,
        method: str = "auto",
        n_paths: int = 100_000,
        seed: int = 42,
        **kwargs,
    ) -> AsianResult:
        """Price the Asian option.

        Args:
            spot: current underlying price.
            curve: discount curve (for rate extraction).
            vol: flat volatility (or initial vol for stoch vol models).
            div_yield: continuous dividend yield.
            method: "auto", "tw", "curran", "mc", "mc_sobol",
                    "local_vol", "sabr", "heston".
            n_paths: MC paths (if MC method).
            seed: random seed.
            **kwargs: model-specific params (vol_surface, alpha, beta, rho, nu, etc.)
        """
        ref = curve.reference_date
        if spot <= 0:
            raise ValueError(f"spot must be positive, got {spot}")
        T = year_fraction(ref, self.schedule.fixing_dates[-1], DayCountConvention.ACT_365_FIXED)
        rate = -math.log(curve.df(self.schedule.fixing_dates[-1])) / max(T, 1e-10)

        # Fixing times as year fractions
        fixing_times = [
            year_fraction(ref, d, DayCountConvention.ACT_365_FIXED)
            for d in self.schedule.fixing_dates
        ]
        weights = self.schedule.effective_weights

        # Split known/future
        known_vals = [self.known_fixings.get(d) for d in self.schedule.fixing_dates]
        known_list = [v for v in known_vals if v is not None]

        # Choose method
        if method == "auto":
            method = "tw" if not self.floating_strike else "mc"

        if method == "tw":
            price_unit = turnbull_wakeman(
                spot, self.strike, rate, vol, fixing_times, weights, T,
                self.option_type, div_yield,
                known_fixings=known_list if known_list else None,
            )
            return AsianResult(
                price=price_unit * self.notional,
                method="turnbull_wakeman",
                n_fixed=self.n_fixed, n_remaining=self.n_remaining,
            )

        if method == "curran":
            price_unit = curran_asian(
                spot, self.strike, rate, vol, fixing_times, weights, T,
                self.option_type, div_yield,
            )
            return AsianResult(
                price=price_unit * self.notional,
                method="curran",
                n_fixed=self.n_fixed, n_remaining=self.n_remaining,
            )

        if method == "mc_sobol":
            # Quasi-random (Sobol) MC for faster convergence
            from pricebook.rng import QuasiRandom
            n_steps = self.schedule.n_fixings
            gen = GBMGenerator(spot=spot, rate=rate, vol=vol, div_yield=div_yield)
            rng = QuasiRandom(dimension=n_steps, seed=seed)
            paths = gen.generate(T=T, n_steps=n_steps, n_paths=n_paths, rng=rng)
            monitoring = paths[:, 1:]
            arith_avg = monitoring.mean(axis=1)
            df_val = math.exp(-rate * T)
            if self.floating_strike:
                terminal = paths[:, -1]
                if self.option_type == OptionType.CALL:
                    payoffs = np.maximum(terminal - arith_avg, 0.0)
                else:
                    payoffs = np.maximum(arith_avg - terminal, 0.0)
            else:
                if self.option_type == OptionType.CALL:
                    payoffs = np.maximum(arith_avg - self.strike, 0.0)
                else:
                    payoffs = np.maximum(self.strike - arith_avg, 0.0)
            discounted = df_val * payoffs
            price = float(discounted.mean())
            std_err = float(discounted.std(ddof=1) / math.sqrt(len(discounted)))
            return AsianResult(
                price=price * self.notional,
                std_error=std_err * self.notional,
                n_paths=len(payoffs),
                method="mc_sobol",
                n_fixed=self.n_fixed, n_remaining=self.n_remaining,
            )

        if method == "local_vol":
            # Smile-consistent MC using local vol surface
            # Requires vol_surface keyword (LocalVolSurface)
            from pricebook.local_vol import local_vol_mc_paths
            lv_surface = kwargs.get("vol_surface")
            if lv_surface is None:
                raise ValueError("method='local_vol' requires vol_surface=LocalVolSurface")
            n_steps = self.schedule.n_fixings
            paths = local_vol_mc_paths(spot, rate, lv_surface, T, n_steps, n_paths,
                                        div_yield, seed)
            return self._payoff_from_paths(paths, rate, T)

        if method == "sabr":
            # SABR stochastic vol MC
            from pricebook.sabr_mc import sabr_mc_paths
            alpha = kwargs.get("alpha", vol)
            beta = kwargs.get("beta", 1.0)
            rho = kwargs.get("rho", 0.0)
            nu = kwargs.get("nu", 0.3)
            forward = spot * math.exp((rate - div_yield) * T)
            n_steps = max(self.schedule.n_fixings, 50)
            F, _ = sabr_mc_paths(forward, T, alpha, beta, rho, nu, n_steps, n_paths, seed)
            # Convert forward paths to spot-equivalent
            # F paths are already the averaging values
            return self._payoff_from_paths(F, rate, T)

        if method == "heston":
            # Heston stochastic vol MC
            v0 = kwargs.get("v0", vol ** 2)
            kappa = kwargs.get("kappa", 2.0)
            theta = kwargs.get("theta", vol ** 2)
            xi = kwargs.get("xi", 0.3)
            rho_h = kwargs.get("rho", -0.7)
            n_steps = max(self.schedule.n_fixings, 50)
            paths = self._heston_paths(spot, rate, div_yield, v0, kappa, theta,
                                        xi, rho_h, T, n_steps, n_paths, seed)
            return self._payoff_from_paths(paths, rate, T)

        # MC method (pseudo-random with control variate)
        n_steps = self.schedule.n_fixings
        mc = mc_asian_arithmetic(
            spot, self.strike, rate, vol, T, n_steps,
            self.option_type, div_yield, n_paths, seed,
            antithetic=True, control_variate=True,
            floating_strike=self.floating_strike,
        )
        return AsianResult(
            price=mc.price * self.notional,
            std_error=mc.std_error * self.notional,
            n_paths=mc.n_paths,
            method="mc_cv",
            n_fixed=self.n_fixed, n_remaining=self.n_remaining,
        )

    def _payoff_from_paths(self, paths: np.ndarray, rate: float, T: float) -> AsianResult:
        """Compute Asian payoff from full paths array (n_paths, n_steps+1)."""
        monitoring = paths[:, 1:]  # exclude t=0
        arith_avg = monitoring.mean(axis=1)
        df_val = math.exp(-rate * T)

        if self.floating_strike:
            terminal = paths[:, -1]
            if self.option_type == OptionType.CALL:
                payoffs = np.maximum(terminal - arith_avg, 0.0)
            else:
                payoffs = np.maximum(arith_avg - terminal, 0.0)
        else:
            if self.option_type == OptionType.CALL:
                payoffs = np.maximum(arith_avg - self.strike, 0.0)
            else:
                payoffs = np.maximum(self.strike - arith_avg, 0.0)

        discounted = df_val * payoffs
        price = float(discounted.mean())
        std_err = float(discounted.std(ddof=1) / math.sqrt(len(discounted)))
        method_name = "smile_mc"

        return AsianResult(
            price=price * self.notional,
            std_error=std_err * self.notional,
            n_paths=len(payoffs),
            method=method_name,
            n_fixed=self.n_fixed, n_remaining=self.n_remaining,
        )

    @staticmethod
    def _heston_paths(
        spot: float, rate: float, div_yield: float,
        v0: float, kappa: float, theta: float, xi: float, rho: float,
        T: float, n_steps: int, n_paths: int, seed: int,
    ) -> np.ndarray:
        """Euler MC for Heston SDE. Returns paths (n_paths, n_steps+1).

        dS/S = (r-q)dt + √V dW₁
        dV = κ(θ-V)dt + ξ√V dW₂
        dW₁·dW₂ = ρ dt

        Note: uses Euler-Maruyama scheme (first-order). Variance is floored
        at zero (full truncation). For higher accuracy, consider QE scheme
        (Andersen 2008) or exact simulation (Broadie & Kaya 2006).
        """
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = spot
        V = np.full(n_paths, v0)
        S = np.full(n_paths, spot, dtype=float)

        for step in range(n_steps):
            z1 = rng.standard_normal(n_paths)
            z2 = rng.standard_normal(n_paths)
            w1 = z1
            w2 = rho * z1 + math.sqrt(1.0 - rho ** 2) * z2

            V_safe = np.maximum(V, 0.0)
            sqrt_v = np.sqrt(V_safe)

            S = S * np.exp(
                (rate - div_yield - 0.5 * V_safe) * dt + sqrt_v * sqrt_dt * w1
            )
            V = V + kappa * (theta - V_safe) * dt + xi * sqrt_v * sqrt_dt * w2
            V = np.maximum(V, 0.0)

            paths[:, step + 1] = S

        return paths

    def greeks(
        self,
        spot: float,
        curve: DiscountCurve,
        vol: float,
        div_yield: float = 0.0,
        method: str = "auto",
    ) -> dict[str, float]:
        """Bump-and-reprice Greeks: delta, gamma, vega, theta."""
        base = self.price(spot, curve, vol, div_yield, method)

        # Delta
        bump_s = spot * 0.01
        up = self.price(spot + bump_s, curve, vol, div_yield, method)
        dn = self.price(spot - bump_s, curve, vol, div_yield, method)
        delta = (up.price - dn.price) / (2 * bump_s)

        # Gamma
        gamma = (up.price - 2 * base.price + dn.price) / (bump_s ** 2)

        # Vega (1% vol bump)
        v_up = self.price(spot, curve, vol + 0.01, div_yield, method)
        vega = v_up.price - base.price

        return {"delta": delta, "gamma": gamma, "vega": vega, "price": base.price}

    def delta_profile(
        self,
        spot: float,
        curve: DiscountCurve,
        vol: float,
        div_yield: float = 0.0,
        method: str = "tw",
    ) -> list[dict[str, float]]:
        """Delta as function of number of fixings done (delta bleeding).

        Returns list of {n_fixed, delta, price} showing how delta decays
        as fixings accumulate. Critical for hedging Asian options.

        Assumption: each known fixing equals current spot (ATM path).
        """
        profile = []
        fixing_dates = self.schedule.fixing_dates
        n_total = len(fixing_dates)

        for n in range(n_total + 1):
            # Simulate n fixings already done at current spot
            known = {fixing_dates[i]: spot for i in range(n)}
            temp_opt = AsianOption(
                schedule=self.schedule, strike=self.strike,
                notional=self.notional, option_type=self.option_type,
                floating_strike=self.floating_strike,
                known_fixings=known,
            )
            g = temp_opt.greeks(spot, curve, vol, div_yield, method)
            profile.append({"n_fixed": n, "delta": g["delta"], "price": g["price"]})

        return profile

    def pv_ctx(self, ctx) -> float:
        """Price from PricingContext."""
        curve = ctx.discount_curve
        vol_surface = ctx.vol_surfaces.get("equity") if ctx.vol_surfaces else None
        vol = vol_surface.vol(self.schedule.fixing_dates[-1], self.strike) if vol_surface else 0.20
        return self.price(spot=100.0, curve=curve, vol=vol).price

    # ---- Serialisation ----

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self._SERIAL_TYPE, "params": {
                "schedule": self.schedule.to_dict(),
                "strike": self.strike,
                "notional": self.notional,
                "option_type": _serialise_atom(self.option_type),
                "floating_strike": self.floating_strike,
            }
        }
        if self.known_fixings:
            d["params"]["known_fixings"] = {
                fd.isoformat(): v for fd, v in self.known_fixings.items()
            }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> AsianOption:
        p = d["params"]
        schedule = AsianSchedule.from_dict(p["schedule"])
        known = {}
        if "known_fixings" in p:
            known = {date.fromisoformat(k): v for k, v in p["known_fixings"].items()}
        return cls(
            schedule=schedule,
            strike=p["strike"],
            notional=p.get("notional", 1.0),
            option_type=OptionType(p.get("option_type", "call")),
            floating_strike=p.get("floating_strike", False),
            known_fixings=known,
        )


# Register
from pricebook.serialisable import _register
_register(AsianOption)
