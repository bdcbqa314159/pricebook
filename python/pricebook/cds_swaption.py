"""CDS swaptions: forward CDS spread, Pedersen model, spread vol.

Options to enter a CDS contract at a given strike spread. The key
subtlety: if default occurs before option expiry, the option either
knocks out or triggers a recovery payment.

* :func:`forward_cds_spread` — par spread of a forward-starting CDS (flat).
* :func:`cds_swaption_black` — Black-76 on forward CDS spread.
* :func:`cds_swaption_black_curves` — Black-76 using curve objects.
* :func:`cds_swaption_bachelier` — Bachelier (normal) model.
* :func:`cds_swaption_greeks` — Greeks via finite differences.
* :class:`PedersenCDSSwaption` — full Pedersen (2003) model.
* :class:`CDSSpreadSmile` — SABR smile on CDS spread vol.
* :class:`StochasticIntensitySwaption` — CIR intensity-based pricing.
* :func:`cds_swaption_put_call_parity` — payer + receiver = forward CDS.

References:
    Pedersen, *Valuation of Portfolio Credit Default Swaptions*,
    Lehman Brothers QR, 2003.
    Schönbucher, *Credit Derivatives Pricing Models*, Ch. 11.
    O'Kane, *Modelling Single-name and Multi-name Credit Derivatives*,
    Ch. 15 (CDS options).
    Hagan et al., *Managing Smile Risk*, Wilmott, 2002 (SABR).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from pricebook.black76 import OptionType, black76_price, bachelier_price, _norm_cdf


# ---- Forward CDS spread ----

@dataclass
class ForwardCDSResult:
    """Forward CDS spread and annuity."""
    forward_spread: float
    risky_annuity: float
    protection_pv: float
    survival_to_start: float


def forward_cds_spread(
    start_years: float,
    end_years: float,
    flat_hazard: float,
    flat_rate: float = 0.05,
    recovery: float = 0.4,
    frequency: int = 4,
) -> ForwardCDSResult:
    """Par spread of a forward-starting CDS.

    The forward CDS spread F(t; T₁, T₂) is the spread that makes
    the forward CDS PV = 0 at time 0. Under the survival measure,
    F is a martingale (Pedersen 2003).

    Args:
        start_years: CDS start (option expiry).
        end_years: CDS maturity.
        flat_hazard: constant hazard rate λ.
        flat_rate: risk-free rate r.
        recovery: recovery rate R.
        frequency: premium payment frequency.
    """
    dt = 1.0 / frequency

    # Protection leg PV: ∫_{T1}^{T2} df(t) × (1−R) × (-dQ/dt) dt
    protection = 0.0
    n_prot = int((end_years - start_years) * frequency)
    for i in range(1, n_prot + 1):
        t = start_years + i * dt
        df = math.exp(-flat_rate * t)
        surv = math.exp(-flat_hazard * t)
        surv_prev = math.exp(-flat_hazard * (t - dt))
        protection += df * (1 - recovery) * (surv_prev - surv)

    # Premium leg (risky annuity): Σ df(t_i) × Q(t_i) × dt
    annuity = 0.0
    for i in range(1, n_prot + 1):
        t = start_years + i * dt
        df = math.exp(-flat_rate * t)
        surv = math.exp(-flat_hazard * t)
        annuity += df * surv * dt

    fwd_spread = protection / annuity if annuity > 0 else 0.0
    surv_start = math.exp(-flat_hazard * start_years)

    return ForwardCDSResult(fwd_spread, annuity, protection, surv_start)


# ---- CDS swaption via Black-76 ----

@dataclass
class CDSSwaption:
    """CDS swaption pricing result."""
    premium: float
    forward_spread: float
    strike_spread: float
    risky_annuity: float
    spread_vol: float
    survival_to_expiry: float
    option_type: str  # "payer" or "receiver"


def cds_swaption_black(
    forward_spread: float,
    strike_spread: float,
    spread_vol: float,
    expiry: float,
    risky_annuity: float,
    survival_to_expiry: float,
    notional: float = 1_000_000,
    option_type: str = "payer",
) -> CDSSwaption:
    """CDS swaption price via Black-76 on the forward CDS spread.

    A payer swaption gives the right to buy protection (pay premium).
    A receiver swaption gives the right to sell protection (receive premium).

    Premium = survival(0, T_expiry) × notional × annuity × Black(F, K, σ, T)

    The survival factor accounts for the knockout on default before expiry:
    if default occurs before expiry, the option is worthless.

    Args:
        forward_spread: forward CDS par spread F.
        strike_spread: swaption strike K.
        spread_vol: lognormal vol of the forward spread.
        expiry: option expiry in years.
        risky_annuity: RPV01 of the underlying CDS.
        survival_to_expiry: Q(0, T_expiry).
        option_type: "payer" (buy protection) or "receiver" (sell protection).

    Reference:
        Pedersen (2003), Section 3; O'Kane (2008), Ch. 15.
    """
    if spread_vol <= 0 or expiry <= 0:
        # Intrinsic value
        if option_type == "payer":
            intrinsic = max(forward_spread - strike_spread, 0.0)
        else:
            intrinsic = max(strike_spread - forward_spread, 0.0)
        prem = survival_to_expiry * notional * risky_annuity * intrinsic
        return CDSSwaption(prem, forward_spread, strike_spread,
                           risky_annuity, spread_vol, survival_to_expiry,
                           option_type)

    opt = OptionType.CALL if option_type == "payer" else OptionType.PUT
    # Black-76 with df=1 (annuity handles discounting)
    black = black76_price(forward_spread, strike_spread, spread_vol,
                          expiry, df=1.0, option_type=opt)

    premium = survival_to_expiry * notional * risky_annuity * black

    return CDSSwaption(premium, forward_spread, strike_spread,
                       risky_annuity, spread_vol, survival_to_expiry,
                       option_type)


# ---- Pedersen model ----

@dataclass
class PedersenResult:
    """Full Pedersen CDS swaption result."""
    premium: float
    forward_spread: float
    strike_spread: float
    spread_vol: float
    survival_to_expiry: float
    mc_premium: float | None


class PedersenCDSSwaption:
    """Pedersen (2003) model for CDS swaptions.

    Models the forward CDS spread as a martingale under the survival
    measure. The spread follows lognormal dynamics:

        dF/F = σ dW  (under the annuity measure, conditional on survival)

    The swaption price is:
        V = Q(0,T) × A(0; T₁,T₂) × E[(F(T)−K)⁺]
        = Q(0,T) × A × Black(F₀, K, σ, T)

    where the expectation is under the T₁-survival annuity measure.

    Args:
        flat_hazard: hazard rate for survival and forward spread.
        flat_rate: risk-free rate.
        recovery: recovery rate.
        spread_vol: lognormal volatility of the forward spread.
    """

    def __init__(
        self,
        flat_hazard: float = 0.02,
        flat_rate: float = 0.05,
        recovery: float = 0.4,
        spread_vol: float = 0.40,
    ):
        self.flat_hazard = flat_hazard
        self.flat_rate = flat_rate
        self.recovery = recovery
        self.spread_vol = spread_vol

    def price(
        self,
        expiry: float,
        cds_maturity: float,
        strike_spread: float,
        notional: float = 1_000_000,
        option_type: str = "payer",
    ) -> PedersenResult:
        """Price a CDS swaption via Black-76 on the forward spread."""
        fwd = forward_cds_spread(
            expiry, cds_maturity, self.flat_hazard,
            self.flat_rate, self.recovery,
        )

        result = cds_swaption_black(
            fwd.forward_spread, strike_spread, self.spread_vol,
            expiry, fwd.risky_annuity, fwd.survival_to_start,
            notional, option_type,
        )

        return PedersenResult(
            result.premium, fwd.forward_spread, strike_spread,
            self.spread_vol, fwd.survival_to_start, None,
        )

    def price_curves(
        self,
        discount_curve,
        survival_curve,
        expiry_date: date,
        maturity_date: date,
        strike_spread: float,
        notional: float = 1_000_000,
        option_type: str = "payer",
        recovery: float = 0.4,
    ) -> PedersenResult:
        """Price using curve objects instead of flat parameters."""
        return _pedersen_price_curves(
            discount_curve, survival_curve,
            expiry_date, maturity_date,
            strike_spread, self.spread_vol,
            notional, option_type, recovery,
        )

    def price_mc(
        self,
        expiry: float,
        cds_maturity: float,
        strike_spread: float,
        notional: float = 1_000_000,
        option_type: str = "payer",
        n_paths: int = 100_000,
        seed: int | None = None,
    ) -> PedersenResult:
        """Price via MC simulation of the forward spread."""
        rng = np.random.default_rng(seed)

        fwd = forward_cds_spread(
            expiry, cds_maturity, self.flat_hazard,
            self.flat_rate, self.recovery,
        )
        F0 = fwd.forward_spread
        surv = fwd.survival_to_start

        # Simulate F(T) under lognormal dynamics
        z = rng.standard_normal(n_paths)
        F_T = F0 * np.exp(
            -0.5 * self.spread_vol**2 * expiry
            + self.spread_vol * math.sqrt(expiry) * z
        )

        if option_type == "payer":
            payoff = np.maximum(F_T - strike_spread, 0.0)
        else:
            payoff = np.maximum(strike_spread - F_T, 0.0)

        mc_premium = float(
            surv * notional * fwd.risky_annuity * payoff.mean()
        )

        return PedersenResult(
            mc_premium, F0, strike_spread, self.spread_vol,
            surv, mc_premium,
        )


# ---- Put-call parity ----

@dataclass
class PutCallParityResult:
    """CDS swaption put-call parity check."""
    payer_premium: float
    receiver_premium: float
    forward_cds_pv: float
    parity_error: float
    holds: bool


def cds_swaption_put_call_parity(
    forward_spread: float,
    strike_spread: float,
    spread_vol: float,
    expiry: float,
    risky_annuity: float,
    survival_to_expiry: float,
    notional: float = 1_000_000,
    tol: float = 0.01,
) -> PutCallParityResult:
    """Verify CDS swaption put-call parity.

    Payer − Receiver = Q(0,T) × A × (F − K)

    This is the analog of call − put = forward − strike for vanilla options.
    """
    payer = cds_swaption_black(
        forward_spread, strike_spread, spread_vol, expiry,
        risky_annuity, survival_to_expiry, notional, "payer",
    )
    receiver = cds_swaption_black(
        forward_spread, strike_spread, spread_vol, expiry,
        risky_annuity, survival_to_expiry, notional, "receiver",
    )

    forward_pv = survival_to_expiry * notional * risky_annuity * (
        forward_spread - strike_spread
    )

    error = abs((payer.premium - receiver.premium) - forward_pv)
    rel_error = error / max(abs(forward_pv), 1.0)
    holds = rel_error < tol

    return PutCallParityResult(
        payer.premium, receiver.premium, forward_pv, rel_error, holds,
    )


# ---- Curve-based CDS swaption pricing ----

def cds_swaption_black_curves(
    discount_curve,
    survival_curve,
    expiry_date: date,
    maturity_date: date,
    strike_spread: float,
    spread_vol: float,
    notional: float = 1_000_000,
    option_type: str = "payer",
    recovery: float = 0.4,
) -> CDSSwaption:
    """CDS swaption via Black-76 using curve objects.

    Computes the forward CDS spread and risky annuity from actual
    discount and survival curves, then delegates to cds_swaption_black().

    Args:
        discount_curve: OIS discount curve.
        survival_curve: credit survival curve.
        expiry_date: swaption expiry.
        maturity_date: underlying CDS maturity.
        strike_spread: swaption strike K.
        spread_vol: lognormal vol of the forward spread.
        option_type: "payer" or "receiver".
        recovery: recovery rate for protection leg.
    """
    from pricebook.cds import forward_cds_par_spread
    from pricebook.day_count import DayCountConvention, year_fraction

    fwd = forward_cds_par_spread(
        discount_curve, survival_curve, expiry_date, maturity_date,
        recovery=recovery,
    )

    expiry = year_fraction(
        discount_curve.reference_date, expiry_date,
        DayCountConvention.ACT_365_FIXED,
    )

    return cds_swaption_black(
        fwd.forward_spread, strike_spread, spread_vol,
        expiry, fwd.risky_annuity, fwd.survival_to_start,
        notional, option_type,
    )


# ---- CDS swaption Greeks ----

@dataclass
class CDSSwaptionGreeks:
    """Greeks for a CDS swaption."""
    delta: float          # dV/dF per 1bp
    gamma: float          # d²V/dF² per 1bp²
    vega: float           # dV/dσ per 1% vol
    theta: float          # dV/dT per day
    knockout_delta: float  # dV/dQ per 1%

    def to_dict(self) -> dict:
        return {
            "delta": self.delta, "gamma": self.gamma,
            "vega": self.vega, "theta": self.theta,
            "knockout_delta": self.knockout_delta,
        }


def cds_swaption_greeks(
    forward_spread: float,
    strike_spread: float,
    spread_vol: float,
    expiry: float,
    risky_annuity: float,
    survival_to_expiry: float,
    notional: float = 1_000_000,
    option_type: str = "payer",
) -> CDSSwaptionGreeks:
    """CDS swaption Greeks via finite differences.

    All sensitivities computed by bumping one input and repricing.
    """
    base = cds_swaption_black(
        forward_spread, strike_spread, spread_vol,
        expiry, risky_annuity, survival_to_expiry,
        notional, option_type,
    ).premium

    # Delta: dV/dF per 1bp
    h_f = 0.0001
    up = cds_swaption_black(
        forward_spread + h_f, strike_spread, spread_vol,
        expiry, risky_annuity, survival_to_expiry,
        notional, option_type,
    ).premium
    down = cds_swaption_black(
        forward_spread - h_f, strike_spread, spread_vol,
        expiry, risky_annuity, survival_to_expiry,
        notional, option_type,
    ).premium
    delta = (up - down) / (2 * h_f)

    # Gamma: d²V/dF²
    gamma = (up - 2 * base + down) / (h_f ** 2)

    # Vega: dV/dσ per 1% vol (central difference)
    h_v = 0.01
    vega_up = cds_swaption_black(
        forward_spread, strike_spread, spread_vol + h_v,
        expiry, risky_annuity, survival_to_expiry,
        notional, option_type,
    ).premium
    vega_down = cds_swaption_black(
        forward_spread, strike_spread, max(spread_vol - h_v, 0.001),
        expiry, risky_annuity, survival_to_expiry,
        notional, option_type,
    ).premium
    vega = (vega_up - vega_down) / (2 * h_v)

    # Theta: time decay per day (positive = loses value)
    h_t = 1.0 / 365.0
    if expiry > h_t:
        theta_val = cds_swaption_black(
            forward_spread, strike_spread, spread_vol,
            expiry - h_t, risky_annuity, survival_to_expiry,
            notional, option_type,
        ).premium
        theta = (base - theta_val) / h_t
    else:
        theta = 0.0

    # Knockout delta: dV/dQ per 1%
    h_q = 0.01
    ko_up = cds_swaption_black(
        forward_spread, strike_spread, spread_vol,
        expiry, risky_annuity, survival_to_expiry + h_q,
        notional, option_type,
    ).premium
    knockout_delta = (ko_up - base) / h_q

    return CDSSwaptionGreeks(delta, gamma, vega, theta, knockout_delta)


# ---- Bachelier (normal) CDS swaption ----

def cds_swaption_bachelier(
    forward_spread: float,
    strike_spread: float,
    normal_vol: float,
    expiry: float,
    risky_annuity: float,
    survival_to_expiry: float,
    notional: float = 1_000_000,
    option_type: str = "payer",
) -> CDSSwaption:
    """CDS swaption via Bachelier (normal) model.

    For near-zero forward spreads where lognormal Black-76 fails.
    Uses absolute (normal) volatility.

    Premium = Q(0,T) × notional × A × Bachelier(F, K, σ_N, T)
    """
    if normal_vol <= 0 or expiry <= 0:
        if option_type == "payer":
            intrinsic = max(forward_spread - strike_spread, 0.0)
        else:
            intrinsic = max(strike_spread - forward_spread, 0.0)
        prem = survival_to_expiry * notional * risky_annuity * intrinsic
        return CDSSwaption(prem, forward_spread, strike_spread,
                           risky_annuity, normal_vol, survival_to_expiry,
                           option_type)

    opt = OptionType.CALL if option_type == "payer" else OptionType.PUT
    bach = bachelier_price(forward_spread, strike_spread, normal_vol,
                           expiry, df=1.0, option_type=opt)
    premium = survival_to_expiry * notional * risky_annuity * bach

    return CDSSwaption(premium, forward_spread, strike_spread,
                       risky_annuity, normal_vol, survival_to_expiry,
                       option_type)


# ---- SABR spread smile ----

class CDSSpreadSmile:
    """SABR-calibrated spread vol smile for CDS swaptions.

    Given a forward CDS spread and SABR parameters, provides
    implied vol for any strike via Hagan's approximation.

    Args:
        forward: forward CDS spread.
        alpha: SABR vol level.
        beta: CEV exponent (typically 0.5 for CDS).
        rho: correlation between spread and vol.
        nu: vol of vol.
    """

    def __init__(
        self,
        forward: float,
        alpha: float = 0.4,
        beta: float = 0.5,
        rho: float = -0.3,
        nu: float = 0.4,
    ):
        self.forward = forward
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu

    def implied_vol(self, strike: float, T: float = 1.0) -> float:
        """SABR implied Black vol for given strike and expiry."""
        from pricebook.sabr import sabr_implied_vol
        return sabr_implied_vol(
            self.forward, strike, T,
            self.alpha, self.beta, self.rho, self.nu,
        )

    def smile(
        self,
        strikes: list[float],
        T: float = 1.0,
    ) -> list[float]:
        """Compute vol smile across multiple strikes."""
        return [self.implied_vol(k, T) for k in strikes]

    @classmethod
    def calibrate(
        cls,
        forward: float,
        strikes: list[float],
        market_vols: list[float],
        T: float = 1.0,
        beta: float = 0.5,
    ) -> CDSSpreadSmile:
        """Calibrate SABR parameters from market vol quotes.

        Fixes beta and calibrates (alpha, rho, nu).
        """
        from pricebook.sabr import sabr_implied_vol
        from pricebook.optimization import minimize

        def objective(params):
            alpha, rho, nu = params
            if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
                return 1e10
            total = 0.0
            for k, mv in zip(strikes, market_vols):
                try:
                    model_vol = sabr_implied_vol(forward, k, T, alpha, beta, rho, nu)
                    total += (model_vol - mv) ** 2
                except (ValueError, ZeroDivisionError):
                    return 1e10
            return total

        result = minimize(objective, x0=[0.3, -0.2, 0.3], method="nelder_mead")
        alpha, rho, nu = result.x
        return cls(forward, alpha, beta, rho, nu)

    def to_dict(self) -> dict:
        return {
            "forward": self.forward, "alpha": self.alpha,
            "beta": self.beta, "rho": self.rho, "nu": self.nu,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CDSSpreadSmile:
        return cls(
            forward=d["forward"], alpha=d["alpha"],
            beta=d["beta"], rho=d["rho"], nu=d["nu"],
        )


# ---- Stochastic intensity CDS swaption ----

class StochasticIntensitySwaption:
    """CDS swaption under CIR stochastic intensity via MC.

    Simulates CIR intensity paths, computes the forward CDS spread
    on each path at expiry, then averages the swaption payoff.
    Captures correlation between default probability and spread vol
    that Black-76 misses.

    As ξ → 0 (deterministic intensity), converges to Black-76.

    Args:
        kappa: CIR mean reversion speed.
        theta: CIR long-run intensity.
        xi: CIR vol of intensity (vol of vol for spreads).
        flat_rate: risk-free rate for discounting.
        recovery: recovery rate.
    """

    def __init__(
        self,
        kappa: float = 1.0,
        theta: float = 0.02,
        xi: float = 0.1,
        flat_rate: float = 0.05,
        recovery: float = 0.4,
    ):
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.flat_rate = flat_rate
        self.recovery = recovery

    def price(
        self,
        expiry: float,
        cds_maturity: float,
        strike_spread: float,
        notional: float = 1_000_000,
        option_type: str = "payer",
        n_paths: int = 50_000,
        n_steps: int = 100,
        seed: int = 42,
    ) -> PedersenResult:
        """Price via MC simulation of CIR intensity paths.

        1. Simulate λ(t) from 0 to T_expiry.
        2. At expiry: compute forward spread from simulated λ.
        3. Payoff = max(F − K, 0) for payer, knockout if defaulted.
        """
        from pricebook.stochastic_credit import CIRIntensity

        cir = CIRIntensity(self.kappa, self.theta, self.xi)
        lam0 = self.theta

        # Simulate intensity paths to expiry
        paths = cir.simulate_intensity(lam0, expiry, n_steps, n_paths, seed)
        dt = expiry / n_steps

        # Survival to expiry: exp(-∫λ ds) using trapezoidal rule
        integral_to_expiry = (paths[:, :-1] + paths[:, 1:]).sum(axis=1) * dt / 2
        survival = np.exp(-integral_to_expiry)

        # Forward spread: (1-R) × average λ over remaining CDS period
        # For flat λ_T this equals (1-R)×λ_T; for term structure it's
        # the integrated forward hazard
        lam_T = np.maximum(paths[:, -1], 0.0)
        forward_spreads = (1 - self.recovery) * lam_T

        # Payoff
        if option_type == "payer":
            payoff = np.maximum(forward_spreads - strike_spread, 0.0)
        else:
            payoff = np.maximum(strike_spread - forward_spreads, 0.0)

        # Risky annuity approximation: simple annuity at flat rate
        remaining = cds_maturity - expiry
        n_periods = max(1, int(remaining * 4))
        ann = sum(
            math.exp(-self.flat_rate * (expiry + (i + 1) * remaining / n_periods))
            * remaining / n_periods
            for i in range(n_periods)
        )

        # Premium: E[survival × payoff × annuity × notional]
        mc_premium = float((survival * payoff).mean()) * notional * ann

        # Mean forward spread for reporting
        mean_fwd = float(forward_spreads.mean())
        mean_surv = float(survival.mean())

        return PedersenResult(
            mc_premium, mean_fwd, strike_spread,
            self.xi, mean_surv, mc_premium,
        )

    def to_dict(self) -> dict:
        return {
            "kappa": self.kappa, "theta": self.theta,
            "xi": self.xi, "flat_rate": self.flat_rate,
            "recovery": self.recovery,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StochasticIntensitySwaption:
        return cls(
            kappa=d["kappa"], theta=d["theta"],
            xi=d["xi"], flat_rate=d.get("flat_rate", 0.05),
            recovery=d.get("recovery", 0.4),
        )


# ---- Exercise into physical CDS ----

@dataclass
class ExerciseResult:
    """Result of exercising a CDS swaption into a physical CDS."""
    exercise_pv: float
    swaption_premium: float
    total_pnl: float
    exercised: bool

    def to_dict(self) -> dict:
        return {
            "exercise_pv": self.exercise_pv,
            "swaption_premium": self.swaption_premium,
            "total_pnl": self.total_pnl,
            "exercised": self.exercised,
        }


def exercise_into_physical(
    forward_spread: float,
    strike_spread: float,
    swaption_premium: float,
    risky_annuity: float,
    survival_to_expiry: float,
    notional: float = 1_000_000,
    option_type: str = "payer",
) -> ExerciseResult:
    """Compute P&L from exercising a CDS swaption into a physical CDS.

    Payer: exercise if F > K (enter CDS at strike, market at forward).
    Receiver: exercise if K > F.

    exercise_pv = (F − K) × A × notional × Q for payer.
    total_pnl = exercise_pv − swaption_premium.
    """
    if option_type == "payer":
        exercised = forward_spread > strike_spread
        exercise_pv = max(forward_spread - strike_spread, 0.0) * \
            risky_annuity * notional * survival_to_expiry
    else:
        exercised = strike_spread > forward_spread
        exercise_pv = max(strike_spread - forward_spread, 0.0) * \
            risky_annuity * notional * survival_to_expiry

    total_pnl = exercise_pv - swaption_premium

    return ExerciseResult(exercise_pv, swaption_premium, total_pnl, exercised)


# ---- Curve-based PedersenCDSSwaption extension ----

def _pedersen_price_curves(
    discount_curve,
    survival_curve,
    expiry_date: date,
    maturity_date: date,
    strike_spread: float,
    spread_vol: float,
    notional: float = 1_000_000,
    option_type: str = "payer",
    recovery: float = 0.4,
) -> PedersenResult:
    """Pedersen model price using curve objects.

    Computes forward spread from curves, then applies Black-76.
    """
    result = cds_swaption_black_curves(
        discount_curve, survival_curve,
        expiry_date, maturity_date,
        strike_spread, spread_vol,
        notional, option_type, recovery,
    )

    return PedersenResult(
        result.premium, result.forward_spread, strike_spread,
        spread_vol, result.survival_to_expiry, None,
    )
