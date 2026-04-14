"""CDS swaptions: forward CDS spread, Pedersen model, spread vol.

Options to enter a CDS contract at a given strike spread. The key
subtlety: if default occurs before option expiry, the option either
knocks out or triggers a recovery payment.

* :func:`forward_cds_spread` — par spread of a forward-starting CDS.
* :func:`cds_swaption_black` — Black-76 on forward CDS spread.
* :class:`PedersenCDSSwaption` — full Pedersen (2003) model.
* :func:`cds_swaption_put_call_parity` — payer + receiver = forward CDS.

References:
    Pedersen, *Valuation of Portfolio Credit Default Swaptions*,
    Lehman Brothers QR, 2003.
    Schönbucher, *Credit Derivatives Pricing Models*, Ch. 11.
    O'Kane, *Modelling Single-name and Multi-name Credit Derivatives*,
    Ch. 15 (CDS options).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.black76 import OptionType, black76_price, _norm_cdf


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
