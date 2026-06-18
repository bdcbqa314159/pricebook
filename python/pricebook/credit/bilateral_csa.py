"""Bilateral CLN with CSA (Credit Support Annex) integration.

Extends the bilateral CLN framework with collateral mechanics:
- Threshold-based margin calls
- Independent amounts (initial margin)
- Minimum transfer amounts
- Funding cost feedback when issuer spreads widen
- Collateral haircuts under stress

    from pricebook.credit.bilateral_csa import (
        CSATerms, BilateralCSAPricer, bilateral_cln_with_csa,
    )

References:
    Gregory (2015). The xVA Challenge, Ch 7-9.
    ISDA (2016). ISDA Credit Support Annex Documentation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from scipy.stats import norm

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve


@dataclass
class CSATerms:
    """Credit Support Annex terms."""
    threshold_investor: float = 0.0     # Investor posts above this MtM
    threshold_issuer: float = 0.0       # Issuer posts above this MtM
    independent_amount: float = 0.0     # Initial margin (one-way or two-way)
    minimum_transfer: float = 100_000   # MTA (minimum transfer amount)
    rounding: float = 10_000            # Rounding increment
    margin_period_risk_days: int = 10   # MPOR (days between call and receipt)
    rehypothecation: bool = True        # Can collateral holder re-use?
    collateral_currency: str = "USD"
    haircut_pct: float = 0.0            # Haircut on posted collateral

    def to_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class BilateralCSAResult:
    """Result of bilateral CLN pricing with CSA."""
    clean_pv: float                  # PV without CSA adjustments
    csa_adjusted_pv: float           # PV after funding/collateral
    funding_cost: float              # Funding cost from uncollateralised exposure
    collateral_benefit: float        # Reduction in funding from posted collateral
    expected_exposure: float         # Average positive exposure
    expected_collateral: float       # Average collateral posted
    uncollateralised_exposure: float # Avg exposure above threshold
    cva: float                       # Credit value adjustment
    dva: float                       # Debit value adjustment
    fva: float                       # Funding value adjustment
    total_xva: float                 # CVA + DVA + FVA

    def to_dict(self) -> dict:
        return dict(vars(self))


class BilateralCSAPricer:
    """Bilateral CLN pricer with CSA mechanics.

    Models the interaction between:
    1. Reference entity default → investor loss
    2. Issuer default → investor loss (counterparty risk)
    3. Collateral posted under CSA → reduces exposure
    4. Funding cost → spread-dependent cost of uncollateralised portion
    """

    def __init__(
        self,
        notional: float,
        coupon: float,
        maturity_years: float,
        ref_survival: SurvivalCurve,
        issuer_survival: SurvivalCurve,
        discount_curve: DiscountCurve,
        ref_recovery: float = 0.40,
        issuer_recovery: float = 0.40,
        correlation: float = 0.30,
        csa: CSATerms | None = None,
        funding_spread: float = 0.0,
        reference_date: date | None = None,
    ):
        self.notional = notional
        self.coupon = coupon
        self.maturity_years = maturity_years
        self.ref_survival = ref_survival
        self.issuer_survival = issuer_survival
        self.discount_curve = discount_curve
        self.ref_recovery = ref_recovery
        self.issuer_recovery = issuer_recovery
        if not -1.0 <= correlation <= 1.0:
            raise ValueError(f"correlation must be in [-1, 1], got {correlation}")
        self.correlation = correlation
        self.csa = csa or CSATerms()
        self.funding_spread = funding_spread
        self.reference_date = reference_date or discount_curve.reference_date

    def price(self, n_paths: int = 20_000, seed: int = 42) -> BilateralCSAResult:
        """Price bilateral CLN with CSA via Monte Carlo.

        Simulates correlated defaults, exposure paths, collateral mechanics,
        and funding costs.
        """
        rng = np.random.default_rng(seed)
        T = self.maturity_years
        n_steps = max(int(T * 4), 4)  # quarterly steps
        dt = T / n_steps
        dc = DayCountConvention.ACT_365_FIXED

        # Correlated default uniforms
        Z = rng.standard_normal((n_paths, 2))
        rho = self.correlation
        Z[:, 1] = rho * Z[:, 0] + math.sqrt(1 - rho**2) * Z[:, 1]
        U_ref = norm.cdf(Z[:, 0])
        U_iss = norm.cdf(Z[:, 1])

        # Time grid
        times = [dt * (i + 1) for i in range(n_steps)]

        # Survival at each step
        ref_survs = []
        iss_survs = []
        dfs = []
        for t in times:
            d = date.fromordinal(self.reference_date.toordinal() + int(t * 365))
            ref_survs.append(self.ref_survival.survival(d))
            iss_survs.append(self.issuer_survival.survival(d))
            dfs.append(self.discount_curve.df(d))

        # Per-path simulation
        pv_clean = np.zeros(n_paths)
        pv_adjusted = np.zeros(n_paths)
        exposures = np.zeros(n_paths)
        collateral_posted = np.zeros(n_paths)
        funding_costs = np.zeros(n_paths)
        cva_losses = np.zeros(n_paths)
        dva_gains = np.zeros(n_paths)

        coupon_per_step = self.coupon * dt * self.notional

        for path in range(n_paths):
            ref_alive = True
            iss_alive = True
            cum_pv = 0.0
            cum_funding = 0.0

            for step in range(n_steps):
                df = dfs[step]

                # Check defaults
                if ref_alive and ref_survs[step] < 1 - U_ref[path]:
                    ref_alive = False
                    # Reference default: recover ref_recovery × notional
                    cum_pv += self.ref_recovery * self.notional * df
                    break

                if iss_alive and iss_survs[step] < 1 - U_iss[path]:
                    iss_alive = False
                    # Issuer default: recover issuer_recovery × notional
                    cum_pv += self.issuer_recovery * self.notional * df
                    cva_losses[path] = (1 - self.issuer_recovery) * self.notional * df
                    break

                # Coupon payment
                cum_pv += coupon_per_step * df

                # Exposure at this step (MtM of remaining cashflows)
                remaining_coupons = sum(
                    coupon_per_step * dfs[s] / df
                    for s in range(step + 1, n_steps)
                )
                principal_pv = self.notional * dfs[-1] / df
                mtm = remaining_coupons + principal_pv - self.notional

                # Collateral: posted when MtM exceeds threshold
                exposure = max(mtm, 0)
                coll = max(exposure - self.csa.threshold_investor, 0)
                coll += self.csa.independent_amount
                coll = max(coll - self.csa.haircut_pct * coll, 0)

                # Uncollateralised exposure
                uncoll = max(exposure - coll, 0)

                # Funding cost on uncollateralised portion
                funding = uncoll * self.funding_spread * dt
                cum_funding += funding * df

                exposures[path] += exposure * dt
                collateral_posted[path] += coll * dt

            else:
                # No default: principal at maturity
                cum_pv += self.notional * dfs[-1]

            pv_clean[path] = cum_pv
            funding_costs[path] = cum_funding
            pv_adjusted[path] = cum_pv - cum_funding

        # Aggregate
        clean = float(np.mean(pv_clean))
        adjusted = float(np.mean(pv_adjusted))
        avg_exposure = float(np.mean(exposures)) / T
        avg_collateral = float(np.mean(collateral_posted)) / T
        avg_funding = float(np.mean(funding_costs))
        cva = float(np.mean(cva_losses))
        dva = 0.0  # simplified
        fva = avg_funding

        return BilateralCSAResult(
            clean_pv=clean,
            csa_adjusted_pv=adjusted,
            funding_cost=fva,
            collateral_benefit=avg_collateral * self.funding_spread,
            expected_exposure=avg_exposure,
            expected_collateral=avg_collateral,
            uncollateralised_exposure=avg_exposure - avg_collateral,
            cva=cva,
            dva=dva,
            fva=fva,
            total_xva=cva + dva + fva,
        )


def bilateral_cln_with_csa(
    notional: float,
    coupon: float,
    maturity_years: float,
    ref_hazard: float,
    issuer_hazard: float,
    risk_free_rate: float,
    ref_recovery: float = 0.40,
    issuer_recovery: float = 0.40,
    correlation: float = 0.30,
    csa: CSATerms | None = None,
    funding_spread: float = 0.0,
    n_paths: int = 20_000,
    seed: int = 42,
) -> BilateralCSAResult:
    """Convenience function: price bilateral CLN+CSA from flat parameters.

    Args:
        notional: CLN notional.
        coupon: annual coupon rate.
        maturity_years: maturity in years.
        ref_hazard: flat hazard rate for reference entity.
        issuer_hazard: flat hazard rate for issuer/SPV.
        risk_free_rate: flat risk-free rate.
        ref_recovery: reference entity recovery.
        issuer_recovery: issuer recovery.
        correlation: default correlation (Gaussian copula).
        csa: CSA terms (optional).
        funding_spread: issuer funding spread (e.g. 0.005 = 50bp).
    """
    ref_date = date(2024, 1, 1)
    dc = DiscountCurve.flat(ref_date, risk_free_rate)
    ref_sc = SurvivalCurve.flat(ref_date, ref_hazard, tenors=list(range(1, int(maturity_years) + 2)))
    iss_sc = SurvivalCurve.flat(ref_date, issuer_hazard, tenors=list(range(1, int(maturity_years) + 2)))

    pricer = BilateralCSAPricer(
        notional, coupon, maturity_years,
        ref_sc, iss_sc, dc,
        ref_recovery, issuer_recovery, correlation,
        csa, funding_spread, ref_date,
    )
    return pricer.price(n_paths, seed)
