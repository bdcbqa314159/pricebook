"""Constant Proportion Debt Obligation (CPDO) pricing and simulation.

* :func:`cpdo_simulate`      — single-path NAV simulation.
* :func:`cpdo_monte_carlo`   — Monte Carlo distribution over many paths.
* :func:`cpdo_rating`        — map default probability to an approximate rating.

CPDO mechanics
--------------
A CPDO issues a note at par and targets a fixed coupon above Libor.  The
vehicle invests in credit (typically index CDS) with leverage that is reset
each period so that the *shortfall* between current NAV and the present value
of outstanding coupons is covered by spread income.  As NAV rises toward the
target the leverage shrinks (``cash-in``) and as NAV falls the leverage grows
(``catch-up`''), potentially breaching the maximum leverage cap and defaulting
(``gap event``).

The key difference from CPPI: CPPI *reduces* risk as the cushion shrinks;
CPDO *increases* risk as the shortfall grows — hence the ``suicide'' leverage
label in the literature.

References
----------
Bentata, A. and Cont, R. (2012). Short-time asymptotics for marginal
    distributions of semimartingales. *arXiv:0807.1706*.
Torresetti, R. and Pallavicini, A. (2007). Stressing Rating Criteria Allowing
    for Default Clustering: the CPDO Case. *SSRN 1015382*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CPDOResult:
    """Single-path CPDO simulation result."""

    final_nav: float
    coupon_paid: float
    defaulted: bool
    leverage_path: np.ndarray
    nav_path: np.ndarray
    target_nav: float

    def to_dict(self) -> dict:
        return {
            "final_nav": self.final_nav,
            "coupon_paid": self.coupon_paid,
            "defaulted": self.defaulted,
            "target_nav": self.target_nav,
            "n_periods": len(self.nav_path),
        }


@dataclass
class CPDOMCResult:
    """Monte Carlo summary for a CPDO structure."""

    success_prob: float
    expected_nav: float
    expected_loss: float
    avg_leverage: float
    default_prob: float
    rating_implied: str

    def to_dict(self) -> dict:
        return vars(self)


# ---------------------------------------------------------------------------
# 1. Single-path simulation
# ---------------------------------------------------------------------------

def cpdo_simulate(
    initial_nav: float,
    target_coupon: float,
    spread_income: float,
    spread_vol: float,
    max_leverage: float,
    n_periods: int,
    gap_risk: float = 0.02,
    seed: int = 42,
) -> CPDOResult:
    """Simulate a single CPDO NAV path.

    Each period the vehicle earns ``spread_income × leverage × notional``
    minus a random spread shock.  Leverage is set each period as::

        shortfall = target_nav - nav
        leverage  = min(shortfall / (spread_income × dt × notional), max_leverage)

    where ``target_nav`` is the PV of all remaining coupons.

    A *cash-in* event occurs when ``nav >= target_nav`` (success).
    A *default* event occurs when ``nav <= 0`` (gap event).

    Args:
        initial_nav:    starting NAV (e.g. 100).
        target_coupon:  annualised coupon rate above funding (e.g. 0.01 = 100bp).
        spread_income:  expected credit spread earned each period (annualised).
        spread_vol:     volatility of spread income shocks (annualised).
        max_leverage:   cap on gross leverage (e.g. 15).
        n_periods:      number of simulation steps (e.g. 252 daily).
        gap_risk:       additional jump-to-default probability per period.
        seed:           random seed for reproducibility.

    Returns:
        :class:`CPDOResult` with full path history.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / n_periods
    notional = initial_nav

    nav_path = np.empty(n_periods + 1)
    leverage_path = np.empty(n_periods + 1)

    nav = float(initial_nav)
    nav_path[0] = nav

    # Target NAV = PV of remaining coupon stream (simplified flat discounting)
    # At each step, target shrinks as coupons are paid.
    coupon_paid = 0.0
    defaulted = False

    for t in range(n_periods):
        periods_remaining = n_periods - t
        # PV of remaining coupons at spread_income ~ proxy for target
        target_nav = notional + notional * target_coupon * periods_remaining * dt

        shortfall = max(target_nav - nav, 0.0)

        # Leverage to close shortfall with one period of spread income
        denom = spread_income * dt * notional
        leverage = min(shortfall / denom, max_leverage) if denom > 0 else max_leverage

        leverage_path[t] = leverage

        # Spread income this period
        z = rng.standard_normal()
        spread_return = (spread_income + spread_vol * z / math.sqrt(1.0 / dt)) * dt
        income = leverage * notional * spread_return

        # Gap-risk: sudden credit blow-up wipes leveraged notional
        gap_event = rng.random() < gap_risk * dt
        if gap_event:
            nav = 0.0
            leverage_path[t + 1] = leverage
            nav_path[t + 1] = nav
            defaulted = True
            coupon_paid += notional * target_coupon * dt
            break

        nav = nav + income
        coupon_paid += notional * target_coupon * dt

        if nav >= target_nav:
            # Cash-in: success
            nav_path[t + 1] = nav
            leverage_path[t + 1] = 0.0
            for remaining in range(t + 2, n_periods + 1):
                nav_path[remaining] = nav
                leverage_path[remaining] = 0.0
            return CPDOResult(
                final_nav=nav,
                coupon_paid=coupon_paid,
                defaulted=False,
                leverage_path=leverage_path,
                nav_path=nav_path,
                target_nav=target_nav,
            )

        if nav <= 0.0:
            nav = 0.0
            nav_path[t + 1] = nav
            leverage_path[t + 1] = 0.0
            defaulted = True
            for remaining in range(t + 2, n_periods + 1):
                nav_path[remaining] = 0.0
                leverage_path[remaining] = 0.0
            break

        nav_path[t + 1] = nav

    leverage_path[n_periods] = 0.0

    return CPDOResult(
        final_nav=max(nav, 0.0),
        coupon_paid=coupon_paid,
        defaulted=defaulted,
        leverage_path=leverage_path,
        nav_path=nav_path,
        target_nav=notional,
    )


# ---------------------------------------------------------------------------
# 2. Monte Carlo
# ---------------------------------------------------------------------------

def cpdo_monte_carlo(
    initial_nav: float,
    target_coupon: float,
    spread_income: float,
    spread_vol: float,
    max_leverage: float,
    n_periods: int,
    gap_risk: float = 0.02,
    n_paths: int = 10_000,
    seed: int = 42,
) -> CPDOMCResult:
    """Monte Carlo distribution of CPDO outcomes.

    Runs :func:`cpdo_simulate` over ``n_paths`` independent paths using
    a vectorised RNG stream (each path gets a unique sub-seed derived from
    the master seed to maintain reproducibility).

    Args:
        n_paths:  number of Monte Carlo paths (default 10 000).
        seed:     master seed; path seeds are ``seed + i``.

    Returns:
        :class:`CPDOMCResult` with aggregate statistics and implied rating.
    """
    successes = 0
    defaults = 0
    nav_sum = 0.0
    loss_sum = 0.0
    leverage_sum = 0.0

    for i in range(n_paths):
        res = cpdo_simulate(
            initial_nav=initial_nav,
            target_coupon=target_coupon,
            spread_income=spread_income,
            spread_vol=spread_vol,
            max_leverage=max_leverage,
            n_periods=n_periods,
            gap_risk=gap_risk,
            seed=seed + i,
        )
        nav_sum += res.final_nav
        loss_sum += max(initial_nav - res.final_nav, 0.0)
        leverage_sum += float(res.leverage_path.mean())
        if res.defaulted:
            defaults += 1
        elif res.final_nav >= initial_nav:
            successes += 1

    default_prob = defaults / n_paths
    success_prob = successes / n_paths
    expected_nav = nav_sum / n_paths
    expected_loss = loss_sum / n_paths
    avg_leverage = leverage_sum / n_paths
    rating = cpdo_rating(default_prob)

    return CPDOMCResult(
        success_prob=success_prob,
        expected_nav=expected_nav,
        expected_loss=expected_loss,
        avg_leverage=avg_leverage,
        default_prob=default_prob,
        rating_implied=rating,
    )


# ---------------------------------------------------------------------------
# 3. Rating mapping
# ---------------------------------------------------------------------------

def cpdo_rating(default_prob: float) -> str:
    """Map a 1-year default probability to an approximate S&P/Moody's rating.

    Thresholds are indicative; see S&P Global default study (2023) and
    Moody's Annual Default Study for calibrated cohort rates.

    Args:
        default_prob: annualised probability of default in [0, 1].

    Returns:
        Rating string such as ``'AAA'``, ``'AA'``, …, ``'CCC/CC'``, ``'D'``.
    """
    if default_prob < 0.001:
        return "AAA"
    elif default_prob < 0.002:
        return "AA+"
    elif default_prob < 0.004:
        return "AA"
    elif default_prob < 0.007:
        return "AA-"
    elif default_prob < 0.010:
        return "A+"
    elif default_prob < 0.015:
        return "A"
    elif default_prob < 0.020:
        return "A-"
    elif default_prob < 0.030:
        return "BBB+"
    elif default_prob < 0.050:
        return "BBB"
    elif default_prob < 0.075:
        return "BBB-"
    elif default_prob < 0.100:
        return "BB+"
    elif default_prob < 0.150:
        return "BB"
    elif default_prob < 0.200:
        return "BB-"
    elif default_prob < 0.300:
        return "B+"
    elif default_prob < 0.400:
        return "B"
    elif default_prob < 0.500:
        return "B-"
    elif default_prob < 0.700:
        return "CCC/CC"
    else:
        return "D"
