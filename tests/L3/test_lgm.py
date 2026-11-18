"""The Linear Gauss-Markov model (L3) — tranche 57, PRE-REGISTERED. `LICHTERS-2015` ch 11.1.

LGM is Hull-White in the parameterisation whose Monte-Carlo is **exact**: the state `z(t)` is a
**driftless** Gaussian with **deterministic** variance `ζ(t)`, so a simulation draws
`z ~ N(0, ζ(t))` in one step — no SDE to discretise, no stepping error — and reads bond prices off
it. That is what an exposure engine wants, and it is the pull behind this tranche (T58).

**The source, transcribed from both renderings and reconciled (§4).** Ch 11.1:

    N(t)      = exp(H_t z_t + ½H_t² ζ_t) / P(0,t)                                   (11.1)
    P(t,T,z)  = (P(0,T)/P(0,t)) · exp[ −(H_T − H_t) z_t − ½(H_T² − H_t²) ζ_t ]      (11.4)

with `z` driftless, `ζ_t = ∫₀ᵗ α_s² ds`, and `H` a deterministic function tied to the Hull-White
mean reversion.

**The spec's transcription of (11.4) is wrong, and wrong in a way that is invisible at `t = 0`.**
It writes the convexity term as `½H̃²ζ` with `H̃ = H(T) − H(t)`; the source has
`½(H_T² − H_t²)ζ`. Those agree **only when `H_t = 0`**, which is `t = 0` — the one date at which
nothing has been simulated, and therefore the one date an LGM is never used at. Measured on this
tranche's parameters: identical at `t = 0`, **2.27× apart at `t = 3`**.

Both oracles the spec itself named catch it, which is the useful part:

| oracle | source (11.4) | the spec's version |
|---|---|---|
| numéraire martingale `E[P(t,T)/N(t)] = P(0,T)` | 0.1 SE | **7.5 SE** |
| MC zero-bond option vs analytic Hull-White | 0.10 SE | **20.8 SE** |

**The reduction is exact by construction, and that is also its blind spot (§4, T39).** The LGM here
is built *from* a `HullWhiteParams` pair — `H(t) = hw_beta(κ, 0, t)`, `ζ(t) = σ²(e^{2κt} − 1)/(2κ)`
— so agreeing with Hull-White analytically is near-tautological: it cannot see the **simulation**,
which is the entire point of the model. Hence the mandatory MC companion. `κ ≠ 0` is structural
rather than asserted, since `HullWhiteParams` already refuses a non-positive mean reversion.
"""

import math
from datetime import date, timedelta

import numpy as np
import pytest

from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.inflation import CapFloorKind
from pricebook.market import DiscountCurve
from pricebook.models.black_mc import MonteCarloConfig
from pricebook.models.hull_white import (
    HullWhiteParams,
    bond_option_variance,
    hw_beta,
    zero_bond_option,
)
from pricebook.models.lgm import (
    lgm_H,
    lgm_numeraire,
    lgm_zero_bond,
    lgm_zeta,
    simulate_lgm_states,
)

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
#: A **non-flat** curve: a flat one would hide an error in the `P(0,T)/P(0,t)` ratio.
CURVE = DiscountCurve(
    __import__("pricebook.foundation.time_measure", fromlist=["TimeMeasure"]).TimeMeasure(VAL, DC),
    (0.0, 1.0, 3.0, 5.0, 10.0),
    (0.025, 0.025, 0.029, 0.032, 0.035),
)
PARAMS = HullWhiteParams(kappa=0.08, sigma=0.012)


def _at(years: float) -> date:
    return VAL + timedelta(days=round(365 * years))


class TestThePremises:
    def test_the_mean_reversion_is_non_zero_STRUCTURALLY(self) -> None:
        """T39's premise, and here it is enforced by the type rather than by this test: the
        reduction is exact *because* it collapses the parameterisation, so a degenerate `κ` would
        make the companion vacuous. `HullWhiteParams` already refuses it."""
        assert PARAMS.kappa > 0.0
        with pytest.raises(ValueError):
            HullWhiteParams(kappa=0.0, sigma=0.012)

    def test_the_initial_curve_is_NOT_flat(self) -> None:
        """Else an error in the `P(0,T)/P(0,t)` ratio would cancel."""
        assert len(set(CURVE.zero_rates)) > 1


class TestTheParameterisation:
    """`H` and `ζ`, transcribed and required to behave like what they are."""

    @pytest.mark.parametrize("years", [0.5, 1.0, 3.0, 7.0])
    def test_H_is_the_hull_white_beta_from_zero(self, years: float) -> None:
        assert lgm_H(PARAMS, years) == pytest.approx(
            (1.0 - math.exp(-PARAMS.kappa * years)) / PARAMS.kappa, rel=1e-15
        )
        assert lgm_H(PARAMS, years) == hw_beta(PARAMS.kappa, 0.0, years)

    def test_zeta_starts_at_zero_and_never_decreases(self) -> None:
        """It is a variance — a requirement oracle that survives any transcription error."""
        assert lgm_zeta(PARAMS, 0.0) == 0.0
        previous = -1.0
        for i in range(40):
            current = lgm_zeta(PARAMS, i * 0.25)
            assert current >= previous, f"zeta decreased at t={i * 0.25}"
            previous = current

    @pytest.mark.parametrize("years", [0.5, 2.0, 5.0])
    def test_zeta_is_the_integral_it_claims_to_be(self, years: float) -> None:
        """`ζ(t) = ∫₀ᵗ σ²e^{2κs} ds`, integrated numerically here rather than read back."""
        steps = 20_000
        step = years / steps
        numeric = sum(
            PARAMS.sigma**2 * math.exp(2.0 * PARAMS.kappa * (i + 0.5) * step) * step
            for i in range(steps)
        )
        assert lgm_zeta(PARAMS, years) == pytest.approx(numeric, rel=1e-7)


class TestTheExactReductionToHullWhite:
    """The anchor — and near-tautological by construction, which is why the companion below is
    mandatory rather than decorative."""

    @pytest.mark.parametrize("expiry,maturity", [(1.0, 5.0), (2.0, 7.0), (0.5, 3.0), (3.0, 10.0)])
    def test_the_state_reproduces_hull_whites_bond_variance(
        self, expiry: float, maturity: float
    ) -> None:
        """`ln P(t,T)` has variance `(H_T − H_t)²ζ_t` under the LGM, and that **is** Hull-White's
        `ν(0,S)`. Measured agreement: machine precision."""
        lgm = (lgm_H(PARAMS, maturity) - lgm_H(PARAMS, expiry)) ** 2 * lgm_zeta(PARAMS, expiry)
        assert lgm == pytest.approx(bond_option_variance(PARAMS, expiry, maturity), rel=1e-13)

    def test_the_bond_at_t_zero_is_todays_curve(self) -> None:
        """"By construction calibrated to today's yield term structure" (ch 11.1) — `P(0,T,z₀) =
        P(0,T)` with `z₀ = 0`."""
        for years in (1.0, 3.0, 7.0):
            assert lgm_zero_bond(PARAMS, 0.0, years, 0.0, CURVE) == pytest.approx(
                CURVE.df_at(years), rel=1e-14
            )


class TestTheNumeraireMartingale:
    """§4's free control: a tradeable over the numéraire is a martingale, so
    `E[P(t,T)/N(t)] = P(0,T)` **by construction** — no model result involved, which is what makes it
    survive a transcription error the Hull-White reduction would share.

    This is the oracle that catches the spec's convexity term (7.5 SE)."""

    PATHS = 60_000

    @pytest.mark.parametrize("t,maturity", [(1.0, 5.0), (3.0, 10.0)])
    def test_the_discounted_bond_is_a_martingale(self, t: float, maturity: float) -> None:
        states = simulate_lgm_states(PARAMS, t, MonteCarloConfig(paths=self.PATHS, seed=20_571))
        ratios = np.array(
            [
                lgm_zero_bond(PARAMS, t, maturity, z, CURVE) / lgm_numeraire(PARAMS, t, z, CURVE)
                for z in states
            ]
        )
        mean = float(ratios.mean())
        se = float(ratios.std(ddof=1) / math.sqrt(self.PATHS))
        target = CURVE.df_at(maturity)
        assert abs(mean - target) < 4.0 * se, (
            f"E[P/N] = {mean:.8f} +- {se:.8f} against P(0,T) = {target:.8f} "
            f"({abs(mean - target) / se:.1f} SE)"
        )

    def test_THE_RESOLUTION_this_band_sees_the_convexity_term(self) -> None:
        """What a green rules out, stated: the band is ~4·SE, and dropping the `−½(H_T²−H_t²)ζ`
        convexity moves the mean far outside it. Measured against the spec's own (wrong) variant,
        which differs only in that term: **7.5 SE**."""
        t, maturity = 3.0, 10.0
        states = simulate_lgm_states(PARAMS, t, MonteCarloConfig(paths=self.PATHS, seed=20_571))
        target = CURVE.df_at(maturity)

        def ratio(convexity) -> tuple[float, float]:
            values = np.array(
                [
                    CURVE.df_at(maturity)
                    / CURVE.df_at(t)
                    * math.exp(
                        -(lgm_H(PARAMS, maturity) - lgm_H(PARAMS, t)) * z - convexity(z)
                    )
                    / lgm_numeraire(PARAMS, t, z, CURVE)
                    for z in states
                ]
            )
            return float(values.mean()), float(values.std(ddof=1) / math.sqrt(self.PATHS))

        zeta = lgm_zeta(PARAMS, t)
        wrong = 0.5 * (lgm_H(PARAMS, maturity) - lgm_H(PARAMS, t)) ** 2 * zeta
        mean, se = ratio(lambda _z: wrong)
        assert abs(mean - target) > 4.0 * se, (
            "the spec's convexity term must fail this band, or the oracle cannot see the term"
        )


class TestTheMonteCarloCompanion:
    """**Mandatory (§4, T39).** The analytic reduction is exact *because* LGM ≡ Hull-White in closed
    form, so it is blind to the simulation — which is the whole reason the model exists. This prices
    a zero-bond option by simulating the state and matches the analytic Hull-White price."""

    PATHS = 200_000

    @pytest.mark.parametrize(
        "expiry,maturity,moneyness", [(1.0, 5.0, 1.0), (2.0, 7.0, 0.98), (3.0, 10.0, 1.02)]
    )
    def test_a_zero_bond_option_by_MC_matches_the_analytic_price(
        self, expiry: float, maturity: float, moneyness: float
    ) -> None:
        strike = moneyness * CURVE.df_at(maturity) / CURVE.df_at(expiry)
        analytic = zero_bond_option(
            CURVE.df_at(expiry),
            CURVE.df_at(maturity),
            strike,
            bond_option_variance(PARAMS, expiry, maturity),
            CapFloorKind.CAP,
        )
        states = simulate_lgm_states(PARAMS, expiry, MonteCarloConfig(paths=self.PATHS, seed=4_242))
        payoffs = np.array(
            [
                max(lgm_zero_bond(PARAMS, expiry, maturity, z, CURVE) - strike, 0.0)
                / lgm_numeraire(PARAMS, expiry, z, CURVE)
                for z in states
            ]
        )
        mean = float(payoffs.mean())
        se = float(payoffs.std(ddof=1) / math.sqrt(self.PATHS))
        assert abs(mean - analytic) < 4.0 * se, (
            f"MC {mean:.8f} +- {se:.8f} vs analytic {analytic:.8f} "
            f"({abs(mean - analytic) / se:.2f} SE)"
        )
        assert analytic > 0.0, "the premise: a live option, not a zero"


class TestTheSimulatedState:
    """The exactness claim itself: one draw per path, no stepping."""

    def test_the_state_is_driftless_with_the_right_variance(self) -> None:
        t = 3.0
        states = np.array(simulate_lgm_states(PARAMS, t, MonteCarloConfig(paths=200_000, seed=7)))
        zeta = lgm_zeta(PARAMS, t)
        se_mean = math.sqrt(zeta / len(states))
        assert abs(states.mean()) < 4.0 * se_mean, "the state must be driftless"
        assert states.var(ddof=1) == pytest.approx(zeta, rel=0.02)

    def test_the_state_at_time_zero_is_degenerate(self) -> None:
        states = simulate_lgm_states(PARAMS, 0.0, MonteCarloConfig(paths=100, seed=1))
        assert all(z == 0.0 for z in states), "zeta(0) = 0, so z(0) = 0 exactly"

    def test_a_negative_horizon_is_refused(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            simulate_lgm_states(PARAMS, -1.0, MonteCarloConfig(paths=10, seed=1))

    def test_the_simulation_is_REPRODUCIBLE(self) -> None:
        """§3.1: identical inputs, identical output — the seed is explicit config."""
        config = MonteCarloConfig(paths=1_000, seed=99)
        assert simulate_lgm_states(PARAMS, 2.0, config) == simulate_lgm_states(PARAMS, 2.0, config)
