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

    def test_THE_RESOLUTION_this_band_sees_the_EQ_11_4_convexity_defect(self) -> None:
        """What a green rules out — and **which defect**, because that is the whole content of a
        resolution statement (§4, F-5.3's per-defect rule).

        The band is ~4·SE, and dropping the eq-(11.4) `−½(H_T²−H_t²)ζ` convexity moves the mean far
        outside it: **7.5 SE**, measured against the spec's own (wrong) variant, which differs only
        in that term.

        **It does NOT follow that the band sees every defect in this formula family** — see
        `test_THIS_BAND_CANNOT_see_the_eq_11_1_half` below. Re-audit #6 found exactly that: a
        correct, §4-compliant resolution test for one defect reading as "resolution asserted" while
        a sibling sat inside the band."""
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

    def test_THIS_BAND_CANNOT_see_the_eq_11_1_half(self) -> None:
        """**Re-audit #6, F-6.1 — the band's limit, stated rather than left to look covered.**

        Dropping the `½` in eq (11.1)'s own exponent shifts this mean by only
        `exp(−½H_t²ζ_t) − 1` — **0.197%** at `t = 3` — which is **1.60 SE** here, comfortably
        inside the 4·SE band. The defect is *weakly* observable (T17's shape, not T34's: the
        sensitivity is nonzero, just small), so no realistic path count rescues this oracle —
        reaching 4 SE needs ~6.25× the paths, and §4's T51 clause says the first question is
        whether a **closed-form** quantity needs a band at all. It does not:
        `test_the_numeraire_matches_eq_11_1_POINTWISE` pins it exactly, at ~1e-15 and no paths.

        This test asserts the gap is real — it is the proof that the pointwise oracle is
        *necessary*, in the same shape §4 demands for a Box-Muller angle: assert that the wrong
        variant **does** satisfy the weaker check."""
        t, maturity = 3.0, 10.0
        states = simulate_lgm_states(PARAMS, t, MonteCarloConfig(paths=self.PATHS, seed=20_571))
        target = CURVE.df_at(maturity)
        ratios = np.array(
            [
                lgm_zero_bond(PARAMS, t, maturity, z, CURVE) / lgm_numeraire(PARAMS, t, z, CURVE)
                for z in states
            ]
        )
        se = float(ratios.std(ddof=1) / math.sqrt(self.PATHS))

        # Dropping the `½` multiplies N by exp(+½H²ζ), so the ratio falls by exp(−½H²ζ).
        shortfall = math.exp(-0.5 * lgm_H(PARAMS, t) ** 2 * lgm_zeta(PARAMS, t))
        halved = float((ratios * shortfall).mean())
        assert abs(halved - target) < 4.0 * se, (
            f"the eq-(11.1) half-defect must sit INSIDE this band, or the claim that the band "
            f"cannot see it is false: {abs(halved - target) / se:.2f} SE"
        )
        assert 1.0 - shortfall < 0.01, (
            f"and it is inside because the defect is small, not because the band is wide: "
            f"{1.0 - shortfall:.4%}"
        )


class TestEq111Pointwise:
    """**The §4 obligation eq (11.1) did not carry until re-audit #6 (F-6.1).**

    `lgm_numeraire` appeared in this file only ever as a **divisor** inside the martingale ratio —
    never on its own against an independently computed value — so the numéraire had its requirement
    oracle and no pointwise transcription. §4 is explicit that the two catch different failures and
    neither substitutes for the other, and the `½` surviving the whole suite is what that costs.

    Transcribed from `LICHTERS-2015` eq (11.1) in **both** renderings and reconciled:

        -layout:  N(t) = (1/P(0,t)) · exp{ H_t z_t + ½ H_t² ζ_t }
        -raw:     N(t) = 1/P(0,t) exp[ H_t z_t + 1/2 H²_t ζ_t ]

    (The `-raw` pass splits `H²_t` across lines and renders the braces as `%`/`+`; the two agree on
    the content, which is what reconciliation means.) The reference below is built from the printed
    closed forms — `H(t) = (1−e^{−κt})/κ` and `ζ(t) = σ²(e^{2κt}−1)/(2κ)` — not from `lgm_H` and
    `lgm_zeta`, so it shares no code path with the implementation."""

    @pytest.mark.parametrize("years", [0.5, 1.0, 3.0, 7.0])
    @pytest.mark.parametrize("state", [-1.4, -0.3, 0.0, 0.6, 2.1])
    def test_the_numeraire_matches_eq_11_1_POINTWISE(self, years: float, state: float) -> None:
        kappa, sigma = PARAMS.kappa, PARAMS.sigma
        h = (1.0 - math.exp(-kappa * years)) / kappa
        zeta = sigma * sigma * (math.exp(2.0 * kappa * years) - 1.0) / (2.0 * kappa)
        expected = math.exp(h * state + 0.5 * h * h * zeta) / CURVE.df_at(years)
        assert lgm_numeraire(PARAMS, years, state, CURVE) == pytest.approx(expected, rel=1e-14)

    def test_the_POINTWISE_fixture_is_NON_DEGENERATE_in_the_half_term(self) -> None:
        """The premise §4 demands: the fixture must make the two branches produce **different**
        numbers. At `z = 0` and small `t` the `½H²ζ` term is tiny, so a parametrisation that only
        sampled there would let the `½` hide — the same `τ=1` shape one guard over. Measured at the
        widest point of the grid above."""
        kappa, sigma = PARAMS.kappa, PARAMS.sigma
        h = (1.0 - math.exp(-kappa * 7.0)) / kappa
        zeta = sigma * sigma * (math.exp(2.0 * kappa * 7.0) - 1.0) / (2.0 * kappa)
        correct, doubled = 0.5 * h * h * zeta, h * h * zeta
        assert abs(math.exp(correct) / math.exp(doubled) - 1.0) > 1e-3, (
            f"the half must be worth something at t=7: {correct} vs {doubled}"
        )

    def test_the_numeraire_STARTS_AT_ONE(self) -> None:
        """`H(0) = 0` and `ζ(0) = 0`, so `N(0) = 1/P(0,0) = 1` — a requirement the finance forces,
        and the reduction companion to the transcription above."""
        assert lgm_numeraire(PARAMS, 0.0, 0.0, CURVE) == pytest.approx(1.0, rel=1e-15)


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

    def test_a_bond_MATURING_BEFORE_it_is_observed_is_refused(self) -> None:
        """Found by the sweep: the guard was written and nothing reached it. It is a real
        precondition rather than dead code — `P(3, 1)` asks for a bond that matured two years
        before the observation date, and eq (11.4) would happily return a number above 1 for it
        (a plausible wrong number, §4's standing concern) instead of refusing."""
        with pytest.raises(ValueError, match="before it is observed"):
            lgm_zero_bond(PARAMS, 3.0, 1.0, 0.0, CURVE)

    def test_THE_PREMISE_the_same_call_with_the_dates_ordered_is_fine(self) -> None:
        """So the refusal is about the ordering and not about those arguments being unusable."""
        assert lgm_zero_bond(PARAMS, 1.0, 3.0, 0.0, CURVE) > 0.0

    def test_a_negative_horizon_is_refused(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            simulate_lgm_states(PARAMS, -1.0, MonteCarloConfig(paths=10, seed=1))

    def test_the_simulation_is_REPRODUCIBLE(self) -> None:
        """§3.1: identical inputs, identical output — the seed is explicit config."""
        config = MonteCarloConfig(paths=1_000, seed=99)
        assert simulate_lgm_states(PARAMS, 2.0, config) == simulate_lgm_states(PARAMS, 2.0, config)
