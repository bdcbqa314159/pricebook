"""The defaultable-equity MC (L4) — tranche 35's independent anchor.

The closed form is Black on a scaled strike; this simulates the model's **definition**,
`S_t = S̃_t·1{τ>t}`, with the default indicator drawn independently of the diffusion. They share no
expression, so agreement is evidence rather than restatement — and it is the only oracle that can
see an inverted or missing strike scaling, since the `h → 0` reduction is blind to every term that
vanishes with the hazard.
"""

import math

import pytest

from pricebook.engine.defaultable_equity import defaultable_call, defaultable_put
from pricebook.engine.defaultable_equity_mc import (
    defaultable_call_estimate,
    defaultable_forward_estimate,
    defaultable_put_estimate,
    simulate_defaultable_spots,
)
from pricebook.models.black_mc import MonteCarloConfig

from test_defaultable_equity import (  # one fixture, so the two files cannot drift
    FORWARD,
    STRIKE,
    SURVIVAL_PROB,
    VOL,
    YEARS,
)

CONFIG = MonteCarloConfig(paths=120_000, seed=20261113)
_SPOTS = None


def _spots():
    """One simulation, reused — deterministic, and it keeps the file inside the `slow` ceiling."""
    global _SPOTS
    if _SPOTS is None:
        _SPOTS = simulate_defaultable_spots(FORWARD, SURVIVAL_PROB, VOL, YEARS, CONFIG)
    return _SPOTS


class TestTheDefaultWeightingIsPinnedFirst:
    def test_the_simulated_forward_is_the_PLAIN_forward(self) -> None:
        """`E[S_t] = p·F̃ = F` — checked before any price, so a wrong default weighting cannot be
        read as a volatility or strike-scaling error below."""
        got = defaultable_forward_estimate(_spots())
        assert abs(got.mean - FORWARD) <= 4.0 * got.standard_error, (
            f"E[S_t]={got.mean} ± {got.standard_error} vs F={FORWARD}"
        )

    def test_the_realised_default_frequency_matches_the_survival_probability(self) -> None:
        """Pinned where it is directly observable rather than through a price (T17's rule): the
        fraction of paths that survive must be `p`, which a price only sees in combination."""
        spots = _spots()
        realised = float((spots > 0.0).mean())
        assert realised == pytest.approx(SURVIVAL_PROB, abs=4.0 * math.sqrt(
            SURVIVAL_PROB * (1 - SURVIVAL_PROB) / len(spots)
        ))


class TestTheClosedFormAgainstTheSimulation:
    @pytest.mark.parametrize("moneyness", [0.7, 1.0, 1.3])
    def test_the_CALL_agrees_within_the_simulation_error(self, moneyness: float) -> None:
        strike = moneyness * FORWARD
        simulated = defaultable_call_estimate(_spots(), strike)
        analytic = defaultable_call(FORWARD, strike, SURVIVAL_PROB, VOL, YEARS)
        assert abs(simulated.mean - analytic) <= 4.0 * simulated.standard_error

    @pytest.mark.parametrize("moneyness", [0.7, 1.0, 1.3])
    def test_the_PUT_agrees_within_the_simulation_error(self, moneyness: float) -> None:
        strike = moneyness * FORWARD
        simulated = defaultable_put_estimate(_spots(), strike)
        analytic = defaultable_put(FORWARD, strike, SURVIVAL_PROB, VOL, YEARS)
        assert abs(simulated.mean - analytic) <= 4.0 * simulated.standard_error

    def test_the_band_RESOLVES_an_INVERTED_strike_scaling(self) -> None:
        """`K/p` where `p·K` belongs — what the reduction cannot see, since the two coincide at
        `p = 1`. Stating the band's power is what makes a green mean something (§4)."""
        simulated = defaultable_call_estimate(_spots(), STRIKE)
        inverted = defaultable_call(FORWARD, STRIKE / (SURVIVAL_PROB * SURVIVAL_PROB), SURVIVAL_PROB, VOL, YEARS)
        assert abs(inverted - simulated.mean) > 4.0 * simulated.standard_error

    def test_the_band_RESOLVES_an_UNSCALED_strike(self) -> None:
        """Dropping the scaling entirely — the classic degenerate-fixture mutant, alive only
        because the companion fixture has `p < 1`."""
        simulated = defaultable_call_estimate(_spots(), STRIKE)
        unscaled = defaultable_call(FORWARD, STRIKE, 1.0, VOL, YEARS)
        assert abs(unscaled - simulated.mean) > 4.0 * simulated.standard_error

    def test_the_band_RESOLVES_a_MISSING_default_lump_in_the_put(self) -> None:
        simulated = defaultable_put_estimate(_spots(), STRIKE)
        without_lump = defaultable_put(FORWARD, STRIKE, SURVIVAL_PROB, VOL, YEARS) - (
            1.0 - SURVIVAL_PROB
        ) * STRIKE
        assert abs(without_lump - simulated.mean) > 4.0 * simulated.standard_error
