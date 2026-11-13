"""The correlated-default MC (L5) — tranche 36b's independent anchor.

The closed form is four bivariate normals; this simulates the model's **definitions** —
`{τ>T} = {U < e^{−H_T}}` with `H_T = (a_T + σ_H Y)⁺` and `Y` correlated to the equity's `X`. They
share no expression, so agreement is evidence rather than restatement, and it is the only oracle
that can see a wrong term in the four-term assembly: the `ρ → 0` reduction is blind to everything
that survives at zero correlation.

**What this isolates, stated rather than implied.** The simulation uses the *same* truncated-Gaussian
`H_T` as the closed form, so what it tests is the **bivariate-normal transcription** — analytic
versus simulated, with a known answer. The truncated Gaussian's own error against a genuinely
non-Gaussian hazard is a modelling question this tranche does not settle, and pretending the MC
settles it would be the kind of borrowed authority §4 exists to refuse.
"""

import math

import pytest

from pricebook.models.black_mc import MonteCarloConfig
from pricebook.risk.wrong_way_mc import (
    forward_estimate,
    realised_correlation,
    realised_survival,
    risky_call_estimate,
    simulate_correlated_default,
)
from pricebook.risk.wrong_way_risk import GaussianHazard, risky_call_value

from test_wrong_way_risk import FORWARD, SIGMA_H, SIGMA_S, STRIKE, SURVIVAL  # one fixture

CONFIG = MonteCarloConfig(paths=200_000, seed=20261113)
RHOS = (-0.8, -0.4, 0.0, 0.4, 0.8)
_PATHS: dict[float, object] = {}


def _paths(rho: float):
    """One simulation per `ρ`, reused — deterministic, and it holds the `slow` ceiling."""
    if rho not in _PATHS:
        _PATHS[rho] = simulate_correlated_default(
            FORWARD, SIGMA_S, GaussianHazard(SURVIVAL, SIGMA_H, rho), CONFIG
        )
    return _PATHS[rho]


class TestTheInputsArePinnedBeforeAnyPrice:
    """Each ingredient checked where it is directly observable, before the price that combines them."""

    @pytest.mark.parametrize("rho", RHOS)
    def test_the_EQUITY_DRIFT_is_pinned_by_the_forward(self, rho: float) -> None:
        got = forward_estimate(_paths(rho))
        assert abs(got.mean - FORWARD) <= 4.0 * got.standard_error

    @pytest.mark.parametrize("rho", RHOS)
    def test_the_EQ_13_4_SOLVE_is_pinned_by_the_realised_survival_frequency(self, rho: float) -> None:
        """`a_T` was solved from `p(T)`; the simulation must default at that rate. This observes
        the solve directly rather than through a price that also depends on everything else."""
        realised = realised_survival(_paths(rho))
        error = math.sqrt(SURVIVAL * (1.0 - SURVIVAL) / CONFIG.paths)
        assert realised == pytest.approx(SURVIVAL, abs=4.0 * error)

    @pytest.mark.parametrize("rho", RHOS)
    def test_RHO_is_pinned_where_it_is_DIRECTLY_observable(self, rho: float) -> None:
        """`ρ` lives in the joint law, so a CVA scalar is a weak test of it (T17). The realised
        correlation of the two Gaussians is the direct pin."""
        assert realised_correlation(_paths(rho)) == pytest.approx(rho, abs=0.01)


class TestTheClosedFormAgainstTheSimulation:
    @pytest.mark.parametrize("rho", RHOS)
    def test_the_two_methods_AGREE_within_the_simulation_error(self, rho: float) -> None:
        hazard = GaussianHazard(SURVIVAL, SIGMA_H, rho)
        simulated = risky_call_estimate(_paths(rho), STRIKE)
        analytic = risky_call_value(FORWARD, STRIKE, SIGMA_S, hazard)
        assert abs(simulated.mean - analytic) <= 4.0 * simulated.standard_error, (
            f"ρ={rho}: MC {simulated.mean} ± {simulated.standard_error} vs {analytic}"
        )

    def test_the_band_RESOLVES_a_FLIPPED_correlation(self) -> None:
        """What a green here rules out, measured: at `ρ = 0.8` the flipped-sign closed form is
        **many** standard errors away, so agreement is a real statement about the sign."""
        simulated = risky_call_estimate(_paths(0.8), STRIKE)
        flipped = risky_call_value(
            FORWARD, STRIKE, SIGMA_S, GaussianHazard(SURVIVAL, SIGMA_H, -0.8)
        )
        assert abs(flipped - simulated.mean) > 20.0 * simulated.standard_error

    def test_the_band_RESOLVES_the_INDEPENDENT_approximation_at_live_correlation(self) -> None:
        """The mutant that ignores `ρ` entirely — which the `ρ → 0` reduction cannot see."""
        for rho in (-0.8, 0.8):
            simulated = risky_call_estimate(_paths(rho), STRIKE)
            independent = risky_call_value(
                FORWARD, STRIKE, SIGMA_S, GaussianHazard(SURVIVAL, SIGMA_H, 0.0)
            )
            assert abs(independent - simulated.mean) > 10.0 * simulated.standard_error

    def test_the_WRONG_WAY_DIRECTION_is_confirmed_by_the_SIMULATION(self) -> None:
        """The direction derived from the model, now read off an independent method rather than
        off the formula that encodes it — which is what makes it a check and not a restatement."""
        values = [risky_call_estimate(_paths(rho), STRIKE).mean for rho in RHOS]
        assert all(b < a for a, b in zip(values, values[1:])), values
