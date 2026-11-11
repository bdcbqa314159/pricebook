"""The Heston characteristic function's requirement oracles (L3) — tranche 33, PRE-REGISTERED.

**Written before a line of the implementation, deliberately.** Tranche 32's RED-first lapse had a
specific cause: the chain-rule algebra was long enough that I wanted to know it was right before
deciding how to assert it. The fix, agreed in thread 0169, is that the oracles which need **no
implementation to state** get written first and sit red while the derivation happens. These are
those oracles: every one is a property the characteristic function must satisfy whatever its
transcription, so none of them can be reverse-engineered from the code they test.

The three that need nothing but the definition `φ_T(u) = E[exp(iu·log(S_T/S₀))]`:

* **`φ(0) = 1`** — it is an expectation of 1. Catches a mis-normalised `C` or `D` term.
* **`φ(−i) = F(T)/S₀`** — because `exp(-i·i·x) = exp(x)`, so `φ(−i) = E[S_T]/S₀`. This is the
  **martingale/forward identity**, and it pins the *drift* regardless of how the stochastic-variance
  part is transcribed. A wrong `κ`, `θ`, `σ_v` or `ρ` leaves it intact; a wrong carry breaks it.
* **Hermitian symmetry `φ(−u) = conj(φ(u))`** — forced because `log(S_T/S₀)` is real-valued. It is
  the cheapest check that the complex branch cuts have not been crossed, which is the failure mode
  this tranche is most exposed to.

**And the trap, named in the spec and pre-registered here too.** The Heston CF contains a complex
square root and logarithm; the *naïve* branch (`g₁`, `e^{+dT}`) is discontinuous and silently wrong
at long maturity — Albrecher et al.'s "Little Heston Trap". The stable branch (`g₂`, `e^{−dT}`) is
not. That is a *property* failure, not a value one: the naïve form still returns a plausible number,
so only an oracle that watches the property across maturities can see it.
"""

import cmath
import math

import pytest

from pricebook.models.heston import (
    HestonParams,
    heston_characteristic_function,
    naive_characteristic_function,
)

#: A genuinely stochastic, correlated parameter set — not the degenerate `σ_v → 0` corner.
PARAMS = HestonParams(kappa=1.5, theta=0.04, sigma_v=0.6, rho=-0.7, v0=0.05)
CARRY = 0.03


class TestTheFixtureIsNotDegenerate:
    def test_the_variance_is_genuinely_STOCHASTIC_and_CORRELATED(self) -> None:
        """`σ_v = 0` collapses Heston to a deterministic-variance Black, and `ρ = 0` removes the
        leverage effect — the two things that make this model worth having."""
        assert PARAMS.sigma_v > 0.3
        assert abs(PARAMS.rho) > 0.5

    def test_the_FELLER_condition_is_stated_rather_than_assumed(self) -> None:
        """`2κθ ≥ σ_v²` keeps the variance strictly positive. This fixture **violates** it
        (2·1.5·0.04 = 0.12 < 0.36), which is the realistic equity regime and the one where a
        variance simulation must handle the zero boundary rather than assume it away."""
        assert 2 * PARAMS.kappa * PARAMS.theta < PARAMS.sigma_v**2


class TestTheDefinitionalIdentities:
    """True of any characteristic function of a real random variable, whatever the transcription."""

    @pytest.mark.parametrize("years", [0.25, 1.0, 5.0, 10.0])
    def test_phi_at_ZERO_is_exactly_one(self, years: float) -> None:
        got = heston_characteristic_function(PARAMS, 0.0 + 0j, years, CARRY)
        assert got == pytest.approx(1.0 + 0j, abs=1e-12)

    @pytest.mark.parametrize("years", [0.25, 1.0, 5.0, 10.0])
    def test_phi_at_MINUS_I_is_the_forward_ratio(self, years: float) -> None:
        """`φ(−i) = E[S_T]/S₀ = exp(carry·T)` — the martingale identity, pinning the drift alone."""
        got = heston_characteristic_function(PARAMS, -1j, years, CARRY)
        assert got.real == pytest.approx(math.exp(CARRY * years), rel=1e-10)
        assert got.imag == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("u", [0.5, 1.0, 3.0, 12.0])
    def test_the_function_is_HERMITIAN_because_the_log_return_is_real(self, u: float) -> None:
        plus = heston_characteristic_function(PARAMS, complex(u), 2.0, CARRY)
        minus = heston_characteristic_function(PARAMS, complex(-u), 2.0, CARRY)
        assert minus == pytest.approx(plus.conjugate(), abs=1e-12)

    @pytest.mark.parametrize("u", [0.5, 2.0, 8.0])
    def test_the_modulus_never_exceeds_one(self, u: float) -> None:
        """`|E[e^{iuX}]| ≤ E|e^{iuX}| = 1` for real `X` — a bound no valid CF can breach."""
        assert abs(heston_characteristic_function(PARAMS, complex(u), 3.0, CARRY)) <= 1.0 + 1e-12


class TestTheLittleHestonTrap:
    """The naive complex branch is discontinuous at long maturity; the stable one is not.

    Albrecher, Mayer, Schoutens & Tistaert. The naive form still *returns a number*, so a value
    oracle at a single maturity sees nothing — only watching a property across maturities does.
    """

    @pytest.mark.parametrize("years", [0.5, 2.0, 6.0, 12.0, 20.0])
    def test_the_STABLE_branch_stays_continuous_and_bounded_at_every_maturity(self, years: float) -> None:
        for u in (0.5, 2.0, 5.0, 15.0):
            got = heston_characteristic_function(PARAMS, complex(u), years, CARRY)
            assert math.isfinite(got.real) and math.isfinite(got.imag)
            assert abs(got) <= 1.0 + 1e-12, f"T={years}, u={u}: |phi|={abs(got)}"

    def test_the_stable_branch_is_SMOOTH_in_maturity_where_the_naive_one_JUMPS(self) -> None:
        """The discriminating property. Both branches agree at short maturity and diverge once the
        naive one crosses a branch cut, so the jump — not the level — is what identifies the trap.
        """
        u = 4.0

        def biggest_jump(fn) -> float:
            grid = [0.5 + 0.25 * k for k in range(60)]
            values = [fn(PARAMS, complex(u), t, CARRY) for t in grid]
            return max(abs(b - a) for a, b in zip(values, values[1:]))

        stable = biggest_jump(heston_characteristic_function)
        naive = biggest_jump(naive_characteristic_function)
        assert stable < 0.05, f"the stable branch must not jump, got {stable}"
        assert naive > 5.0 * stable, (
            f"the fixture must actually trip the naive branch, else this proves nothing: "
            f"stable={stable}, naive={naive}"
        )

    def test_the_two_branches_AGREE_at_short_maturity(self) -> None:
        """So the stable form is not simply a different function — it is the same one, continued."""
        for u in (1.0, 3.0):
            stable = heston_characteristic_function(PARAMS, complex(u), 0.25, CARRY)
            naive = naive_characteristic_function(PARAMS, complex(u), 0.25, CARRY)
            assert stable == pytest.approx(naive, abs=1e-10)


class TestTheDegenerateReduction:
    """`σ_v → 0` makes the variance deterministic, so the CF collapses to Black's."""

    def test_a_ZERO_vol_of_vol_gives_the_deterministic_variance_CF(self) -> None:
        """With `σ_v = 0`, `v_t` is the deterministic `θ + (v₀−θ)e^{−κt}`, so

            `φ(u) = exp(iu(carry·T − ½W) − ½u²W)`,  `W = ∫₀ᵀ v_t dt`

        which the test integrates itself rather than reading back from the model.
        """
        params = HestonParams(kappa=1.5, theta=0.04, sigma_v=0.0, rho=0.0, v0=0.05)
        years = 2.0
        integrated = params.theta * years + (params.v0 - params.theta) * (
            1.0 - math.exp(-params.kappa * years)
        ) / params.kappa
        for u in (0.5, 2.0, 5.0):
            expected = cmath.exp(
                1j * u * (CARRY * years - 0.5 * integrated) - 0.5 * u * u * integrated
            )
            got = heston_characteristic_function(params, complex(u), years, CARRY)
            assert got == pytest.approx(expected, abs=1e-10)
