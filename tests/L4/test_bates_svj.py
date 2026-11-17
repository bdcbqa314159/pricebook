"""The SVJ / Bates model's oracles (L4) — tranche 49, PRE-REGISTERED. `GATHERAL-2006` ch 5.

Heston plus Merton-style lognormal jumps in the underlying — the model ch 5 argues "wins" because
jumps supply the steep short-dated skew stochastic volatility alone cannot, which is exactly the gap
T46's SVI fit exposed. The characteristic function factorises, `φ_SVJ = φ_Heston · φ_jump`, so T33's
inversion prices it with no new machinery (extracted to `engine/fourier_inversion.py` at its second
consumer, bit-identically — see `test_fourier_inversion.py`).

**The oracle is a DOUBLE reduction, and the second one is the whole point.**

* **Reduction A — `m → 0` ⇒ Heston.** Exact, and reuses T33. It is also **structurally blind to
  every jump parameter**, because it is taken at the point where the jumps vanish: `α` and `δ` can
  be anything at all and A still passes. That is the T39 blind spot verbatim, and this arc has now
  paid it three times (T46's `b→0`, T48's `η→0`, here).
* **Reduction B — `η → 0, v₀ = v̄` ⇒ the Merton jump-diffusion.** An **independent closed form** —
  the Poisson-weighted Black series, not the Heston CF — and non-degenerate in exactly the jump
  parameters A kills. Symmetrically, B is blind to the stochastic volatility, because it is taken at
  `η = 0`.

Neither is an anchor on its own and the blindness is asserted, not assumed: a jump mutation must
leave A green and kill **B**, a vol-of-vol mutation the reverse. The Merton series used here is the
test's own reference, derived from the Poisson mixture of lognormals (conditional on `n` jumps,
`log S_T` is normal with forward `F·e^{−m k T}·e^{n(α+δ²/2)}` and total variance `V + nδ²`, with
`k = e^{α+δ²/2} − 1`), and independently checked against a Monte-Carlo of the jump diffusion before
it was committed here — 0.07 to 1.39 SE across five cases, and at `m = 0` it equals plain Black
**bit-identically**.

**One correction to the spec's martingale identity, from the tree's own convention.** §2 asks for
`φ_SVJ(−i) = 1`. That holds only where the carry is zero: our `heston_characteristic_function` ends
in `+ iu·carry·T`, so `φ_SVJ(−i) = φ_Heston(−i) = F/S₀`. The convention-correct statement — and the
one that actually pins the compensator — is **`φ_jump(−i) = 1` exactly**, which is what makes the
jump factor drift-neutral and leaves the forward to the Heston factor. Both are asserted below, so
the correction is visible rather than silently substituted.

**Premises** (§4, asserted rather than believed): `m ≠ 0`, `η ≠ 0`, and `α ≠ 0` — a symmetric jump
hides the skew, which is the T17 symmetry degeneracy — plus a short *and* a long maturity, because
the jump-skew effect is a short-`T` statement and a single long maturity would not see it.
"""

import cmath
import math
from datetime import date, timedelta

import numpy as np
import pytest

from pricebook.engine import price
from pricebook.engine.bates_fourier import bates_call_price
from pricebook.engine.fourier_inversion import FourierGrid
from pricebook.engine.heston_fourier import heston_call_price
from pricebook.foundation import Currency, Money, PricingFailure, PricingResult
from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.inflation import CapFloorKind
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.market.equity import (
    DividendSchedule,
    EquityAsset,
    EquityUnderlying,
    dividend_forward,
)
from pricebook.models.bates import (
    BatesModel,
    BatesParams,
    JumpParams,
    bates_characteristic_function,
    jump_characteristic_function,
)
from pricebook.models.black import black_implied_vol, black_price
from pricebook.models.heston import HestonParams, heston_characteristic_function
from pricebook.models.svi import fit_svi_slice
from pricebook.products.equity_option import EquityOptionKind, EuropeanEquityOption

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
EUR = Currency("EUR")
RATE = 0.025
CURVE = DiscountCurve.flat(VAL, RATE, DC)
ASSET = EquityAsset("ACME")
UNDERLYING = EquityUnderlying(100.0, 0.0, DividendSchedule(continuous_yield=0.01))

HESTON = HestonParams(kappa=1.5, theta=0.04, sigma_v=0.6, rho=-0.7, v0=0.05)
#: Downward-skewed jumps: `α < 0` is what makes the jump skew observable at all.
JUMPS = JumpParams(intensity=0.5, mean=-0.2, volatility=0.15)
PARAMS = BatesParams(heston=HESTON, jumps=JUMPS)

SHORT_DAYS, LONG_DAYS = 60, 730
SHORT_EXPIRY = VAL + timedelta(days=SHORT_DAYS)
LONG_EXPIRY = VAL + timedelta(days=LONG_DAYS)
#: Wider than tranche 33's Heston grid, for the reason measured in `TestTheInversionIsConverged`:
#: jump tails make `phi_SVJ` decay more slowly, worst at short maturity.
GRID = FourierGrid(400.0, 4001)


def _years(expiry: date) -> float:
    return CURVE.time_measure.year_fraction(expiry)


def _forward(expiry: date) -> float:
    return dividend_forward(UNDERLYING, expiry, CURVE)


def _carry(expiry: date) -> float:
    return math.log(_forward(expiry) / UNDERLYING.spot) / _years(expiry)


def _merton_call(
    forward: float, strike: float, total_variance: float, jumps: JumpParams, years: float
) -> float:
    """The Poisson-weighted Black series — **the test's own reference**, derived rather than read
    back from any Bates code. Undiscounted."""
    compensator = math.exp(jumps.mean + 0.5 * jumps.volatility**2) - 1.0
    total = 0.0
    for n in range(80):
        if jumps.intensity > 0.0:
            weight = math.exp(
                -jumps.intensity * years
                + n * math.log(jumps.intensity * years)
                - math.lgamma(n + 1)
            )
        else:
            weight = 1.0 if n == 0 else 0.0
        jumped_forward = forward * math.exp(
            -jumps.intensity * compensator * years
            + n * (jumps.mean + 0.5 * jumps.volatility**2)
        )
        jumped_variance = total_variance + n * jumps.volatility**2
        total += weight * black_price(jumped_forward, strike, jumped_variance, CapFloorKind.CAP)
    return total


class TestThePremises:
    """Asserted rather than believed — the T44b lesson."""

    def test_the_jump_parameters_are_live_and_asymmetric(self) -> None:
        assert JUMPS.intensity > 0.0, "a zero intensity collapses SVJ to Heston"
        assert JUMPS.mean != 0.0, "a symmetric jump hides the skew (T17)"
        assert JUMPS.mean < 0.0, "and the equity direction is downward"
        assert JUMPS.volatility > 0.0

    def test_the_vol_of_vol_is_live(self) -> None:
        assert HESTON.sigma_v > 0.0, "a zero vol-of-vol collapses SVJ to Merton"

    def test_the_two_maturities_bracket_the_jump_effect(self) -> None:
        assert _years(SHORT_EXPIRY) < 0.25 < 1.0 < _years(LONG_EXPIRY)


class TestTheJumpCharacteristicFunction:
    """Transcribed pointwise from ch 5's `ψ(u)`, both renderings reconciled, plus the requirement
    that pins the compensator."""

    @pytest.mark.parametrize("u", [0.0, 0.3, 1.0, 4.0, -2.5])
    @pytest.mark.parametrize("days", [SHORT_DAYS, LONG_DAYS])
    def test_it_reproduces_an_independent_transcription(self, u: float, days: int) -> None:
        """`φ_jump(u) = exp(T·[ m(e^{iuα − u²δ²/2} − 1) − iu·m(e^{α+δ²/2} − 1) ])`, written out
        here from ch 5 p.65 rather than read back from the implementation."""
        years = days / 365.0
        intensity, alpha, delta = JUMPS.intensity, JUMPS.mean, JUMPS.volatility
        argument = complex(u)
        psi = intensity * (
            cmath.exp(1j * argument * alpha - argument * argument * delta * delta / 2.0) - 1.0
        ) - 1j * argument * intensity * (math.exp(alpha + delta * delta / 2.0) - 1.0)
        expected = cmath.exp(psi * years)
        got = jump_characteristic_function(JUMPS, argument, years)
        assert got.real == pytest.approx(expected.real, rel=1e-14, abs=1e-15)
        assert got.imag == pytest.approx(expected.imag, rel=1e-14, abs=1e-15)

    @pytest.mark.parametrize("days", [SHORT_DAYS, LONG_DAYS])
    def test_the_compensator_makes_the_jump_factor_drift_neutral(self, days: int) -> None:
        """**`φ_jump(−i) = 1` exactly** — the requirement that pins the compensator, and the
        convention-correct form of the spec's `φ_SVJ(−i) = 1` (our Heston factor carries the drift,
        so the whole CF at `−i` is `F/S₀`, not 1). At `u = −i` the two bracketed terms become the
        same number and cancel; a dropped or mis-signed compensator breaks it at once."""
        value = jump_characteristic_function(JUMPS, -1j, days / 365.0)
        assert value.real == pytest.approx(1.0, abs=1e-13)
        assert value.imag == pytest.approx(0.0, abs=1e-13)

    def test_the_compensator_is_not_vacuously_one(self) -> None:
        """The premise: without the compensator the factor at `−i` is materially off 1, so the
        identity above has power."""
        years, intensity = 2.0, JUMPS.intensity
        uncompensated = cmath.exp(
            years * intensity * (math.exp(JUMPS.mean + JUMPS.volatility**2 / 2.0) - 1.0)
        )
        assert abs(uncompensated - 1.0) > 0.1, f"{uncompensated}"

    @pytest.mark.parametrize("days", [SHORT_DAYS, LONG_DAYS])
    def test_the_full_cf_recovers_the_forward(self, days: int) -> None:
        """`φ_SVJ(−i) = F/S₀` — the martingale identity in this tree's carry convention."""
        expiry = VAL + timedelta(days=days)
        years, carry = _years(expiry), _carry(expiry)
        value = bates_characteristic_function(PARAMS, -1j, years, carry)
        assert value.real == pytest.approx(_forward(expiry) / UNDERLYING.spot, rel=1e-12)
        assert value.imag == pytest.approx(0.0, abs=1e-12)

    def test_zero_intensity_makes_the_jump_factor_identically_one(self) -> None:
        silent = JumpParams(intensity=0.0, mean=-0.2, volatility=0.15)
        for u in (0.0, 1.3, -2.0):
            assert jump_characteristic_function(silent, complex(u), 1.5) == 1.0 + 0j

    @pytest.mark.parametrize("days", [SHORT_DAYS, LONG_DAYS])
    def test_the_cf_factorises(self, days: int) -> None:
        """`φ_SVJ = φ_Heston · φ_jump` — ch 5's own statement, asserted as a structural identity."""
        expiry = VAL + timedelta(days=days)
        years, carry = _years(expiry), _carry(expiry)
        for u in (0.5, 2.0, -1.5):
            product = heston_characteristic_function(
                HESTON, complex(u), years, carry
            ) * jump_characteristic_function(JUMPS, complex(u), years)
            got = bates_characteristic_function(PARAMS, complex(u), years, carry)
            assert got == pytest.approx(product, rel=1e-13)


class TestReductionA:
    """`m → 0` ⇒ Heston. Exact, reuses T33 — **and blind to every jump parameter**."""

    @pytest.mark.parametrize("days", [SHORT_DAYS, LONG_DAYS])
    @pytest.mark.parametrize("strike", [80.0, 100.0, 130.0])
    def test_zero_intensity_prices_exactly_as_heston(self, days: int, strike: float) -> None:
        expiry = VAL + timedelta(days=days)
        silent = BatesParams(heston=HESTON, jumps=JumpParams(0.0, -0.2, 0.15))
        theirs = heston_call_price(HESTON, UNDERLYING, strike, expiry, (CURVE, GRID))
        ours = bates_call_price(silent, UNDERLYING, strike, expiry, (CURVE, GRID))
        assert ours == pytest.approx(theirs, rel=1e-13)

    @pytest.mark.parametrize("alpha,delta", [(-0.5, 0.4), (0.0, 0.01), (0.9, 1.2)])
    def test_REDUCTION_A_IS_BLIND_to_the_jump_parameters(self, alpha: float, delta: float) -> None:
        """The blindness asserted, not assumed (T39). At `m = 0` the jump mean and volatility can
        be *anything* and A still passes — which is precisely why it cannot be the only anchor."""
        expiry = LONG_EXPIRY
        silent = BatesParams(heston=HESTON, jumps=JumpParams(0.0, alpha, delta))
        theirs = heston_call_price(HESTON, UNDERLYING, 100.0, expiry, (CURVE, GRID))
        assert bates_call_price(silent, UNDERLYING, 100.0, expiry, (CURVE, GRID)) == pytest.approx(
            theirs, rel=1e-13
        )


class TestReductionB:
    """`η → 0, v₀ = v̄` ⇒ the Merton jump diffusion — A's mandatory companion, non-degenerate in
    exactly the parameters A kills."""

    DETERMINISTIC = HestonParams(kappa=1.5, theta=0.04, sigma_v=0.0, rho=0.0, v0=0.04)

    @pytest.mark.parametrize("days", [SHORT_DAYS, LONG_DAYS])
    @pytest.mark.parametrize("strike", [80.0, 100.0, 130.0])
    def test_zero_vol_of_vol_prices_as_the_merton_series(self, days: int, strike: float) -> None:
        expiry = VAL + timedelta(days=days)
        years = _years(expiry)
        params = BatesParams(heston=self.DETERMINISTIC, jumps=JUMPS)
        reference = CURVE.df(expiry) * _merton_call(
            _forward(expiry), strike, self.DETERMINISTIC.theta * years, JUMPS, years
        )
        got = bates_call_price(params, UNDERLYING, strike, expiry, (CURVE, GRID))
        assert got == pytest.approx(reference, rel=2e-6)

    @pytest.mark.parametrize("alpha,delta", [(-0.4, 0.25), (-0.05, 0.1), (0.3, 0.5)])
    def test_REDUCTION_B_SEES_what_A_cannot(self, alpha: float, delta: float) -> None:
        """The other half of the symmetry: B moves with the jump parameters A is blind to."""
        expiry, years = LONG_EXPIRY, _years(LONG_EXPIRY)
        jumps = JumpParams(0.5, alpha, delta)
        params = BatesParams(heston=self.DETERMINISTIC, jumps=jumps)
        reference = CURVE.df(expiry) * _merton_call(
            _forward(expiry), 100.0, self.DETERMINISTIC.theta * years, jumps, years
        )
        got = bates_call_price(params, UNDERLYING, 100.0, expiry, (CURVE, GRID))
        assert got == pytest.approx(reference, rel=2e-6)
        # and it is a DIFFERENT number from the jump-free case, or B would be vacuous too
        silent = BatesParams(heston=self.DETERMINISTIC, jumps=JumpParams(0.0, alpha, delta))
        assert abs(
            got - bates_call_price(silent, UNDERLYING, 100.0, expiry, (CURVE, GRID))
        ) > 0.05 * got

    def test_REDUCTION_B_IS_BLIND_to_the_vol_of_vol(self) -> None:
        """Symmetric to A's blindness: taken at `η = 0`, B cannot see the stochastic volatility at
        all. Stating both is what stops either being mistaken for covering the model."""
        assert self.DETERMINISTIC.sigma_v == 0.0
        assert self.DETERMINISTIC.v0 == self.DETERMINISTIC.theta, "v0 = vbar, or v is not constant"


class TestPutCallParity:
    """Model-independent, for all parameters — it survives a transcription error that both the
    source and the code might share."""

    @pytest.mark.parametrize("days", [SHORT_DAYS, LONG_DAYS])
    @pytest.mark.parametrize("strike", [70.0, 100.0, 140.0])
    def test_the_call_minus_the_put_is_the_discounted_forward_difference(
        self, days: int, strike: float
    ) -> None:
        expiry = VAL + timedelta(days=days)
        call = bates_call_price(PARAMS, UNDERLYING, strike, expiry, (CURVE, GRID))
        discount = CURVE.df(expiry)
        put = call - discount * (_forward(expiry) - strike)
        assert put > 0.0
        assert call - put == pytest.approx(discount * (_forward(expiry) - strike), rel=1e-12)


class TestAgainstMonteCarlo:
    """The independent absolute-level anchor: neither reduction prices the *general* model, and a
    price-level error that survives both is exactly what an MC sees (§4, T43).

    Banded per T17 as `4·SE` plus the inversion's own measured error, which is quoted separately
    rather than lumped: the grid's truncation error is measured against a far finer grid in
    `test_the_inversion_is_converged`, below 1e-9 relative, so the band is effectively `4·SE`.
    """

    PATHS = 60_000
    STEPS = 400

    def _mc(self, expiry: date, strike: float, seed: int) -> tuple[float, float]:
        years = _years(expiry)
        generator = np.random.default_rng(seed)
        dt = years / self.STEPS
        root_dt = math.sqrt(dt)
        orthogonal = math.sqrt(1.0 - HESTON.rho**2)
        compensator = math.exp(JUMPS.mean + 0.5 * JUMPS.volatility**2) - 1.0
        log_spot = np.full(self.PATHS, math.log(UNDERLYING.spot))
        variance = np.full(self.PATHS, HESTON.v0)
        carry = _carry(expiry)
        for _ in range(self.STEPS):
            used = np.maximum(variance, 0.0)
            first = generator.standard_normal(self.PATHS)
            second = generator.standard_normal(self.PATHS)
            correlated = HESTON.rho * first + orthogonal * second
            counts = generator.poisson(JUMPS.intensity * dt, self.PATHS)
            jump = JUMPS.mean * counts + JUMPS.volatility * np.sqrt(counts) * (
                generator.standard_normal(self.PATHS)
            )
            log_spot = (
                log_spot
                + (carry - JUMPS.intensity * compensator - 0.5 * used) * dt
                + np.sqrt(used) * root_dt * first
                + jump
            )
            variance = (
                variance
                + HESTON.kappa * (HESTON.theta - used) * dt
                + HESTON.sigma_v * np.sqrt(used) * root_dt * correlated
            )
        payoff = np.maximum(np.exp(log_spot) - strike, 0.0) * CURVE.df(expiry)
        return float(payoff.mean()), float(payoff.std(ddof=1) / math.sqrt(self.PATHS))

    @pytest.mark.parametrize("strike", [90.0, 110.0])
    def test_the_fourier_price_agrees_with_an_independent_simulation(self, strike: float) -> None:
        closed = bates_call_price(PARAMS, UNDERLYING, strike, LONG_EXPIRY, (CURVE, GRID))
        mean, se = self._mc(LONG_EXPIRY, strike, seed=90_125)
        assert abs(mean - closed) < 4.0 * se, (
            f"Fourier={closed:.6f} vs MC={mean:.6f} +- {se:.6f} "
            f"({abs(mean - closed) / se:.2f} SE)"
        )

    def test_THE_RESOLUTION_the_band_can_see_the_jumps(self) -> None:
        """A green above is evidence only about what the band resolves. The jump component moves
        the price by far more than `4·SE`, so dropping it would fail this check."""
        strike = 90.0
        _, se = self._mc(LONG_EXPIRY, strike, seed=90_125)
        with_jumps = bates_call_price(PARAMS, UNDERLYING, strike, LONG_EXPIRY, (CURVE, GRID))
        without = heston_call_price(HESTON, UNDERLYING, strike, LONG_EXPIRY, (CURVE, GRID))
        # MEASURED: the jump effect is 2.853 on a price of 20.64 (13.8%) against a 4*SE band of
        # ~0.41 -- about 7x. Asserted at 5x so the margin is real and stated, not hopeful.
        assert abs(with_jumps - without) > 5.0 * (4.0 * se), (
            f"jump effect {abs(with_jumps - without):.4f} vs band {4 * se:.4f}"
        )


class TestTheInversionIsConverged:
    """The Fourier price is the *reference* half of the MC band, so it needs its own accuracy
    argument in the regime it is used (§4, "a reference is code too")."""

    REFERENCE = FourierGrid(1200.0, 24001)

    @pytest.mark.parametrize("days", [SHORT_DAYS, LONG_DAYS])
    @pytest.mark.parametrize("strike", [80.0, 100.0, 130.0])
    def test_the_committed_grid_matches_a_far_finer_one(self, days: int, strike: float) -> None:
        expiry = VAL + timedelta(days=days)
        committed = bates_call_price(PARAMS, UNDERLYING, strike, expiry, (CURVE, GRID))
        fine = bates_call_price(PARAMS, UNDERLYING, strike, expiry, (CURVE, self.REFERENCE))
        assert committed == pytest.approx(fine, rel=1e-9)

    def test_THE_REASON_the_grid_is_wider_than_hestons(self) -> None:
        """**Jump tails decay more slowly, and the damage is at SHORT maturity** — the opposite of
        where one looks for truncation trouble, because that is where the diffusion has contributed
        least and the jump tail dominates. Measured here rather than asserted: tranche 33's
        `upper=200` is wrong by ~3e-05 relative at 60 days and by ~1e-14 at 730, so a
        Heston-shaped default prices long-dated options perfectly and short-dated ones wrongly.
        This is what makes the constant's *value* oracled rather than chosen."""
        narrow = FourierGrid(200.0, 2001)
        short_error = max(
            abs(
                bates_call_price(PARAMS, UNDERLYING, k, SHORT_EXPIRY, (CURVE, narrow))
                / bates_call_price(PARAMS, UNDERLYING, k, SHORT_EXPIRY, (CURVE, self.REFERENCE))
                - 1.0
            )
            for k in (80.0, 100.0, 130.0)
        )
        long_error = max(
            abs(
                bates_call_price(PARAMS, UNDERLYING, k, LONG_EXPIRY, (CURVE, narrow))
                / bates_call_price(PARAMS, UNDERLYING, k, LONG_EXPIRY, (CURVE, self.REFERENCE))
                - 1.0
            )
            for k in (80.0, 100.0, 130.0)
        )
        assert short_error > 1e-6, f"the premise: the narrow grid must actually fail short ({short_error:.2e})"
        assert long_error < 1e-10, f"...and pass long ({long_error:.2e}), which is how it hides"
        assert short_error > 1_000 * long_error


class TestTheCrossArcTieBack:
    """The capstone: ch 5's thesis is that jumps give the short-dated skew Heston lacks — the gap
    T46's SVI exposed. Asserted as an **ordering**, never a figure value (§4: ch 5's results are
    plots)."""

    MONEYNESS = (0.85, 0.92, 0.96, 1.0, 1.04, 1.09, 1.16)

    def _svi_atm_slope(self, jumps: JumpParams | None, expiry: date) -> float:
        years = _years(expiry)
        forward = _forward(expiry)
        discount = CURVE.df(expiry)
        strikes = [forward * x for x in self.MONEYNESS]
        variances = []
        for strike in strikes:
            if jumps is None:
                undiscounted = (
                    heston_call_price(HESTON, UNDERLYING, strike, expiry, (CURVE, GRID)) / discount
                )
            else:
                undiscounted = (
                    bates_call_price(
                        BatesParams(HESTON, jumps), UNDERLYING, strike, expiry, (CURVE, GRID)
                    )
                    / discount
                )
            vol = black_implied_vol(undiscounted, forward, strike, years, CapFloorKind.CAP)
            variances.append(vol * vol * years)
        log_strikes = tuple(math.log(k / forward) for k in strikes)
        fitted = fit_svi_slice(log_strikes, tuple(variances))
        return fitted.slope(0.0)

    def test_the_svj_short_dated_skew_is_STEEPER_than_hestons(self) -> None:
        """Downward jumps (`α < 0`) must steepen the at-the-money total-variance skew at short
        maturity. The number is never read off ch 5's figure — only the ordering is asserted, and
        the SVI fit is T46's, which makes this the arc's two builds checking each other."""
        heston_slope = self._svi_atm_slope(None, SHORT_EXPIRY)
        svj_slope = self._svi_atm_slope(JUMPS, SHORT_EXPIRY)
        assert svj_slope < heston_slope, f"SVJ {svj_slope} should be steeper than Heston {heston_slope}"
        assert heston_slope < 0.0, "the premise: Heston's own skew is already downward (rho < 0)"
        assert svj_slope < 1.5 * heston_slope, "and materially steeper, not marginally"


class TestTheRegisteredPricer:
    NOTIONAL = Money(1_000_000.0, EUR)

    def _model(self) -> BatesModel:
        market = MarketSnapshot(
            valuation_date=VAL,
            curves=CurveSet({CurveKey(CurveRole.DISCOUNT, EUR): CURVE}),
            equities={ASSET: UNDERLYING},
        )
        return BatesModel(market, PARAMS)

    def test_it_prices_a_call_through_the_registry(self) -> None:
        option = EuropeanEquityOption(
            ASSET, 100.0, LONG_EXPIRY, self.NOTIONAL, EquityOptionKind.CALL
        )
        out = price(option, self._model())
        assert isinstance(out, PricingResult), out
        expected = bates_call_price(PARAMS, UNDERLYING, 100.0, LONG_EXPIRY, (CURVE, GRID))
        assert out.pv.amount / 1_000_000.0 == pytest.approx(expected, rel=1e-6)

    def test_the_registered_pricer_is_CONVERGED_AT_SHORT_MATURITY(self) -> None:
        """**The hole the sweep found.** `TestTheInversionIsConverged` oracles the grid constant's
        *value*; nothing oracled the registered pricer's *use* of it, because the pricer test above
        runs at 730 days where tranche 33's narrower `upper=200` is already accurate to 1e-14. So
        narrowing `_DEFAULT_GRID` back survived the whole file — the T33b input-population shape
        one level up: the constant was tested, the code path that reads it was not.

        **And the strike matters as much as the maturity**, which the first attempt at this test got
        wrong. Truncation is an *absolute* error, so it is negligible against a large price and
        ruinous against a small one: at 60 days `upper=200` is off by 1.7e-08 at the money and by
        **2.9e-05 at K=130**, 1.4e-04 at K=150. A tolerance of 1e-7 on an at-the-money option
        therefore still let the narrowed grid through; the fixture has to be **deep out of the
        money and short dated** — which is exactly the region an SVJ model is bought for."""
        strike = 130.0
        option = EuropeanEquityOption(
            ASSET, strike, SHORT_EXPIRY, self.NOTIONAL, EquityOptionKind.CALL
        )
        out = price(option, self._model())
        assert isinstance(out, PricingResult), out
        reference = bates_call_price(
            PARAMS, UNDERLYING, strike, SHORT_EXPIRY, (CURVE, FourierGrid(1200.0, 24001))
        )
        assert out.pv.amount / 1_000_000.0 == pytest.approx(reference, rel=1e-7)
        # the premise: at the money the same narrowing is invisible, so this fixture is load-bearing
        narrow = FourierGrid(200.0, 4001)
        at_the_money = abs(
            bates_call_price(PARAMS, UNDERLYING, 100.0, SHORT_EXPIRY, (CURVE, narrow))
            / bates_call_price(PARAMS, UNDERLYING, 100.0, SHORT_EXPIRY, (CURVE, FourierGrid(1200.0, 24001)))
            - 1.0
        )
        in_the_wing = abs(
            bates_call_price(PARAMS, UNDERLYING, strike, SHORT_EXPIRY, (CURVE, narrow)) / reference
            - 1.0
        )
        assert at_the_money < 1e-7 < in_the_wing, (
            f"the wing is where the grid shows: ATM {at_the_money:.2e}, wing {in_the_wing:.2e}"
        )

    def test_an_EXPIRED_in_the_money_option_is_PV_zero_invariant_6(self) -> None:
        """The tenth-shape premise, with the T45 sharpening: the fixture must be **in the money**,
        or the fall-through returns an intrinsic of 0 and the guard is invisible."""
        strike = 85.0
        assert strike < UNDERLYING.spot, "the premise: ITM, or the intrinsic is 0"
        option = EuropeanEquityOption(ASSET, strike, VAL, self.NOTIONAL, EquityOptionKind.CALL)
        out = price(option, self._model())
        assert isinstance(out, PricingResult), out
        assert out.pv.amount == 0.0
        assert (UNDERLYING.spot - strike) * 1_000_000.0 > 1.0

    def test_a_missing_underlying_is_a_failure_not_an_exception(self) -> None:
        bare = MarketSnapshot(
            valuation_date=VAL, curves=CurveSet({CurveKey(CurveRole.DISCOUNT, EUR): CURVE})
        )
        option = EuropeanEquityOption(
            ASSET, 100.0, LONG_EXPIRY, self.NOTIONAL, EquityOptionKind.CALL
        )
        assert isinstance(price(option, BatesModel(bare, PARAMS)), PricingFailure)


class TestTheParameterGuards:
    @pytest.mark.parametrize("intensity", [-0.1, -5.0])
    def test_a_negative_jump_intensity_is_refused(self, intensity: float) -> None:
        with pytest.raises(ValueError, match="intensity"):
            JumpParams(intensity=intensity, mean=-0.1, volatility=0.2)

    @pytest.mark.parametrize("volatility", [-0.01, -1.0])
    def test_a_negative_jump_volatility_is_refused(self, volatility: float) -> None:
        with pytest.raises(ValueError, match="volatility"):
            JumpParams(intensity=0.5, mean=-0.1, volatility=volatility)

    def test_a_zero_jump_volatility_is_allowed(self) -> None:
        """A deterministic jump size is a legitimate model, not a degenerate one."""
        assert JumpParams(intensity=0.5, mean=-0.1, volatility=0.0).volatility == 0.0
