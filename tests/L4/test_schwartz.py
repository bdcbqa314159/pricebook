"""The Schwartz (1997) one-factor mean-reverting commodity model — tranche 64, PRE-REGISTERED.
`CLARK-2014` ch 2.10.1.1. The commodity arc's capstone.

The model the earlier tranches' flat vol was waiting for: mean reversion in **log-spot**, which gives
a commodity the *term structure* of volatility a flat Black vol cannot. It is where
`CommodityBlackModel`'s own docstring prophecy comes true — "a commodity's volatility is on a path to
diverge from an equity's within this very arc".

**The source, transcribed from both renderings and reconciled (§4).** Clark pp.79–82:

    dS = κ(α_S − ln S)·S dt + σ S dW                                      (2.147a)
    dX = κ(α_X − X) dt + σ dW,   X = ln S,   α_X = α_S − σ²/(2κ)          (2.147b)
    E[X_T]   = X₀e^{−κT} + α_X(1 − e^{−κT})                               (2.150a)
    Var[X_T] = (σ²/(2κ))·(1 − e^{−2κT})                                   (2.150b)
    V = ω·e^{−r^dT}[F_{0,T}·N(ωd₁) − K·N(ωd₂)],  d₁,₂ = [ln(F/K) ± ½σ²_{0,T}T]/(σ_{0,T}√T)
                                                                          (2.151a/b)
    F_{0,T} = S₀^{e^{−κT}}·exp( α̂_X(1 − e^{−κT}) + (σ²/(4κ))(1 − e^{−2κT}) )   (2.156)
    lim_{T→∞} F_{0,T} = exp(α̂_X + σ²/(4κ)) = exp(α_S − σ²/(4κ) − λ/κ)     (2.155)

with `α̂_X = α_X − λ/κ` and `λ` the market price of risk, so the level this model actually carries is

    α̂_X = α_S − σ²/(2κ) − λ/κ

**Three parameters go in and only this one composite comes out, which is a T11 identifiability point
made at design time rather than discovered at calibration.** `α_S` (the physical long-run level) and
`λ` (the market price of risk) appear *only* through `α̂_X`, so no set of forward prices can separate
them. The model therefore carries the **risk-neutral** level and says so, rather than carrying three
parameters of which two are unidentifiable. That is the opposite of folding the adjustment in
silently: the decomposition is named here and in the module docstring, and what is *carried* is what
is observable.

**The κ → 0 reduction is NOT to the cost-of-carry forward, and the spec's claim that it was is
corrected here.** The Schwartz drift is pure mean reversion — there is no `r − δ` in (2.147) — so as
`κ → 0` the log-spot becomes driftless arithmetic Brownian motion and

    Var[X_T] → σ²T          (exact)
    F_{0,T}  → S₀·exp(σ²T/2)

Measured on this fixture: the limit forward is **88.5136**, while T60's cost-of-carry forward at
`r = 4.2%`, `δ = 1.5%` is **83.9785**. They are different numbers because they are different models,
and Clark says as much on p.82 — matching market forwards is a *calibration* of `α̂_X` ("to choose
`α_X` such that `E^d[S_T] = F_{0,T}`"), not a property the model has for free.

**The cross-engine reduction is still available, with the condition stated.** The two agree exactly
when T60's carry is chosen to reproduce the κ → 0 Schwartz drift, i.e. at **`δ = r − σ²/2`**, since
`S₀e^{(r−δ)T} = S₀e^{σ²T/2}` there. So the anchor the spec wanted exists — it just needed the true
condition rather than an assumed one, and with it the two engines agree to the last ULP available.

**The MC companion's reuse of `simulate_lgm_states` is EXACT, not approximate.** The LGM state is
`z_T ~ N(0, ζ(T))` with `ζ(T) = σ²(e^{2κT} − 1)/(2κ)`, so

    Var(e^{−κT}·z_T) = e^{−2κT}·ζ(T) = σ²(1 − e^{−2κT})/(2κ) = Var[X_T]

identically — the same `X̂_t = e^{κt}X_t` integrating factor Clark uses to integrate the SDE. The
Schwartz state is the LGM state scaled by `e^{−κT}` plus a deterministic mean, which is the reuse the
whole arc was built toward. It is a **consistency guard rather than independence** (one simulator,
two readings — the T17 same-object caveat) and is labelled as such.
"""

import math
from datetime import date, timedelta

import numpy as np
import pytest

from pricebook.engine import price
from pricebook.engine.schwartz import schwartz_futures, schwartz_total_variance
from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.money import Currency, Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.market.commodity import CommodityAsset, CommodityUnderlying
from pricebook.market.curve import DiscountCurve
from pricebook.market.curve_set import CurveKey, CurveRole, CurveSet
from pricebook.market.snapshot import MarketSnapshot
from pricebook.models.black import black_price
from pricebook.models.black_mc import MonteCarloConfig
from pricebook.models.commodity_black import CommodityBlackModel
from pricebook.models.hull_white import HullWhiteParams
from pricebook.models.lgm import lgm_zeta, simulate_lgm_states
from pricebook.models.schwartz import SchwartzModel, SchwartzParams
from pricebook.products.commodity_option import CommodityOption, CommodityOptionKind

VAL = date(2026, 1, 15)
DC = DayCountConvention.ACT_365_FIXED
USD = Currency("USD")
WTI = CommodityAsset("WTI")

RATE = 0.042
SPOT, VOL = 78.5, 0.31
#: **κ materially non-zero** — the capstone's whole point is the parameter the reduction zeroes, so
#: a near-zero κ companion would be the reduction in disguise (T39, premise asserted below).
KAPPA, LEVEL = 0.55, math.log(85.0)
#: Not one year, so the `(1 − e^{−2κT})/(2κ)` factor cannot be mistaken for `σ²`.
EXPIRY = VAL + timedelta(days=912)
YEARS = 912 / 365.0
FLAT = DiscountCurve.flat(VAL, RATE, DC)


def _params(kappa: float = KAPPA, level: float = LEVEL, sigma: float = VOL) -> SchwartzParams:
    return SchwartzParams(kappa=kappa, level=level, sigma=sigma)


def _market(convenience: float = 0.015) -> MarketSnapshot:
    return MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet({CurveKey(CurveRole.DISCOUNT, USD): FLAT}),
        commodities={WTI: CommodityUnderlying(WTI, SPOT, convenience)},
    )


def _option(
    strike: float = 85.0, kind: CommodityOptionKind = CommodityOptionKind.CALL
) -> CommodityOption:
    return CommodityOption(
        asset=WTI, strike=strike, expiry=EXPIRY, notional=Money(1_000.0, USD), kind=kind
    )


def _pv(product, params: SchwartzParams | None = None, convenience: float = 0.015) -> float:
    out = price(product, SchwartzModel(_market(convenience), params or _params()))
    assert isinstance(out, PricingResult), out
    return out.pv.amount


# ── references written here from the print, never read back from the engine ────────────────────


def _reference_variance(kappa: float, sigma: float, years: float) -> float:
    """Eq (2.150b): `Var[X_T] = (σ²/(2κ))(1 − e^{−2κT})` — the Black-76 TOTAL variance."""
    return sigma * sigma / (2.0 * kappa) * (1.0 - math.exp(-2.0 * kappa * years))


def _reference_futures(
    spot: float, kappa: float, level: float, sigma: float, years: float
) -> float:
    """Eq (2.156): `S₀^{e^{−κT}}·exp(α̂_X(1−e^{−κT}) + (σ²/(4κ))(1−e^{−2κT}))`."""
    decay = math.exp(-kappa * years)
    return spot**decay * math.exp(
        level * (1.0 - decay)
        + sigma * sigma / (4.0 * kappa) * (1.0 - math.exp(-2.0 * kappa * years))
    )


class TestThePremisesTheOraclesRestOn:
    def test_the_mean_reversion_is_MATERIALLY_non_zero(self) -> None:
        """**T39, and this is the arc's sharpest instance.** The κ → 0 anchor is exact *because* it
        kills the mean reversion — which is the entire reason this model exists — so the anchor is
        structurally blind to the whole model. The companion's κ must be materially non-zero, and
        this measures how far: at κ = 0.55 the total variance is **37%** of the GBM value."""
        assert KAPPA > 0.1
        ratio = _reference_variance(KAPPA, VOL, YEARS) / (VOL * VOL * YEARS)
        assert ratio < 0.5, ratio

    def test_the_horizon_is_NOT_one_year(self) -> None:
        assert abs(YEARS - 1.0) > 1.4

    def test_the_LEVEL_differs_from_the_SPOT(self) -> None:
        """A mutation swapping `α̂_X` for the current log-spot is invisible if the two coincide."""
        assert abs(math.exp(LEVEL) / SPOT - 1.0) > 0.05


class TestTheClosedFormsPointwise:
    """§4: the two genuinely-new formulae, each transcribed pointwise against a reference this file
    forms itself. Both are named surface for the F-6.1 reason."""

    @pytest.mark.parametrize("kappa", [0.05, 0.25, 0.55, 2.0, 8.0])
    @pytest.mark.parametrize("years", [0.25, 1.0, 2.4986, 10.0])
    def test_the_TOTAL_VARIANCE_matches_eq_2_150b(self, kappa: float, years: float) -> None:
        assert schwartz_total_variance(_params(kappa=kappa), years) == pytest.approx(
            _reference_variance(kappa, VOL, years), rel=1e-14
        )

    @pytest.mark.parametrize("kappa", [0.05, 0.55, 2.0])
    @pytest.mark.parametrize("years", [0.25, 2.4986, 10.0])
    def test_the_FUTURES_price_matches_eq_2_156(self, kappa: float, years: float) -> None:
        assert schwartz_futures(_params(kappa=kappa), SPOT, years) == pytest.approx(
            _reference_futures(SPOT, kappa, LEVEL, VOL, years), rel=1e-13
        )

    def test_the_futures_price_STARTS_AT_THE_SPOT(self) -> None:
        """At `T = 0` the decay is 1, both exponents vanish and `F = S₀^1 = S₀` — a requirement the
        finance forces, and the companion to the transcription above."""
        assert schwartz_futures(_params(), SPOT, 0.0) == pytest.approx(SPOT, rel=1e-14)

    def test_the_variance_VANISHES_at_zero_horizon(self) -> None:
        assert schwartz_total_variance(_params(), 0.0) == pytest.approx(0.0, abs=1e-18)


class TestTheVarianceTermStructure:
    """The structural fact that separates a mean-reverting commodity from GBM: the variance
    **saturates** rather than growing without bound.

    Per §4's obligation-not-form clause, what is asserted is that the measured term structure
    matches the closed form and that the finite limit holds. The *shape* — the Samuelson effect,
    near-dated futures being more volatile than far-dated — is reported below as a measured
    consequence rather than pre-asserted.
    """

    def test_the_STATIONARY_variance_is_sigma_squared_over_two_kappa(self) -> None:
        """`Var[X_T] → σ²/(2κ)` as `T → ∞` (2.150b). A GBM's variance is `σ²T` and diverges, so
        this is the qualitative difference the model exists to express."""
        stationary = VOL * VOL / (2.0 * KAPPA)
        assert schwartz_total_variance(_params(), 200.0) == pytest.approx(stationary, rel=1e-12)
        assert schwartz_total_variance(_params(), 20.0) / stationary > 0.99

    def test_the_ANNUALISED_variance_term_structure_is_MEASURED_against_the_closed_form(
        self,
    ) -> None:
        """Measured across well-spread maturities — a single maturity cannot see a term structure
        at all. The reported consequence: the annualised variance falls from **0.0840** at three
        months through 0.0583, 0.0327 and 0.0174 to **0.0044** at twenty years — a factor of 19.2,
        i.e. the **Samuelson effect**. That is a statement about what was measured, not an asserted
        monotonicity (§4's obligation-not-form clause)."""
        measured = [
            (years, schwartz_total_variance(_params(), years) / years)
            for years in (0.25, 1.0, 2.4986, 5.0, 20.0)
        ]
        for years, annualised in measured:
            assert annualised == pytest.approx(
                _reference_variance(KAPPA, VOL, years) / years, rel=1e-14
            )
        # Reported, from the measurement above: the near end carries far more variance per year.
        assert measured[0][1] / measured[-1][1] > 15.0, measured

    def test_a_LARGER_kappa_pulls_the_variance_down_at_every_maturity(self) -> None:
        """A requirement the algebra forces — `(1−e^{−2κT})/(2κ)` is decreasing in `κ` — so it
        survives a transcription error that the pointwise oracle would share with the source."""
        for years in (0.5, 2.4986, 10.0):
            values = [
                schwartz_total_variance(_params(kappa=k), years) for k in (0.1, 0.55, 2.0, 8.0)
            ]
            assert all(b < a for a, b in zip(values, values[1:])), (years, values)


class TestTheZeroMeanReversionLimit:
    """The anchor — and the correction to the spec, which assumed it landed on the cost-of-carry
    forward. It does not: the Schwartz drift is pure mean reversion."""

    def test_the_variance_tends_to_SIGMA_SQUARED_T(self) -> None:
        """`(σ²/(2κ))(1 − e^{−2κT}) → σ²T` as `κ → 0` — exact in the limit, and approached from
        below because the mean reversion only ever removes variance."""
        assert schwartz_total_variance(_params(kappa=1e-9), YEARS) == pytest.approx(
            VOL * VOL * YEARS, rel=1e-8
        )
        for kappa in (1e-4, 1e-2, 0.1):
            assert schwartz_total_variance(_params(kappa=kappa), YEARS) < VOL * VOL * YEARS

    def test_the_FUTURES_price_tends_to_the_DRIFTLESS_LOG_limit_not_the_carry_forward(
        self,
    ) -> None:
        """**The spec's claim corrected, measured.** As `κ → 0` the log-spot is driftless arithmetic
        Brownian motion, so `E[S_T] = S₀e^{σ²T/2}` — and that is **not** T60's cost-of-carry
        forward, because (2.147) contains no `r − δ` at all. On this fixture the two are 88.5136
        against 83.9785: different models, not a discrepancy. Clark says so on p.82 — matching
        market forwards is a calibration of `α̂_X`, not something the model gives for free."""
        driftless = SPOT * math.exp(0.5 * VOL * VOL * YEARS)
        assert schwartz_futures(_params(kappa=1e-9), SPOT, YEARS) == pytest.approx(
            driftless, rel=1e-7
        )
        carry_forward = SPOT * math.exp((RATE - 0.015) * YEARS)
        assert abs(driftless / carry_forward - 1.0) > 0.04, (driftless, carry_forward)

    def test_at_the_MATCHING_carry_the_two_ENGINES_agree(self) -> None:
        """**The cross-engine anchor the spec wanted, with the true condition supplied.** The
        Schwartz κ → 0 drift is `σ²/2`, so it coincides with T60's `r − δ` exactly when
        `δ = r − σ²/2`. At that convenience yield the two engines — different code, different
        models — price the same number.

        Banded rather than `==`: the two compose `exp`/`log` differently (a power `S₀^{e^{−κT}}`
        against `S₀·e^{(r−δ)T}`), which is T61's lesson and the case §4 says must not be asserted as
        bit-identity."""
        matching_convenience = RATE - 0.5 * VOL * VOL
        schwartz = _pv(_option(), _params(kappa=1e-9), convenience=matching_convenience)
        out = price(
            _option(),
            CommodityBlackModel(_market(matching_convenience), VOL),
        )
        assert isinstance(out, PricingResult)
        assert schwartz == pytest.approx(out.pv.amount, rel=1e-7)

    def test_the_two_engines_DISAGREE_at_a_genuine_kappa_so_the_anchor_is_not_vacuous(
        self,
    ) -> None:
        """The companion that gives the anchor above its teeth: at κ = 0.55 the same comparison
        must fail, or the Schwartz engine is not doing anything a flat Black vol does not."""
        matching_convenience = RATE - 0.5 * VOL * VOL
        schwartz = _pv(_option(), _params(), convenience=matching_convenience)
        out = price(_option(), CommodityBlackModel(_market(matching_convenience), VOL))
        assert isinstance(out, PricingResult)
        assert abs(schwartz / out.pv.amount - 1.0) > 0.2, (schwartz, out.pv.amount)


class TestTheSimulatedOUMomentsArePinnedFIRST:
    """**Before the price companion is credited with anything** (T19's unbiased-before-trusted
    discipline, and the T17 pin). The Schwartz state's conditional mean and variance are *both*
    closed form, so each is pinned against the simulated increments directly — a price would see
    them only through one scalar.

    The simulator is `simulate_lgm_states`, reused: `Var(e^{−κT}·z_T) = e^{−2κT}·ζ(T)` is
    identically `σ²(1−e^{−2κT})/(2κ)`, so the reuse is **exact** rather than an approximation. It is
    a consistency guard, not independence — one simulator read two ways (T17's same-object caveat).
    """

    PATHS = 200_000

    def _states(self, params: SchwartzParams, years: float, seed: int) -> np.ndarray:
        """The Schwartz log-spot at `years`, built from the LGM state."""
        decay = math.exp(-params.kappa * years)
        lgm = np.array(
            simulate_lgm_states(
                HullWhiteParams(params.kappa, params.sigma),
                years,
                MonteCarloConfig(paths=self.PATHS, seed=seed),
            )
        )
        mean = math.log(SPOT) * decay + params.level * (1.0 - decay)
        return mean + decay * lgm

    def test_the_LGM_SCALING_reproduces_the_schwartz_variance_EXACTLY(self) -> None:
        """The algebraic identity the reuse rests on, checked before any simulation: the LGM's
        `ζ(T)` scaled by `e^{−2κT}` *is* eq (2.150b). Not a tolerance on a measurement — an
        identity on two closed forms."""
        for kappa in (0.1, 0.55, 2.0):
            for years in (0.5, 2.4986, 10.0):
                decay_squared = math.exp(-2.0 * kappa * years)
                scaled = decay_squared * lgm_zeta(HullWhiteParams(kappa, VOL), years)
                assert scaled == pytest.approx(
                    _reference_variance(kappa, VOL, years), rel=1e-13
                )

    def test_the_realised_MEAN_reproduces_eq_2_150a(self) -> None:
        states = self._states(_params(), YEARS, seed=64_001)
        decay = math.exp(-KAPPA * YEARS)
        expected = math.log(SPOT) * decay + LEVEL * (1.0 - decay)
        standard_error = math.sqrt(_reference_variance(KAPPA, VOL, YEARS) / self.PATHS)
        assert abs(float(states.mean()) - expected) < 4.0 * standard_error, (
            f"realised {states.mean():.8f} vs {expected:.8f} "
            f"({abs(float(states.mean()) - expected) / standard_error:.2f} SE)"
        )

    def test_the_realised_VARIANCE_reproduces_eq_2_150b(self) -> None:
        states = self._states(_params(), YEARS, seed=64_001)
        expected = _reference_variance(KAPPA, VOL, YEARS)
        realised = float(states.var(ddof=1))
        # A chi-squared standard error on a variance estimate from n samples.
        standard_error = expected * math.sqrt(2.0 / (self.PATHS - 1))
        assert abs(realised - expected) < 4.0 * standard_error, (
            f"realised {realised:.8f} vs {expected:.8f} "
            f"({abs(realised - expected) / standard_error:.2f} SE)"
        )

    def test_the_moment_error_IMPROVES_as_the_root_of_the_path_count(self) -> None:
        """A **bias** would not improve with paths, so failure-to-improve is the detector (T17),
        averaged over seeds because a null needs averaging as much as an effect does (T21)."""
        expected = _reference_variance(KAPPA, VOL, YEARS)
        decay = math.exp(-KAPPA * YEARS)

        def rms(paths: int) -> float:
            errors = []
            for seed in (64_101, 64_102, 64_103, 64_104, 64_105, 64_106):
                lgm = np.array(
                    simulate_lgm_states(
                        HullWhiteParams(KAPPA, VOL),
                        YEARS,
                        MonteCarloConfig(paths=paths, seed=seed),
                    )
                )
                errors.append((float((decay * lgm).var(ddof=1)) - expected) ** 2)
            return math.sqrt(sum(errors) / len(errors))

        coarse, fine = rms(12_500), rms(200_000)
        assert fine < coarse / 2.0, f"expected ~4x improvement, got {coarse} -> {fine}"


class TestTheMonteCarloPriceCompanion:
    """The oracle that sees what `κ → 0` cannot: the option priced under **genuine mean reversion**.
    The closed form is exact, so the band is `k·SE` with no approximation term."""

    PATHS = 200_000
    SEED = 64_303

    def _mc(self, strike: float, params: SchwartzParams, paths: int, seed: int):
        decay = math.exp(-params.kappa * YEARS)
        lgm = np.array(
            simulate_lgm_states(
                HullWhiteParams(params.kappa, params.sigma),
                YEARS,
                MonteCarloConfig(paths=paths, seed=seed),
            )
        )
        log_spot = math.log(SPOT) * decay + params.level * (1.0 - decay) + decay * lgm
        # The model's own futures price is `E[S_T]`, so the simulated terminal spot must be
        # recentred on it — the martingale property the price rests on, not a free choice.
        terminal = np.exp(log_spot)
        terminal = terminal * (
            schwartz_futures(params, SPOT, YEARS) / float(terminal.mean())
        )
        payoff = np.maximum(terminal - strike, 0.0)
        df = FLAT.df(EXPIRY)
        return (
            float(df * payoff.mean()),
            float(df * payoff.std(ddof=1) / math.sqrt(paths)),
        )

    @pytest.mark.parametrize("strike", [70.0, 85.0, 105.0])
    def test_the_closed_form_sits_INSIDE_the_band(self, strike: float) -> None:
        mean, se = self._mc(strike, _params(), self.PATHS, self.SEED)
        closed = _pv(_option(strike)) / 1_000.0
        assert abs(closed - mean) < 4.0 * se, (
            f"closed {closed:.6f} vs MC {mean:.6f} ± {se:.6f} "
            f"({abs(closed - mean) / se:.2f} SE)"
        )

    def test_THE_RESOLUTION_this_band_rules_out(self) -> None:
        """Named per defect (§4, F-5.3). The band resolves a defect in the **mean-reversion
        factor**: replacing `(1−e^{−2κT})/(2κ)` by `T` — the GBM variance, which is what dropping
        the mean reversion gives — moves the price far outside `4·SE`. It is silent about defects
        smaller than that, and the two closed forms are pinned pointwise at `1e-14` anyway, so the
        band is not their oracle (the T51 clause)."""
        mean, se = self._mc(85.0, _params(), self.PATHS, self.SEED)
        closed = _pv(_option(85.0)) / 1_000.0
        gbm_variance = VOL * VOL * YEARS
        inflated = FLAT.df(EXPIRY) * black_price(
            schwartz_futures(_params(), SPOT, YEARS), 85.0, gbm_variance,
            CommodityOptionKind.CALL,
        )
        assert abs(inflated - closed) > 4.0 * se, (
            f"dropping the mean reversion ({abs(inflated - closed):.6f}) must fall outside the "
            f"{4.0 * se:.6f} band, or this oracle cannot see the model at all"
        )


class TestTheContractAndItsGuards:
    @pytest.mark.parametrize("strike", [55.0, 70.0, 105.0, 130.0])
    def test_PUT_CALL_PARITY_on_the_models_own_futures_price(self, strike: float) -> None:
        """`C − P = df·(F − K)`, model-free given `F`, swept off the money — it survives a
        transcription error in the variance entirely, since the variance cancels."""
        call = _pv(_option(strike, CommodityOptionKind.CALL))
        put = _pv(_option(strike, CommodityOptionKind.PUT))
        forward = schwartz_futures(_params(), SPOT, YEARS)
        assert call - put == pytest.approx(
            1_000.0 * FLAT.df(EXPIRY) * (forward - strike), rel=1e-12
        )

    def test_the_parity_SWEEP_is_genuinely_off_the_money(self) -> None:
        forward = schwartz_futures(_params(), SPOT, YEARS)
        for strike in (55.0, 70.0, 105.0, 130.0):
            assert abs(strike / forward - 1.0) > 0.1, (strike, forward)

    def test_an_EXPIRED_option_is_PV_ZERO_even_DEEP_IN_THE_MONEY(self) -> None:
        for strike in (300.0, 5.0):
            expired = CommodityOption(
                asset=WTI, strike=strike, expiry=VAL - timedelta(days=1),
                notional=Money(1_000.0, USD), kind=CommodityOptionKind.CALL,
            )
            assert _pv(expired) == 0.0

    def test_a_MISSING_commodity_is_a_PricingFailure(self) -> None:
        bare = MarketSnapshot(
            valuation_date=VAL, curves=CurveSet({CurveKey(CurveRole.DISCOUNT, USD): FLAT})
        )
        out = price(_option(), SchwartzModel(bare, _params()))
        assert isinstance(out, PricingFailure) and "WTI" in out.reason

    def test_a_NON_POSITIVE_mean_reversion_is_refused(self) -> None:
        """Every closed form divides by `κ`, so `κ = 0` is not a slow model but an undefined one —
        and the κ → 0 *limit* is approached analytically, never evaluated at zero."""
        for kappa in (0.0, -0.1):
            with pytest.raises(ValueError, match="mean reversion"):
                _params(kappa=kappa)

    def test_a_NEGATIVE_volatility_and_a_NON_FINITE_level_are_refused(self) -> None:
        with pytest.raises(ValueError, match="volatility"):
            _params(sigma=-0.1)
        with pytest.raises(ValueError, match="level"):
            _params(level=float("nan"))

    def test_the_PARAMS_are_their_own_type_not_HullWhiteParams_plus_a_level(self) -> None:
        """§6, and the ruling the spec left to Code. The algebra is shared with Hull-White — both
        are OU — but a *rate's* mean reversion and a *log-commodity-price's* are different ideas,
        and the charter's DRY rule is about knowledge rather than characters. Reusing the rates type
        would make a commodity model carry a type named for an interest-rate model, which is the
        shared-word-not-shared-shape error. T57's LGM reuse was the opposite case: the LGM's numbers
        literally *are* Hull-White's."""
        assert not isinstance(_params(), HullWhiteParams)
        assert _params().kappa == KAPPA and _params().sigma == VOL
