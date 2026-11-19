"""Commodity spread options — Margrabe (exact) and Kirk (approximate). Tranche 63, PRE-REGISTERED.
`CLARK-2014` ch 2.9.1, 2.9.2.

An option on `S⁽¹⁾ − S⁽²⁾ − K`: the crack, spark and calendar spreads a commodity desk actually
trades. Both named methods collapse to **1-D Black-76 on a combined volatility**, so both reuse
`black_price` and no bivariate normal is needed.

**The source, transcribed from both renderings and reconciled (§4).** pp.74–76:

    Margrabe (K = 0), eq (2.138):
        [σ^{(1;2)}]² = [σ⁽¹⁾]² + [σ⁽²⁾]² − 2ρ₁₂·σ⁽¹⁾·σ⁽²⁾
        V = S₀⁽¹⁾e^{−r^{f;1}T}·N(d₁) − S₀⁽²⁾e^{−r^{f;2}T}·N(d₂)

    Kirk (K ≠ 0), eq (2.139)/(2.140) and p.76:
        Y_t = S_t⁽²⁾ + K·e^{−r^d(T−t)},    σ̃⁽²⁾ = σ⁽²⁾·S_t⁽²⁾/Y_t
        σ_Z = √( [σ⁽¹⁾]² + [σ̃⁽²⁾]² − 2ρ₁₂·σ⁽¹⁾·σ̃⁽²⁾ )
        V₀  = e^{−r^dT}·[ F⁽¹⁾_{0,T}·N(d₁^Z) − Y₀·N(d₂^Z) ],   Z₀ = F₁/(F₂ + K)

**`S₀e^{−r^fT}` is `F·df(T)`**, since `F = S·e^{(r^d−r^f)T}` (T60) — so Margrabe is precisely
`df(T)·black_price(F₁, F₂, σ_M²T, call)`, Black-76 with the *second* forward as the strike. Clark's
own driftless restatement on p.76 makes the same point for Kirk with the strike `F₂ + K`.

**One engine serves both, and the reduction is structural rather than a second code path.** Kirk's
weight `w = F₂/(F₂+K)` is exactly `1` at `K = 0`, where `σ_Z` becomes `σ_M` and the strike becomes
`F₂` — so Kirk *is* Margrabe at zero strike, and the same code is **exact** there and an
**approximation** away from it. That is worth stating precisely because it is the tranche's main
reduction anchor and §4 warns that such an anchor is blind to what it degenerates: `K = 0` kills the
very weight Kirk introduces, so the finite-strike companion is mandatory (T39).

**The weight multiplies BOTH the squared term and the cross term**, which is the detail a mutation
most plausibly gets half-right: `σ̃⁽²⁾` appears in `[σ̃⁽²⁾]²` *and* in `−2ρσ⁽¹⁾σ̃⁽²⁾`.

**Two instruments for the correlation, and neither substitutes for the other.** This is §4's
pointwise-plus-requirement pairing one level up — at the level of the *instrument* rather than the
formula:

- `ρ` enters the combined vol only through the **product** `σ₁σ₂ρ`, so a set of oracles that
  establish `ρ` is present and correctly signed is **silent about its magnitude** (T43's trap,
  stated as "complete about presence and sign, silent about magnitude"). The combined vol is
  therefore transcribed **by value** at non-trivial `ρ`.
- A price pins only one `ρ`-weighted scalar, so it sees `ρ` *weakly* (T17). The MC companion pins
  `ρ` **where it is directly observable** instead: the realised covariance of the two simulated
  log-increments must reproduce `σ₁σ₂ρ·t`, which pins the entry itself rather than a contraction of
  it — and a factorisation bias shows up as failure-to-improve as `√N`.
"""

import math
from datetime import date, timedelta

import numpy as np
import pytest

from pricebook.engine import price
from pricebook.engine.commodity_spread import kirk_weight, spread_total_variance
from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.linalg import pseudo_square_root
from pricebook.foundation.money import Currency, Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.market.commodity import CommodityAsset, CommodityUnderlying, commodity_forward
from pricebook.market.curve import DiscountCurve
from pricebook.market.curve_set import CurveKey, CurveRole, CurveSet
from pricebook.market.snapshot import MarketSnapshot
from pricebook.models.black import black_price
from pricebook.models.commodity_black import CommodityBlackModel
from pricebook.models.commodity_spread import CommoditySpreadModel
from pricebook.products.commodity_option import CommodityOption, CommodityOptionKind
from pricebook.products.commodity_spread import CommoditySpreadOption, SpreadLegs

VAL = date(2026, 1, 15)
DC = DayCountConvention.ACT_365_FIXED
USD = Currency("USD")
WTI, BRENT = CommodityAsset("WTI"), CommodityAsset("BRENT")

RATE = 0.042
#: **Different carries as well as different vols**, so no mutation that confuses the two legs can
#: survive by symmetry.
SPOT_LONG, CONV_LONG = 82.0, 0.012
SPOT_SHORT, CONV_SHORT = 76.5, 0.028
#: **σ₁ ≠ σ₂ and ρ ≠ 0**: at ρ = 0 the cross term vanishes and a dropped `−2ρσ₁σ₂` survives; at
#: σ₁ = σ₂ with ρ = 1 the combined vol is exactly zero and the structure is invisible.
VOL_LONG, VOL_SHORT, RHO = 0.34, 0.26, 0.45
EXPIRY = VAL + timedelta(days=912)
YEARS = 912 / 365.0
FLAT = DiscountCurve.flat(VAL, RATE, DC)


def _market() -> MarketSnapshot:
    return MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet({CurveKey(CurveRole.DISCOUNT, USD): FLAT}),
        commodities={
            WTI: CommodityUnderlying(WTI, SPOT_LONG, CONV_LONG),
            BRENT: CommodityUnderlying(BRENT, SPOT_SHORT, CONV_SHORT),
        },
    )


def _model(
    vol_long: float = VOL_LONG, vol_short: float = VOL_SHORT, rho: float = RHO
) -> CommoditySpreadModel:
    return CommoditySpreadModel(_market(), vol_long, vol_short, rho)


def _spread(
    strike: float = 4.0, kind: CommodityOptionKind = CommodityOptionKind.CALL
) -> CommoditySpreadOption:
    return CommoditySpreadOption(
        legs=SpreadLegs(long=WTI, short=BRENT),
        strike=strike,
        expiry=EXPIRY,
        notional=Money(1_000.0, USD),
        kind=kind,
    )


def _pv(product, model: CommoditySpreadModel | None = None) -> float:
    out = price(product, model or _model())
    assert isinstance(out, PricingResult), out
    return out.pv.amount


def _forwards() -> tuple[float, float]:
    market = _market()
    return (
        commodity_forward(market.commodities[WTI], EXPIRY, FLAT),
        commodity_forward(market.commodities[BRENT], EXPIRY, FLAT),
    )


def _reference_variance(
    vol_long: float, vol_short: float, rho: float, strike: float
) -> float:
    """`σ_Z²·T` from the print: the weight on the short leg, then the three-term combination."""
    _, forward_short = _forwards()
    weight = forward_short / (forward_short + strike)
    weighted = vol_short * weight
    return (
        vol_long * vol_long + weighted * weighted - 2.0 * rho * vol_long * weighted
    ) * YEARS


class TestThePremisesTheOraclesRestOn:
    def test_the_correlation_is_NON_ZERO_and_the_vols_are_UNEQUAL(self) -> None:
        """At `ρ = 0` the cross term vanishes, so a dropped `−2ρσ₁σ₂` survives; at `σ₁ = σ₂` with
        `ρ = 1` the combined vol is exactly zero and every structural error hides behind it."""
        assert abs(RHO) > 0.2
        assert abs(VOL_LONG - VOL_SHORT) > 0.05

    def test_the_two_LEGS_differ_in_carry_as_well_as_in_vol(self) -> None:
        """So a mutation that swaps the legs cannot survive by symmetry — the forwards differ for
        two independent reasons (spot and convenience yield), not one."""
        long_forward, short_forward = _forwards()
        assert abs(long_forward / short_forward - 1.0) > 0.05
        assert CONV_LONG != CONV_SHORT

    def test_the_horizon_is_NOT_one_year_and_the_strike_is_AWAY_from_zero(self) -> None:
        assert abs(YEARS - 1.0) > 1.4
        assert kirk_weight(_forwards()[1], 4.0) < 0.96


class TestTheCombinedVolatilityPointwise:
    """§4: the one new load-bearing formula, transcribed by **value** — which is the T43 point. An
    oracle set establishing that `ρ` is present and correctly signed would be *complete about
    presence and sign and silent about magnitude*, and `ρ` enters only through the product
    `σ₁σ₂ρ`, so the magnitude is exactly what must be pinned."""

    @pytest.mark.parametrize("rho", [-0.8, -0.3, 0.0, 0.45, 0.9])
    @pytest.mark.parametrize("strike", [0.0, 4.0, 20.0])
    def test_the_total_variance_matches_the_print(self, rho: float, strike: float) -> None:
        _, forward_short = _forwards()
        got = spread_total_variance(
            VOL_LONG, VOL_SHORT, rho, kirk_weight(forward_short, strike), YEARS
        )
        assert got == pytest.approx(
            _reference_variance(VOL_LONG, VOL_SHORT, rho, strike), rel=1e-14
        )

    def test_the_WEIGHT_multiplies_BOTH_the_squared_and_the_CROSS_term(self) -> None:
        """The detail a mutation most plausibly gets half-right. `σ̃⁽²⁾ = σ⁽²⁾·F₂/(F₂+K)` appears in
        `[σ̃⁽²⁾]²` *and* in `−2ρσ⁽¹⁾σ̃⁽²⁾`; weighting only one gives a different number, which this
        measures rather than asserts."""
        _, forward_short = _forwards()
        weight = kirk_weight(forward_short, 20.0)
        correct = spread_total_variance(VOL_LONG, VOL_SHORT, RHO, weight, YEARS)
        squared_only = (
            VOL_LONG**2
            + (VOL_SHORT * weight) ** 2
            - 2.0 * RHO * VOL_LONG * VOL_SHORT
        ) * YEARS
        cross_only = (
            VOL_LONG**2
            + VOL_SHORT**2
            - 2.0 * RHO * VOL_LONG * VOL_SHORT * weight
        ) * YEARS
        assert correct != pytest.approx(squared_only, rel=1e-6)
        assert correct != pytest.approx(cross_only, rel=1e-6)
        assert weight < 0.85, weight

    def test_the_KIRK_WEIGHT_is_one_at_zero_strike_and_falls_with_the_strike(self) -> None:
        _, forward_short = _forwards()
        assert kirk_weight(forward_short, 0.0) == 1.0
        weights = [kirk_weight(forward_short, k) for k in (0.0, 5.0, 20.0, 60.0)]
        assert all(b < a for a, b in zip(weights, weights[1:])), weights


class TestTheCorrelationLimitsAreRequirementOracles:
    """Closed forms the finance forces, so they survive a transcription error the source-value
    oracle would share — and they pin the **cross term's sign**, which no single-ρ check does."""

    def test_PERFECT_correlation_gives_the_VOL_DIFFERENCE(self) -> None:
        """At `ρ = +1` the two logs move together, so `σ_M = |σ₁ − σ₂|`. A flipped cross-term sign
        would give `σ₁ + σ₂` here and the two are a factor of five apart on this fixture."""
        got = spread_total_variance(VOL_LONG, VOL_SHORT, 1.0, 1.0, YEARS)
        assert got == pytest.approx((VOL_LONG - VOL_SHORT) ** 2 * YEARS, rel=1e-13)
        assert (VOL_LONG + VOL_SHORT) ** 2 / (VOL_LONG - VOL_SHORT) ** 2 > 20.0

    def test_PERFECT_ANTI_correlation_gives_the_VOL_SUM(self) -> None:
        got = spread_total_variance(VOL_LONG, VOL_SHORT, -1.0, 1.0, YEARS)
        assert got == pytest.approx((VOL_LONG + VOL_SHORT) ** 2 * YEARS, rel=1e-13)

    def test_EQUAL_vols_at_perfect_correlation_give_EXACTLY_zero(self) -> None:
        """The degenerate corner the fixture deliberately avoids: a perfectly-correlated pair with
        equal vols has a riskless spread. Asserted here precisely so the main fixture's avoidance
        of it is a stated premise rather than an accident."""
        assert spread_total_variance(0.3, 0.3, 1.0, 1.0, YEARS) == pytest.approx(0.0, abs=1e-18)

    def test_the_variance_is_MONOTONE_DECREASING_in_the_correlation(self) -> None:
        values = [
            spread_total_variance(VOL_LONG, VOL_SHORT, rho, 1.0, YEARS)
            for rho in (-1.0, -0.5, 0.0, 0.5, 1.0)
        ]
        assert all(b < a for a, b in zip(values, values[1:])), values

    def test_the_variance_is_NEVER_NEGATIVE_across_the_whole_correlation_range(self) -> None:
        """`σ₁² + σ₂² − 2ρσ₁σ₂ ≥ (σ₁−σ₂)² ≥ 0` for `|ρ| ≤ 1`. A requirement oracle: a combined
        variance that went negative would make the method undefined, not merely inaccurate."""
        for rho in np.linspace(-1.0, 1.0, 41):
            for weight in (0.3, 0.7, 1.0):
                assert (
                    spread_total_variance(VOL_LONG, VOL_SHORT, float(rho), weight, YEARS) >= 0.0
                )


class TestMargrabeIsExactAndKirkReducesToIt:
    def test_ZERO_strike_is_EXACTLY_margrabe_black_on_the_second_forward(self) -> None:
        """Eq (2.138) rearranged through `S₀e^{−r^fT} = F·df(T)`: Margrabe is Black-76 with the
        *second* forward as the strike. Exact, and the whole reduction anchor."""
        long_forward, short_forward = _forwards()
        variance = (
            VOL_LONG**2 + VOL_SHORT**2 - 2.0 * RHO * VOL_LONG * VOL_SHORT
        ) * YEARS
        direct = black_price(long_forward, short_forward, variance, CommodityOptionKind.CALL)
        assert _pv(_spread(strike=0.0)) == 1_000.0 * FLAT.df(EXPIRY) * direct

    def test_the_COMPANION_is_non_degenerate_in_what_the_reduction_KILLS(self) -> None:
        """**T39.** The `K = 0` anchor is exact *because* the weight is 1 — which is precisely the
        quantity Kirk introduces, so the anchor is structurally blind to it. The finite-strike
        companion exercises a weight measurably below 1, and this states by how much."""
        _, forward_short = _forwards()
        assert kirk_weight(forward_short, 20.0) < 0.8
        zero = spread_total_variance(VOL_LONG, VOL_SHORT, RHO, 1.0, YEARS)
        finite = spread_total_variance(
            VOL_LONG, VOL_SHORT, RHO, kirk_weight(forward_short, 20.0), YEARS
        )
        assert abs(finite / zero - 1.0) > 0.1, (zero, finite)

    def test_a_ZERO_SHORT_VOL_reduces_to_the_T60_VANILLA(self) -> None:
        """A spread against a deterministic leg is an ordinary option on the first leg, struck at
        `F₂ + K`. **A cross-product check against a different engine** — T60's — so it catches a
        shared reading error no within-product oracle can (§4)."""
        long_forward, short_forward = _forwards()
        vanilla = CommodityOption(
            asset=WTI,
            strike=short_forward + 4.0,
            expiry=EXPIRY,
            notional=Money(1_000.0, USD),
            kind=CommodityOptionKind.CALL,
        )
        out = price(vanilla, CommodityBlackModel(_market(), VOL_LONG))
        assert isinstance(out, PricingResult)
        assert _pv(_spread(strike=4.0), _model(vol_short=0.0)) == pytest.approx(
            out.pv.amount, rel=1e-13
        )

    @pytest.mark.parametrize("strike", [-6.0, 0.0, 4.0, 15.0])
    def test_PUT_CALL_PARITY_on_the_spread(self, strike: float) -> None:
        """`C − P = df·(F₁ − F₂ − K)`, model-free — it survives a transcription error in the
        combined vol entirely, since the vol cancels."""
        long_forward, short_forward = _forwards()
        call = _pv(_spread(strike, CommodityOptionKind.CALL))
        put = _pv(_spread(strike, CommodityOptionKind.PUT))
        assert call - put == pytest.approx(
            1_000.0 * FLAT.df(EXPIRY) * (long_forward - short_forward - strike), rel=1e-11
        )


#: Correlated-GBM runs are deterministic in their arguments and several oracles want the same one
#: twice, so they are memoised — a lossless reduction (T50/T53), not a weakening.
_MC_CACHE: dict[tuple, tuple[float, float]] = {}


def _correlated_increments(rho: float, paths: int, seed: int) -> np.ndarray:
    """Two correlated standard normals per path, factorised through `pseudo_square_root`.

    Genuine reuse of the tranche-16 factorisation rather than a hand-rolled `ρZ₁ + √(1−ρ²)Z₂`: at
    two factors on a 2×2 correlation the pseudo square root is **exact** (full rank, nothing
    truncated), so the reuse costs no accuracy and the factorisation is the one already oracled by
    `A·Aᵀ = C` at L0.
    """
    factor = np.array(pseudo_square_root(((1.0, rho), (rho, 1.0)), 2))
    rng = np.random.default_rng(seed)
    return rng.standard_normal((paths, 2)) @ factor.T


def _mc_spread(strike: float, rho: float, paths: int, seed: int) -> tuple[float, float]:
    """The discounted spread-call price by simulation, and its standard error."""
    key = (strike, rho, paths, seed)
    if key in _MC_CACHE:
        return _MC_CACHE[key]
    normals = _correlated_increments(rho, paths, seed)
    long_forward, short_forward = _forwards()
    root = math.sqrt(YEARS)
    terminal_long = long_forward * np.exp(
        -0.5 * VOL_LONG**2 * YEARS + VOL_LONG * root * normals[:, 0]
    )
    terminal_short = short_forward * np.exp(
        -0.5 * VOL_SHORT**2 * YEARS + VOL_SHORT * root * normals[:, 1]
    )
    payoff = np.maximum(terminal_long - terminal_short - strike, 0.0)
    df = FLAT.df(EXPIRY)
    out = (
        float(df * payoff.mean()),
        float(df * payoff.std(ddof=1) / math.sqrt(paths)),
    )
    _MC_CACHE[key] = out
    return out


class TestTheSimulatedCorrelationIsUnbiasedFIRST:
    """**Before the companion is credited with anything (T19).** A factorisation must be proved
    unbiased before a price built on it is trusted — otherwise a correlation bias is read as a Kirk
    approximation error, and the two are indistinguishable at the price level.

    This is also the T17 pin: a *price* sees `ρ` only through one `ρ`-weighted scalar, so it pins ρ
    weakly. The **realised covariance** of the log-increments pins the entry itself.
    """

    PATHS = 200_000

    @pytest.mark.parametrize("rho", [-0.7, 0.0, 0.45, 0.9])
    def test_the_realised_COVARIANCE_reproduces_sigma_1_sigma_2_rho_t(self, rho: float) -> None:
        """`Cov(ln S₁(t), ln S₂(t)) = σ₁σ₂ρ·t` — the quantity `ρ` *is*, rather than a contraction
        of it. This is what pins the correlation itself, which no price can do (T17)."""
        normals = _correlated_increments(rho, self.PATHS, 63_001)
        root = math.sqrt(YEARS)
        log_long = VOL_LONG * root * normals[:, 0]
        log_short = VOL_SHORT * root * normals[:, 1]
        realised = float(np.cov(log_long, log_short, ddof=1)[0, 1])
        expected = VOL_LONG * VOL_SHORT * rho * YEARS
        standard_error = (
            VOL_LONG * VOL_SHORT * YEARS / math.sqrt(self.PATHS)
        ) * math.sqrt(1.0 + rho * rho)
        assert abs(realised - expected) < 4.0 * standard_error, (
            f"realised {realised:.8f} vs {expected:.8f} "
            f"({abs(realised - expected) / standard_error:.2f} SE)"
        )

    def test_the_covariance_error_HALVES_as_the_root_of_the_path_count(self) -> None:
        """A **bias** would not improve with paths, so failure-to-improve is the detector (T17).
        Averaged over seeds, because a null needs averaging as much as an effect does (T21) — one
        seed can make a biased factorisation look convergent."""
        expected = VOL_LONG * VOL_SHORT * RHO * YEARS

        def rms(paths: int) -> float:
            errors = []
            for seed in (63_101, 63_102, 63_103, 63_104, 63_105, 63_106):
                normals = _correlated_increments(RHO, paths, seed)
                root = math.sqrt(YEARS)
                realised = float(
                    np.cov(
                        VOL_LONG * root * normals[:, 0],
                        VOL_SHORT * root * normals[:, 1],
                        ddof=1,
                    )[0, 1]
                )
                errors.append((realised - expected) ** 2)
            return math.sqrt(sum(errors) / len(errors))

        coarse, fine = rms(12_500), rms(200_000)
        assert fine < coarse / 2.0, f"expected ~4x improvement, got {coarse} -> {fine}"

    def test_the_factorisation_is_EXACT_at_two_factors(self) -> None:
        """The premise the reuse rests on: a 2×2 correlation at two factors is full rank, so
        `pseudo_square_root` truncates nothing and `A·Aᵀ = C` holds exactly — the reuse costs no
        accuracy and inherits the L0 oracle."""
        for rho in (-0.9, 0.0, 0.45, 0.9):
            factor = np.array(pseudo_square_root(((1.0, rho), (rho, 1.0)), 2))
            reconstructed = factor @ factor.T
            assert reconstructed == pytest.approx(
                np.array([[1.0, rho], [rho, 1.0]]), abs=1e-14
            )


class TestKirksApproximationErrorIsMEASURED:
    """Kirk is an approximation (`K ≪ S⁽²⁾` is its stated assumption), so the deliverable is its
    measured error — banded as `k·SE + the measured Kirk error`, the two terms kept apart (§4)."""

    PATHS = 80_000
    BIG_PATHS = 600_000
    SEED = 63_303

    def test_MARGRABE_is_exact_against_the_simulation(self) -> None:
        """At `K = 0` the method is Margrabe and therefore **exact**, so the only error is sampling
        — the band is `4·SE` with no approximation term, which is what "exact" means operationally
        and what separates this from every Kirk point below."""
        mean, se = _mc_spread(0.0, RHO, self.BIG_PATHS, self.SEED)
        closed = _pv(_spread(strike=0.0)) / 1_000.0
        assert abs(closed - mean) < 4.0 * se, (
            f"Margrabe {closed:.6f} vs MC {mean:.6f} ± {se:.6f} "
            f"({abs(closed - mean) / se:.2f} SE) — this one has no approximation term"
        )

    def test_the_KIRK_error_GROWS_with_the_strike_away_from_zero(self) -> None:
        """Kirk's assumption is `K ≪ S⁽²⁾`, so the error must grow as the strike leaves zero —
        measured, with every point resolved above the reference run's own noise."""
        measured: list[tuple[float, float]] = []
        for strike in (4.0, 15.0, 40.0):
            mean, se = _mc_spread(strike, RHO, self.BIG_PATHS, self.SEED)
            kirk = _pv(_spread(strike)) / 1_000.0
            assert abs(kirk - mean) > 2.0 * se, (
                f"at K={strike} the Kirk error is not resolved above the reference SE "
                f"({abs(kirk - mean) / se:.2f} SE) — that point measures noise"
            )
            measured.append((strike, abs(kirk - mean) / mean))
        assert all(b[1] > a[1] for a, b in zip(measured, measured[1:])), measured

    @pytest.mark.parametrize("strike", [4.0, 15.0])
    def test_the_KIRK_price_sits_inside_FOUR_SE_PLUS_the_measured_error(
        self, strike: float
    ) -> None:
        mean, se = _mc_spread(strike, RHO, self.PATHS, 63_404)
        reference, reference_se = _mc_spread(strike, RHO, self.BIG_PATHS, self.SEED)
        kirk = _pv(_spread(strike)) / 1_000.0
        bound = abs(kirk - reference) + 2.0 * reference_se
        assert abs(kirk - mean) < 4.0 * se + bound, (
            f"Kirk {kirk:.6f} vs MC {mean:.6f} ± {se:.6f}; error bounded by {bound:.6f}"
        )

    def test_THE_RESOLUTION_this_band_rules_out(self) -> None:
        """Named per defect (§4, F-5.3), and per the T51 clause: the combined **variance** is closed
        form and pinned pointwise at `1e-14`, so it is not this band's business. What the band is
        for is the one thing no exact check can see — **Kirk's lognormal assumption about
        `S⁽²⁾ + K`**, to which an oracle built from the combined vol is structurally blind."""
        mean, se = _mc_spread(15.0, RHO, self.PATHS, 63_404)
        reference, reference_se = _mc_spread(15.0, RHO, self.BIG_PATHS, self.SEED)
        kirk = _pv(_spread(15.0)) / 1_000.0
        band = 4.0 * se + abs(kirk - reference) + 2.0 * reference_se

        long_forward, short_forward = _forwards()
        inflated = FLAT.df(EXPIRY) * black_price(
            long_forward,
            short_forward + 15.0,
            1.25 * spread_total_variance(
                VOL_LONG, VOL_SHORT, RHO, kirk_weight(short_forward, 15.0), YEARS
            ),
            CommodityOptionKind.CALL,
        )
        assert abs(inflated - kirk) > band, (
            f"a 25% combined-variance defect ({abs(inflated - kirk):.6f}) must fall outside the "
            f"{band:.6f} band, or this oracle resolves nothing"
        )


class TestTheContractAndItsGuards:
    def test_an_EXPIRED_spread_is_PV_ZERO_even_DEEP_IN_THE_MONEY(self) -> None:
        """Invariant 6, with the F-4.1 sub-shape: an out-of-the-money expired option returns `0`
        either way, so the guard is only visible in the money. `K = −50` against a ~+6 spread."""
        for strike in (200.0, -50.0):
            expired = CommoditySpreadOption(
                legs=SpreadLegs(long=WTI, short=BRENT),
                strike=strike,
                expiry=VAL - timedelta(days=1),
                notional=Money(1_000.0, USD),
                kind=CommodityOptionKind.CALL,
            )
            assert _pv(expired) == 0.0

    def test_a_NEGATIVE_strike_is_LEGITIMATE_for_a_spread(self) -> None:
        """Unlike every other option in the tree, a spread option's strike may be zero or negative:
        `S₁ − S₂ − K` with `K < 0` just means the holder is paid to enter. So there is no
        positive-strike construction guard here — the real constraint is `F₂ + K > 0`, which needs
        market data and is therefore a `PricingFailure`, not a `ValueError` (§3.4)."""
        assert _pv(_spread(strike=-10.0)) > _pv(_spread(strike=10.0))

    def test_a_strike_that_makes_the_SHIFTED_FORWARD_non_positive_is_REFUSED(self) -> None:
        """Kirk's weight is `F₂/(F₂+K)`, so `F₂ + K ≤ 0` has no lognormal to match and the method
        is undefined rather than inaccurate. Refused in domain vocabulary, as a value."""
        _, short_forward = _forwards()
        out = price(_spread(strike=-short_forward - 1.0), _model())
        assert isinstance(out, PricingFailure)
        assert "shifted" in out.reason or "non-positive" in out.reason, out.reason

    def test_a_MISSING_leg_is_a_PricingFailure_naming_WHICH_leg(self) -> None:
        bare = MarketSnapshot(
            valuation_date=VAL,
            curves=CurveSet({CurveKey(CurveRole.DISCOUNT, USD): FLAT}),
            commodities={WTI: CommodityUnderlying(WTI, SPOT_LONG, CONV_LONG)},
        )
        out = price(_spread(), CommoditySpreadModel(bare, VOL_LONG, VOL_SHORT, RHO))
        assert isinstance(out, PricingFailure) and "BRENT" in out.reason

    def test_a_CORRELATION_outside_the_unit_interval_is_refused(self) -> None:
        for rho in (-1.0001, 1.0001, float("nan")):
            with pytest.raises(ValueError, match="correlation"):
                CommoditySpreadModel(_market(), VOL_LONG, VOL_SHORT, rho)

    def test_a_NEGATIVE_volatility_is_refused(self) -> None:
        with pytest.raises(ValueError, match="volatilit"):
            CommoditySpreadModel(_market(), -0.1, VOL_SHORT, RHO)

    def test_the_two_LEGS_must_DIFFER(self) -> None:
        """A spread of an asset against itself is identically `−K`, not an option — and it would
        price with a combined vol of `σ√(2−2ρ)` that no longer describes anything."""
        with pytest.raises(ValueError, match="differ"):
            SpreadLegs(long=WTI, short=WTI)

    def test_the_spread_round_trips_and_refuses_an_unknown_version(self) -> None:
        product = _spread()
        assert CommoditySpreadOption.from_dict(product.to_dict()) == product
        bad = product.to_dict()
        bad["schema_version"] = 99
        with pytest.raises(ValueError, match="reads up to"):
            CommoditySpreadOption.from_dict(bad)

    def test_the_LEGS_are_an_ORDERED_pair_and_equality_sees_the_order(self) -> None:
        """§6: the legs are a pair, not a set — `WTI − BRENT` is not `BRENT − WTI`, and the
        bundling exists to meet the field limit without losing that."""
        assert SpreadLegs(long=WTI, short=BRENT) != SpreadLegs(long=BRENT, short=WTI)
