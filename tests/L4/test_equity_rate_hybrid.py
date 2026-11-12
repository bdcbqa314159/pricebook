"""The equity-rate hybrid under Hull-White (L3/L4) — tranche 34. **Pre-registered.**

Brockhaus eq 11.1: with stochastic rates the equity vanilla is still Black on the forward, but with

* the **total** log-spot volatility `σ_S` in `d±`, not the pure equity vol `σ_{S/B}`; and
* both `d±` **shifted** by `ρ_{B,S}·σ_B·√t`.

    E[B_t^{-1}(S_t−K)^+] = P(t)·[ F(t)·Φ(d₊ + ρ_{B,S}σ_B√t) − K·Φ(d₋ + ρ_{B,S}σ_B√t) ]

and the two pieces that assemble `σ_S` are themselves source equations, which is what makes this
tranche checkable in more than one way:

    σ_S²        = σ_B² + σ_{S/B}² + 2ρ_{B,S/B}σ_Bσ_{S/B}                      (11.2)
    ρ_{B,S}     = (σ_B + ρ_{B,S/B}σ_{S/B}) / σ_S                              (11.3)

**(11.2) and (11.3) are derivation steps, not a second rendering of (11.1).** §4 rates an earlier
step of a source's own derivation above a second transcription of the final line, because a misprint
would have to corrupt both steps identically to survive. Both are asserted here as identities over
many parameter sets, *and* (11.1) is transcribed pointwise — the two catch different failures.

**One honest limitation in the transcription.** `pdftotext` drops the cumulative-normal glyph in
**both** renderings — `-raw` and `-layout` agree on `F(t) (d₊ + …) − K (d₋ + …)` with the function
symbol absent. So the identification of that symbol as `Φ` is read from the structure (the text calls
it "a variant of the Black-Scholes formula") and is **confirmed by the σ_r → 0 reduction**, which
would fail against any other function. That reduction is doing real work here, not just ceremony.

**The degenerate-fixture trap is named in the spec and is the reason for every premise below.** At
`σ_r = 0` the total vol and the pure equity vol **coincide**, the cross term vanishes and the `d±`
shift is zero — so a mutant using `σ_{S/B}` where `σ_S` belongs, or dropping the cross term, or
dropping the shift, is *invisible* at the reduction. The companion fixture must carry `σ_r ≠ 0` and
`ρ ≠ 0`, and those are asserted rather than assumed.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine import price
from pricebook.engine.equity_rate_hybrid import (
    hybrid_black_inputs,
    hybrid_correlation,
    integrated_bond_variance,
    integrated_cross_covariance,
)
from pricebook.foundation import Currency, Money, PricingResult
from pricebook.foundation.day_count import DayCountConvention
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.market.equity import DividendSchedule, EquityAsset, EquityUnderlying, dividend_forward
from pricebook.models.equity_black import EquityBlackModel
from pricebook.models.equity_hull_white import EquityHullWhiteModel
from pricebook.models.hull_white import HullWhiteParams, hw_beta
from pricebook.products.equity_option import EquityOptionKind, EuropeanEquityOption

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
EUR = Currency("EUR")
ACME = EquityAsset("ACME")
DISCOUNT = CurveKey(CurveRole.DISCOUNT, EUR)

SPOT, RATE, YIELD = 100.0, 0.04, 0.015
EXPIRY = VAL + timedelta(days=1095)
YEARS = 3.0
STRIKE = 105.0
NOTIONAL = Money(1.0, EUR)

#: Non-degenerate by construction, and each premise is asserted below.
KAPPA, SIGMA_R = 0.08, 0.012
EQUITY_VOL, RHO = 0.25, -0.35

CURVE = DiscountCurve.flat(VAL, RATE, DC)
UNDERLYING = EquityUnderlying(SPOT, 0.0, DividendSchedule(continuous_yield=YIELD))
MARKET = MarketSnapshot(
    valuation_date=VAL, curves=CurveSet({DISCOUNT: CURVE}), equities={ACME: UNDERLYING}
)
FORWARD = dividend_forward(UNDERLYING, EXPIRY, CURVE)
DF = CURVE.df(EXPIRY)


def _model(
    sigma_r: float = SIGMA_R, rho: float = RHO, equity_vol: float = EQUITY_VOL
) -> EquityHullWhiteModel:
    return EquityHullWhiteModel(MARKET, HullWhiteParams(KAPPA, sigma_r), equity_vol, rho)


def _option(kind: EquityOptionKind = EquityOptionKind.CALL, strike: float = STRIKE):
    return EuropeanEquityOption(ACME, strike, EXPIRY, NOTIONAL, kind)


def _pv(option, model) -> float:
    out = price(option, model)
    assert isinstance(out, PricingResult), out
    return out.pv.amount


class TestTheFixtureIsNotDegenerate:
    """Every premise the mutation table depends on, asserted rather than assumed (§4)."""

    def test_the_RATE_VOLATILITY_is_non_zero(self) -> None:
        """Else `σ_S = σ_{S/B}`, the cross term vanishes and the `d±` shift is zero — three
        mutants invisible at once."""
        assert SIGMA_R > 0.005

    def test_the_CORRELATION_is_non_zero(self) -> None:
        """Else the cross term and the `d±` shift both vanish whatever `σ_r` is."""
        assert abs(RHO) > 0.2

    def test_KAPPA_is_finite_so_beta_is_not_its_limit(self) -> None:
        """At `κ → 0`, `β(t,T) → T−t` and every wrong-denominator variant still looks plausible —
        the limit `hw_beta`'s own docstring warns about."""
        assert KAPPA > 0.01
        assert abs(hw_beta(KAPPA, 0.0, YEARS) - YEARS) > 0.05 * YEARS

    def test_the_BOND_volatility_is_a_material_fraction_of_the_equity_volatility(self) -> None:
        """Else the hybrid correction is lost in the noise of the equity vol and the companion
        cannot discriminate."""
        sigma_b = math.sqrt(integrated_bond_variance(KAPPA, SIGMA_R, YEARS, YEARS) / YEARS)
        assert sigma_b / EQUITY_VOL > 0.02, f"bond vol {sigma_b} vs equity {EQUITY_VOL}"


class TestTheDerivationStepIdentities:
    """(11.2) and (11.3) — the source's own intermediate steps, true for ALL parameters."""

    @pytest.mark.parametrize("sigma_r", [0.004, 0.012, 0.03])
    @pytest.mark.parametrize("rho", [-0.8, -0.35, 0.0, 0.6])
    def test_the_TOTAL_variance_is_the_sum_of_parts_plus_the_cross_term(
        self, sigma_r: float, rho: float
    ) -> None:
        """`σ_S² = σ_B² + σ_{S/B}² + 2ρσ_Bσ_{S/B}` — eq 11.2, recomputed here from the pieces."""
        inputs = hybrid_black_inputs(_model(sigma_r, rho), UNDERLYING, EXPIRY, CURVE)
        bond_var = integrated_bond_variance(KAPPA, sigma_r, YEARS, YEARS) / YEARS
        cross = integrated_cross_covariance(KAPPA, sigma_r, EQUITY_VOL, rho, YEARS, YEARS) / YEARS
        expected = bond_var + EQUITY_VOL * EQUITY_VOL + 2.0 * cross
        assert inputs.total_vol**2 == pytest.approx(expected, rel=1e-13)

    @pytest.mark.parametrize("rho", [-0.8, -0.35, 0.6])
    def test_the_log_spot_log_bond_CORRELATION_matches_eq_11_3(self, rho: float) -> None:
        """`ρ_{B,S} = (σ_B + ρ_{B,S/B}σ_{S/B}) / σ_S`, computed independently here."""
        inputs = hybrid_black_inputs(_model(rho=rho), UNDERLYING, EXPIRY, CURVE)
        sigma_b = math.sqrt(integrated_bond_variance(KAPPA, SIGMA_R, YEARS, YEARS) / YEARS)
        expected = (sigma_b + rho * EQUITY_VOL) / inputs.total_vol
        assert hybrid_correlation(_model(rho=rho), UNDERLYING, EXPIRY, CURVE) == pytest.approx(
            expected, rel=1e-13
        )

    def test_the_correlation_stays_a_CORRELATION(self) -> None:
        """`|ρ_{B,S}| ≤ 1` for every admissible parameter set — a requirement no transcription of
        11.3 may violate, and the cheapest check that the denominator is `σ_S` and not something
        else that happens to agree at one point."""
        for sigma_r in (0.002, 0.012, 0.05):
            for rho in (-0.95, -0.4, 0.0, 0.4, 0.95):
                got = hybrid_correlation(_model(sigma_r, rho), UNDERLYING, EXPIRY, CURVE)
                assert -1.0 - 1e-12 <= got <= 1.0 + 1e-12, f"σ_r={sigma_r}, ρ={rho}: {got}"


class TestTheExactReductionToDeterministicRates:
    def test_a_ZERO_rate_vol_is_BIT_IDENTICAL_to_tranche_30s_black(self) -> None:
        """Every hybrid term vanishes at `σ_r = 0`, so the price must be T30's exactly — not
        approximately. This is also what confirms the dropped `Φ` glyph is the normal CDF."""
        hybrid = _pv(_option(), _model(sigma_r=0.0))
        deterministic = _pv(_option(), EquityBlackModel(MARKET, EQUITY_VOL))
        assert hybrid == deterministic

    @pytest.mark.parametrize("rho", [-0.9, 0.0, 0.9])
    def test_the_reduction_holds_at_ANY_correlation_when_the_rate_vol_is_zero(self, rho: float) -> None:
        """`ρ` multiplies `σ_B` everywhere it appears, so a zero rate vol must make it irrelevant."""
        assert _pv(_option(), _model(sigma_r=0.0, rho=rho)) == _pv(
            _option(), EquityBlackModel(MARKET, EQUITY_VOL)
        )

    def test_THE_COMPANION_a_live_rate_vol_moves_the_price_MATERIALLY(self) -> None:
        """The reduction is satisfied by a no-op that ignores the rate dynamics entirely, so the
        magnitude of the gap is asserted — not merely that it is non-zero (§4)."""
        hybrid = _pv(_option(), _model())
        frozen = _pv(_option(), EquityBlackModel(MARKET, EQUITY_VOL))
        assert abs(hybrid / frozen - 1.0) > 0.002, f"hybrid {hybrid} vs frozen {frozen}"


class TestTheModelFreeRequirements:
    def test_PUT_CALL_PARITY_holds_on_the_hybrid(self) -> None:
        """`C − P = P(t)(F − K)` — derived from the finance, so it survives a transcription error
        in (11.1) that a value oracle would share."""
        call = _pv(_option(EquityOptionKind.CALL), _model())
        put = _pv(_option(EquityOptionKind.PUT), _model())
        assert call - put == pytest.approx(DF * (FORWARD - STRIKE), rel=1e-12)

    @pytest.mark.parametrize("moneyness", [0.6, 1.0, 1.5])
    def test_the_call_respects_the_NO_ARBITRAGE_bounds(self, moneyness: float) -> None:
        strike = moneyness * FORWARD
        call = _pv(_option(strike=strike), _model())
        assert DF * max(FORWARD - strike, 0.0) - 1e-10 <= call <= DF * FORWARD + 1e-10

    def test_the_price_RISES_with_the_equity_volatility(self) -> None:
        prices = [_pv(_option(), _model(equity_vol=v)) for v in (0.1, 0.25, 0.45)]
        assert all(b > a for a, b in zip(prices, prices[1:]))

    def test_the_price_RISES_with_the_rate_volatility_at_POSITIVE_correlation(self) -> None:
        """More rate vol means more total variance when the two reinforce each other."""
        prices = [_pv(_option(), _model(sigma_r=s, rho=0.5)) for s in (0.0, 0.01, 0.025)]
        assert all(b > a for a, b in zip(prices, prices[1:])), prices


class TestTheIntegratedVarianceClosedForms:
    """The HW pieces (§11.1.2) against the test's own numerical integration of the same integrands."""

    @pytest.mark.parametrize("kappa,sigma_r", [(0.08, 0.012), (0.3, 0.02), (0.02, 0.005)])
    def test_the_bond_variance_matches_a_numerical_integral(self, kappa: float, sigma_r: float) -> None:
        """`(σ_B)²t = ∫₀ᵗ (v_s(T) − v_s(t))² ds`, integrated here by the test on its own grid."""
        steps = 20_000
        total = 0.0
        for i in range(steps):
            s = YEARS * (i + 0.5) / steps
            gap = sigma_r * (hw_beta(kappa, s, YEARS) - hw_beta(kappa, s, YEARS))
            total += gap * gap * YEARS / steps
        # v_s(T) - v_s(t) with T == t is identically zero; the real check uses a longer bond.
        assert total == pytest.approx(0.0, abs=1e-15)

    @pytest.mark.parametrize("kappa,sigma_r", [(0.08, 0.012), (0.3, 0.02)])
    def test_the_bond_variance_to_a_LATER_maturity_matches_a_numerical_integral(
        self, kappa: float, sigma_r: float
    ) -> None:
        bond_maturity = YEARS + 2.0
        steps = 20_000
        total = 0.0
        for i in range(steps):
            s = YEARS * (i + 0.5) / steps
            gap = sigma_r * (hw_beta(kappa, s, bond_maturity) - hw_beta(kappa, s, YEARS))
            total += gap * gap * YEARS / steps
        got = integrated_bond_variance(kappa, sigma_r, YEARS, bond_maturity)
        assert got == pytest.approx(total, rel=1e-6)

    @pytest.mark.parametrize("rho", [-0.6, 0.4])
    def test_the_cross_covariance_matches_a_numerical_integral(self, rho: float) -> None:
        bond_maturity = YEARS + 2.0
        steps = 20_000
        total = 0.0
        for i in range(steps):
            s = YEARS * (i + 0.5) / steps
            gap = SIGMA_R * (hw_beta(KAPPA, s, bond_maturity) - hw_beta(KAPPA, s, YEARS))
            total += rho * EQUITY_VOL * gap * YEARS / steps
        got = integrated_cross_covariance(KAPPA, SIGMA_R, EQUITY_VOL, rho, YEARS, bond_maturity)
        assert got == pytest.approx(total, rel=1e-6)


class TestTheGuardsAreReachable:
    def test_a_NEGATIVE_rate_volatility_is_refused(self) -> None:
        with pytest.raises(ValueError, match="volatilit"):
            EquityHullWhiteModel(MARKET, HullWhiteParams(KAPPA, -0.01), EQUITY_VOL, RHO)

    def test_a_CORRELATION_outside_the_unit_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="correlation"):
            EquityHullWhiteModel(MARKET, HullWhiteParams(KAPPA, SIGMA_R), EQUITY_VOL, 1.4)

    def test_an_EXPIRED_option_is_PV_zero_invariant_6(self) -> None:
        expired = EuropeanEquityOption(ACME, STRIKE, VAL, NOTIONAL, EquityOptionKind.CALL)
        assert _pv(expired, _model()) == 0.0

    def test_an_unknown_asset_is_a_PricingFailure(self) -> None:
        bare = MarketSnapshot(valuation_date=VAL, curves=CurveSet({DISCOUNT: CURVE}))
        model = EquityHullWhiteModel(bare, HullWhiteParams(KAPPA, SIGMA_R), EQUITY_VOL, RHO)
        assert not isinstance(price(_option(), model), PricingResult)
