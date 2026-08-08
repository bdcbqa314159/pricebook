"""L4 oracle — cross-currency basis curve under a USD CSA (slice 6c, C1 close).

The foreign-collateral EUR curve (EUR discounted under USD collateral, keyed `(DISCOUNT, EUR, USD)`)
is bootstrapped so the xccy basis swaps reprice to zero. Oracles:

1. **Reprice-to-zero** — every calibrating xccy basis swap prices to ~0 through the L4 engine off
   the bootstrapped curve (doc 18 §8), for zero AND a real basis.
2. **CIP closed-form anchor** — at ZERO basis the curve reproduces `df_eur` (EUR-OIS), so the FX
   forward `F = S·df_eur^usd/df_usd` equals the textbook `S·df_eur/df_usd` to 1e-10 (stronger than
   reprice-to-par self-consistency). A real basis moves the curve below CIP.

Note (design finding, flagged to Cowork): the constant-notional both-legs-OIS-flat xccy basis swap
has the FX spot and USD discount CANCEL out of the reprice-to-zero condition (the domestic leg
telescopes to par; `N_eur = N_usd/S`). Spot/USD-DF enter only the CIP FX-forward oracle, not the
bootstrap — so the curve is pinned by `df_eur` + the basis alone.
"""

from datetime import date

from pricebook_ng.calibration.calibrate import (
    CalibrationSpec,
    CurrencyCurves,
    CurveBuild,
    ParSwapQuote,
    XccyBasisQuote,
    XccyBuild,
    calibrate,
    xccy_swaps,
)
from pricebook_ng.engine import price
from pricebook_ng.foundation import (
    Currency,
    DayCountConvention,
    Frequency,
    PricingResult,
    Tenor,
    TenorUnit,
    fx_pair,
    get_rate_index,
)
from pricebook_ng.market.snapshot import ScalarKey

VAL = date(2026, 1, 15)
SOFR = get_rate_index("SOFR")
ESTR = get_rate_index("ESTR")
EURIBOR_3M = get_rate_index("EURIBOR_3M")
DC = DayCountConvention.ACT_365_FIXED
Y = TenorUnit.YEAR
A = Frequency.ANNUAL
S = 1.08  # EURUSD spot


def _q(pairs):
    return tuple(ParSwapQuote(Tenor(t, Y), r) for t, r in pairs)


USD = CurrencyCurves(
    Currency.USD,
    CurveBuild(SOFR, A, DC, _q([(1, .040), (2, .041), (3, .042), (5, .043)])),
    CurveBuild(SOFR, A, DC, _q([(1, .040), (2, .041), (3, .042), (5, .043)])),
)
EUR = CurrencyCurves(
    Currency.EUR,
    CurveBuild(ESTR, A, DC, _q([(1, .030), (2, .032), (3, .034), (5, .036)])),
    CurveBuild(EURIBOR_3M, A, DC, _q([(1, .0312), (2, .0332), (3, .0352), (5, .0372)])),
)
FX = {ScalarKey(fx_pair(Currency.EUR, Currency.USD)): S}


def _spec(spread: float) -> CalibrationSpec:
    quotes = tuple(XccyBasisQuote(Tenor(t, Y), spread) for t in (1, 2, 3, 5))
    return CalibrationSpec(
        VAL, (USD, EUR), xccy=XccyBuild(Currency.EUR, Currency.USD, quotes), fx=FX
    )


def test_xccy_swaps_reprice_to_zero() -> None:
    for spread in (0.0, 0.0010):
        out = calibrate(_spec(spread))
        assert not isinstance(out, tuple) or out[1].converged, out
        model, _ = out
        for sw in xccy_swaps(_spec(spread)):
            r = price(sw, model)
            assert isinstance(r, PricingResult)
            assert abs(r.pv.amount) < 1e-9, (spread, r.pv.amount)
            assert r.basis == Currency.USD


def test_cip_zero_basis_reproduces_textbook_forward() -> None:
    model, _ = calibrate(_spec(0.0))
    curves = model.market.curves
    eur_ois = curves.discount(Currency.EUR)  # df_eur
    eur_usd = curves.discount(Currency.EUR, Currency.USD)  # df_eur^usd
    usd = curves.discount(Currency.USD)  # df_usd
    for t in (1, 2, 3, 5):
        d = VAL + Tenor(t, Y)
        assert abs(eur_usd.df(d) - eur_ois.df(d)) < 1e-10  # zero basis reproduces EUR-OIS
        forward = S * eur_usd.df(d) / usd.df(d)
        cip = S * eur_ois.df(d) / usd.df(d)  # textbook covered-interest-parity forward
        assert abs(forward - cip) < 1e-10


def test_nonzero_basis_moves_the_curve_below_cip() -> None:
    model, _ = calibrate(_spec(0.0010))
    curves = model.market.curves
    d = VAL + Tenor(5, Y)
    eur_usd = curves.discount(Currency.EUR, Currency.USD).df(d)
    eur_ois = curves.discount(Currency.EUR).df(d)
    assert eur_usd < eur_ois  # positive basis on the received leg lowers the collateral DFs
