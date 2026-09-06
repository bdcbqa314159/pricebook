"""L3 oracle — SABR smile calibration (T1 slice 21).

Known-value oracle (stronger than self-consistency):
1. ROUND-TRIP — a smile generated from a KNOWN seed `SabrParams` at the model-derived forward is
   calibrated back to that seed to tol. Recovers a known answer AND verifies the §3d F-identity by
   construction (the fit's F is the one `SABRModel.black_vol` reads).
2. REPRICE — the fitted params, written to a `SabrSurface`, make `SABRModel.black_vol` reproduce each
   target market vol: calibrate-to-quote == price-to-smile, one fact (§3d).
3. FAILURE-AS-VALUE — a degenerate smile (non-positive strike, lognormal SABR undefined) returns
   `CalibrationFailure`, never a raise or a silent bad fit.
"""

from datetime import date

from pricebook_ng.calibration.calibrate import CalibrationFailure, CalibrationResult, SolveConfig
from pricebook_ng.calibration.sabr_calibrate import SabrCalibrationSpec, calibrate_sabr_smile
from pricebook_ng.foundation import (
    Accrual,
    DayCountConvention,
    Tenor,
    TenorUnit,
    TimeMeasure,
    get_rate_index,
)
from pricebook_ng.market.building_blocks import forward
from pricebook_ng.market.curve import DiscountCurve
from pricebook_ng.market.curve_set import CurveKey, CurveRole, CurveSet
from pricebook_ng.market.quotes import SmileQuote
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.market.vol_surface import SabrParams, SabrSurface, SurfaceKey
from pricebook_ng.models.sabr import SABRModel, sabr_vol

VAL = date(2026, 1, 15)
INDEX = get_rate_index("EURIBOR_3M")
CCY = INDEX.id.currency
DC365 = DayCountConvention.ACT_365_FIXED
TM = TimeMeasure(VAL, DC365)
EXPIRY = VAL + Tenor(1, TenorUnit.YEAR)
BETA = 0.5
SEED = SabrParams(alpha=0.20, beta=BETA, rho=-0.30, nu=0.40)


def _model() -> SABRModel:
    disc = DiscountCurve.flat(TM, 0.030, until=EXPIRY + Tenor(6, TenorUnit.MONTH))
    proj = DiscountCurve.flat(TM, 0.035, until=EXPIRY + Tenor(6, TenorUnit.MONTH))
    curves = CurveSet(
        {CurveKey(CurveRole.DISCOUNT, CCY): disc, CurveKey(CurveRole.PROJECTION, INDEX): proj}
    )
    return SABRModel(MarketSnapshot(VAL, curves))


def _forward(model: SABRModel) -> float:
    accrual = Accrual(EXPIRY, EXPIRY + INDEX.id.tenor, INDEX.accrual.day_count)
    return forward(model.market.curves.projection(INDEX), accrual)


def _seed_smile(model: SABRModel) -> SmileQuote:
    fwd = _forward(model)
    t = TM.year_fraction(EXPIRY)
    strikes = tuple(fwd + d for d in (-0.015, -0.007, 0.0, 0.007, 0.015, 0.025))
    vols = tuple(sabr_vol(fwd, k, t, SEED) for k in strikes)
    return SmileQuote(EXPIRY, strikes, vols)


def test_sabr_calibration_round_trip_recovers_seed() -> None:
    model = _model()
    spec = SabrCalibrationSpec(model, INDEX, _seed_smile(model), BETA, SolveConfig(tolerance=1e-12))
    out = calibrate_sabr_smile(spec)
    assert not isinstance(out, CalibrationFailure)
    params, result = out
    assert isinstance(result, CalibrationResult) and result.converged
    assert abs(params.alpha - SEED.alpha) < 1e-6
    assert abs(params.rho - SEED.rho) < 1e-6
    assert abs(params.nu - SEED.nu) < 1e-6
    assert params.beta == BETA  # β is fixed, not fitted
    assert all(abs(r) < 1e-8 for r in result.residuals)


def test_sabr_calibration_reprice_identity_through_model() -> None:
    # the fitted params, read back by SABRModel, reproduce the target smile at the SAME forward (§3d)
    model = _model()
    smile = _seed_smile(model)
    out = calibrate_sabr_smile(SabrCalibrationSpec(model, INDEX, smile, BETA))
    assert not isinstance(out, CalibrationFailure)
    params, _ = out
    fitted = SABRModel(
        MarketSnapshot(VAL, model.market.curves, surfaces={SurfaceKey(INDEX): SabrSurface({EXPIRY: params})})
    )
    for strike, target in zip(smile.strikes, smile.vols):
        assert abs(fitted.black_vol(INDEX, EXPIRY, strike) - target) < 1e-6


def test_sabr_calibration_degenerate_smile_is_a_failure_value() -> None:
    # a non-positive strike makes the lognormal SABR vol undefined → CalibrationFailure, not a raise
    model = _model()
    fwd = _forward(model)
    t = TM.year_fraction(EXPIRY)
    strikes = (-0.005, 0.0, fwd, fwd + 0.01)  # ascending, but two non-positive strikes
    vols = tuple(0.2 for _ in strikes)
    spec = SabrCalibrationSpec(model, INDEX, SmileQuote(EXPIRY, strikes, vols), BETA)
    assert isinstance(calibrate_sabr_smile(spec), CalibrationFailure)


def test_smile_quote_validates_its_shape() -> None:
    import pytest

    with pytest.raises(ValueError):
        SmileQuote(EXPIRY, (0.02, 0.03), (0.2, 0.2))  # <3 points
    with pytest.raises(ValueError):
        SmileQuote(EXPIRY, (0.02, 0.03, 0.04), (0.2, 0.2))  # length mismatch
    with pytest.raises(ValueError):
        SmileQuote(EXPIRY, (0.04, 0.03, 0.02), (0.2, 0.2, 0.2))  # not ascending
