"""The nominal cap/floor pricer. MCINERNEY-ZASTAWNIAK §3.5 eq 3.21.

The strip walk, the curve lookups and the valuation-date partition. The caplet analytics have their
own oracles against an independent transcription in `tests/L3/test_hull_white_caplet.py`; here the
question is whether the *engine* composes them correctly.

The headline oracle is **cap - floor = the forward-rate swap value**, a cross-product identity
derived from the payoff (`[ω(L−K)]⁺` for ω=±1 differences to `L−K`), computed on the other side from
discount factors alone with no option formula in sight. It catches the leg/sign error that no
magnitude check notices, and it does not telescope: the right-hand side never calls `hw_caplet`.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine import price
from pricebook.engine.nominal_cap import nominal_caplet_value
from pricebook.foundation import (
    CapFloorKind,
    CouponSchedule,
    Currency,
    DayCountConvention,
    Money,
    PricingFailure,
    PricingResult,
)
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.models.hull_white import (
    CapletPeriod,
    HullWhiteModel,
    HullWhiteParams,
    bond_option_variance,
    hw_caplet,
)
from pricebook.products import NominalCap

VAL = date(2026, 1, 15)
EUR = Currency("EUR")
DC = DayCountConvention.ACT_365_FIXED
MEASURE = TimeMeasure(VAL, DC)
N = 10_000_000.0
KEY = CurveKey(CurveRole.DISCOUNT, EUR)
# SEMI-ANNUAL (180-day) periods, so every accrual is ~0.4932 and NOT 1.0. On annual ACT/365F
# periods `tau` is exactly 1.0, which makes the `(1 + tau*K)` scaling of eq 3.21 invisible: a
# mutation dropping the accrual passed all 312 tests. The fixture's non-unit accrual is what gives
# these oracles power over it, and it is asserted below rather than assumed.
DATES = tuple(VAL + timedelta(days=180 * k) for k in range(5))   # anchor + 4 semi-annual periods
PARAMS = HullWhiteParams(0.05, 0.010)

# SLOPED, so the caplets are not all equal and a mis-indexed strip would show.
CURVE = DiscountCurve(MEASURE, (0.0, 1.0, 2.0, 4.0), (0.026, 0.030, 0.034, 0.038))


def _model(curve: DiscountCurve = CURVE, params: HullWhiteParams = PARAMS,
           vd: date = VAL) -> HullWhiteModel:
    return HullWhiteModel(
        MarketSnapshot(valuation_date=vd, curves=CurveSet({KEY: curve})), params
    )


def _cap(strike: float = 0.03, kind: CapFloorKind = CapFloorKind.CAP,
         n_periods: int = 4) -> NominalCap:
    return NominalCap(CouponSchedule(DATES[: n_periods + 1], DC), strike, Money(N, EUR), kind)


def _pv(cap: NominalCap, model: HullWhiteModel) -> float:
    out = price(cap, model)
    assert isinstance(out, PricingResult), out
    return out.pv.amount


# ---- the strip IS the sum of its caplets ----

def test_the_strip_equals_the_SUM_of_its_caplets_priced_independently():
    cap = _cap()
    model = _model()
    expected = 0.0
    for start, pay, accrual in cap.schedule.periods():
        nu = bond_option_variance(PARAMS, MEASURE.year_fraction(start), MEASURE.year_fraction(pay))
        unit = CapletPeriod(CURVE.df(start), CURVE.df(pay), accrual)
        expected += N * hw_caplet(unit, cap.strike, nu, CapFloorKind.CAP)
    assert _pv(cap, model) == pytest.approx(expected, rel=1e-14)


def test_the_fixture_accruals_are_NOT_unity():
    """The premise the `(1 + tau*K)` oracles rest on. At `tau = 1` the accrual cannot be seen."""
    accruals = [p[2] for p in _cap().schedule.periods()]
    assert all(abs(a - 1.0) > 0.4 for a in accruals), accruals


def test_the_FIRST_caplet_has_already_FIXED_so_it_is_exactly_its_intrinsic():
    """Its period starts at the anchor, so the option expires at once: `nu(0,0) = 0` because
    `(1 - e^{-2*kappa*0})` vanishes. The caplet is therefore its intrinsic on a forward that is no
    longer uncertain — and on this fixture that intrinsic is exactly zero, the rate being below the
    3% strike.

    Discovered by the distinctness test below failing its `> 0` assertion, which was the wrong
    requirement: a caplet whose rate has already fixed and is out of the money is worth nothing,
    and asserting otherwise would have forced a fixture that hid the `nu(0,0) = 0` reduction."""
    cap, model = _cap(), _model()
    first = next(iter(cap.schedule.periods()))
    assert MEASURE.year_fraction(first[0]) == 0.0
    assert bond_option_variance(PARAMS, 0.0, MEASURE.year_fraction(first[1])) == 0.0
    forward = (CURVE.df(first[0]) / CURVE.df(first[1]) - 1.0) / first[2]
    assert forward < cap.strike, "the fixture's first forward is no longer out of the money"
    assert nominal_caplet_value(cap, model, first) == 0.0


def test_the_LIVE_caplets_are_DISTINCT_so_a_mis_indexed_strip_would_show():
    """The fixture-power premise, over the periods that have NOT already fixed. On a flat curve
    every caplet is nearly equal and a reordered or duplicated period is invisible in the total."""
    cap, model = _cap(), _model()
    live = list(cap.schedule.periods())[1:]
    values = [nominal_caplet_value(cap, model, p) for p in live]
    assert len(set(round(v, 4) for v in values)) == len(live), values
    assert all(v > 0.0 for v in values)
    gaps = [b - a for a, b in zip(values, values[1:])]
    assert all(g > 1.0 for g in gaps), f"consecutive caplets are nearly equal: {gaps}"


def test_a_LONGER_strip_is_worth_MORE():
    model = _model()
    pvs = [_pv(_cap(n_periods=n), model) for n in (1, 2, 3, 4)]
    assert pvs == sorted(pvs), pvs
    assert len(set(round(p, 4) for p in pvs)) == 4


# ---- the headline cross-product identity: cap - floor = the forward-rate swap ----

def test_cap_MINUS_floor_IS_the_forward_rate_swap_value():
    """`[ω(L−K)]⁺` for ω=±1 differences to `L−K`, so the cap less the floor is the swap that pays
    the floating rate and receives `K`. The right-hand side is built from discount factors ONLY —
    no option formula, no `hw_caplet` — so a shared Black or variance error cannot hide in it."""
    model = _model()
    for strike in (0.01, 0.03, 0.05):
        cap = _pv(_cap(strike, CapFloorKind.CAP), model)
        floor = _pv(_cap(strike, CapFloorKind.FLOOR), model)
        swap = 0.0
        for start, pay, accrual in _cap().schedule.periods():
            # N*tau*(L - K)*df(pay), with L the simply-compounded forward: tau*L = df(start)/df(pay) - 1
            swap += N * ((CURVE.df(start) / CURVE.df(pay) - 1.0) - accrual * strike) * CURVE.df(pay)
        assert cap - floor == pytest.approx(swap, rel=1e-12), strike


def test_the_parity_identity_is_NOT_trivially_zero():
    """The companion: at these strikes cap and floor are both substantial, so the identity is
    comparing two real numbers rather than two near-zeros."""
    model = _model()
    cap = _pv(_cap(0.03, CapFloorKind.CAP), model)
    floor = _pv(_cap(0.03, CapFloorKind.FLOOR), model)
    assert cap > 1e-3 * N and floor > 1e-3 * N, (cap, floor)


# ---- the anchor reduction and its companion, at the engine level ----

def test_at_ZERO_vol_the_strip_IS_its_intrinsic():
    model = _model(params=HullWhiteParams(0.05, 0.0))
    expected = 0.0
    for start, pay, accrual in _cap().schedule.periods():
        forward = (CURVE.df(start) / CURVE.df(pay) - 1.0) / accrual
        expected += N * accrual * CURVE.df(pay) * max(forward - 0.03, 0.0)
    assert _pv(_cap(), model) == pytest.approx(expected, rel=1e-12)


def test_sigma_MOVES_the_strip_so_the_reduction_is_not_vacuous():
    pvs = [_pv(_cap(), _model(params=HullWhiteParams(0.05, s)))
           for s in (0.0, 0.004, 0.010, 0.025)]
    assert pvs == sorted(pvs), pvs
    assert len(set(round(p, 6) for p in pvs)) == 4
    assert pvs[-1] > pvs[0] * 1.05


def test_KAPPA_moves_the_strip_too():
    """`κn` is the other free parameter of the (b) fit, so it must not be inert — otherwise the
    calibration would be solving for a parameter the price cannot see."""
    pvs = [_pv(_cap(), _model(params=HullWhiteParams(k, 0.010))) for k in (0.01, 0.05, 0.30)]
    assert len(set(round(p, 6) for p in pvs)) == 3, pvs


# ---- invariant 6 and failure-as-a-value ----

def test_a_fully_SETTLED_cap_is_PV_ZERO_and_needs_no_curve():
    bare = HullWhiteModel(
        MarketSnapshot(valuation_date=DATES[-1] + timedelta(days=1), curves=CurveSet({})), PARAMS
    )
    out = price(_cap(), bare)
    assert isinstance(out, PricingResult), out
    assert out.pv.amount == 0.0


def test_a_PARTLY_elapsed_cap_drops_only_the_settled_periods():
    vd = VAL + timedelta(days=180 * 2 + 5)
    model = _model(curve=DiscountCurve(TimeMeasure(vd, DC), (0.0, 1.0, 2.0), (0.030, 0.034, 0.038)),
                   vd=vd)
    remaining = [p for p in _cap().schedule.periods() if p[1] > vd]
    assert len(remaining) == 2, "the fixture must leave SOME periods settled and some live"
    expected = sum(nominal_caplet_value(_cap(), model, p) for p in remaining)
    assert _pv(_cap(), model) == pytest.approx(expected, rel=1e-14)


def test_a_MISSING_curve_is_a_FAILURE_not_a_raise():
    bare = HullWhiteModel(MarketSnapshot(valuation_date=VAL, curves=CurveSet({})), PARAMS)
    out = price(_cap(), bare)
    assert isinstance(out, PricingFailure), out
    assert "nominal cap" in out.reason


# ---- the representation contract ----

def test_the_cap_round_trips_and_refuses_an_unknown_version():
    cap = _cap(0.025, CapFloorKind.FLOOR)
    assert NominalCap.from_dict(cap.to_dict()) == cap
    payload = cap.to_dict()
    payload["schema_version"] = 99
    with pytest.raises(ValueError, match="reads up to"):
        NominalCap.from_dict(payload)


def test_the_cap_refuses_a_payload_of_another_KIND():
    payload = _cap().to_dict()
    payload["kind_"] = "inflation_capfloor"
    with pytest.raises(ValueError, match="not a nominal_cap"):
        NominalCap.from_dict(payload)


def test_a_FLOOR_round_trips_as_a_FLOOR_not_a_cap():
    """The enum must survive the round trip — serialising the omega as a bool or dropping it would
    silently turn a floor into a cap."""
    floor = _cap(0.03, CapFloorKind.FLOOR)
    assert NominalCap.from_dict(floor.to_dict()).kind is CapFloorKind.FLOOR
    assert _pv(floor, _model()) != _pv(_cap(0.03, CapFloorKind.CAP), _model())
