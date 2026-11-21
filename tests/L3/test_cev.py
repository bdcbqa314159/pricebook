"""The CEV model's parameter type (L3). MCINERNEY-ZASTAWNIAK §8.3 eq (8.5); BROCKHAUS-2016 §6.1.

`dF = sigma(t) F^beta dW` with `0 <= beta <= 1` and zero an absorbing state. Frozen data and the
refusals; the pricing is oracled in `tests/L4/test_cev_caplet.py`.

**On `beta`, and the one unit question that is not cosmetic.** Two held sources name the exponent
differently — MCINERNEY-ZASTAWNIAK calls it `beta` (eq 8.5), BROCKHAUS-2016 calls it `rho` (eq 6.8).
`beta` is kept: it is the rates-standard letter, it is the arc's own source's letter, and `rho` would
collide with correlation everywhere in this tree. What the two letters agree on is the consequence:
**`sigma` carries units of rate^(1-beta)**, so a CEV vol is NOT comparable across `beta`. At
`beta = 1` it is the familiar dimensionless 22%; at `beta = 0` it is an absolute rate, ~85bp. Nothing
can be shared between two `CevModel`s with different exponents and the same `vol` — they are not the
same volatility.
"""

from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from pricebook.foundation import Currency, DayCountConvention
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.models.cev import CevModel

VAL = date(2026, 1, 15)
EUR = Currency("EUR")
MEASURE = TimeMeasure(VAL, DayCountConvention.ACT_365_FIXED)
CURVE = DiscountCurve(MEASURE, (0.0, 1.0, 2.0, 4.0), (0.026, 0.030, 0.034, 0.038))
SNAPSHOT = MarketSnapshot(
    valuation_date=VAL, curves=CurveSet({CurveKey(CurveRole.DISCOUNT, EUR): CURVE})
)


@pytest.mark.parametrize("beta", (0.0, 0.25, 0.5, 0.75, 0.999))
def test_an_exponent_in_the_sources_RANGE_is_accepted(beta):
    assert CevModel(SNAPSHOT, 0.1, beta).beta == beta


@pytest.mark.parametrize("beta", (-0.01, 1.0, 1.5))
def test_an_exponent_OUTSIDE_the_closed_forms_domain_is_REFUSED(beta):
    # The source's SDE admits `0 <= beta <= 1`, but Cox's closed form is stated for `0 < beta < 1`
    # STRICTLY and is undefined at `beta = 1` (`1/(1-beta)` diverges — measured: the degrees of
    # freedom overflow and the formula returns `nan` by `beta = 0.99999`). A model that cannot be
    # priced is refused at construction rather than returning a `nan` later: §3's failure-is-a-value
    # rule applied one layer down, where the input is simply outside the domain.
    with pytest.raises(ValueError, match="exponent"):
        CevModel(SNAPSHOT, 0.1, beta)


@pytest.mark.parametrize("vol", (0.0, -0.1))
def test_a_non_positive_vol_is_REFUSED(vol):
    with pytest.raises(ValueError, match="volatility"):
        CevModel(SNAPSHOT, vol, 0.5)


def test_the_model_is_FROZEN():
    model = CevModel(SNAPSHOT, 0.1, 0.5)
    with pytest.raises(FrozenInstanceError):
        setattr(model, "beta", 0.7)


def test_it_is_DISTINCT_from_the_tranche_71_smile_models():
    # CEV is a different idea, with `beta` its defining parameter (§6) — and the displaced-diffusion
    # model is an APPROXIMATION to this one (§8.4 derives it by a Taylor expansion of `F^beta`), so a
    # shared base class would suggest an interchangeability the mathematics denies.
    from pricebook.models.smile import BachelierModel, DisplacedDiffusionModel

    assert not issubclass(CevModel, (BachelierModel, DisplacedDiffusionModel))
    assert not issubclass(BachelierModel, CevModel)
    assert not issubclass(DisplacedDiffusionModel, CevModel)
