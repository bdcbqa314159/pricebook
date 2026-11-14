"""Survival curves keyed by reference entity (L1) — tranche 41b. Pre-registered RED.

**Why the key had to change.** `CurveSet.survivals` was keyed by `Currency`, so a portfolio of N
names in one currency was **not representable in the snapshot**. Tranche 41's copula therefore held
its per-name curves on the *model*, which is fine with no engine (§3 binds engines) and a real
violation once tranche 42's basket engine reads market data from outside `model.market`. The
currency key conflated two different things: a CDS has a **currency** (whose discount curve it uses)
and a **reference entity** (whose survival curve it uses).

**The bit-identity baseline below was captured BEFORE the re-key and is committed as hex
literals.** That is what makes this a pre-registration rather than a re-run: a refactor compared
against a freshly recomputed value proves only that two runs of the same code agree. These numbers
came out of the currency-keyed tree, so reproducing them exactly is evidence the key moved and the
arithmetic did not (§4 — a key change is a no-op on the operations, so bit-identity is available and
is therefore the bar, not ULP).
"""

from datetime import date, timedelta

import pytest

from pricebook.engine import price
from pricebook.engine.credit_default_swap import par_spread, protection_leg, risky_pv01
from pricebook.foundation import CouponSchedule, Currency, Money
from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.market.survival_curve import ReferenceEntity, SurvivalCurve
from pricebook.models.credit import CreditModel
from pricebook.models.credit_swaption import CreditSwaptionModel
from pricebook.products.credit_default_swap import CreditDefaultSwap, ProtectionDirection
from pricebook.products.credit_default_swaption import CreditDefaultSwaption, SwaptionDirection
from pricebook.products.forward_cds import ForwardCDS, ForwardPeriod

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
TM = TimeMeasure(VAL, DC)
EUR = Currency("EUR")
DISCOUNT = CurveKey(CurveRole.DISCOUNT, EUR)
CURVE = DiscountCurve(TM, (0.0, 1.0, 2.0, 3.0, 5.0), (0.020, 0.026, 0.030, 0.033, 0.036))

ACME = ReferenceEntity("ACME")
BRAVO = ReferenceEntity("BRAVO")

#: The curve the baseline was captured against, under the old `Currency` key.
HAZARD = SurvivalCurve(TM, (0.0, 1.0, 3.0), (0.010, 0.025, 0.045))
#: A visibly different name, so "each CDS sees its own curve" has content.
WIDER = SurvivalCurve(TM, (0.0, 1.0, 3.0), (0.040, 0.075, 0.110))

DATES = tuple(VAL + timedelta(days=round(182.5 * i)) for i in range(11))
SCHEDULE = CouponSchedule(DATES, DC)
NOTIONAL = Money(10_000_000.0, EUR)
SPLIT = DATES[4]

#: Captured from the currency-keyed tree at v0.48.0, before `ReferenceEntity` existed.
BASELINE = {
    "cds|0.01|BUYER": {
        "pv": "0x1.3ad2c5f7d250ep+18", "rpv01": "0x1.1543bff6085d6p+2",
        "prot": "0x1.70f2a3604e6a3p+19", "par": "0x1.1dc25e3aceb2bp-6"},
    "cds|0.01|SELLER": {
        "pv": "-0x1.3ad2c5f7d250ep+18", "rpv01": "0x1.1543bff6085d6p+2",
        "prot": "0x1.70f2a3604e6a3p+19", "par": "0x1.1dc25e3aceb2bp-6"},
    "cds|0.0229|BUYER": {
        "pv": "-0x1.cde13b351e2c4p+17", "rpv01": "0x1.1543bff6085d6p+2",
        "prot": "0x1.70f2a3604e6a3p+19", "par": "0x1.1dc25e3aceb2bp-6"},
    "cds|0.0229|SELLER": {
        "pv": "0x1.cde13b351e2c4p+17", "rpv01": "0x1.1543bff6085d6p+2",
        "prot": "0x1.70f2a3604e6a3p+19", "par": "0x1.1dc25e3aceb2bp-6"},
    "forward": {"pv": "0x1.d1d5d759e7871p+16"},
    "swaption": {"pv": "0x1.770b7e7090566p+17"},
}


def _snapshot(**survivals: SurvivalCurve) -> MarketSnapshot:
    return MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet(
            {DISCOUNT: CURVE},
            survivals={ReferenceEntity(name): curve for name, curve in survivals.items()}))


def _model(**survivals: SurvivalCurve) -> CreditModel:
    return CreditModel(_snapshot(**survivals), 0.40)


def _cds(entity: ReferenceEntity, spread: float = 0.0100,
         direction: ProtectionDirection = ProtectionDirection.BUYER) -> CreditDefaultSwap:
    return CreditDefaultSwap(SCHEDULE, spread, NOTIONAL, direction, entity)


class TestTheSingleNamePathIsBitIdentical:
    """§4 pure refactor: the key moved, no operation did."""

    @pytest.mark.parametrize("spread", [0.0100, 0.0229])
    @pytest.mark.parametrize("direction", list(ProtectionDirection))
    def test_the_cds_reprices_BIT_IDENTICALLY_to_the_currency_keyed_tree(
        self, spread: float, direction: ProtectionDirection) -> None:
        model = _model(ACME=HAZARD)
        product = _cds(ACME, spread, direction)
        want = BASELINE[f"cds|{spread}|{direction.name}"]
        assert price(product, model).pv.amount.hex() == want["pv"]
        assert risky_pv01(product, model).hex() == want["rpv01"]
        assert protection_leg(product, model).hex() == want["prot"]
        assert par_spread(product, model).hex() == want["par"]

    def test_the_forward_cds_reprices_BIT_IDENTICALLY(self) -> None:
        product = ForwardCDS(
            ForwardPeriod(SCHEDULE, SPLIT), 0.0180, NOTIONAL, ProtectionDirection.BUYER, ACME)
        assert price(product, _model(ACME=HAZARD)).pv.amount.hex() == BASELINE["forward"]["pv"]

    def test_the_swaption_reprices_BIT_IDENTICALLY(self) -> None:
        product = CreditDefaultSwaption(
            ForwardPeriod(SCHEDULE, SPLIT), 0.0180, NOTIONAL, SwaptionDirection.PAYER, ACME)
        model = CreditSwaptionModel(_model(ACME=HAZARD), 0.45)
        assert price(product, model).pv.amount.hex() == BASELINE["swaption"]["pv"]


class TestTheNewCapabilityTwoEntitiesInONECurrency:
    """The thing the currency key could not express — and the reason for the tranche."""

    def test_TWO_entities_in_one_currency_are_REPRESENTABLE(self) -> None:
        snapshot = _snapshot(ACME=HAZARD, BRAVO=WIDER)
        assert set(snapshot.curves.survivals) == {ACME, BRAVO}
        assert snapshot.curves.survival(ACME) is HAZARD
        assert snapshot.curves.survival(BRAVO) is WIDER

    def test_each_CDS_prices_against_ITS_OWN_entitys_curve(self) -> None:
        """A CDS on ACME must not see BRAVO's curve — the §6 "another name's data cannot be
        reached" guard, as the equity snapshot already enforces for tickers."""
        model = _model(ACME=HAZARD, BRAVO=WIDER)
        acme = price(_cds(ACME), model).pv.amount
        bravo = price(_cds(BRAVO), model).pv.amount
        assert acme != bravo
        # and each equals what it prices against on its own
        assert acme == price(_cds(ACME), _model(ACME=HAZARD)).pv.amount
        assert bravo == price(_cds(BRAVO), _model(BRAVO=WIDER)).pv.amount

    def test_THE_PREMISE_the_two_curves_are_MATERIALLY_different(self) -> None:
        """Without this the test above could pass with both CDS reading one curve."""
        acme = 1.0 - HAZARD.survival(5.0)
        bravo = 1.0 - WIDER.survival(5.0)
        # measured: 13.93% against 33.63%, so BRAVO's default probability is 2.41x ACME's
        assert bravo / acme - 1.0 > 1.0, (acme, bravo)

    def test_the_old_CURRENCY_key_no_longer_resolves(self) -> None:
        """The conflation is gone, not merely deprecated: a currency is not an entity."""
        snapshot = _snapshot(ACME=HAZARD)
        with pytest.raises(KeyError, match="EUR"):
            snapshot.curves.survival(EUR)


class TestTheGuardsAreReachable:
    def test_an_UNKNOWN_entity_is_a_PricingFailure_not_another_names_curve(self) -> None:
        """§3.4 — silently substituting a different name's curve is a plausible wrong number."""
        model = _model(ACME=HAZARD)
        out = price(_cds(BRAVO), model)
        assert not hasattr(out, "pv"), out
        assert "BRAVO" in out.reason

    def test_an_UNKNOWN_entity_raises_from_the_curve_set_in_its_own_vocabulary(self) -> None:
        with pytest.raises(KeyError, match="BRAVO"):
            _snapshot(ACME=HAZARD).curves.survival(BRAVO)

    def test_an_EMPTY_entity_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="reference entity"):
            ReferenceEntity("")

    def test_the_FORWARD_PERIODs_split_must_be_a_schedule_date(self) -> None:
        """The validation that was duplicated in two products now lives in one place."""
        with pytest.raises(ValueError, match="schedule date"):
            ForwardPeriod(SCHEDULE, VAL + timedelta(days=7))

    def test_the_FORWARD_PERIODs_split_must_precede_the_maturity(self) -> None:
        with pytest.raises(ValueError, match="before the maturity"):
            ForwardPeriod(SCHEDULE, DATES[-1])
