"""The engine registry's dispatch key (tranche 33b) — PRE-REGISTERED, red before the refactor.

Until now `_PRICERS` was keyed on the **product type alone**, which carried a silent footgun: a
second `@register(SomeProduct)` *replaced* the first with no warning. Tranche 33 hit it for real —
registering the Heston pricer for `EuropeanEquityOption` would have removed tranche 30's Black
pricer from the table, and the tree would have priced every equity vanilla under Heston without a
single test noticing, because the two signatures are identical.

That is not a hypothetical: one product legitimately has several models. The equity vanilla already
has three (Black at T30, the local-vol MC at T32, Heston at T33), and §3 has always taken the model
as the second argument — so the **model type is already part of what identifies a pricer**, and the
registry was the one place that did not know it.

**Two things are asserted here that could not both be true before:**

* **coexistence** — two pricers for one product under different models each resolve to their own;
* **loud failure** — registering the *same* `(product, model)` twice raises, rather than silently
  overwriting. A footgun that announces itself is not a footgun.

And one that must not change: **every existing product prices bit-identically**, because this
widens a dictionary key and touches no arithmetic.
"""

from typing import Protocol, runtime_checkable

import pytest

from pricebook.engine.registry import (
    AmbiguousRegistration,
    DuplicateRegistration,
    dispatch,
    register,
    registered_pricers,
)
from pricebook.foundation.results import PricingFailure


class _ProductA:
    pass


class _ModelOne:
    pass


class _ModelTwo:
    pass


@runtime_checkable
class _RatioLike(Protocol):
    """A protocol with a REAL member — an empty one is satisfied by everything, which would make
    the "does not satisfy" test below unable to fail."""

    def ratio(self) -> float: ...


class TestCoexistenceTheThingThatWasImpossibleBefore:
    """Two models, one product — the case that silently clobbered on the product-only registry."""

    def test_two_pricers_for_ONE_product_under_DIFFERENT_models_both_survive(self) -> None:
        @register(_ProductA, _ModelOne)
        def _price_one(product: object, model: object) -> str:
            return "one"

        @register(_ProductA, _ModelTwo)
        def _price_two(product: object, model: object) -> str:
            return "two"

        assert dispatch(_ProductA(), _ModelOne()) == "one"
        assert dispatch(_ProductA(), _ModelTwo()) == "two"

    def test_registering_a_NEW_pair_disturbs_no_existing_one(self) -> None:
        """The precise failure tranche 33 proved empirically, now impossible.

        A fresh model type, because re-registering `(_ProductA, _ModelOne)` would now correctly
        **raise** — which is the other half of this tranche and is asserted in its own test below.
        """

        class _ModelThree:
            pass

        before = dict(registered_pricers())

        @register(_ProductA, _ModelThree)
        def _late(product: object, model: object) -> str:
            return "late"

        after = registered_pricers()
        assert set(after) - set(before) == {(_ProductA, _ModelThree)}, "exactly one key added"
        for key, pricer in before.items():
            assert after[key] is pricer, f"{key} was disturbed by an unrelated registration"


class TestTheFailuresAreLoud:
    def test_registering_the_SAME_pair_twice_RAISES(self) -> None:
        """§8's shape: a footgun that announces itself is not a footgun. Silently overwriting a
        pricer is debt incurred without a ledger entry — the registry refuses instead."""

        class _Dup:
            pass

        @register(_Dup, _ModelOne)
        def _first(product: object, model: object) -> str:
            return "first"

        with pytest.raises(DuplicateRegistration, match="already registered"):

            @register(_Dup, _ModelOne)
            def _second(product: object, model: object) -> str:
                return "second"

        assert dispatch(_Dup(), _ModelOne()) == "first", "the original must survive the attempt"

    def test_an_UNREGISTERED_pair_is_a_PricingFailure_naming_BOTH_types(self) -> None:
        """A product registered under *some other* model must not silently fall through to it."""

        class _Unpriced:
            pass

        @register(_Unpriced, _ModelOne)
        def _only(product: object, model: object) -> str:
            return "only"

        out = dispatch(_Unpriced(), _ModelTwo())
        assert isinstance(out, PricingFailure)
        assert "_Unpriced" in out.reason and "_ModelTwo" in out.reason, out.reason


class TestProtocolPolymorphismIsPreserved:
    """A pricer typed against a protocol serves every model satisfying it — deliberately.

    This is what the product-only key *hid* and what a naive concrete-type key would have
    destroyed: `IndexRatioModel` is a `@runtime_checkable` protocol, and the inflation cap/floor
    and year-on-year pricers are polymorphic over it on purpose. Keying them to one concrete model
    would have broken 63 tests and, worse, would have encoded a falsehood about what those pricers
    are for.
    """

    def test_ONE_protocol_registration_serves_EVERY_model_satisfying_it(self) -> None:
        from pricebook.foundation.results import PricingResult
        from pricebook.models.index_ratio import IndexRatioModel
        from pricebook.products.inflation import InflationCapFloor

        keys = [k for k in registered_pricers() if k[0] is InflationCapFloor]
        assert keys == [(InflationCapFloor, IndexRatioModel)], (
            f"the cap/floor must be registered ONCE, under the protocol: {keys}"
        )

    def test_a_model_satisfying_the_protocol_needs_NO_registration_of_its_own(self) -> None:
        """Which is why a test's own stub model dispatches without `src/` knowing it exists.

        The protocol carries a **real member**, deliberately. An empty `Protocol` is satisfied by
        every object, so a fixture using one would also pass against a dispatcher that never
        checked `isinstance` at all — the companion test below proves that check is load-bearing,
        and it can only do so if satisfying the protocol is a real condition.
        """

        class _Satisfies:
            def ratio(self) -> float:
                return 1.0

        @register(_ProductA, _RatioLike)
        def _by_protocol(product: object, model: object) -> str:
            return "protocol"

        assert dispatch(_ProductA(), _Satisfies()) == "protocol"

    def test_a_model_NOT_satisfying_the_protocol_does_NOT_dispatch_to_it(self) -> None:
        """The `isinstance` check is load-bearing, and the mutation sweep found it unguarded.

        Dropping it survived all 1140 tests, because every model the suite hands to a
        protocol-registered pricer happens to satisfy that protocol. Without the check, a model
        that does **not** satisfy it would be handed to the pricer anyway and fail somewhere deep
        inside — an `AttributeError` from the middle of a valuation instead of a clean
        `PricingFailure` at the boundary (§3.4). The degenerate fixture here is the *population of
        models the suite uses*, not any one value.
        """

        class _Unrelated:
            pass

        class _OtherProduct:
            pass

        @register(_OtherProduct, _RatioLike)
        def _needs_ratio(product: object, model: object) -> str:
            return "protocol"

        out = dispatch(_OtherProduct(), _Unrelated())
        assert isinstance(out, PricingFailure), f"expected a clean failure, got {out!r}"
        assert "_Unrelated" in out.reason


class TestExactBeatsProtocol:
    """A concrete registration wins over a protocol one that would also match."""

    def test_the_CONCRETE_pricer_is_chosen_when_both_could_serve(self) -> None:
        class _Both:
            pass

        class _AnyModel(Protocol):
            pass

        loose = runtime_checkable(_AnyModel)

        @register(_Both, loose)
        def _loose(product: object, model: object) -> str:
            return "protocol"

        @register(_Both, _ModelOne)
        def _exact(product: object, model: object) -> str:
            return "exact"

        assert dispatch(_Both(), _ModelOne()) == "exact", "exact must beat protocol"
        assert dispatch(_Both(), _ModelTwo()) == "protocol", "and the protocol still serves others"

    def test_TWO_matching_protocols_are_a_LOUD_ambiguity_not_a_coin_flip(self) -> None:
        class _Ambiguous:
            pass

        class _ProtoA(Protocol):
            pass

        class _ProtoB(Protocol):
            pass

        first, second = runtime_checkable(_ProtoA), runtime_checkable(_ProtoB)

        @register(_Ambiguous, first)
        def _a(product: object, model: object) -> str:
            return "a"

        @register(_Ambiguous, second)
        def _b(product: object, model: object) -> str:
            return "b"

        with pytest.raises(AmbiguousRegistration, match="matches"):
            dispatch(_Ambiguous(), _ModelOne())


class TestTheRefactorDisturbsNothing:
    def test_every_PRODUCT_registers_each_model_at_most_once(self) -> None:
        """The eight registrations the tree had before the dispatch refactor, each still present,
        and no product registering the same model type twice."""

        products = [key[0].__name__ for key in registered_pricers()]
        expected = {
            "EuropeanEquityOption",
            "FixedCashflow",
            "InflationCapFloor",
            "InflationLinkedBond",
            "NominalCap",
            "VarianceSwap",
            "YearOnYearInflationSwap",
            "ZeroCouponInflationSwap",
        }
        assert expected <= set(products), f"missing: {expected - set(products)}"
        # **The invariant is "distinct models per product", NOT "one pricer per product".** It was
        # first written as the latter with `EuropeanEquityOption` exempted by name, because that was
        # the only multi-model product when this test was written. Tranche 71 made `NominalCap` the
        # second (Hull-White, Bachelier, displaced-diffusion), so the exemption list would have had
        # to grow -- which is exactly what the original comment warned against: "a test that has to
        # be edited on every addition is a test that will eventually be edited without thought."
        #
        # The hard-coded `expected` set had also quietly stopped covering the tree: `CommodityOption`
        # carries two pricers and is not in the set, so nothing checked it either way. A fixed list
        # of names that silently skips whatever is not on it is tranche 26b's floor-discovery defect
        # one level down -- the oracle's domain excluding the thing that could be wrong. So this now
        # walks EVERY registered product rather than a named eight.
        for name in sorted(set(products)):
            models = [key[1] for key in registered_pricers() if key[0].__name__ == name]
            assert len(set(models)) == len(models), f"{name} has duplicate model keys: {models}"
        equity_models = [
            key[1] for key in registered_pricers() if key[0].__name__ == "EuropeanEquityOption"
        ]
        assert len(equity_models) > 1, "several models must coexist on the equity vanilla"
        cap_models = [key[1] for key in registered_pricers() if key[0].__name__ == "NominalCap"]
        assert len(cap_models) > 1, "several models must coexist on the nominal cap"
    def test_the_equity_vanilla_resolves_to_a_DIFFERENT_pricer_per_model(self) -> None:
        """The production instance of coexistence, not a synthetic one."""
        from pricebook.engine.registry import _resolve
        from pricebook.models.equity_black import EquityBlackModel
        from pricebook.models.heston import HestonModel
        from pricebook.products.equity_option import EuropeanEquityOption

        table = registered_pricers()
        black = table[(EuropeanEquityOption, EquityBlackModel)]
        heston = table[(EuropeanEquityOption, HestonModel)]
        assert black is not heston
        assert black.__name__ == "price_european_equity_option"
        assert heston.__name__ == "price_european_equity_option_heston"

    def test_the_nominal_cap_resolves_to_a_DIFFERENT_pricer_per_model(self) -> None:
        """The second production instance of coexistence, added at tranche 71 — the smile models
        price the SAME contract under a different measure, which is the capability the registry's
        `(product, model)` key exists for."""
        from pricebook.models.hull_white import HullWhiteModel
        from pricebook.models.smile import BachelierModel, DisplacedDiffusionModel
        from pricebook.products.nominal_cap import NominalCap

        table = registered_pricers()
        pricers = [
            table[(NominalCap, model)]
            for model in (HullWhiteModel, BachelierModel, DisplacedDiffusionModel)
        ]
        assert len({p.__name__ for p in pricers}) == 3, [p.__name__ for p in pricers]

    def test_every_key_is_a_PRODUCT_MODEL_pair(self) -> None:

        for key in registered_pricers():
            assert isinstance(key, tuple) and len(key) == 2, f"{key} is not a (product, model) pair"
            assert all(isinstance(part, type) for part in key), f"{key} holds a non-type"
