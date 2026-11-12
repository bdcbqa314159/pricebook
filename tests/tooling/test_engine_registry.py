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

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pricebook.engine.registry import (  # noqa: E402
    DuplicateRegistration,
    dispatch,
    register,
    registered_pricers,
)
from pricebook.foundation.results import PricingFailure  # noqa: E402


class _ProductA:
    pass


class _ModelOne:
    pass


class _ModelTwo:
    pass


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

    def test_registering_the_second_does_not_REMOVE_the_first(self) -> None:
        """The precise failure tranche 33 proved empirically, now impossible."""
        before = dict(registered_pricers())

        @register(_ProductA, _ModelOne)
        def _late(product: object, model: object) -> str:
            return "late"

        after = registered_pricers()
        for key, pricer in before.items():
            if key != (_ProductA, _ModelOne):
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


class TestTheRefactorDisturbsNothing:
    def test_every_PRODUCTION_pricer_is_still_registered_exactly_once(self) -> None:
        """The eight registrations the tree had before this tranche, each still present and each
        now carrying its model type."""
        import pricebook.engine  # noqa: F401  (importing runs every @register)

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
        for name in expected:
            assert products.count(name) == 1, f"{name} is registered {products.count(name)} times"

    def test_every_key_is_a_PRODUCT_MODEL_pair(self) -> None:
        import pricebook.engine  # noqa: F401

        for key in registered_pricers():
            assert isinstance(key, tuple) and len(key) == 2, f"{key} is not a (product, model) pair"
            assert all(isinstance(part, type) for part in key), f"{key} holds a non-type"
