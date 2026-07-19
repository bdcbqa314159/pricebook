"""PricingResult / PricingFailure oracles (L0) — Topic 0 Slice 6.

Engine I/O: a result is a decomposition (dirty PV, accrued ⇒ clean) that records its
basis — the value currency AND the collateral/discounting basis — so a PV is never
ambiguous (settlement ruling §1). Failure is a value.
"""

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.results import DiscountBasis, PricingFailure, PricingResult

USD, EUR = Currency.USD, Currency.EUR


def test_clean_is_pv_minus_accrued():
    r = PricingResult(pv=Money(102.0, USD), accrued=Money(2.0, USD))
    assert r.clean == Money(100.0, USD)


def test_no_accrual_clean_equals_pv():
    r = PricingResult(pv=Money(100.0, USD))
    assert r.clean == Money(100.0, USD)


def test_result_records_currency_and_collateral_basis():
    # a EUR value discounted on USD collateral (CSA) — the PV is unambiguous
    r = PricingResult(pv=Money(100.0, EUR), basis=DiscountBasis(collateral=USD))
    assert r.pv.currency is EUR
    assert r.basis.collateral is USD


def test_failure_is_a_value():
    f = PricingFailure(reason="no curve for index")
    assert f.reason == "no curve for index"


def test_full_decomposition_vocabulary_optional():
    # S6: the full vocabulary exists up front (fields optional, unpopulated until a consumer)
    r = PricingResult(
        pv=Money(100.0, USD),
        accrued=Money(2.0, USD),
        cashflow_breakdown=(Money(60.0, USD), Money(40.0, USD)),
        sensitivities={"dv01": 12.5},
        diagnostics={"solver_iters": 4},
    )
    assert r.clean == Money(98.0, USD)
    assert r.sensitivities["dv01"] == 12.5
    # a bare result still works — everything but pv defaults to None
    bare = PricingResult(pv=Money(5.0, USD))
    assert bare.cashflow_breakdown is None and bare.sensitivities is None
