"""Rate-quotation basis oracles (L0) — Topic 0 gate rework (S12).

A rate is meaningless without its basis (compounding + day count). `Compounding` is the
quotation basis (SIMPLE…CONTINUOUS) — a *different* concept from the index `AccrualMethod`
(compounded/exponential/averaging), which is why the index enum was renamed to kill the
collision. `convert_rate` moves a rate between bases; treating semi-annual as continuous
is silently wrong by ~r²/2, the failure class this project prevents.
"""

import math

import pytest

from pricebook_ng.foundation.rate_basis import Compounding, convert_rate, growth_factor


def test_growth_factor_by_basis():
    assert growth_factor(0.05, 1.0, Compounding.CONTINUOUS) == pytest.approx(math.exp(0.05), abs=1e-15)
    assert growth_factor(0.05, 1.0, Compounding.SIMPLE) == pytest.approx(1.05, abs=1e-15)
    assert growth_factor(0.05, 1.0, Compounding.ANNUAL) == pytest.approx(1.05, abs=1e-15)
    assert growth_factor(0.05, 1.0, Compounding.SEMI_ANNUAL) == pytest.approx(1.025 ** 2, abs=1e-15)


def test_convert_semi_to_annual_exact():
    # a 5% semi-annual rate = 5.0625% annual (same growth over a year)
    assert convert_rate(0.05, 1.0, Compounding.SEMI_ANNUAL, Compounding.ANNUAL) == pytest.approx(0.050625, abs=1e-12)


def test_convert_semi_to_continuous_and_wrongness():
    r = convert_rate(0.05, 1.0, Compounding.SEMI_ANNUAL, Compounding.CONTINUOUS)
    assert r == pytest.approx(math.log(1.025 ** 2), abs=1e-12)
    assert r != pytest.approx(0.05, abs=1e-4)          # NOT the same as 5% — the ~r²/2 error


def test_round_trip_and_identity():
    r0 = 0.037
    there = convert_rate(r0, 2.0, Compounding.QUARTERLY, Compounding.CONTINUOUS)
    back = convert_rate(there, 2.0, Compounding.CONTINUOUS, Compounding.QUARTERLY)
    assert back == pytest.approx(r0, abs=1e-12)
    assert convert_rate(r0, 2.0, Compounding.MONTHLY, Compounding.MONTHLY) == r0   # identity
