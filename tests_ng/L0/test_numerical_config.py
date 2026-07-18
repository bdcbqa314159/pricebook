"""NumericalConfig oracle (L0) — Topic 0 gate rework (F1/S15).

Decomposed by method family (§3b — a config never earns `fields-exempt`): each sub-config
≤5 fields, so `verify.py fields` passes on merit. RNG family is pinned (S15). Reads as
`numerics.monte_carlo.paths`. Serialisation pattern: nested to_dict/from_dict + schema.
"""

import pytest

from pricebook_ng.foundation.numerical_config import (
    RNG_FAMILY,
    IntegrationConfig,
    LatticeConfig,
    MonteCarloConfig,
    NumericalConfig,
    SolverConfig,
)


def test_decomposed_by_method_family():
    c = NumericalConfig()
    assert isinstance(c.monte_carlo, MonteCarloConfig)
    assert isinstance(c.lattice, LatticeConfig)
    assert isinstance(c.integration, IntegrationConfig)
    assert isinstance(c.solver, SolverConfig)
    # reads as what it is
    assert c.monte_carlo.paths > 0
    assert c.lattice.tree_steps > 0          # tree folded into the lattice (discretisation grids)
    assert c.integration.cos_n > 0
    assert c.solver.fd_bump > 0


def test_rng_family_pinned():
    # S15: a later RNG switch silently shifts every MC oracle — pin it as an invariant
    assert RNG_FAMILY == "pcg64"


def test_positive_knobs_validated():
    with pytest.raises(ValueError):
        MonteCarloConfig(paths=0)
    with pytest.raises(ValueError):
        SolverConfig(fd_bump=0.0)


def test_round_trips_through_dict():
    c = NumericalConfig(monte_carlo=MonteCarloConfig(paths=10_000),
                        integration=IntegrationConfig(cos_n=256))
    assert NumericalConfig.from_dict(c.to_dict()) == c
    assert c.to_dict()["schema_version"] == 1


def test_future_schema_rejected():
    data = NumericalConfig().to_dict()
    data["schema_version"] = 99
    with pytest.raises(ValueError):
        NumericalConfig.from_dict(data)
