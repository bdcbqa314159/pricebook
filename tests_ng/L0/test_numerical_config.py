"""NumericalConfig oracle (L0) — Topic 0 Slice 6.

The FULL reproducibility knob set, designed up front so it never retrofits a
foundational value type (the 12 knobs deferred at CP-3 retire #1). Serialisation
pattern: per-class to_dict/from_dict + schema_version, no framework.
"""

import pytest

from pricebook_ng.foundation.numerical_config import NumericalConfig


def test_full_knob_set_defaults():
    c = NumericalConfig()
    # MC · PDE · tree · quadrature · COS · root-finder · fd bump — all present
    for knob in ("fd_bump", "mc_paths", "mc_seed", "mc_antithetic", "mc_sobol", "mc_brownian_bridge",
                 "pde_time_steps", "pde_space_steps", "pde_n_std_devs", "tree_steps",
                 "quadrature_tol", "quadrature_max_iter", "cos_n", "cos_l",
                 "rootfinder_tol", "rootfinder_max_iter"):
        assert hasattr(c, knob), knob


def test_replace_and_immutability():
    c = NumericalConfig(mc_paths=1000)
    d = c.replace(mc_paths=2000)
    assert d.mc_paths == 2000 and c.mc_paths == 1000


def test_positive_knobs_validated():
    with pytest.raises(ValueError):
        NumericalConfig(fd_bump=0.0)
    with pytest.raises(ValueError):
        NumericalConfig(mc_paths=0)


def test_round_trips_through_dict():
    c = NumericalConfig(mc_paths=10_000, cos_n=128, pde_time_steps=200)
    assert NumericalConfig.from_dict(c.to_dict()) == c
    assert c.to_dict()["schema_version"] == 1


def test_future_schema_rejected():
    data = NumericalConfig().to_dict()
    data["schema_version"] = 99
    with pytest.raises(ValueError):
        NumericalConfig.from_dict(data)
