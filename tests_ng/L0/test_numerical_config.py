"""NumericalConfig serialisation oracle (L0) — CP-3 #1, first quarry retire.

The genuine residual that stood between ng's `NumericalConfig` and superseding the quarry's
`core/numerical_config.py`: `to_dict`/`from_dict` (round-trippable wire form with a schema
version) and a `replace` convenience. The quarry's 12 extra knobs are shed `dead` (no production
consumer — see the retire note in `quarry_reconciliation.md`), so this closes the residual.
"""

import pytest

from pricebook_ng.foundation.numerical_config import NumericalConfig


def test_round_trips_through_dict():
    cfg = NumericalConfig(fd_bump=1e-5, mc_paths=10_000, mc_seed=7)
    assert NumericalConfig.from_dict(cfg.to_dict()) == cfg


def test_to_dict_carries_schema_version():
    assert NumericalConfig().to_dict()["schema_version"] == 1


def test_missing_version_reads_as_v1():
    data = NumericalConfig().to_dict()
    del data["schema_version"]  # legacy payload with no version
    assert NumericalConfig.from_dict(data) == NumericalConfig()


def test_future_version_is_rejected_loudly():
    data = NumericalConfig().to_dict()
    data["schema_version"] = 99
    with pytest.raises(ValueError):
        NumericalConfig.from_dict(data)


def test_replace_returns_a_new_config():
    cfg = NumericalConfig(mc_paths=1000, mc_seed=3)
    changed = cfg.replace(mc_paths=2000)
    assert changed.mc_paths == 2000
    assert changed.mc_seed == cfg.mc_seed  # others preserved
    assert cfg.mc_paths == 1000  # original untouched (frozen)
