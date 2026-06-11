"""Tests for schema versioning on @serialisable / Serialisable (G1 P3 Slice 2, closes G1)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pricebook.core.serialisable import (
    Serialisable,
    _SCHEMA_VERSION_KEY,
    _SCHEMA_VERSION_KEY_FLAT,
    serialisable,
    serialisable_convention,
)


# ============================================================
# Test fixtures (declared once so registration only happens once)
# ============================================================

class _BaseMixinSample(Serialisable):
    """Sample using the Serialisable inheritance path."""
    _SERIAL_TYPE = "_test_base_mixin_sample"
    _SERIAL_FIELDS = ["x", "y"]

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


@serialisable("_test_decorator_sample", ["a", "b"])
class _DecoratorSample:
    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b


@serialisable("_test_decorator_v2_sample", ["a", "b"], schema_version=2)
class _DecoratorV2Sample:
    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b


@serialisable_convention("_test_convention_sample")
@dataclass(frozen=True)
class _ConventionSample:
    name: str
    n: int


@serialisable_convention("_test_convention_v3_sample", schema_version=3)
@dataclass(frozen=True)
class _ConventionV3Sample:
    name: str
    n: int


# ============================================================
# Default version is 1
# ============================================================

class TestDefaultVersion:
    def test_serialisable_base_default(self):
        assert Serialisable._SERIAL_SCHEMA_VERSION == 1
        assert _BaseMixinSample._SERIAL_SCHEMA_VERSION == 1

    def test_decorator_default(self):
        assert _DecoratorSample._SERIAL_SCHEMA_VERSION == 1

    def test_convention_default(self):
        assert _ConventionSample._SERIAL_SCHEMA_VERSION == 1


# ============================================================
# to_dict() emits the version
# ============================================================

class TestToDictEmitsVersion:
    def test_serialisable_base(self):
        obj = _BaseMixinSample(1, 2)
        d = obj.to_dict()
        assert d[_SCHEMA_VERSION_KEY] == 1
        assert d["type"] == "_test_base_mixin_sample"
        assert d["params"] == {"x": 1, "y": 2}

    def test_decorator_emits_v1(self):
        obj = _DecoratorSample(3, 4)
        d = obj.to_dict()
        assert d[_SCHEMA_VERSION_KEY] == 1

    def test_decorator_with_explicit_v2(self):
        obj = _DecoratorV2Sample(5, 6)
        d = obj.to_dict()
        assert d[_SCHEMA_VERSION_KEY] == 2

    def test_convention_emits_flat_key(self):
        obj = _ConventionSample(name="foo", n=7)
        d = obj.to_dict()
        # Flat — no "type"/"params" envelope.
        assert "type" not in d
        assert "params" not in d
        # Underscore-prefixed key
        assert d[_SCHEMA_VERSION_KEY_FLAT] == 1
        # Real fields still present
        assert d["name"] == "foo"
        assert d["n"] == 7

    def test_convention_explicit_v3(self):
        obj = _ConventionV3Sample(name="bar", n=8)
        d = obj.to_dict()
        assert d[_SCHEMA_VERSION_KEY_FLAT] == 3


# ============================================================
# from_dict() round-trips
# ============================================================

class TestRoundTrip:
    def test_serialisable_base(self):
        obj = _BaseMixinSample(10, 20)
        rebuilt = _BaseMixinSample.from_dict(obj.to_dict())
        assert (rebuilt.x, rebuilt.y) == (10, 20)

    def test_decorator(self):
        obj = _DecoratorSample(11, 22)
        rebuilt = _DecoratorSample.from_dict(obj.to_dict())
        assert (rebuilt.a, rebuilt.b) == (11, 22)

    def test_decorator_v2(self):
        obj = _DecoratorV2Sample(12, 24)
        rebuilt = _DecoratorV2Sample.from_dict(obj.to_dict())
        assert (rebuilt.a, rebuilt.b) == (12, 24)

    def test_convention(self):
        obj = _ConventionSample(name="x", n=7)
        rebuilt = _ConventionSample.from_dict(obj.to_dict())
        assert rebuilt == obj

    def test_convention_v3(self):
        obj = _ConventionV3Sample(name="y", n=8)
        rebuilt = _ConventionV3Sample.from_dict(obj.to_dict())
        assert rebuilt == obj


# ============================================================
# Backward compatibility — absent version is treated as v1
# ============================================================

class TestBackwardCompat:
    def test_serialisable_no_version_reads_as_v1(self):
        # Hand-crafted dict (no schema_version) — what existing on-disk
        # payloads serialised before this slice carry.
        d = {"type": "_test_base_mixin_sample", "params": {"x": 100, "y": 200}}
        rebuilt = _BaseMixinSample.from_dict(d)
        assert (rebuilt.x, rebuilt.y) == (100, 200)

    def test_decorator_no_version_reads_as_v1(self):
        d = {"type": "_test_decorator_sample", "params": {"a": 1, "b": 2}}
        rebuilt = _DecoratorSample.from_dict(d)
        assert (rebuilt.a, rebuilt.b) == (1, 2)

    def test_convention_no_version_reads_as_v1(self):
        # Flat dict without _schema_version — pre-slice format
        d = {"name": "old", "n": 7}
        rebuilt = _ConventionSample.from_dict(d)
        assert rebuilt == _ConventionSample(name="old", n=7)


# ============================================================
# Future-version detection — clear, actionable error
# ============================================================

class TestFutureVersionDetection:
    def test_serialisable_rejects_future_version(self):
        d = {
            "type": "_test_base_mixin_sample",
            "params": {"x": 1, "y": 2},
            _SCHEMA_VERSION_KEY: 99,
        }
        with pytest.raises(ValueError, match="schema_version=99"):
            _BaseMixinSample.from_dict(d)

    def test_decorator_rejects_future_version(self):
        d = {
            "type": "_test_decorator_sample",
            "params": {"a": 1, "b": 2},
            _SCHEMA_VERSION_KEY: 7,
        }
        with pytest.raises(ValueError, match="upgrade this environment"):
            _DecoratorSample.from_dict(d)

    def test_convention_rejects_future_version(self):
        d = {"name": "future", "n": 0, _SCHEMA_VERSION_KEY_FLAT: 999}
        with pytest.raises(ValueError, match="schema_version=999"):
            _ConventionSample.from_dict(d)

    def test_v2_class_still_rejects_v3_payload(self):
        d = {
            "type": "_test_decorator_v2_sample",
            "params": {"a": 1, "b": 2},
            _SCHEMA_VERSION_KEY: 3,
        }
        with pytest.raises(ValueError):
            _DecoratorV2Sample.from_dict(d)

    def test_v2_class_accepts_v1_and_v2_payloads(self):
        # Older payload accepted (no migration logic needed when fields haven't
        # actually changed — this is just the versioning machinery).
        d_v1 = {"type": "_test_decorator_v2_sample", "params": {"a": 1, "b": 2}}
        d_v2 = {**d_v1, _SCHEMA_VERSION_KEY: 2}
        assert _DecoratorV2Sample.from_dict(d_v1).a == 1
        assert _DecoratorV2Sample.from_dict(d_v2).b == 2


# ============================================================
# Wire-format invariants (the things consumers rely on)
# ============================================================

class TestWireFormatInvariants:
    def test_top_level_keys_for_envelope_format(self):
        obj = _BaseMixinSample(1, 2)
        d = obj.to_dict()
        # Envelope keys: type, params, schema_version. Nothing else expected.
        assert set(d.keys()) == {"type", "params", _SCHEMA_VERSION_KEY}

    def test_flat_convention_keys(self):
        obj = _ConventionSample(name="x", n=1)
        d = obj.to_dict()
        # Flat: real fields + _schema_version. Nothing else.
        assert set(d.keys()) == {"name", "n", _SCHEMA_VERSION_KEY_FLAT}

    def test_schema_version_does_not_leak_into_constructor(self):
        # _schema_version is metadata, not a constructor field. The convention
        # has no `_schema_version` parameter, so the deserialiser MUST strip it.
        obj = _ConventionSample(name="x", n=7)
        d = obj.to_dict()
        # Round-trip succeeds — proves the key is stripped before __init__.
        rebuilt = _ConventionSample.from_dict(d)
        assert rebuilt == obj
