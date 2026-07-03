"""Tests for fixings manager."""

import json
import os
import tempfile
import pytest
from datetime import date, datetime

from pricebook.core.fixings import FixingsStore, create_sample_fixings


REF = date(2024, 6, 15)


# ---- Basic operations ----

class TestFixingsStore:
    def test_set_and_get(self):
        store = FixingsStore()
        store.set("SOFR", date(2024, 1, 15), 0.043)
        assert store.get("SOFR", date(2024, 1, 15)) == 0.043

    def test_get_missing_returns_none(self):
        store = FixingsStore()
        assert store.get("SOFR", date(2024, 1, 15)) is None

    def test_get_or_raise(self):
        store = FixingsStore()
        store.set("SOFR", date(2024, 1, 15), 0.043)
        assert store.get_or_raise("SOFR", date(2024, 1, 15)) == 0.043

    def test_get_or_raise_missing(self):
        store = FixingsStore()
        with pytest.raises(KeyError):
            store.get_or_raise("SOFR", date(2024, 1, 15))

    def test_has(self):
        store = FixingsStore()
        store.set("SOFR", date(2024, 1, 15), 0.043)
        assert store.has("SOFR", date(2024, 1, 15))
        assert not store.has("SOFR", date(2024, 1, 16))

    def test_rate_names(self):
        store = FixingsStore()
        store.set("SOFR", date(2024, 1, 15), 0.043)
        store.set("ESTR", date(2024, 1, 15), 0.035)
        assert store.rate_names() == ["ESTR", "SOFR"]

    def test_dates_for(self):
        store = FixingsStore()
        store.set("SOFR", date(2024, 1, 15), 0.043)
        store.set("SOFR", date(2024, 1, 16), 0.044)
        dates = store.dates_for("SOFR")
        assert dates == [date(2024, 1, 15), date(2024, 1, 16)]

    def test_series(self):
        store = FixingsStore()
        store.set("SOFR", date(2024, 1, 15), 0.043)
        store.set("SOFR", date(2024, 1, 16), 0.044)
        store.set("SOFR", date(2024, 1, 17), 0.045)
        s = store.series("SOFR", start=date(2024, 1, 16))
        assert len(s) == 2
        assert s[0] == (date(2024, 1, 16), 0.044)

    def test_bulk_set(self):
        store = FixingsStore()
        fixings = [(date(2024, 1, d), 0.04 + d * 0.001) for d in range(1, 6)]
        store.bulk_set("SOFR", fixings)
        assert len(store.dates_for("SOFR")) == 5

    def test_overwrite(self):
        store = FixingsStore()
        store.set("SOFR", date(2024, 1, 15), 0.043)
        store.set("SOFR", date(2024, 1, 15), 0.044)
        assert store.get("SOFR", date(2024, 1, 15)) == 0.044


# ---- Persistence ----

class TestPersistence:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FixingsStore(tmpdir)
            store.set("SOFR", date(2024, 1, 15), 0.043)
            store.set("SOFR", date(2024, 1, 16), 0.044)
            store.save()

            # Reload
            store2 = FixingsStore(tmpdir)
            assert store2.get("SOFR", date(2024, 1, 15)) == 0.043
            assert store2.get("SOFR", date(2024, 1, 16)) == 0.044

    def test_save_creates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FixingsStore(tmpdir)
            store.set("SOFR", date(2024, 1, 15), 0.043)
            store.save()
            assert os.path.exists(os.path.join(tmpdir, "SOFR.json"))

    def test_multiple_rates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FixingsStore(tmpdir)
            store.set("SOFR", date(2024, 1, 15), 0.043)
            store.set("ESTR", date(2024, 1, 15), 0.035)
            store.save()

            store2 = FixingsStore(tmpdir)
            assert store2.get("SOFR", date(2024, 1, 15)) == 0.043
            assert store2.get("ESTR", date(2024, 1, 15)) == 0.035


# ---- Sample fixings ----

class TestSampleFixings:
    def test_creates_sofr(self):
        store = create_sample_fixings(REF)
        assert "SOFR" in store.rate_names()
        assert len(store.dates_for("SOFR")) > 100

    def test_creates_estr(self):
        store = create_sample_fixings(REF)
        assert "ESTR" in store.rate_names()

    def test_creates_cpi(self):
        store = create_sample_fixings(REF)
        assert "CPI" in store.rate_names()
        # CPI should be around 310+
        series = store.series("CPI")
        assert series[-1][1] > 300

    def test_reasonable_rates(self):
        store = create_sample_fixings(REF)
        for d, v in store.series("SOFR"):
            assert 0 < v < 0.20

    def test_deterministic(self):
        s1 = create_sample_fixings(REF)
        s2 = create_sample_fixings(REF)
        assert s1.series("SOFR") == s2.series("SOFR")


class TestSaveSafety:
    def test_save_rejects_path_traversal_rate_name(self, tmp_path):
        """A rate name with path separators must not be written outside the dir."""
        store = FixingsStore()
        store.set("../evil", date(2024, 1, 15), 0.05)
        with pytest.raises(ValueError, match="Unsafe rate name"):
            store.save(str(tmp_path))
        # nothing escaped the target dir
        assert not (tmp_path.parent / "evil.json").exists()

    def test_save_accepts_normal_rate_name(self, tmp_path):
        store = FixingsStore()
        store.set("SOFR", date(2024, 1, 15), 0.043)
        store.save(str(tmp_path))
        assert (tmp_path / "SOFR.json").exists()


class TestDatetimeKeyNormalisation:
    """datetime is a subclass of date — keys must collapse to the calendar day."""

    def test_set_datetime_get_date(self):
        s = FixingsStore()
        s.set("SOFR", datetime(2024, 1, 15, 9, 0), 0.043)
        assert s.get("SOFR", date(2024, 1, 15)) == 0.043

    def test_set_date_get_datetime(self):
        s = FixingsStore()
        s.set("SOFR", date(2024, 1, 15), 0.043)
        assert s.get("SOFR", datetime(2024, 1, 15, 23, 59)) == 0.043

    def test_same_day_datetime_and_date_are_one_entry(self):
        s = FixingsStore()
        s.set("X", date(2024, 1, 15), 1.0)
        s.set("X", datetime(2024, 1, 15, 9, 0), 2.0)  # same day → overwrites
        assert s.series("X") == [(date(2024, 1, 15), 2.0)]

    def test_datetime_key_does_not_poison_series(self):
        s = FixingsStore()
        s.set("X", datetime(2024, 1, 15, 9, 0), 1.0)
        # would previously TypeError: can't compare datetime to date
        assert s.series("X") == [(date(2024, 1, 15), 1.0)]
        assert s.dates_for("X") == [date(2024, 1, 15)]


class TestLoadErrorContext:
    def test_corrupt_file_names_the_file(self, tmp_path):
        (tmp_path / "SOFR.json").write_text("{ not valid json")
        with pytest.raises(ValueError, match="SOFR.json"):
            FixingsStore(str(tmp_path))

    def test_wrong_shape_file_names_the_file(self, tmp_path):
        (tmp_path / "config.json").write_text('{"foo": [1, 2, 3]}')
        with pytest.raises(ValueError, match="config.json"):
            FixingsStore(str(tmp_path))
