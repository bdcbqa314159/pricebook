"""Tests for ts._io: from_csv (db tests would need fixtures)."""
import pytest
from pricebook.ts._io import from_csv

class TestFromCSV:
    def test_import(self):
        # Just verify the function exists and is callable
        assert callable(from_csv)
