"""Tests for schema adapter: analyse, translate, custom aliases."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.schema_adapter import analyse_json, SchemaAdapter, SchemaHint
from pricebook.serialisable import from_dict


# ---- Analysis ----

class TestAnalyseJSON:

    def test_already_pricebook_format(self):
        d = {"type": "irs", "params": {"start": "2026-04-28", "end": "2031-04-28", "fixed_rate": 0.035}}
        hint = analyse_json(d)
        assert hint.suggested_type == "irs"
        assert hint.confidence == 1.0
        assert len(hint.unmapped_fields) == 0

    def test_camel_case_irs(self):
        d = {"fixedRate": 0.035, "startDate": "2026-04-28", "endDate": "2031-04-28", "notional": 10_000_000}
        hint = analyse_json(d)
        assert hint.suggested_type == "irs"
        assert hint.confidence > 0.5
        assert hint.field_mapping["fixedRate"] == "fixed_rate"
        assert hint.field_mapping["startDate"] == "start"

    def test_bond_detection(self):
        d = {"couponRate": 0.05, "maturityDate": "2036-04-28", "issueDate": "2026-04-28", "faceValue": 100}
        hint = analyse_json(d)
        assert hint.suggested_type == "bond"
        assert hint.field_mapping["couponRate"] == "coupon_rate"

    def test_cds_detection(self):
        d = {"spread": 0.01, "recoveryRate": 0.4, "start": "2026-04-28", "end": "2031-04-28"}
        hint = analyse_json(d)
        assert hint.suggested_type == "cds"

    def test_nested_unwrap(self):
        d = {"instrument": {"type": "swap", "fixedRate": 0.035, "start": "2026-04-28", "end": "2031-04-28"}}
        hint = analyse_json(d)
        assert hint.suggested_type == "irs"
        assert "Unwrapped from 'instrument'" in hint.warnings

    def test_type_alias(self):
        d = {"type": "InterestRateSwap", "params": {"start": "2026-04-28", "end": "2031-04-28", "fixed_rate": 0.035}}
        hint = analyse_json(d)
        assert hint.suggested_type == "irs"

    def test_unknown_type_guesses(self):
        d = {"type": "exotic_widget", "fixedRate": 0.035, "start": "2026-04-28", "end": "2031-04-28", "notional": 1000000}
        hint = analyse_json(d)
        assert hint.suggested_type is not None
        assert any("Unknown type" in w for w in hint.warnings)

    def test_missing_required(self):
        d = {"fixedRate": 0.035}
        hint = analyse_json(d)
        assert len(hint.missing_required) > 0

    def test_curve_detection(self):
        d = {"referenceDate": "2026-04-28", "dates": ["2027-04-28"], "discountFactors": [0.97]}
        hint = analyse_json(d)
        assert hint.suggested_type == "discount_curve"
        assert hint.field_mapping["discountFactors"] == "dfs"

    def test_to_dict(self):
        d = {"fixedRate": 0.035, "startDate": "2026-04-28", "endDate": "2031-04-28"}
        hint = analyse_json(d)
        hd = hint.to_dict()
        assert "suggested_type" in hd
        assert "confidence" in hd


# ---- Translation ----

class TestSchemaAdapter:

    def test_translate_camel_case(self):
        adapter = SchemaAdapter()
        external = {"fixedRate": 0.035, "startDate": "2026-04-28", "endDate": "2031-04-28",
                     "notional": 1_000_000, "type": "swap"}
        pb = adapter.translate(external)
        assert pb["type"] == "irs"
        assert pb["params"]["fixed_rate"] == 0.035
        assert pb["params"]["start"] == "2026-04-28"

    def test_translate_and_construct(self):
        """Full pipeline: external JSON → translate → from_dict → instrument."""
        adapter = SchemaAdapter()
        external = {"type": "swap", "fixedRate": 0.035, "startDate": "2026-04-28",
                     "endDate": "2031-04-28", "notional": 10_000_000}
        pb = adapter.translate(external)
        irs = from_dict(pb)
        assert irs.fixed_rate == 0.035
        assert irs.notional == 10_000_000

    def test_passthrough_pricebook_format(self):
        adapter = SchemaAdapter()
        d = {"type": "cds", "params": {"start": "2026-04-28", "end": "2031-04-28", "spread": 0.01}}
        pb = adapter.translate(d)
        assert pb == d

    def test_custom_aliases(self):
        adapter = SchemaAdapter(source="murex")
        adapter.add_alias("TAUX_FIXE", "fixed_rate")
        adapter.add_alias("DATE_DEBUT", "start")
        adapter.add_alias("DATE_FIN", "end")
        adapter.add_type_alias("IRS_VANILLA", "irs")

        external = {"type": "IRS_VANILLA", "TAUX_FIXE": 0.035,
                     "DATE_DEBUT": "2026-04-28", "DATE_FIN": "2031-04-28",
                     "notional": 1_000_000}
        pb = adapter.translate(external)
        assert pb["type"] == "irs"
        assert pb["params"]["fixed_rate"] == 0.035

    def test_batch_aliases(self):
        adapter = SchemaAdapter()
        adapter.add_aliases({"PRIX_FIXE": "fixed_rate", "DEBUT": "start", "FIN": "end"})
        assert adapter._field_aliases["PRIX_FIXE"] == "fixed_rate"

    def test_unknown_type_raises(self):
        adapter = SchemaAdapter()
        with pytest.raises(ValueError, match="Cannot determine"):
            adapter.translate({"foo": "bar"})

    def test_nested_translate(self):
        adapter = SchemaAdapter()
        external = {"instrument": {"fixedRate": 0.035, "startDate": "2026-04-28",
                                    "endDate": "2031-04-28", "notional": 1_000_000}}
        pb = adapter.translate(external)
        assert pb["type"] == "irs"

    def test_bond_translate(self):
        adapter = SchemaAdapter()
        external = {"type": "fixed_rate_bond", "couponRate": 0.05,
                     "maturityDate": "2036-04-28", "issueDate": "2026-04-28",
                     "faceValue": 100}
        pb = adapter.translate(external)
        assert pb["type"] == "bond"
        bond = from_dict(pb)
        assert bond.coupon_rate == 0.05

    def test_pv_after_translate(self):
        """Translate external → construct → price → verify finite."""
        from pricebook.discount_curve import DiscountCurve
        adapter = SchemaAdapter()
        external = {"type": "swap", "fixedRate": 0.035, "startDate": "2026-04-28",
                     "endDate": "2031-04-28", "notional": 10_000_000}
        pb = adapter.translate(external)
        irs = from_dict(pb)
        curve = DiscountCurve.flat(date(2026, 4, 28), 0.03)
        pv = irs.pv(curve)
        assert math.isfinite(pv)
