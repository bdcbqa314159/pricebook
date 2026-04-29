"""Schema adapter: translate external JSON into pricebook format.

When receiving data from another system (risk engine, trading platform,
market data provider), the field names and structure may not match
pricebook's `{"type": ..., "params": {...}}` convention. This adapter
analyses incoming JSON and suggests mappings.

    from pricebook.schema_adapter import SchemaAdapter, analyse_json

    # Analyse unknown JSON and get hints
    hints = analyse_json(external_data)
    print(hints.suggested_type)      # "irs"
    print(hints.field_mapping)       # {"fixedRate": "fixed_rate", ...}
    print(hints.confidence)          # 0.85

    # Translate
    adapter = SchemaAdapter()
    adapter.add_alias("swap_rate", "fixed_rate")
    adapter.add_alias("startDate", "start")
    pricebook_dict = adapter.translate(external_data)
    instrument = from_dict(pricebook_dict)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pricebook.serialisable import _REGISTRY


# ---------------------------------------------------------------------------
# Known field signatures per type (for detection)
# ---------------------------------------------------------------------------

_TYPE_SIGNATURES: dict[str, set[str]] = {
    "irs": {"fixed_rate", "start", "end", "notional"},
    "bond": {"coupon_rate", "maturity", "face_value", "issue_date"},
    "cds": {"spread", "recovery", "start", "end"},
    "fra": {"strike", "start", "end"},
    "deposit": {"rate", "start", "end"},
    "ois": {"fixed_rate", "start", "end"},
    "swaption": {"expiry", "strike", "swap_end"},
    "capfloor": {"strike", "option_type", "start", "end"},
    "term_loan": {"spread", "amort_rate", "start", "end"},
    "revolver": {"max_commitment", "drawn_amount"},
    "cln": {"coupon_rate", "leverage", "recovery"},
    "fx_forward": {"pair", "strike", "maturity"},
    "trs": {"underlying", "notional", "repo_spread"},
    "discount_curve": {"dates", "dfs", "reference_date"},
    "survival_curve": {"dates", "survival_probs", "reference_date"},
}

# Common field name aliases: external → pricebook
_COMMON_ALIASES: dict[str, str] = {
    # camelCase → snake_case
    "fixedRate": "fixed_rate",
    "floatRate": "float_rate",
    "fixedFrequency": "fixed_frequency",
    "floatFrequency": "float_frequency",
    "fixedDayCount": "fixed_day_count",
    "floatDayCount": "float_day_count",
    "startDate": "start",
    "endDate": "end",
    "effectiveDate": "start",
    "maturityDate": "maturity",
    "terminationDate": "end",
    "expiryDate": "expiry",
    "swapEnd": "swap_end",
    "issueDate": "issue_date",
    "couponRate": "coupon_rate",
    "faceValue": "face_value",
    "dayCount": "day_count",
    "dayCountFraction": "day_count",
    "amortRate": "amort_rate",
    "maxCommitment": "max_commitment",
    "drawnAmount": "drawn_amount",
    "drawnSpread": "drawn_spread",
    "undrawnFee": "undrawn_fee",
    "repoSpread": "repo_spread",
    "recoveryRate": "recovery",
    "optionType": "option_type",
    "swaptionType": "swaption_type",
    "tradeId": "trade_id",
    "tradeID": "trade_id",
    "referenceDate": "reference_date",
    "valuationDate": "valuation_date",
    "survivalProbs": "survival_probs",
    "survivalProbabilities": "survival_probs",
    "discountFactors": "dfs",
    "paymentDelay": "payment_delay_days",
    "observationShift": "observation_shift_days",
    # Common abbreviations
    "ccy": "currency",
    "curr": "currency",
    "freq": "frequency",
    "dc": "day_count",
    "nom": "notional",
    "nominal": "notional",
    "principal": "notional",
    "cpn": "coupon_rate",
    "coupon": "coupon_rate",
    "vol": "sigma",
    "volatility": "sigma",
    # Bloomberg-style
    "SW_FIXED_RATE": "fixed_rate",
    "SW_NOTIONAL": "notional",
    "SW_MATURITY": "end",
    "SW_EFF_DT": "start",
    "CPN": "coupon_rate",
    "MATURITY": "maturity",
    "ISSUE_DT": "issue_date",
    "PAR_AMT": "face_value",
    "CDS_SPREAD": "spread",
    "RECOVERY_RATE": "recovery",
}

# Type name aliases
_TYPE_ALIASES: dict[str, str] = {
    "swap": "irs",
    "interest_rate_swap": "irs",
    "InterestRateSwap": "irs",
    "fixed_rate_bond": "bond",
    "FixedRateBond": "bond",
    "credit_default_swap": "cds",
    "CreditDefaultSwap": "cds",
    "forward_rate_agreement": "fra",
    "ForwardRateAgreement": "fra",
    "ois_swap": "ois",
    "OISSwap": "ois",
    "cap": "capfloor",
    "floor": "capfloor",
    "Cap": "capfloor",
    "Floor": "capfloor",
    "TermLoan": "term_loan",
    "RevolvingFacility": "revolver",
    "CreditLinkedNote": "cln",
    "FXForward": "fx_forward",
    "fx_fwd": "fx_forward",
    "TotalReturnSwap": "trs",
    "total_return_swap": "trs",
    "DiscountCurve": "discount_curve",
    "SurvivalCurve": "survival_curve",
    "SpreadCurve": "spread_curve",
}


def _normalise_key(key: str) -> str:
    """Convert camelCase/PascalCase/UPPER_CASE to snake_case."""
    # camelCase → snake_case
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    return s.lower().strip("_")


# ---------------------------------------------------------------------------
# Analysis result
# ---------------------------------------------------------------------------

@dataclass
class SchemaHint:
    """Result of analysing external JSON."""
    suggested_type: str | None       # best-guess pricebook type key
    confidence: float                 # 0.0 to 1.0
    field_mapping: dict[str, str]     # external_key → pricebook_key
    unmapped_fields: list[str]        # fields we couldn't map
    missing_required: list[str]       # pricebook fields not found in input
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "suggested_type": self.suggested_type,
            "confidence": round(self.confidence, 3),
            "field_mapping": self.field_mapping,
            "unmapped_fields": self.unmapped_fields,
            "missing_required": self.missing_required,
            "warnings": self.warnings,
        }


def analyse_json(data: dict[str, Any]) -> SchemaHint:
    """Analyse external JSON and suggest pricebook type + field mappings."""
    return _analyse_json_with_aliases(data, _COMMON_ALIASES, _TYPE_ALIASES)


def _analyse_json_with_aliases(
    data: dict[str, Any],
    field_aliases: dict[str, str],
    type_aliases: dict[str, str],
) -> SchemaHint:
    """Internal: analyse with explicit alias dicts (thread-safe).

    Works with:
    - Already pricebook-formatted: {"type": "irs", "params": {...}}
    - Flat dict: {"fixedRate": 0.035, "startDate": "2026-04-28", ...}
    - Nested: {"instrument": {"type": "swap", ...}, "trade": {...}}
    """
    warnings = []

    # Already in pricebook format?
    if "type" in data and "params" in data:
        t = type_aliases.get(data["type"], data["type"])
        if t in _REGISTRY:
            return SchemaHint(
                suggested_type=t, confidence=1.0,
                field_mapping={k: k for k in data["params"]},
                unmapped_fields=[], missing_required=[], warnings=[],
            )

    # Extract the payload (might be nested under "instrument", "trade", "data", etc.)
    payload = data
    for wrapper_key in ("instrument", "trade", "data", "params", "body"):
        if wrapper_key in data and isinstance(data[wrapper_key], dict):
            payload = data[wrapper_key]
            warnings.append(f"Unwrapped from '{wrapper_key}'")
            break

    # Resolve type if present
    explicit_type = None
    for type_key in ("type", "instrumentType", "instrument_type", "product", "productType"):
        if type_key in payload:
            raw_type = str(payload[type_key])
            explicit_type = type_aliases.get(raw_type, raw_type)
            break

    # Map fields
    field_mapping = {}
    unmapped = []
    normalised_payload = {}

    for ext_key, value in payload.items():
        if ext_key in ("type", "instrumentType", "instrument_type", "product", "productType"):
            continue
        # Try direct alias
        if ext_key in field_aliases:
            pb_key = field_aliases[ext_key]
            field_mapping[ext_key] = pb_key
            normalised_payload[pb_key] = value
        # Try normalised
        elif _normalise_key(ext_key) in field_aliases:
            pb_key = field_aliases[_normalise_key(ext_key)]
            field_mapping[ext_key] = pb_key
            normalised_payload[pb_key] = value
        # Try snake_case directly as pricebook key
        elif _normalise_key(ext_key) != ext_key:
            snake = _normalise_key(ext_key)
            field_mapping[ext_key] = snake
            normalised_payload[snake] = value
        else:
            # Already snake_case — pass through
            field_mapping[ext_key] = ext_key
            normalised_payload[ext_key] = value

    # Guess type from field signatures if not explicit
    if explicit_type and explicit_type in _REGISTRY:
        best_type = explicit_type
        confidence = 0.9
    else:
        best_type, confidence = _guess_type(set(normalised_payload.keys()))
        if explicit_type:
            warnings.append(f"Unknown type '{explicit_type}', guessed '{best_type}'")

    # Check what's missing
    missing = []
    if best_type and best_type in _TYPE_SIGNATURES:
        required = _TYPE_SIGNATURES[best_type]
        present = set(normalised_payload.keys())
        missing = sorted(required - present)

    # Find unmapped (fields that don't match any known pricebook field)
    if best_type and best_type in _REGISTRY:
        cls = _REGISTRY[best_type]
        known_fields = set(getattr(cls, "_SERIAL_FIELDS", []))
        unmapped = [k for k in normalised_payload if k not in known_fields and known_fields]

    return SchemaHint(
        suggested_type=best_type,
        confidence=confidence,
        field_mapping=field_mapping,
        unmapped_fields=unmapped,
        missing_required=missing,
        warnings=warnings,
    )


def _guess_type(fields: set[str]) -> tuple[str | None, float]:
    """Guess the pricebook type from a set of field names.

    Scores by overlap ratio, breaking ties with absolute overlap count
    (prefer the type that matches more fields).
    """
    best_type = None
    best_score = 0.0
    best_overlap = 0

    for type_key, signature in _TYPE_SIGNATURES.items():
        overlap = len(fields & signature)
        total = len(signature)
        if total == 0:
            continue
        score = overlap / total
        if score > best_score or (score == best_score and overlap > best_overlap):
            best_score = score
            best_type = type_key
            best_overlap = overlap

    return best_type, best_score


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------

class SchemaAdapter:
    """Translates external JSON into pricebook format.

    Maintains custom alias mappings on top of the built-in ones.
    Can be configured per source system.

        adapter = SchemaAdapter(source="murex")
        adapter.add_alias("TAUX_FIXE", "fixed_rate")
        adapter.add_type_alias("IRS_VANILLA", "irs")
        pb_dict = adapter.translate(external_data)
    """

    def __init__(self, source: str = "unknown"):
        self.source = source
        self._field_aliases: dict[str, str] = dict(_COMMON_ALIASES)
        self._type_aliases: dict[str, str] = dict(_TYPE_ALIASES)

    def add_alias(self, external: str, pricebook: str) -> None:
        """Add a field name mapping: external → pricebook."""
        self._field_aliases[external] = pricebook

    def add_type_alias(self, external: str, pricebook: str) -> None:
        """Add a type name mapping."""
        self._type_aliases[external] = pricebook

    def add_aliases(self, mapping: dict[str, str]) -> None:
        """Add multiple field aliases at once."""
        self._field_aliases.update(mapping)

    def translate(self, data: dict[str, Any]) -> dict[str, Any]:
        """Translate external JSON into pricebook `{"type": ..., "params": {...}}`.

        Uses analyse_json for detection, then builds the pricebook dict.
        Returns the translated dict ready for from_dict().
        """
        # Already pricebook format?
        if "type" in data and "params" in data:
            t = self._type_aliases.get(data["type"], data["type"])
            if t in _REGISTRY:
                return {"type": t, "params": data["params"]}

        hint = self.analyse(data)

        if hint.suggested_type is None:
            raise ValueError(f"Cannot determine instrument type from: {list(data.keys())}")

        # Build params from mapped fields
        payload = data
        for wrapper_key in ("instrument", "trade", "data", "params", "body"):
            if wrapper_key in data and isinstance(data[wrapper_key], dict):
                payload = data[wrapper_key]
                break

        params = {}
        for ext_key, value in payload.items():
            if ext_key in ("type", "instrumentType", "instrument_type", "product", "productType"):
                continue
            pb_key = self._resolve_field(ext_key)
            params[pb_key] = value

        return {"type": hint.suggested_type, "params": params}

    def analyse(self, data: dict[str, Any]) -> SchemaHint:
        """Analyse with this adapter's custom aliases (thread-safe)."""
        # Use merged copies instead of mutating globals
        merged_field = {**_COMMON_ALIASES, **self._field_aliases}
        merged_type = {**_TYPE_ALIASES, **self._type_aliases}
        return _analyse_json_with_aliases(data, merged_field, merged_type)

    def _resolve_field(self, ext_key: str) -> str:
        """Resolve an external field name to pricebook."""
        if ext_key in self._field_aliases:
            return self._field_aliases[ext_key]
        normalised = _normalise_key(ext_key)
        if normalised in self._field_aliases:
            return self._field_aliases[normalised]
        return normalised

    def to_dict(self) -> dict[str, Any]:
        """Serialise adapter config (custom aliases only, not built-ins)."""
        custom_fields = {k: v for k, v in self._field_aliases.items()
                         if k not in _COMMON_ALIASES or self._field_aliases[k] != _COMMON_ALIASES.get(k)}
        custom_types = {k: v for k, v in self._type_aliases.items()
                        if k not in _TYPE_ALIASES or self._type_aliases[k] != _TYPE_ALIASES.get(k)}
        return {"source": self.source, "field_aliases": custom_fields,
                "type_aliases": custom_types}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SchemaAdapter:
        adapter = cls(source=d.get("source", "unknown"))
        adapter.add_aliases(d.get("field_aliases", {}))
        for k, v in d.get("type_aliases", {}).items():
            adapter.add_type_alias(k, v)
        return adapter
