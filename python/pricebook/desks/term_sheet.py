"""Term sheet generator: markdown-based structured output from any instrument.

    from pricebook.desks.term_sheet import generate_term_sheet

References:
    ISDA (2006). 2006 ISDA Definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


# ═══════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════

@dataclass
class TermSheetSection:
    """One section of a term sheet."""
    title: str
    content: str

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class TermSheet:
    """Complete term sheet document."""
    title: str
    instrument_type: str
    sections: list[TermSheetSection]
    generated_date: date

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", f"*Generated: {self.generated_date}*", ""]
        for s in self.sections:
            lines.append(f"## {s.title}")
            lines.append(s.content)
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "instrument_type": self.instrument_type,
            "generated_date": self.generated_date.isoformat(),
            "sections": [s.to_dict() for s in self.sections],
        }


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Format a markdown table."""
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def _format_number(value: float, decimals: int = 2) -> str:
    if abs(value) >= 1e6:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:,.{decimals}f}"
    return f"{value:.{max(decimals, 4)}f}"


def _extract_key_terms(inst_dict: dict) -> list[tuple[str, str]]:
    """Extract human-readable key terms from instrument dict."""
    terms = []
    key_map = {
        "notional": "Notional", "start": "Start Date", "end": "End Date",
        "maturity": "Maturity", "spread": "Spread", "coupon": "Coupon",
        "strike": "Strike", "expiry": "Expiry", "frequency": "Frequency",
        "day_count": "Day Count", "currency": "Currency",
        "recovery": "Recovery Rate", "direction": "Direction",
    }
    for key, label in key_map.items():
        if key in inst_dict:
            val = inst_dict[key]
            if isinstance(val, float):
                terms.append((label, _format_number(val)))
            else:
                terms.append((label, str(val)))
    return terms


# ═══════════════════════════════════════════════════════════════
# Section Generators
# ═══════════════════════════════════════════════════════════════

def _summary_section(instrument, metadata: dict | None = None) -> TermSheetSection:
    inst_type = type(instrument).__name__
    meta = metadata or {}
    lines = [f"**Instrument Type:** {inst_type}"]
    for k in ["counterparty", "book", "desk", "trader"]:
        if k in meta:
            lines.append(f"**{k.title()}:** {meta[k]}")
    if hasattr(instrument, "notional"):
        lines.append(f"**Notional:** {_format_number(instrument.notional)}")
    return TermSheetSection("Deal Summary", "\n".join(lines))


def _key_terms_section(instrument) -> TermSheetSection:
    if hasattr(instrument, "to_dict"):
        d = instrument.to_dict()
    else:
        d = {}
    terms = _extract_key_terms(d)
    if not terms:
        return TermSheetSection("Key Terms", "*No structured terms available.*")
    rows = [[k, v] for k, v in terms]
    return TermSheetSection("Key Terms", _format_table(["Term", "Value"], rows))


def _risk_section(instrument, pv: float | None = None) -> TermSheetSection:
    lines = []
    if pv is not None:
        lines.append(f"**Present Value:** {_format_number(pv)}")
    for attr in ["delta", "gamma", "vega", "theta", "rho", "dv01", "cs01"]:
        if hasattr(instrument, attr):
            val = getattr(instrument, attr)
            if callable(val):
                continue
            lines.append(f"**{attr.upper()}:** {_format_number(val)}")
    if not lines:
        lines.append("*Risk profile computed via bump-and-reprice on PricingContext.*")
    return TermSheetSection("Risk Profile", "\n".join(lines))


def _scenario_section(scenarios: list[dict] | None = None) -> TermSheetSection:
    if not scenarios:
        return TermSheetSection("Scenario Analysis", "*No scenarios specified.*")
    headers = list(scenarios[0].keys())
    rows = [[str(s.get(h, "")) for h in headers] for s in scenarios]
    return TermSheetSection("Scenario Analysis", _format_table(headers, rows))


# ═══════════════════════════════════════════════════════════════
# Main Functions
# ═══════════════════════════════════════════════════════════════

def generate_term_sheet(
    instrument,
    pv: float | None = None,
    *,
    title: str | None = None,
    metadata: dict | None = None,
    scenarios: list[dict] | None = None,
    generated_date: date | None = None,
) -> TermSheet:
    """Generate a term sheet for any instrument.

    Args:
        instrument: any pricebook instrument with to_dict().
        pv: pre-computed present value (optional).
        title: custom title. Defaults to instrument type.
        metadata: counterparty, book, desk, trader info.
        scenarios: list of scenario result dicts for scenario section.
        generated_date: defaults to today.
    """
    inst_type = type(instrument).__name__
    if generated_date is None:
        generated_date = date.today()

    sections = [
        _summary_section(instrument, metadata),
        _key_terms_section(instrument),
        _risk_section(instrument, pv),
    ]
    if scenarios:
        sections.append(_scenario_section(scenarios))

    return TermSheet(
        title=title or f"{inst_type} Term Sheet",
        instrument_type=inst_type,
        sections=sections,
        generated_date=generated_date,
    )
