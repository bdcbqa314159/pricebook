"""The `verify.py conventions` gate (tranche 41c) — the register cannot silently corrupt.

**Why a gate rather than care.** Tranche 41b found the conventions register carrying **two `C11`
rows**: the tranche-39 protection-leg row took an id the invariant-6 `model.market` row already
held. Two review passes read past it, **including the Cowork pass that ratified "C11" by name at
tranche 39**. The register is cited from `src/` docstrings, from `OPEN.md`, from the CHANGELOG and
from CLAUDE.md, and nothing checks it mechanically — so a duplicate id, or a citation to a row that
does not exist, is invisible to every gate. That is the tranche-26b references-drift shape exactly,
and it is **statically decidable**, which is what makes it gateable under §9's boundary where the
τ=1 fixture guard was not.

**The oracle is seeded drift, which is mutation applied to a gate** (the T26b/T38 pattern): each
corruption shape is reconstructed against a copy of the *real* register and the gate must reject
it, because a gate that cannot fail is empty.

**Scope is the register's text, deliberately.** The gate does not check that a row's *claim* is true
— that a `guarded` row really is guarded is the §9 register-pass's judgement, not a parse — which is
the τ=1 over-reach lesson: a gate needing judgement needs an exemption list, and a gate with an
exemption list cannot fail.
"""

import re
from pathlib import Path

import pytest

from verify import conventions, live_convention_ids, register_citations

ROOT = Path(__file__).resolve().parents[2]
REGISTER = ROOT / "docs" / "conventions_register.md"


class TestTheRealRegisterIsSound:
    def test_the_gate_passes_on_the_real_register(self) -> None:
        assert conventions() == []

    def test_the_live_ids_are_a_GAP_FREE_ASCENDING_run(self) -> None:
        """Derived from the register, not hard-coded: the live rows begin at **C7**, not C10.

        The tranche-41c spec said the live sequence starts at the "stated floor… C10", but C7–C9
        are live too — the register's note only says *new* rows continue at C10. Reading the floor
        out of the table rather than pinning it keeps the gate right about that."""
        ids = live_convention_ids(REGISTER.read_text())
        assert len(ids) >= 7
        assert ids == sorted(ids), ids
        assert ids == list(range(ids[0], ids[0] + len(ids))), ids

    def test_every_CITATION_in_the_register_resolves(self) -> None:
        """Including citations to the **archived** C1–C6 rows, which are legitimate references to
        a documented past and must not be reported as dangling."""
        problems = [p for p in conventions() if "cites" in p]
        assert problems == []
        assert 1 in register_citations(REGISTER.read_text())


class TestTheGateREJECTSEachCorruptionShape:
    """Seeded against the real register, so no fixture can drift away from it."""

    @staticmethod
    def _run(monkeypatch, register: str) -> list[str]:
        real = Path.read_text

        def patched(self, *args, **kwargs):
            if self.name == "conventions_register.md":
                return register
            return real(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", patched)
        return conventions()

    def test_a_DUPLICATE_id_is_rejected(self, monkeypatch) -> None:
        """Tranche 41b's actual defect, reproduced: renumber C13 back to C11."""
        text = REGISTER.read_text()
        seeded = text.replace("| C13 |", "| C11 |", 1)
        assert seeded != text, "the premise: C13 must be there to collide"
        problems = self._run(monkeypatch, seeded)
        assert any("C11" in p and "twice" in p for p in problems), problems

    def test_a_DANGLING_citation_is_rejected(self, monkeypatch) -> None:
        text = REGISTER.read_text()
        seeded = text.replace("Same family as C7/C8", "Same family as C7/C99", 1)
        assert seeded != text, "the premise"
        problems = self._run(monkeypatch, seeded)
        assert any("C99" in p for p in problems), problems

    def test_a_GAP_in_the_live_sequence_is_rejected(self, monkeypatch) -> None:
        """A renumber that drops a row leaves a hole no reader would notice."""
        text = REGISTER.read_text()
        seeded = re.sub(r"\n\| C11 \|.*?(?=\n\| C12 \|)", "", text, count=1, flags=re.S)
        assert "| C11 |" not in seeded, "the premise: the row must actually be gone"
        problems = self._run(monkeypatch, seeded)
        assert any("gap" in p for p in problems), problems

    def test_an_OUT_OF_ORDER_live_sequence_is_rejected(self, monkeypatch) -> None:
        """Which is the state tranche 41b's renumber actually left behind: C13 sat between C10 and
        C11, because the id was edited in place. Harmless to a parser and confusing to a reader,
        so it is caught here and the file was reordered."""
        text = REGISTER.read_text()
        rows = re.findall(r"\n\| C\d+ \|", text)
        assert len(rows) >= 7
        seeded = text.replace("| C12 |", "| C99X |", 1).replace("| C13 |", "| C12 |", 1)
        seeded = seeded.replace("| C99X |", "| C13 |", 1)
        problems = self._run(monkeypatch, seeded)
        assert any("ascending" in p or "gap" in p for p in problems), problems

    def test_an_UNPARSEABLE_register_is_rejected_rather_than_read_as_empty(
        self, monkeypatch) -> None:
        """Silently reading no rows would make every citation dangle *and* the sequence vacuous —
        the T38 lesson about a mangled section being treated as absent."""
        problems = self._run(monkeypatch, "# conventions\n\nnothing here\n")
        assert len(problems) >= 1
        assert any("no live convention rows" in p for p in problems), problems
