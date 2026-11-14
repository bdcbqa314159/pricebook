"""The `verify.py gates` gate (tranche 38) — the charter's gate list cannot diverge from `ci.yml`.

**Why a gate rather than care.** `verify.py references` was built at tranche 26b, named in charter
§9, and celebrated in §7 as *"the gate's first run caught the other two"* — and was **never wired
into `ci.yml`**. For twelve tranches it ran only when someone remembered `verify.py all` by hand,
which makes it a lint, not a gate: a cited key left at `in-hand` could have merged green the whole
time. It caught `OKANE-2008` at T38 purely because `verify.py all` happened to be run locally.

This is the same empty-gate defect as the §9 "a gate that can't fail" rule, one level up — and it
is **statically decidable** (two files, two lists, one comparison), which is precisely what makes
it gateable where the τ=1 fixture-degeneracy guard was not (§9's gateability boundary, ratified in
the same tranche after two prototypes were declined for needing a 7-of-14 exemption list).

**The oracle is seeded drift, which is mutation applied to a gate.** Each of the three divergence
shapes is reconstructed against copies of the two real files, and the gate must reject it. The
third shape is the one a set comparison alone cannot see: the charter and the workflow **agreeing
on a name `verify.py` does not implement**, where both sides are mutually consistent and both
wrong.
"""

import re
from pathlib import Path

from verify import _KNOWN_GATES, gates

ROOT = Path(__file__).resolve().parents[2]
CHARTER = ROOT / "CLAUDE.md"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _listed(charter: str) -> set[str]:
    match = re.search(r"- \*\*`verify\.py`\*\*:(.*?)\. CI runs", charter, re.S)
    assert match is not None, "§9's gate list must open with '- **`verify.py`**: …. CI runs'"
    return set(re.findall(r"`([a-z]+)`", match.group(1)))


def _invoked(workflow: str) -> set[str]:
    return set(re.findall(r"python verify\.py ([a-z]+)", workflow)) - {"all"}


class TestTheRealTreeAgrees:
    def test_the_charter_list_and_the_workflow_name_the_SAME_gates(self) -> None:
        assert gates() == []

    def test_every_named_gate_is_a_command_verify_py_IMPLEMENTS(self) -> None:
        """Neither side is authority on its own: a name both agree on can still be a typo."""
        assert _listed(CHARTER.read_text()) <= _KNOWN_GATES

    def test_the_gate_COVERS_ITSELF(self) -> None:
        """A gate absent from its own list is the defect it exists to catch."""
        assert "gates" in _listed(CHARTER.read_text())
        assert "gates" in _invoked(WORKFLOW.read_text())

    def test_REFERENCES_is_the_gate_whose_absence_motivated_this_one(self) -> None:
        assert "references" in _invoked(WORKFLOW.read_text())


class TestTheGateREJECTSEachRealDriftShape:
    """Seeded drift, run against the real files so a fixture cannot drift from them."""

    def test_a_gate_NAMED_but_never_invoked_is_rejected(self, monkeypatch) -> None:
        """T38's actual defect: `references` in §9, absent from `ci.yml`."""
        workflow = WORKFLOW.read_text()
        seeded = re.sub(r"\n +- run: python verify\.py references[^\n]*", "", workflow)
        assert seeded != workflow, "the premise: the line must be there to remove"
        problems = self._run(monkeypatch, workflow=seeded)
        assert len(problems) == 1
        assert "names `references`" in problems[0] and "never invokes" in problems[0]

    def test_a_gate_INVOKED_but_never_named_is_rejected(self, monkeypatch) -> None:
        """The other direction: CI can reject a push the charter never warned about."""
        charter = CHARTER.read_text()
        seeded = charter.replace("`references` · `gates` ·", "`gates` ·", 1)
        assert seeded != charter, "the premise"
        problems = self._run(monkeypatch, charter=seeded)
        assert len(problems) == 1
        assert "invokes `references`" in problems[0] and "does not name it" in problems[0]

    def test_a_TYPO_the_two_sides_AGREE_on_is_rejected(self, monkeypatch) -> None:
        """The shape a set comparison is blind to: both sides consistent, both wrong.

        `python verify.py referenecs` exits non-zero on argparse, so CI would go red — but it
        would go red *always*, which is a broken build rather than a gate. The set difference is
        empty here; only checking the agreed name against what `verify.py` implements sees it."""
        problems = self._run(
            monkeypatch,
            charter=CHARTER.read_text().replace("`references` · `gates`", "`referenecs` · `gates`", 1),
            workflow=WORKFLOW.read_text().replace("verify.py references", "verify.py referenecs", 1))
        assert len(problems) == 1
        assert "`referenecs`" in problems[0] and "no such command" in problems[0]

    def test_a_MANGLED_section_9_is_rejected_rather_than_read_as_empty(self, monkeypatch) -> None:
        """An unparseable list must fail loudly: silently reading it as `set()` would make every
        gate 'invoked but not named' — or, worse, make a deleted list look like agreement."""
        problems = self._run(
            monkeypatch,
            charter=CHARTER.read_text().replace("- **`verify.py`**:", "- the gates:", 1))
        assert len(problems) == 1 and "no longer opens its gate list" in problems[0]

    @staticmethod
    def _run(monkeypatch, charter: str | None = None, workflow: str | None = None) -> list[str]:
        """Run the real gate with one or both files substituted, touching neither on disk."""
        real = Path.read_text

        def patched(self, *args, **kwargs):
            if charter is not None and self.name == "CLAUDE.md":
                return charter
            if workflow is not None and self.name == "ci.yml":
                return workflow
            return real(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", patched)
        return gates()
