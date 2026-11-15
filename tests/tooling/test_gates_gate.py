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

import pytest
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

    def test_every_named_SUBCOMMAND_is_a_command_verify_py_IMPLEMENTS(self) -> None:
        """Neither side is authority on its own: a name both agree on can still be a typo.

        Scoped to the `verify.py` subcommands since tranche 42b, because the list now also names
        the **standalone** gates (`ruff`, `pyright`), which are not `verify.py` commands and must
        not be required to be."""
        subcommands = _invoked(WORKFLOW.read_text())
        assert _listed(CHARTER.read_text()) & subcommands <= _KNOWN_GATES

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


class TestTheNonVerifyCIGatesAreCoveredToo:
    """Tranche 42b — `pyright` and `ruff` are gates the charter names and `ci.yml` must invoke.

    The tranche-38 gate closed "a `verify.py` gate stops running and nothing notices". It covered
    **`verify.py` subcommands only**, so `ruff` and `pyright` — the two standalone CI gates, one of
    which caught a tranche-42 type defect a local run missed — could be deleted from `ci.yml` with
    nothing noticing. Same defect, one category wider, and statically decidable.

    **An invocation is not an installation, and that distinction is the whole design.** Both names
    also appear in `ci.yml`'s `pip install` line, so a naive "is the name in the file" check would
    pass with the `ruff check` step deleted — a gate satisfied by its own dependency list. The
    check therefore ignores `pip install` steps, and the seeded-drift test below deletes the
    *invocation* while leaving the *install* in place, which is the case that separates a real
    check from a self-satisfying one.

    **`pytest` is deliberately NOT named**, and that is a finding rather than an omission: it has
    no direct `ci.yml` step at all — it runs *through* `verify.py tests` and `verify.py slow`, both
    already named and invoked. Naming it would either fail (no invocation) or, under a looser
    check, pass on the install line. Raised for Cowork rather than guessed.
    """

    def test_PYRIGHT_and_RUFF_are_named_in_section_9(self) -> None:
        listed = _listed(CHARTER.read_text())
        assert {"pyright", "ruff"} <= listed, sorted(listed)

    def test_PYTEST_is_NOT_named_because_it_has_no_direct_invocation(self) -> None:
        """The premise behind excluding it, asserted so the exclusion is evidenced."""
        workflow = WORKFLOW.read_text()
        invocations = [
            line for line in workflow.splitlines()
            if line.strip().startswith("- run:") and "pytest" in line
            and "pip install" not in line]
        assert invocations == [], invocations
        assert "pytest" not in _listed(CHARTER.read_text())
        assert "tests" in _listed(CHARTER.read_text()), "it is covered as `tests`"

    def test_the_real_tree_AGREES_on_the_standalone_gates(self) -> None:
        assert gates() == []

    @pytest.mark.parametrize("gate", ["pyright", "ruff"])
    def test_a_named_standalone_gate_whose_INVOCATION_is_deleted_is_rejected(
        self, monkeypatch, gate: str) -> None:
        """The install line stays, which is what makes this test discriminating."""
        workflow = WORKFLOW.read_text()
        seeded = "\n".join(
            line for line in workflow.splitlines()
            if not (line.strip().startswith("- run:") and gate in line
                    and "pip install" not in line))
        assert seeded != workflow, "the premise: there was an invocation to remove"
        problems = TestTheGateREJECTSEachRealDriftShape._run(
            monkeypatch, workflow=seeded)
        assert any(gate in p and "never invokes" in p for p in problems), problems

    def test_RUFF_is_rejected_even_though_its_INSTALL_line_survives(self, monkeypatch) -> None:
        """The case that separates a real check from a self-satisfying one.

        `ruff` appears twice in `ci.yml` — once as `pip install ruff==…` and once as the
        `ruff check` step — so a check asking only "is the name in the file?" would pass with the
        gate deleted, satisfied by the dependency list. (`pyright` cannot show this: it arrives via
        `npx`, so it has no install line. An earlier draft of the test above assumed both names
        were on it and failed on `pyright` for that reason — the premise was mine, not the gate's.)
        """
        workflow = WORKFLOW.read_text()
        seeded = "\n".join(
            line for line in workflow.splitlines()
            if not (line.strip().startswith("- run:") and "ruff" in line
                    and "pip install" not in line))
        assert "ruff" in seeded, "the install line must survive, or this proves nothing"
        assert "ruff check" not in seeded, "and the invocation must be gone"
        problems = TestTheGateREJECTSEachRealDriftShape._run(monkeypatch, workflow=seeded)
        assert any("ruff" in p and "never invokes" in p for p in problems), problems

    @pytest.mark.parametrize("gate", ["pyright", "ruff"])
    def test_removing_a_gate_CONSISTENTLY_from_both_sides_is_allowed(
        self, monkeypatch, gate: str) -> None:
        """The gate checks *agreement*, not the presence of a fixed set — dropping a gate
        deliberately is a decision the charter records, not an error to block."""
        workflow = "\n".join(
            line for line in WORKFLOW.read_text().splitlines()
            if not (line.strip().startswith("- run:") and gate in line
                    and "pip install" not in line))
        charter = CHARTER.read_text()
        for pattern in (f"`{gate}` · ", f" · `{gate}`"):
            if pattern in charter:
                charter = charter.replace(pattern, "", 1)
                break
        assert gate not in _listed(charter), "the premise: the name is gone from §9"
        problems = TestTheGateREJECTSEachRealDriftShape._run(
            monkeypatch, charter=charter, workflow=workflow)
        assert problems == [], problems
