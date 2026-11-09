"""The `verify.py references` gate (tranche 26b) — the register cannot silently diverge from `src/`.

**Why a gate rather than care.** Committing T26, one word of a Cowork edit (`CREPEY-2026
(worked-from)`) was cross-checked against `references.md` and did not match. Pulling that thread
found **five** stale statuses, not one: `CREPEY-2026` and `CAPINSKI-ZASTAWNIAK` still `in-hand`
though T25 had landed; `JAECKEL-2002` self-contradictory (header `in-hand`, body `worked-from`, 18
citers); `MCINERNEY-ZASTAWNIAK` `in-hand` with **10** citers and a body promising `worked-from`
*"until T10 lands"* — T10 landed sixteen tranches earlier; and `ACKLAM-2000` `in-hand` while
`distributions.py` transcribes its coefficients. None was caught by a person reading carefully. That
is the §9 re-audit's *register accuracy* target failing between re-audits eleven tranches apart.

**The oracle is seeded drift, which is mutation applied to a gate.** A gate is only worth its line
count if it demonstrably **fails** the states it exists to forbid, so each of the real drift shapes
is reconstructed here and the gate must reject it. The three real shapes are the natural fixtures:
the gate retroactively catches exactly what was found by hand.

The checks key on the register's **header status field** and on `src/` citations — never on the prose
body (§4's domain fence: a gate that parsed narrative would be unfalsifiable). `JAECKEL`'s
header/body contradiction is caught because 18 citers force the header to `worked-from`, which its
body already claimed.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verify import (  # noqa: E402
    check_reference_statuses,
    cited_reference_keys,
    reference_statuses,
    references,
)

REGISTER = ROOT / "docs" / "references.md"


class TestTheRegisterParsesMechanically:
    def test_every_header_yields_a_key_and_a_status(self) -> None:
        statuses = reference_statuses(REGISTER.read_text())
        assert statuses, "no reference headers parsed — the format changed"
        assert "JAECKEL-2002" in statuses and "BERHAN-2012" in statuses
        assert all(s in {"in-hand", "worked-from", "sought"} for s in statuses.values()), statuses

    def test_the_citations_resolve_to_real_module_paths(self) -> None:
        citations = cited_reference_keys()
        assert citations, "no source: citations extracted"
        # The strict key pattern must not invent keys out of prose like `OTM-`, `FX-` or `OI-3`.
        assert all(k.replace("-", "").isalnum() and k.isupper() for k in citations), citations
        assert len(citations["BERHAN-2012"]) > 10


class TestTheGateRejectsEachRealDriftShape:
    """Seeded drift. Each case is a state the gate exists to forbid; it must FAIL every one."""

    def test_shape_1_a_CITED_key_left_at_in_hand_is_rejected(self) -> None:
        """CREPEY / C-Z / MCINERNEY / ACKLAM — the commonest and stalest shape."""
        errs = check_reference_statuses({"JAECKEL-2002": "in-hand"}, {"JAECKEL-2002": ["a.py", "b.py"]})
        assert errs and any("JAECKEL-2002" in e and "worked-from" in e for e in errs), errs

    def test_shape_2_a_cited_key_ABSENT_from_the_register_is_rejected(self) -> None:
        errs = check_reference_statuses({}, {"GHOST-1999": ["a.py"]})
        assert errs and any("GHOST-1999" in e for e in errs), errs

    def test_shape_3_a_cited_key_only_REFERENCED_BY_another_source_is_rejected(self) -> None:
        """§7: no citation by inference — a work known only through another is not citable."""
        errs = check_reference_statuses({"JY-2003": "referenced-by"}, {"JY-2003": ["a.py"]})
        assert errs and any("JY-2003" in e for e in errs), errs

    def test_shape_4_an_ORPHAN_worked_from_key_is_rejected(self) -> None:
        """`worked-from` claims something was built against it; zero citers contradicts the claim."""
        errs = check_reference_statuses({"LONELY-2001": "worked-from"}, {})
        assert errs and any("LONELY-2001" in e for e in errs), errs


class TestTheGateAcceptsTheLEGITIMATE_states:
    """A gate that rejected these would block honest work, which is the other way to be useless."""

    def test_a_HELD_BUT_UNBUILT_source_at_in_hand_with_no_citer_is_fine(self) -> None:
        """Exactly CREPEY's state between its registration and T25 — the legitimate `in-hand`."""
        assert check_reference_statuses({"NEW-2026": "in-hand"}, {}) == []

    def test_a_cited_key_at_worked_from_is_fine(self) -> None:
        assert check_reference_statuses({"OK-2000": "worked-from"}, {"OK-2000": ["a.py"]}) == []


class TestTheGatePassesOnTheCorrectedTree:
    def test_the_real_register_and_the_real_tree_AGREE(self) -> None:
        """The end state the five corrections produce. Before them this failed — which is the
        gate's own first finding, and the reason it is not merely restating what was already true."""
        assert references() == []
