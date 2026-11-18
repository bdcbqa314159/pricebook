"""The committed mutation sweep (tranche 52) — PRE-REGISTERED, and tested against known answers.

**Why this exists.** The sweep is a standing §9 obligation — every arc close and every eleven
tranches — and until now it was an ad-hoc script rewritten from scratch each time: the one part of
the method with no provenance, and the one that re-introduced the same tooling bugs twice.

**A sweep that mis-scores is worse than none, because it launders false confidence.** So the
instrument is checked against a *known answer* rather than trusted: a tiny subject module whose
oracle is deliberately weak in one place, where it is known in advance which mutation must be killed
and which must survive.

The three instrument clauses the charter has already paid for, each a regression test here:

1. **Bytecode invalidation** (§9, T46). A mutation written within one filesystem mtime tick leaves
   the `.pyc`'s `(mtime, size)` unchanged, so CPython declines to recompile and the run tests the
   *original* source while printing a confident verdict for the mutant. That produced a false
   *"killed"* — the dangerous direction, which hides a hole exactly as a false green does — and
   flipped three verdicts at T46. The instrument now carries the proof of the bug it was built to
   prevent.
2. **applied / killed / survived** (§9, re-audit #4). A `0` survivors is meaningless without the
   applied count: a run where every mutation errored reports a clean zero. So the result cannot be
   asked whether it is clean without the applied count being complete.
3. **Positional mutation on identical lines** (re-audit #5, bitten **twice** — `yoy.py:136/:152` and
   `credit_legs.py:140/:250`). Two textually identical load-bearing lines in different functions are
   two real sites; a string-replacement sweep refuses them as non-unique and leaves them uncounted.

**What it is not.** Not a gate, and it does not choose its own mutations. Mutation *selection* is the
re-audit's judgement — where a transcription error would land, which blind spot to probe — and a tool
that chose for itself would sweep where it is easy rather than where the failure lives, which is §9's
can't-fail gate wearing an instrument's clothes.
"""

import time
from pathlib import Path

import pytest

from mutation_sweep import Mutation, _apply, sweep

TOOLING = Path(__file__).resolve().parent
FIXTURES = TOOLING / "fixtures"
SUBJECT = FIXTURES / "subject.py"
#: The subject's oracle, passed explicitly because it is named `check_` to stay out of the real suite.
CHECK = ("fixtures/check_subject.py",)


def _mutation(line: int, old: str, new: str, label: str) -> Mutation:
    return Mutation(path=SUBJECT, line=line, old=old, new=new, label=label)


def _line_of(fragment: str) -> int:
    """The 1-based line carrying `fragment`, so the tests name lines by content, not by count —
    otherwise editing the fixture silently moves every mutation off its target."""
    lines = SUBJECT.read_text(encoding="utf-8").splitlines()
    matches = [i for i, text in enumerate(lines, 1) if fragment in text]
    assert len(matches) == 1, f"{fragment!r} appears {len(matches)} times, not once"
    return matches[0]


#: **One sweep, all four verdict categories.** Each inner `pytest` costs ~0.27s of interpreter and
#: pytest startup whatever it runs, so the instrument's own tests share verdicts rather than
#: re-deriving them — a `SweepResult` is frozen, so sharing is safe. It is also the better shape: a
#: scoring instrument's job is to keep *killed*, *survived*, *not applied* and *errored* apart, and
#: one mixed sweep is where that is visible at a glance.
#:
#: The subject is rewritten immediately before the sweep, so every mutation here is also written
#: within one filesystem mtime tick of a fresh `.pyc` — the T46 defect's exact window, which makes
#: `killed == 1` below a regression test for the bytecode guard rather than a separate run.


@pytest.fixture(scope="module")
def mixed() -> object:
    SUBJECT.write_text(SUBJECT.read_text(encoding="utf-8"), encoding="utf-8")
    return sweep(
        [
            _mutation(_line_of("return 2.0 * x + 1.0"), "2.0", "2.5", "killed: guarded"),
            _mutation(_line_of("return 3.0 * x - 7.0"), "3.0", "3.5", "survived: unguarded"),
            _mutation(_line_of("return 2.0 * x + 1.0"), "ABSENT", "x", "not applied: bad fragment"),
            _mutation(_line_of("return 2.0 * x + 1.0"), "+ 1.0", "+", "errored: malformed"),
        ],
        tests=CHECK,
        root=TOOLING,
    )


class TestTheToolScoresAgainstAKnownAnswer:
    """The floor: four verdicts checked against known answers, not trusted."""

    def test_a_guarded_defect_is_KILLED(self, mixed) -> None:
        assert mixed.killed == 1

    def test_a_PLANTED_HOLE_is_reported_as_SURVIVED(self, mixed) -> None:
        """The half that matters: a tool reporting everything as killed would pass the test above
        and be worthless."""
        assert mixed.survived == 1
        assert mixed.survivors == ("survived: unguarded",)

    def test_a_mutation_that_does_not_apply_is_NOT_counted_as_killed(self, mixed) -> None:
        """§9: a `0` survivors is read only against a full applied count."""
        assert mixed.not_applied == ("not applied: bad fragment",)
        assert mixed.total == 4
        assert mixed.applied == 3, "the unapplied mutation must not inflate the applied count"

    def test_an_ERRORED_run_is_NOT_a_kill(self, mixed) -> None:
        """**The clause the instrument's own known-answer test discovered.** Re-audit #4's false
        clean sheet came from 21 runs that all errored on the command line — and the applied-count
        guard cannot catch that shape, because those mutations *did* reach the source. By return code
        alone an import error, a usage error or a collection failure is indistinguishable from a test
        failure, so every errored run reads as *killed*: the dangerous direction again.

        Here a malformed mutation (`+ 1.0` → `+`) makes the subject unparseable, so pytest exits 2
        rather than 1. The mutation applied and the run returned non-zero, and it must still not be
        scored as a kill — it is a mutation to fix, not a defect caught."""
        assert len(mixed.errored) == 1
        assert "errored: malformed" in mixed.errored[0]
        assert mixed.killed == 1, "the malformed mutant must not be counted among the kills"

    def test_a_sweep_with_a_survivor_or_an_error_is_NOT_clean(self, mixed) -> None:
        assert not mixed.is_clean

    def test_the_subject_is_RESTORED_afterwards(self, mixed) -> None:
        """A sweep that left the tree mutated would be a catastrophe of a different kind — and a
        malformed mutation among the set makes this the sharper check."""
        assert mixed.applied == 3, "the premise: sweeps actually ran"
        text = SUBJECT.read_text(encoding="utf-8")
        assert "return 2.0 * x + 1.0" in text
        assert "2.5" not in text

    def test_no_stale_cache_for_the_subject_outlives_a_sweep(self, mixed) -> None:
        """The T46 guard, stated structurally as well as behaviourally: the mutations above were
        written inside one mtime tick of a fresh `.pyc`, and `killed == 1` is the behavioural half."""
        assert mixed.killed == 1
        cache = FIXTURES / "__pycache__"
        cached = list(cache.glob("subject.*.pyc")) if cache.exists() else []
        assert cached == [], f"a stale cache for the subject survived the sweep: {cached}"

    def test_the_summary_states_every_count_it_has(self, mixed) -> None:
        """So a report cannot headline a survivor count alone — the §9 clause, in the one place a
        human reads."""
        summary = mixed.summary()
        assert "applied 3/4" in summary
        assert "killed 1" in summary
        assert "survived 1" in summary
        assert "ERRORED" in summary
        assert "NOT APPLIED" in summary


class TestIdenticalLinesAreTWOSites:
    """The re-audit #5 defect, bitten twice, now impossible.

    **Tested at the apply step rather than through a whole sweep, and that is the better-sized
    test.** Positional addressing is a property of `_apply`: given a line, does it mutate *that*
    line and only that line. Inferring it from a sweep's applied count routes two subprocess
    `pytest` runs through the verdict machinery to observe something that happens *before* any test
    runs — and on the Windows runner a spawn costs ~1s, which is how this file came to sit on the
    `slow` ceiling and then fail on variance (10.2s against 10.0s, having passed the run before).
    The end-to-end counts are already proved by the mixed sweep above; this proves the addressing.
    """

    def _twin_lines(self) -> list[int]:
        lines = SUBJECT.read_text(encoding="utf-8").splitlines()
        twins = [i for i, text in enumerate(lines, 1) if text.strip() == "scaled = x * 1.5"]
        assert len(twins) == 2, "the premise: the fixture must actually contain identical lines"
        return twins

    def test_mutating_one_twin_leaves_the_OTHER_untouched(self) -> None:
        """`twin_a` and `twin_b` share the line `scaled = x * 1.5`. A string replacement sees two
        occurrences and either refuses them as non-unique or hits the first whichever is named;
        positional addressing makes each a real, separate site."""
        for line in self._twin_lines():
            original = _apply(_mutation(line, "1.5", "1.6", f"twin at {line}"))
            try:
                assert original is not None, f"the mutation at line {line} did not apply"
                mutated = SUBJECT.read_text(encoding="utf-8").splitlines()
                changed = [i for i, text in enumerate(mutated, 1) if "1.6" in text]
                assert changed == [line], (
                    f"naming line {line} changed lines {changed} — a string replacement would have "
                    "hit the first occurrence whichever line was named"
                )
            finally:
                assert original is not None
                SUBJECT.write_text(original, encoding="utf-8")

    def test_both_twins_are_ADDRESSABLE_as_distinct_sites(self) -> None:
        """The counting half, without a verdict: two mutations carrying the *same* fragment and
        different lines are two sites, which is exactly what a replace-based sweep cannot express."""
        first_line, second_line = self._twin_lines()
        first = _mutation(first_line, "1.5", "1.6", "twin one")
        second = _mutation(second_line, "1.5", "1.6", "twin two")
        assert first.old == second.old, "the premise: the same fragment, which is what defeats replace"
        assert first.line != second.line
        assert len({(m.path, m.line) for m in (first, second)}) == 2


class TestTheAppliedCountIsLoadBearing:
    """§9: a `0` survivors is read only against a full applied count. These cases cost no inner test
    run at all — a mutation that cannot be applied is never run."""

    def test_a_mutation_naming_the_WRONG_LINE_does_not_apply(self) -> None:
        """Positional addressing has its own failure mode, and it must be loud: a fragment that
        exists in the file but not on the named line is *not applied*, never silently relocated."""
        elsewhere = _line_of("return 3.0 * x - 7.0")
        result = sweep(
            [_mutation(elsewhere, "2.0", "2.5", "right fragment, wrong line")],
            tests=CHECK,
            root=TOOLING,
        )
        assert result.applied == 0
        assert result.not_applied == ("right fragment, wrong line",)

    def test_a_line_outside_the_file_does_not_apply(self) -> None:
        result = sweep(
            [_mutation(10_000, "2.0", "2.5", "past the end")], tests=CHECK, root=TOOLING
        )
        assert result.applied == 0
        assert result.not_applied == ("past the end",)


class TestTheToolRefusesWhatItCannotScore:
    def test_an_empty_mutation_set_is_refused(self) -> None:
        """A sweep of nothing would report `applied 0/0, survived 0` and be trivially "clean" — the
        can't-fail shape, so it raises instead."""
        with pytest.raises(ValueError, match="at least one mutation"):
            sweep([], tests=CHECK, root=TOOLING)

    def test_a_mutation_that_changes_nothing_is_refused(self) -> None:
        """`old == new` would apply, run, and be scored as a mutant while testing the original — a
        guaranteed false `survived`, which is a hole-shaped lie."""
        with pytest.raises(ValueError, match="changes nothing"):
            Mutation(path=SUBJECT, line=1, old="x", new="x", label="no-op")

    def test_a_RED_baseline_is_refused_rather_than_swept(self) -> None:
        """If the chosen tests do not pass unmutated, every mutant "fails" for a reason that is not
        the mutation and the sweep is a false clean sheet. Seeded by mutating the subject *outside*
        the sweep, so the baseline the sweep checks is genuinely red."""
        original = SUBJECT.read_text(encoding="utf-8")
        try:
            SUBJECT.write_text(
                original.replace("return 2.0 * x + 1.0", "return 9.0 * x + 1.0"), encoding="utf-8"
            )
            with pytest.raises(ValueError, match="UNMUTATED tree"):
                sweep(
                    [_mutation(_line_of("return 3.0 * x - 7.0"), "3.0", "3.5", "never reached")],
                    tests=CHECK,
                    root=TOOLING,
                )
        finally:
            SUBJECT.write_text(original, encoding="utf-8")
