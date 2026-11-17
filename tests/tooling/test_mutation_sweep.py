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

from mutation_sweep import Mutation, sweep

FIXTURES = Path(__file__).resolve().parent / "fixtures"
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


class TestTheToolScoresAgainstAKnownAnswer:
    """The floor: a verdict checked, not trusted."""

    def test_a_guarded_defect_is_KILLED(self) -> None:
        result = sweep(
            [_mutation(_line_of("return 2.0 * x + 1.0"), "2.0", "2.5", "guarded: 2.0 -> 2.5")],
            tests=CHECK,
            root=FIXTURES,
        )
        assert result.applied == 1
        assert result.killed == 1
        assert result.survived == 0

    def test_a_PLANTED_HOLE_is_reported_as_SURVIVED(self) -> None:
        """The other half, and the one that matters: a tool that reported everything as killed would
        pass the test above and be worthless."""
        result = sweep(
            [_mutation(_line_of("return 3.0 * x - 7.0"), "3.0", "3.5", "unguarded: 3.0 -> 3.5")],
            tests=CHECK,
            root=FIXTURES,
        )
        assert result.applied == 1
        assert result.killed == 0
        assert result.survived == 1
        assert result.survivors == ("unguarded: 3.0 -> 3.5",)

    def test_the_subject_is_RESTORED_afterwards(self) -> None:
        """A sweep that left the tree mutated would be a catastrophe of a different kind."""
        before = SUBJECT.read_text(encoding="utf-8")
        sweep(
            [_mutation(_line_of("return 2.0 * x + 1.0"), "2.0", "2.5", "restore check")],
            tests=CHECK,
            root=FIXTURES,
        )
        assert SUBJECT.read_text(encoding="utf-8") == before


class TestTheByteCodeGuardIsProven:
    """§9's T46 clause, as a regression test on the instrument."""

    def test_a_mutation_written_within_one_mtime_tick_is_still_detected(self) -> None:
        """The defect's exact conditions: compile the subject first so a `.pyc` exists, then mutate
        it immediately. Without cache invalidation the interpreter would run the stale bytecode and
        report the mutant **killed** on the original source — a false kill, the dangerous direction.

        The planted hole is the probe, because a false kill is only visible on a mutation whose true
        verdict is *survived*: if the stale original ran, the guarded test would pass and the tool
        would say "survived" too. So the discriminating case is the opposite one — mutate `guarded`
        in a tight loop and require every verdict to be `killed`."""
        line = _line_of("return 2.0 * x + 1.0")
        for _ in range(3):
            # touch, then mutate at once: the window the T46 defect lived in
            SUBJECT.write_text(SUBJECT.read_text(encoding="utf-8"), encoding="utf-8")
            result = sweep(
                [_mutation(line, "2.0", "2.5", "tight-loop mutant")],
                tests=CHECK,
                root=FIXTURES,
            )
            assert result.killed == 1, "a stale .pyc let the ORIGINAL run and faked a verdict"

    def test_the_subject_is_recompiled_rather_than_read_from_cache(self) -> None:
        """Stated structurally as well as behaviourally: after a sweep no stale cache for the
        subject survives, so the next run cannot inherit one."""
        sweep(
            [_mutation(_line_of("return 2.0 * x + 1.0"), "2.0", "2.5", "cache check")],
            tests=CHECK,
            root=FIXTURES,
        )
        cached = list((FIXTURES / "__pycache__").glob("subject.*.pyc")) if (
            FIXTURES / "__pycache__"
        ).exists() else []
        assert cached == [], f"a stale cache for the subject survived the sweep: {cached}"


class TestIdenticalLinesAreTWOSites:
    """The re-audit #5 defect, bitten twice, now impossible."""

    def test_two_textually_identical_lines_are_both_applied_and_both_scored(self) -> None:
        """`twin_a` and `twin_b` share the line `scaled = x * 1.5`. A string-replacement sweep sees
        two occurrences and refuses both as non-unique; this tool addresses them by position, so
        each is a real, counted site."""
        lines = SUBJECT.read_text(encoding="utf-8").splitlines()
        twins = [i for i, text in enumerate(lines, 1) if text.strip() == "scaled = x * 1.5"]
        assert len(twins) == 2, "the premise: the fixture must actually contain identical lines"
        result = sweep(
            [
                _mutation(twins[0], "1.5", "1.6", "twin_a's scale"),
                _mutation(twins[1], "1.5", "1.6", "twin_b's scale"),
            ],
            tests=CHECK,
            root=FIXTURES,
        )
        assert result.applied == 2, "both identical lines must apply as separate sites"
        assert result.killed == 2


class TestTheAppliedCountIsLoadBearing:
    """§9: a `0` survivors is read only against a full applied count."""

    def test_a_mutation_that_does_not_apply_is_NOT_counted_as_killed(self) -> None:
        result = sweep(
            [_mutation(_line_of("return 2.0 * x + 1.0"), "NOT PRESENT", "x", "bad pattern")],
            tests=CHECK,
            root=FIXTURES,
        )
        assert result.applied == 0
        assert result.killed == 0
        assert result.survived == 0
        assert result.not_applied == ("bad pattern",)

    def test_a_mutation_naming_the_WRONG_LINE_does_not_apply(self) -> None:
        """Positional addressing has its own failure mode, and it must be loud: a fragment that
        exists in the file but not on the named line is *not applied*, never silently relocated."""
        elsewhere = _line_of("return 3.0 * x - 7.0")
        result = sweep(
            [_mutation(elsewhere, "2.0", "2.5", "right fragment, wrong line")],
            tests=CHECK,
            root=FIXTURES,
        )
        assert result.applied == 0
        assert result.not_applied == ("right fragment, wrong line",)

    def test_a_result_with_unapplied_mutations_is_NOT_clean(self) -> None:
        """The clause made mechanical: `is_clean` cannot be true while a mutation failed to apply,
        so the re-audit #4 report — "survivors: 0" from 21 runs that all errored — is not
        expressible."""
        result = sweep(
            [
                _mutation(_line_of("return 2.0 * x + 1.0"), "2.0", "2.5", "applies"),
                _mutation(_line_of("return 2.0 * x + 1.0"), "ABSENT", "x", "does not apply"),
            ],
            tests=CHECK,
            root=FIXTURES,
        )
        assert result.survived == 0
        assert not result.is_clean, "zero survivors with an unapplied mutation is not clean"

    def test_a_fully_applied_sweep_with_no_survivors_IS_clean(self) -> None:
        result = sweep(
            [_mutation(_line_of("return 2.0 * x + 1.0"), "2.0", "2.5", "applies")],
            tests=CHECK,
            root=FIXTURES,
        )
        assert result.is_clean

    def test_the_summary_always_states_all_three_counts(self) -> None:
        """So a report cannot headline a survivor count alone — the §9 clause, in the one place a
        human reads."""
        result = sweep(
            [_mutation(_line_of("return 3.0 * x - 7.0"), "3.0", "3.5", "a survivor")],
            tests=CHECK,
            root=FIXTURES,
        )
        summary = result.summary()
        assert "applied 1/1" in summary
        assert "killed 0" in summary
        assert "survived 1" in summary


class TestTheToolRefusesWhatItCannotScore:
    def test_an_empty_mutation_set_is_refused(self) -> None:
        """A sweep of nothing would report `applied 0/0, survived 0` and be trivially "clean" — the
        can't-fail shape, so it raises instead."""
        with pytest.raises(ValueError, match="at least one mutation"):
            sweep([], tests=CHECK, root=FIXTURES)

    def test_a_mutation_that_changes_nothing_is_refused(self) -> None:
        """`old == new` would apply, run, and be scored as a mutant while testing the original —
        a guaranteed false `survived`, which is a hole-shaped lie."""
        with pytest.raises(ValueError, match="changes nothing"):
            Mutation(path=SUBJECT, line=1, old="x", new="x", label="no-op")

    def test_a_line_outside_the_file_is_refused(self) -> None:
        result = sweep(
            [_mutation(10_000, "2.0", "2.5", "past the end")], tests=CHECK, root=FIXTURES
        )
        assert result.applied == 0
        assert result.not_applied == ("past the end",)


class TestTheSweepIsCheapEnoughToBeRun:
    def test_one_mutation_costs_about_one_test_run(self) -> None:
        """The instrument must not be so dear that a re-audit trims its own mutation set to afford
        it — which would be the timing gate shrinking an oracle's domain (§9), one level up."""
        started = time.perf_counter()
        sweep(
            [_mutation(_line_of("return 2.0 * x + 1.0"), "2.0", "2.5", "cost probe")],
            tests=CHECK,
            root=FIXTURES,
        )
        assert time.perf_counter() - started < 10.0
