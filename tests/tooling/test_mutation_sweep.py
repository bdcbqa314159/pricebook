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


def _planted_tree(tmp_path: Path) -> tuple[Path, Path]:
    """A two-file tree with a known hole: one module, one test, and a sum the test pins once.

    Deliberately minimal and deliberately *outside* this repo's own tree, so a sweep over it cannot
    be confused by the real suite — the engine is verified against a planted answer, never against
    its own report (this file's governing pattern, T52).
    """
    module = tmp_path / "planted.py"
    module.write_text("def total(a, b):\n    return a + b\n", encoding="utf-8")
    tests = tmp_path / "test_planted.py"
    tests.write_text(
        "from planted import total\n\n\ndef test_the_sum():\n    assert total(2, 3) == 5\n",
        encoding="utf-8",
    )
    return module, tests

# ============================================================================================
# F-7.3 — the single-killer COUNT belongs to the instrument.
#
# §9 makes the single-killer re-run standing, but until tranche 65 `SweepResult` returned only
# applied/killed/survived — so every re-audit since #2 hand-rolled a counting loop *beside* this
# file, inheriting none of its verified baseline or `finally` restore. Re-audit #7 paid for it: a
# command timeout killed the counter between mutating a source and restoring it, and the next run
# measured against the corrupted baseline. Nothing shipped, but it is exactly the failure this file
# exists to prevent, arriving through the one obligation it did not cover.
#
# These are known-answer tests on the new field, in this file's own pattern: the engine is verified
# against a planted hole whose answer is known in advance, never against its own confident report.
# ============================================================================================


def test_the_PARSER_reads_pytests_FAILED_summary_lines() -> None:
    """A known answer: three failures, one of them parametrised twice, plus a line that merely
    *contains* the word. The parametrised pair must count **once** — deleting that one test function
    removes both, which is the honest reading for §9's purpose."""
    from mutation_sweep import _failing_tests

    stdout = "\n".join(
        [
            "tests/L0/test_a.py .F..                                        [ 50%]",
            "FAILED tests/L0/test_a.py::test_one",
            "FAILED tests/L0/test_b.py::test_two[0.5]",
            "FAILED tests/L0/test_b.py::test_two[0.5]",
            "  the message mentions FAILED but is not at the line start",
            "FAILED tests/L0/test_c.py::TestGroup::test_three",
            "3 failed, 1 passed in 0.10s",
        ]
    )
    assert _failing_tests(stdout) == (
        "tests/L0/test_a.py::test_one",
        "tests/L0/test_b.py::test_two[0.5]",
        "tests/L0/test_c.py::TestGroup::test_three",
    )


def test_the_parser_returns_NOTHING_when_no_test_failed() -> None:
    from mutation_sweep import _failing_tests

    assert _failing_tests("12 passed in 0.30s") == ()


def test_a_SURVIVOR_records_no_killers_and_a_KILL_records_who(tmp_path: Path) -> None:
    """The planted-hole pattern: two mutations on one module, one caught by the test file and one
    not, so the recorded map must distinguish them. The known answer is which."""
    module, tests = _planted_tree(tmp_path)
    result = sweep(
        [
            Mutation(module, 2, "a + b", "a - b", "caught: the sum became a difference"),
            Mutation(module, 2, "a + b", "b + a", "not caught: addition commutes"),
        ],
        tests=(str(tests),),
        root=tmp_path,
        count_killers=True,
    )
    assert result.killed == 1 and result.survived == 1
    assert "caught: the sum became a difference" in result.killers
    assert result.killers["caught: the sum became a difference"]
    # A survivor has no killers at all — recording an empty tuple for it would make it look like a
    # mutation that died to zero tests, which is a different (and impossible) thing.
    assert "not caught: addition commutes" not in result.killers


def test_SINGLE_KILLERS_names_the_mutations_that_die_to_exactly_one_test(tmp_path: Path) -> None:
    """The standing §9 question, answered by the instrument. The planted tree has one test for the
    sum, so the caught mutation is a single-point-of-failure oracle by construction — delete that
    one test and the hole reopens with the suite green."""
    module, tests = _planted_tree(tmp_path)
    result = sweep(
        [Mutation(module, 2, "a + b", "a - b", "the sum")],
        tests=(str(tests),),
        root=tmp_path,
        count_killers=True,
    )
    assert result.single_killers() == ("the sum",)
    assert result.killer_counts() == (("the sum", 1),)
    assert "SINGLE-KILLER (1)" in result.summary()


def test_the_SUMMARY_still_leads_with_the_three_COUNTS(tmp_path: Path) -> None:
    """The §9 invariant the new line must not displace: applied/killed/survived come first and
    always, because each alone lies. The single-killer line is an addition, not a replacement."""
    module, tests = _planted_tree(tmp_path)
    result = sweep(
        [Mutation(module, 2, "a + b", "a - b", "the sum")],
        tests=(str(tests),),
        root=tmp_path,
        count_killers=True,
    )
    assert result.summary().startswith("applied 1/1  killed 1  survived 0")


def test_a_mutation_dying_to_SEVERAL_tests_is_NOT_a_single_killer(tmp_path: Path) -> None:
    """The other side, or `single_killers` would flag everything. A second test on the same value
    takes the count to two, and the hole then survives either test's deletion."""
    module = tmp_path / "planted.py"
    module.write_text("def total(a, b):\n    return a + b\n", encoding="utf-8")
    tests = tmp_path / "test_planted.py"
    tests.write_text(
        "from planted import total\n"
        "def test_one():\n    assert total(2, 3) == 5\n"
        "def test_two():\n    assert total(10, 1) == 11\n",
        encoding="utf-8",
    )
    result = sweep(
        [Mutation(module, 2, "a + b", "a - b", "the sum")],
        tests=(str(tests),),
        root=tmp_path,
        count_killers=True,
    )
    assert result.killer_counts() == (("the sum", 2),)
    assert result.single_killers() == ()
    assert "SINGLE-KILLER" not in result.summary()


def test_a_VERDICT_ONLY_sweep_REFUSES_to_answer_the_single_killer_question(tmp_path: Path) -> None:
    """**The design error this file's own known-answer tests caught, pinned so it cannot return.**

    The verdict run carries `-x`, so pytest stops at the first failure and prints exactly one
    `FAILED` line however many tests would have failed. Counting killers from a verdict run would
    therefore report **every** kill as a single-killer — a false-positive machine, and the mirror of
    the false clean sheet this instrument exists to prevent.

    So counting is a separate mode, and a verdict-only result **refuses** the question instead of
    answering `()`. An empty tuple would read as "no single-killer oracles found" when the truth is
    "nobody looked", which is the can't-fail shape §9 forbids.
    """
    module, tests = _planted_tree(tmp_path)
    result = sweep(
        [Mutation(module, 2, "a + b", "a - b", "the sum")],
        tests=(str(tests),),
        root=tmp_path,
    )
    assert result.killed == 1
    assert result.killers is None
    with pytest.raises(ValueError, match="did not count killers"):
        result.single_killers()
    with pytest.raises(ValueError, match="did not count killers"):
        result.killer_counts()
    # And the summary must not quietly imply an answer either.
    assert "SINGLE-KILLER" not in result.summary()


def test_the_COUNTING_mode_drops_the_early_exit_flag() -> None:
    """The mechanism behind the test above, asserted directly rather than inferred from behaviour:
    `-x` and a killer count are incompatible by construction, so the counting flags must not carry
    it while the verdict flags must."""
    from mutation_sweep import _PYTEST, _PYTEST_COUNTING

    assert "-x" in _PYTEST
    assert "-x" not in _PYTEST_COUNTING
