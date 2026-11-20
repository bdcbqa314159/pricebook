"""The sweep instrument's KILLER-COUNTING mode — tranche 65, re-audit #7's F-7.3.

Its own file because it is its own capability, not because of a clock. `sweep()`'s default answers
*did this mutation die?*; this mode answers *to how many tests?* — a separate opt-in flag, a separate
set of pytest arguments, and its own refusal semantics when the question was not asked. The
mechanics of addressing, bytecode invalidation, the verified baseline and the ERRORED verdict live
in `test_mutation_sweep.py`; they are a different subject and share nothing with these fixtures.

**Why the mode exists.** §9 makes the single-killer re-run standing, but until tranche 65
`SweepResult` returned only applied/killed/survived — so every re-audit since #2 hand-rolled a
counting loop *beside* the instrument, inheriting none of its verified baseline or `finally` restore.
Re-audit #7 paid for it: a command timeout killed the counter between mutating a source and
restoring it, and the next run measured against the corrupted baseline. Nothing shipped, but it is
exactly the failure the instrument exists to prevent, arriving through the one obligation it did not
cover. **An obligation that is standing but uninstrumented will be re-implemented until it fails.**

**Why counting is a separate mode, which these tests are the reason for.** The verdict run carries
`-x`, so pytest stops at the first failure and prints one `FAILED` line however many tests would
have failed. Counting from a verdict run would therefore report **every** kill as a single-killer — a
false-positive machine, and the exact mirror of the false clean sheet the instrument was committed to
prevent. The first draft of F-7.3's fix did precisely that, and these known-answer tests caught it
before it merged: the second time they have found a defect in the thing they test.

Known-answer throughout, in the instrument's governing pattern: the engine is verified against a
planted hole whose answer is known in advance, never against its own confident report.

Provenance: tests only — see `tests/tooling/mutation_sweep.py`.
"""

from pathlib import Path

import pytest

from mutation_sweep import Mutation, SweepResult, sweep

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


@pytest.fixture(scope="module")
def counted_sweep(tmp_path_factory: pytest.TempPathFactory) -> SweepResult:
    """**One** counting sweep over a planted module with two mutations — one caught by the planted
    test file and one not — shared by every assertion below that reads these inputs.

    Module-scoped on purpose, and it is a **lossless** reduction (the T50/T53 pattern). Every
    `sweep()` call verifies the baseline by running the suite *before* mutating anything, so a test
    that sweeps costs **two** subprocess runs rather than one — and on Windows a process spawn is
    expensive enough that one sweep per assertion put this file over the §9 ceiling (13.2s against
    10.0s, measured on CI; the local 3.45s was not the measurement, §9). Nothing is given up: the
    assertions below read the same two mutations they would each have swept for themselves. The
    sweeps whose inputs genuinely differ — a two-test tree, and a verdict-only run — keep their own.
    """
    tmp_path = tmp_path_factory.mktemp("counted")
    module, tests = _planted_tree(tmp_path)
    return sweep(
        [
            Mutation(module, 2, "a + b", "a - b", "caught: the sum became a difference"),
            Mutation(module, 2, "a + b", "b + a", "not caught: addition commutes"),
        ],
        tests=(str(tests),),
        root=tmp_path,
        count_killers=True,
    )


def test_a_SURVIVOR_records_no_killers_and_a_KILL_records_who(
    counted_sweep: SweepResult,
) -> None:
    """The planted-hole pattern: one mutation the test file catches and one it does not, so the
    recorded map must distinguish them. The known answer is which."""
    assert counted_sweep.killed == 1 and counted_sweep.survived == 1
    assert counted_sweep.killers is not None
    assert counted_sweep.killers["caught: the sum became a difference"]
    # A survivor has no killers at all — recording an empty tuple for it would make it look like a
    # mutation that died to zero tests, which is a different (and impossible) thing.
    assert "not caught: addition commutes" not in counted_sweep.killers


def test_SINGLE_KILLERS_names_the_mutations_that_die_to_exactly_one_test(
    counted_sweep: SweepResult,
) -> None:
    """The standing §9 question, answered by the instrument. The planted tree has one test for the
    sum, so the caught mutation is a single-point-of-failure oracle by construction — delete that
    one test and the hole reopens with the suite green. The survivor must **not** appear: it died to
    no tests at all, which is a different thing from dying to one."""
    assert counted_sweep.single_killers() == ("caught: the sum became a difference",)
    assert counted_sweep.killer_counts() == (
        ("caught: the sum became a difference", 1),
    )
    assert "SINGLE-KILLER (1)" in counted_sweep.summary()


def test_the_SUMMARY_still_leads_with_the_three_COUNTS(counted_sweep: SweepResult) -> None:
    """The §9 invariant the new line must not displace: applied/killed/survived come first and
    always, because each alone lies. The single-killer line is an addition, not a replacement."""
    assert counted_sweep.summary().startswith("applied 2/2  killed 1  survived 1")


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
