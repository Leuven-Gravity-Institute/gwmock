# Copyright (C) 2026 Leuven Gravity Institute
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

"""The one entry-and-platform relaxation in the reference gate, and its limits.

An entry that generates through LALSimulation does not reproduce across processor
architectures: replayed on Darwin/arm64 against these Linux/x86_64 references, the two such
entries move by 3.90e-06 and 4.17e-06 of peak, while every other entry stays at 8.0e-13 or
below and the Python version moves nothing at all. The difference is present in LAL's own
frequency-domain output before any code in this project runs, so the comparison relaxes for
those entries rather than pretending the run is wrong.

Not marked ``e2e``: none of this needs a generation run, and the relaxation is exactly the kind
of thing that should be checked on every pull request rather than in the job that runs for
minutes. What it guards is a gate getting quietly wider -- the flag spreading to entries that
never touch LAL, the loose bound leaking onto the platform that wrote the reference, or the
integer statistics being swept along with the floats.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .matrix import E2E_MATRIX, MatrixEntry
from .reference_values import (
    FOREIGN_PLATFORM_TOLERANCE,
    STATISTIC_TOLERANCE,
    WRITE_ENVIRONMENT_VARIABLE,
    comparison_bound,
    known_different_platform,
    statistic_differences,
)

pytestmark = pytest.mark.unit

LINUX = {"system": "Linux", "machine": "x86_64", "python": "3.12", "device": "cpu"}
DARWIN = {"system": "Darwin", "machine": "arm64", "python": "3.13", "device": "cpu"}

#: The entries whose samples come from LAL, which is measured to move across architectures.
#: Restated here rather than read from the matrix, so that flagging a new entry has to be a
#: deliberate edit in two places instead of silently widening the gate for it.
#:
#: ``signal/bbh/ligo_hl`` is flagged for its waveform source, not from a measurement of its own: it
#: generates with LAL as the other two do, and was added after the cross-platform measurement.
LAL_ENTRIES = {
    "noise/uncorrelated_gaussian/quick_start",
    "signal/bbh/et_triangle_sardinia",
    "signal/bbh/ligo_hl",
}

#: The largest relative move measured between Linux/x86_64 and Darwin/arm64 on a LAL entry.
MEASURED_PLATFORM_DIFFERENCE = 4.17e-6


def _outputs(peak: float, *, n: int = 16384, nonzero: int = 5734, argmax: int = 4080) -> dict:
    """Return a one-file statistics block of the shape a reference stores."""
    return {
        "signal.hdf5": {
            "n": n,
            "nonzero": nonzero,
            "argmax": argmax,
            "peak": peak,
            "rms": peak / 11.4,
            "signed_peak": peak,
            "mean": 3.5e-26,
        }
    }


class TestKnownDifferentPlatform:
    """Which fingerprint keys establish that two runs are on different machine architectures."""

    def test_a_run_does_not_differ_from_itself(self) -> None:
        assert not known_different_platform(LINUX, dict(LINUX))

    @pytest.mark.parametrize("key", ["python", "device"])
    def test_the_keys_measured_not_to_matter_do_not_establish_a_difference(self, key: str) -> None:
        """The narrowing to ``system`` and ``machine`` is the measurement, not a convenience.

        Both Python versions produce bit-identical output on either platform -- 37 of 37 files --
        and the device difference is 3.16e-13. Letting either key answer this would hand a LAL
        entry the loose tolerance for a reason known not to move it, which is the one thing a
        relaxed gate must not do.
        """
        assert not known_different_platform(LINUX, dict(LINUX, **{key: "something-else"}))

    @pytest.mark.parametrize("key", ["system", "machine"])
    def test_the_keys_measured_to_matter_do_establish_a_difference(self, key: str) -> None:
        assert known_different_platform(LINUX, dict(LINUX, **{key: "something-else"}))

    def test_linux_and_darwin_are_established_as_different(self) -> None:
        assert known_different_platform(LINUX, DARWIN)

    @pytest.mark.parametrize(
        "stored",
        [
            pytest.param(None, id="absent"),
            pytest.param({}, id="empty"),
            pytest.param({"python": "3.12", "device": "cpu"}, id="no-platform-keys"),
            pytest.param({"system": "Linux", "python": "3.12"}, id="machine-missing"),
            pytest.param({"system": "Linux", "machine": "", "python": "3.12"}, id="machine-blank"),
        ],
    )
    def test_an_incomplete_fingerprint_establishes_nothing(self, stored: dict | None) -> None:
        """The fail-closed direction, and the reason this asks the positive question.

        A reviewer found the earlier form -- ``not same_platform(...)``, which answered "same?"
        and took the loose branch on false -- handing every flagged entry the wider bound whenever
        a fingerprint was missing or partial, *including on the machine that wrote the reference*.
        Nothing in the tree was actually widened, because every reference carries a full
        fingerprint; it was a widening waiting for the first one that did not.

        Widening needs evidence. Absence of evidence keeps the tight gate, and a tight gate
        failing somewhere foreign is a visible prompt to look rather than a silent pass.
        """
        assert not known_different_platform(stored, DARWIN)
        assert not known_different_platform(DARWIN, stored)


class TestComparisonBound:
    """The conjunction: a flagged entry *and* an established platform change, both required."""

    @pytest.mark.parametrize(
        ("lal_waveform", "stored", "produced", "expected", "relaxed"),
        [
            pytest.param(True, LINUX, DARWIN, FOREIGN_PLATFORM_TOLERANCE, True, id="flagged-and-foreign"),
            pytest.param(True, LINUX, dict(LINUX), STATISTIC_TOLERANCE, False, id="flagged-but-same-platform"),
            pytest.param(False, LINUX, DARWIN, STATISTIC_TOLERANCE, False, id="foreign-but-unflagged"),
            pytest.param(False, LINUX, dict(LINUX), STATISTIC_TOLERANCE, False, id="neither"),
            pytest.param(True, None, DARWIN, STATISTIC_TOLERANCE, False, id="flagged-but-unknown-platform"),
            pytest.param(
                True,
                dict(LINUX, python="3.14", device="gpu"),
                LINUX,
                STATISTIC_TOLERANCE,
                False,
                id="flagged-but-only-python-and-device-moved",
            ),
        ],
    )
    def test_only_a_flagged_entry_on_an_established_other_platform_is_relaxed(
        self, lal_waveform: bool, stored: dict | None, produced: dict, expected: float, relaxed: bool
    ) -> None:
        """Both halves pinned, in both directions.

        The selection used to be a condition spelled out inside the reference test, where it could
        only be reached by a full generation run -- so a reviewer could see it was right but
        nothing held it there. Dropping either half of the conjunction, or letting an
        unestablished platform count as foreign, now fails here.
        """
        tolerance, because = comparison_bound(lal_waveform, stored, produced)
        assert tolerance == expected
        assert (because is not None) is relaxed

    def test_the_reason_names_both_platforms_and_the_library(self) -> None:
        """The caller prints this verbatim, so what it has to contain is pinned here.

        A note that says only "a looser bound was used" sends the reader back to the source to
        find out which bound, why, and against what. These four facts are what make it act on.
        """
        _, because = comparison_bound(True, LINUX, DARWIN)
        assert because is not None
        assert "LAL" in because
        assert "Darwin/arm64" in because
        assert "Linux/x86_64" in because

    def test_nothing_is_relaxed_without_a_reason_to_report(self) -> None:
        """The tolerance and its justification travel together, so one cannot be had without the other."""
        for lal_waveform, stored, produced in ((True, LINUX, dict(LINUX)), (False, LINUX, DARWIN)):
            tolerance, because = comparison_bound(lal_waveform, stored, produced)
            assert (tolerance == FOREIGN_PLATFORM_TOLERANCE) == (because is not None)


class TestTheToleranceIsHonoured:
    """``statistic_differences`` compares floats against the bound it is given."""

    def test_the_measured_platform_difference_fails_the_tight_gate(self) -> None:
        """The default is unchanged: this is still a failure everywhere it used to be."""
        stored = _outputs(8.942912893531663e-22)
        produced = _outputs(8.942912893531663e-22 * (1 + MEASURED_PLATFORM_DIFFERENCE))
        assert statistic_differences(stored, produced)

    def test_the_measured_platform_difference_passes_the_foreign_bound(self) -> None:
        stored = _outputs(8.942912893531663e-22)
        produced = _outputs(8.942912893531663e-22 * (1 + MEASURED_PLATFORM_DIFFERENCE))
        assert not statistic_differences(stored, produced, FOREIGN_PLATFORM_TOLERANCE)

    def test_a_percent_level_regression_still_fails_the_foreign_bound(self) -> None:
        """What the relaxation is sized to keep catching.

        A different waveform, a rescaled PSD or a lost taper moves these by percent. If the loose
        bound admitted that too it would not be a gate at all, just a comment.
        """
        stored = _outputs(8.942912893531663e-22)
        produced = _outputs(8.942912893531663e-22 * 1.01)
        assert statistic_differences(stored, produced, FOREIGN_PLATFORM_TOLERANCE)

    @pytest.mark.parametrize(("key", "moved"), [("n", 16383), ("nonzero", 5733), ("argmax", 4081)])
    def test_the_integer_statistics_stay_exact_at_the_loose_bound(self, key: str, moved: int) -> None:
        """A relaxed float tolerance must not carry the integers with it.

        No amount of last-bit drift moves a sample count or a peak's position, and the
        measurement confirms it: all 37 files kept every integer statistic across the platform
        change. So these stay exact whatever bound the floats are held to -- otherwise the
        loose comparison would stop noticing a waveform that shifted off its sample.
        """
        stored = _outputs(8.942912893531663e-22)
        produced = _outputs(8.942912893531663e-22, **{key: moved})
        assert statistic_differences(stored, produced, FOREIGN_PLATFORM_TOLERANCE)


class TestWhichEntriesAreFlagged:
    """The flag names the entries whose numbers come out of LAL, and only those."""

    def test_exactly_the_measured_entries_are_flagged(self) -> None:
        """Pinned as a set, so both directions are caught.

        Adding a flag hands an entry a bound 100x looser than the rest of the suite, and dropping
        one puts an entry back under a gate it is measured to fail. Neither should be reachable by
        editing ``matrix.py`` alone.
        """
        assert {entry.label for entry in E2E_MATRIX if entry.lal_waveform} == LAL_ENTRIES

    def test_the_entries_that_reproduce_across_platforms_keep_the_tight_gate(self) -> None:
        """Six of the eight reproduce at 8.0e-13 or better; they are not part of this at all.

        This is the reason the relaxation is per entry rather than per platform: an Apple machine
        still holds most of the matrix to 1e-06, which a suite-wide skip or a widened
        ``STATISTIC_TOLERANCE`` would both have thrown away.
        """
        unflagged = {entry.label for entry in E2E_MATRIX if not entry.lal_waveform}
        assert unflagged, "every entry is flagged, so nothing is left under the tight gate"
        assert not (unflagged & LAL_ENTRIES)


class TestTheBoundsThemselves:
    """The two numbers, and the ordering between them the whole scheme relies on."""

    def test_the_foreign_bound_is_looser_than_the_tight_one(self) -> None:
        assert FOREIGN_PLATFORM_TOLERANCE > STATISTIC_TOLERANCE

    def test_the_foreign_bound_admits_the_measured_difference_with_margin(self) -> None:
        """Margin for LAL builds nobody has measured, not for a spread that was observed.

        The 4.17e-06 comes from one Mac. Whether an x86_64 Mac or a Linux/aarch64 host lands in
        the same place is unmeasured, which is what the margin is for -- so it is worth pinning
        that the margin exists rather than leaving the bound sitting just above the one number
        that happens to have been taken.
        """
        assert FOREIGN_PLATFORM_TOLERANCE > 10 * MEASURED_PLATFORM_DIFFERENCE

    def test_the_tight_gate_still_rejects_the_measured_difference(self) -> None:
        """The relaxation is a second bound, not a replacement for the first."""
        assert MEASURED_PLATFORM_DIFFERENCE > STATISTIC_TOLERANCE


class TestTheNoteSurvivesAFailure:
    """The relaxed bound announces itself even when the comparison then fails.

    The note used to be printed after the assertion, so the one run that most needed it -- a
    flagged entry on a foreign platform whose difference cleared even the *loose* bound -- was
    the single run that printed nothing, because the assertion left the function first. A
    reviewer caught it. These drive the real reference test rather than the helper, because the
    ordering inside that function is the whole of what went wrong; testing
    :func:`comparison_bound` alone would have passed throughout.
    """

    ENTRY = MatrixEntry(label="synthetic/lal/entry", covers="a flagged entry, for this test only", lal_waveform=True)

    @staticmethod
    def _document(fingerprint: dict, peak: float) -> dict:
        return {
            "label": "synthetic/lal/entry",
            "fingerprint": fingerprint,
            "environment": {},
            "outputs": {
                "output/signal/strain.hdf5": {
                    "content_hash": f"sha256:{peak:.17e}",
                    "n": 16384,
                    "nonzero": 5734,
                    "argmax": 4080,
                    "peak": peak,
                    "rms": peak / 11.4,
                    "signed_peak": peak,
                    "mean": 3.5e-26,
                }
            },
        }

    def _run(self, monkeypatch: pytest.MonkeyPatch, stored: dict, produced: dict) -> None:
        """Drive ``test_the_output_matches_its_stored_reference`` without a generation run."""
        from . import test_reference_values as module

        monkeypatch.delenv(WRITE_ENVIRONMENT_VARIABLE, raising=False)
        monkeypatch.setattr(module, "written_files", lambda directory: [])
        monkeypatch.setattr(module, "build_reference", lambda label, directory, files: produced)
        monkeypatch.setattr(module, "load_reference", lambda label: stored)
        module.test_the_output_matches_its_stored_reference(self.ENTRY, lambda entry: Path("unused"))

    def test_the_note_is_printed_when_even_the_loose_bound_fails(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The regression itself: a difference above 1e-04, on the relaxed path.

        The comparison must still fail -- the loose bound is a bound, not an exemption -- and the
        note must still come out, because the reader's first question about that failure is which
        bound it cleared.
        """
        base = 8.942912893531663e-22
        stored = self._document(LINUX, base)
        produced = self._document(DARWIN, base * (1 + 1e-3))

        with pytest.raises(AssertionError) as failure:
            self._run(monkeypatch, stored, produced)

        printed = capsys.readouterr().out
        assert "note: synthetic/lal/entry was compared at 0.0001" in printed
        assert "LAL" in printed
        assert "Darwin/arm64" in printed
        assert "Linux/x86_64" in printed
        # And the failure itself still reports which bound was in force.
        assert "tolerance applied:     0.0001" in str(failure.value)
        assert "LAL entry, off the reference platform" in str(failure.value)

    def test_the_note_is_printed_when_the_loose_bound_passes(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The path that already worked, kept working -- moving the print must not have lost it."""
        base = 8.942912893531663e-22
        stored = self._document(LINUX, base)
        produced = self._document(DARWIN, base * (1 + 4.17e-6))

        self._run(monkeypatch, stored, produced)

        assert "note: synthetic/lal/entry was compared at 0.0001" in capsys.readouterr().out

    def test_no_note_is_printed_on_the_reference_platform(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Silence is the correct output when nothing was relaxed.

        A note on every run would be noise, and would stop the real one being noticed. Here the
        same entry, same difference, replayed where its reference was written: the tight gate
        applies and the run fails without claiming any relaxation.
        """
        base = 8.942912893531663e-22
        stored = self._document(LINUX, base)
        produced = self._document(dict(LINUX), base * (1 + 4.17e-6))

        with pytest.raises(AssertionError) as failure:
            self._run(monkeypatch, stored, produced)

        assert "note:" not in capsys.readouterr().out
        assert "tolerance applied:     1e-06" in str(failure.value)
        assert "off the reference platform" not in str(failure.value)
