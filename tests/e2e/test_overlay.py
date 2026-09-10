"""Invariants the test-time overlay has to hold.

Not marked ``e2e``: these run no simulation, so they belong in the default suite where they guard
the overlay on every pull request. What they protect is subtle and was got wrong once already --
an overlay that quietly fails to apply, or that changes the code path it was meant to leave
alone, makes the end-to-end suite report on something other than what its matrix claims.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from .matrix import E2E_MATRIX
from .overlay import (
    _ALIGNED_START,
    _FIXTURE_EVENT_GPS,
    _GLITCH_RATE_SCALES,
    _OVERLAYS,
    _TEST_SEED,
    CONTAINS_SIGNAL,
    EXAMPLES_DIR,
    NOT_HERMETIC,
    POPULATION_FIXTURE,
    apply_overlay,
)

#: Settings that select a code path or the physics. An overlay may shorten a run or repoint an
#: input; changing any of these would make the entry stop covering what it claims to.
#:
#: ``minimum-frequency``, ``earth-rotation`` and ``waveform-backend-arguments`` are here because
#: each selects behaviour rather than scale: the cutoff taper, the rotating-versus-static
#: projection, and the waveform backend's own configuration.
_PATH_DEFINING_KEYS = (
    "waveform-model",
    "waveform-backend",
    "waveform-backend-arguments",
    "detectors",
    "source-type",
    "backend",
    "minimum-frequency",
    "earth-rotation",
)

#: What an overlay *is* allowed to change, and therefore what these tests do not preserve.
#:
#: Recorded because it is a real limitation rather than an oversight. ``sampling-frequency`` is
#: reduced from 4096 Hz to 1024 Hz, which moves the Nyquist frequency and the discretisation;
#: ``duration`` and ``total-duration`` are cut, which changes how many segments a run produces.
#: So these entries verify that orchestration works and produces sane data at test scale -- they
#: do not reproduce the example's numerical behaviour, and a defect that only appears above
#: 512 Hz or across many segments would not be caught here.
_SCALE_KEYS = ("sampling-frequency", "duration", "total-duration")

#: Matrix entries that are actually run here.
#:
#: A non-hermetic entry may or may not have an overlay. The gengli one has none on purpose --
#: it cannot be exercised at all, so an overlay for it would be untested guesswork, and the
#: KeyError in ``apply_overlay`` is what tells whoever adds its fixture to write one. The
#: DeepExtractor one does have one, because it *can* be exercised: it was, against the real
#: dataset, and CI declines the download rather than the entry. Its overlay is checked by
#: ``TestGlitchRateScaling`` below, which is parametrized over the scales rather than over
#: ``_RUNNABLE`` for exactly that reason.
_RUNNABLE = tuple(entry for entry in E2E_MATRIX if entry.label not in NOT_HERMETIC)


def _example(label: str) -> dict[str, Any]:
    """Load one example configuration."""
    return yaml.safe_load((EXAMPLES_DIR / label / "config.yaml").read_text(encoding="utf-8"))


def test_every_matrix_entry_has_an_overlay():
    """An entry with no overlay would run at full example size -- a day of data for most."""
    missing = [entry.label for entry in _RUNNABLE if entry.label not in _OVERLAYS]
    assert not missing, f"these runnable matrix entries have no test overlay: {missing}"


def test_the_population_fixture_exists():
    """Every remote population is repointed here, so its absence breaks every signal entry."""
    assert POPULATION_FIXTURE.is_file(), f"the population fixture is missing: {POPULATION_FIXTURE}"


def test_the_aligned_start_brackets_the_fixture_event():
    """The signal entries' segment must actually contain the population's only event.

    A segment that misses it still runs and writes correctly-shaped files full of zeros, so this
    arithmetic is the difference between testing the pipeline and testing nothing. Checked
    against the shortest duration any entry uses, since that is the tightest case.
    """
    shortest = min(
        overlay["globals"]["simulator-arguments"]["duration"]
        for label, overlay in _OVERLAYS.items()
        if label in CONTAINS_SIGNAL
    )

    assert _ALIGNED_START <= _FIXTURE_EVENT_GPS < _ALIGNED_START + shortest, (
        f"the fixture event at GPS {_FIXTURE_EVENT_GPS} falls outside "
        f"[{_ALIGNED_START}, {_ALIGNED_START + shortest}), so signal entries would write zeros"
    )


@pytest.mark.parametrize("entry", _RUNNABLE, ids=lambda entry: entry.label)
def test_the_overlay_pins_the_seed_where_the_orchestrator_reads_it(entry, tmp_path: Path):
    """The seed must land in the noise arguments, not only in the globals.

    The orchestrator copies the global seed into the noise arguments with ``setdefault``, so an
    example that pins its own -- several pin 42 -- keeps it and a global-only override does
    nothing. That was measured, not guessed: changing the global left all nine output files
    identical, while changing the noise seed changed all nine.
    """
    merged = apply_overlay(_example(entry.label), entry.label, tmp_path)

    assert merged["globals"]["simulator-arguments"]["seed"] == _TEST_SEED
    noise = merged.get("orchestration", {}).get("noise")
    if isinstance(noise, dict):
        assert noise["arguments"]["seed"] == _TEST_SEED, (
            "the example's own noise seed survived the overlay, so runs are pinned by the example "
            "rather than by the test"
        )


@pytest.mark.parametrize("entry", _RUNNABLE, ids=lambda entry: entry.label)
def test_the_overlay_does_not_change_the_code_path(entry, tmp_path: Path):
    """Shortening a run must not alter what it exercises.

    Compares every path-defining setting before and after the overlay. Without this an overlay
    could, say, swap a waveform model for a faster one and the entry would keep claiming
    coverage it no longer has.
    """
    original = _example(entry.label)
    merged = apply_overlay(original, entry.label, tmp_path)

    for section in ("signal", "population", "noise"):
        before = original.get("orchestration", {}).get(section)
        after = merged.get("orchestration", {}).get(section)
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        for key in _PATH_DEFINING_KEYS:
            if key in before:
                assert after.get(key) == before[key], (
                    f"the overlay changed orchestration.{section}.{key} for '{entry.label}', "
                    f"which changes the code path the entry is meant to cover"
                )


def test_only_scale_settings_are_overridden_in_the_globals():
    """Every global an overlay touches must be a scale knob, not a behaviour switch.

    The complement of ``test_the_overlay_does_not_change_the_code_path``: rather than listing
    what must be preserved, this bounds what may be altered, so a new overlay cannot quietly
    introduce an override of something behavioural. ``seed`` is permitted because pinning it is
    the point.
    """
    permitted = set(_SCALE_KEYS) | {"start-time", "seed"}
    for label, overlay in _OVERLAYS.items():
        touched = set(overlay.get("globals", {}).get("simulator-arguments", {}))
        assert touched <= permitted, (
            f"the overlay for '{label}' overrides {sorted(touched - permitted)}, which is not a "
            f"scale setting. If that is deliberate, it belongs in the matrix entry's description "
            f"rather than hidden in an overlay."
        )


@pytest.mark.parametrize("label", sorted(NOT_HERMETIC), ids=lambda label: label)
def test_a_blocked_entry_is_declared_as_not_run(label: str):
    """An entry that never runs must say so where coverage is read, not only in a skip message.

    A matrix entry reads as coverage. One that is permanently skipped is the most misleading
    thing the matrix can contain -- a reader counts the entries and takes each for a path that is
    exercised. Enforcing the marker keeps that count honest as entries come and go.
    """
    entry = next((entry for entry in E2E_MATRIX if entry.label == label), None)
    assert entry is not None, f"'{label}' is listed as blocked but is not in the matrix"
    assert "not run" in entry.covers.lower(), (
        f"'{label}' cannot be executed here, so its matrix description must say so; it currently "
        f"reads: {entry.covers!r}"
    )


class TestGlitchRateScaling:
    """The one overlay mechanism that reaches inside a list, and what bounds it.

    ``_OVERLAYS`` is deep-merged, and ``noise.arguments.glitches`` is a list -- so an overlay
    that named it would replace the whole glitch model and be free to drift from the example.
    Scaling the rate instead touches one number per model. These tests are what keep it to that.
    """

    @staticmethod
    def _rates(config: dict[str, Any]) -> dict[str, float]:
        return config["orchestration"]["noise"]["arguments"]["glitches"][0]["rate"]

    def test_every_scaled_label_is_a_matrix_entry_with_an_overlay(self):
        """A stale label here scales nothing and says otherwise.

        Membership of the matrix and of ``_OVERLAYS``, not of ``_RUNNABLE``: a scale is applied
        by ``apply_overlay``, which every entry with an overlay goes through, whether or not CI
        chooses to run it.
        """
        labels = {entry.label for entry in E2E_MATRIX}
        unknown = sorted(set(_GLITCH_RATE_SCALES) - labels)
        assert not unknown, f"_GLITCH_RATE_SCALES names entries that are not in the matrix: {unknown}"

        without_overlay = sorted(set(_GLITCH_RATE_SCALES) - set(_OVERLAYS))
        assert not without_overlay, (
            f"_GLITCH_RATE_SCALES names entries with no overlay, so the scale is never applied: {without_overlay}"
        )

    @pytest.mark.parametrize("label", sorted(_GLITCH_RATE_SCALES), ids=lambda label: label)
    def test_the_scale_multiplies_the_examples_own_rates(self, label: str, tmp_path: Path):
        """Every rate must be the example's value times the declared factor -- and nothing else.

        Checked against the example rather than against restated numbers, which is the whole
        reason the overlay scales instead of listing: a rate written out here could drift from
        the configuration it claims to shorten and this test would still pass.
        """
        original = self._rates(_example(label))
        scaled = self._rates(apply_overlay(_example(label), label, tmp_path))
        scale = _GLITCH_RATE_SCALES[label]

        assert set(scaled) == set(original), (
            "the overlay changed which glitch classes are configured; the backend requires the "
            "rate mapping's keys to match `glitch_classes` exactly"
        )
        for name, value in original.items():
            assert scaled[name] == pytest.approx(value * scale, rel=1e-12), (
                f"rate for '{name}' is {scaled[name]}, not {value} * {scale}"
            )

    @pytest.mark.parametrize("label", sorted(_GLITCH_RATE_SCALES), ids=lambda label: label)
    def test_the_scale_leaves_everything_but_the_rate_alone(self, label: str, tmp_path: Path):
        """The complement: bound what the scaling may touch, not only what it must produce.

        A per-class ``rate`` mapping stays a mapping -- rewriting it as a scalar would switch the
        backend from a proportional class draw to a uniform one, retiring the coverage the matrix
        entry claims while every other assertion here still passed.
        """
        original = _example(label)["orchestration"]["noise"]["arguments"]["glitches"]
        merged = apply_overlay(_example(label), label, tmp_path)["orchestration"]["noise"]["arguments"]["glitches"]

        assert len(merged) == len(original), "the overlay added or dropped a glitch model"
        for before, after in zip(original, merged, strict=True):
            assert type(after["rate"]) is type(before["rate"]), (
                "the overlay changed the rate's form; scalar and per-class mapping select "
                "different class-draw code in the backend"
            )
            assert {key: value for key, value in after.items() if key != "rate"} == {
                key: value for key, value in before.items() if key != "rate"
            }, "the overlay changed a glitch-model setting other than the rate"

    @pytest.mark.parametrize("label", sorted(_GLITCH_RATE_SCALES), ids=lambda label: label)
    def test_the_scaled_span_expects_glitches_in_every_output_file(self, label: str, tmp_path: Path):
        """The point of scaling: at the example's own rates the entry would write zeros.

        A run that fires no glitch still completes and writes correctly-shaped files, exactly the
        trap ``test_the_output_contains_signal_where_expected`` exists for on the signal side --
        and that test cannot cover a glitch-only entry, which has no ``signal`` block to look
        under. So the arithmetic is asserted here instead: the expected count per output file,
        which is one segment of one interferometer, has to be comfortably above one.
        """
        merged = apply_overlay(_example(label), label, tmp_path)
        segment = merged["globals"]["simulator-arguments"]["duration"]
        rate = self._rates(merged)
        total = sum(rate.values()) if isinstance(rate, dict) else rate

        expected = total * segment
        assert expected > 3.0, (
            f"'{label}' expects only {expected:.2f} glitches per segment per interferometer, so "
            f"an output file of zeros is a likely outcome rather than a failure"
        )


def test_contains_signal_only_names_matrix_entries():
    """A stale label here would silently stop asserting that a signal was produced."""
    labels = {entry.label for entry in E2E_MATRIX}
    unknown = sorted(CONTAINS_SIGNAL - labels)
    assert not unknown, f"CONTAINS_SIGNAL names entries that are not in the matrix: {unknown}"


def test_the_runner_pins_the_earth_orientation_table():
    """The child process must not be free to fetch a different IERS table.

    A reference is only meaningful if its inputs are determined by things the reference records,
    and which IERS table Astropy loaded is recorded nowhere. `auto_download` defaults to ``True``
    with a 30-day `auto_max_age`, so an installation whose `astropy-iers-data` has aged out
    silently fetches the current table from the IERS server -- two runs with byte-identical
    dependency sets then produce different strain.

    Asserted by launching a child with the runner's environment rather than by reading the shim,
    because what matters is the value Astropy ends up with in the process that generates data.
    """
    pytest.importorskip("astropy")
    from .runner import _deterministic_iers_environment

    completed = subprocess.run(
        [sys.executable, "-c", "from astropy.utils import iers; print(iers.conf.auto_download)"],
        capture_output=True,
        text=True,
        check=False,
        env=_deterministic_iers_environment(),
    )

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.strip() == "False", (
        f"the runner's environment left IERS auto-download enabled: {completed.stdout.strip()!r}"
    )


def test_the_population_fixture_matches_the_example_it_copies():
    """The test-owned catalogue must differ from the example's only in ``coa_time``.

    It exists because the e2e epoch has to sit inside the IERS table while the examples stay in
    2030, so the coalescence time cannot be shared. Everything else -- masses, distance, sky
    position, inclination -- must track the example, or the suite quietly stops testing the
    configuration it claims to and the stored references stop describing the documented example.

    Nothing else enforces that: the neighbouring guard checks the event falls inside the span, not
    that the two files agree. Duplication without a guard is how the copy goes stale.
    """
    example = EXAMPLES_DIR / "signal" / "bbh_population.csv"
    fixture = POPULATION_FIXTURE

    example_rows = example.read_text(encoding="utf-8").strip().splitlines()
    fixture_rows = fixture.read_text(encoding="utf-8").strip().splitlines()

    assert fixture_rows[0] == example_rows[0], (
        f"{fixture.name} and {example.name} disagree on their columns; the fixture is a copy of "
        f"the example and must follow it"
    )
    assert len(fixture_rows) == len(example_rows), (
        f"{example.name} has {len(example_rows) - 1} event(s) and {fixture.name} has "
        f"{len(fixture_rows) - 1}; the e2e suite would test a different population than the example"
    )

    columns = example_rows[0].split(",")
    time_column = columns.index("coa_time")
    paired = zip(fixture_rows[1:], example_rows[1:], strict=True)
    for line_number, (mine, theirs) in enumerate(paired, start=2):
        mine_fields, theirs_fields = mine.split(","), theirs.split(",")
        fields = zip(mine_fields, theirs_fields, strict=True)
        differing = [columns[i] for i, (a, b) in enumerate(fields) if a != b]
        assert differing == ["coa_time"], (
            f"line {line_number} differs in {differing} as well as coa_time; only the coalescence "
            f"time may differ between the fixture and the example"
        )
    # And the coalescence time must actually differ, or the fixture has no reason to exist.
    assert fixture_rows[1].split(",")[time_column] != example_rows[1].split(",")[time_column]


def test_every_overlay_override_survives_the_merge(tmp_path: Path):
    """An override that lands nowhere is worse than no override: it reads as done and is not.

    This catches one of the two ways that happened while writing this module: a `signal` key at the
        overlay's root, which the deep merge accepted while the configuration read
        `orchestration.signal`. Silent -- the config parsed, the run succeeded, the value stayed the
        example's.

        The other way, two `orchestration` keys in one dict literal where Python keeps only the last,
        is *not* covered here and cannot be: the duplicate is discarded at parse time, so the overlay
        this test inspects never contains the lost override. Ruff's `F601` catches that one, and the
        pre-commit hook enforces it -- worth knowing so nobody adds a test here expecting to.
    """

    def leaves(mapping, prefix=()):
        for key, value in mapping.items():
            if isinstance(value, dict):
                yield from leaves(value, (*prefix, key))
            else:
                yield (*prefix, key), value

    for label, overlay in _OVERLAYS.items():
        if label in NOT_HERMETIC:
            continue
        example = _example(label)
        merged = apply_overlay(example, label, tmp_path / label.replace("/", "_"))
        for path, expected in leaves(overlay):
            # The override must attach to a branch the configuration already has. This is the check
            # that catches the real bug: a `signal` key at the overlay root produced
            # `signal.arguments.reference_time_ssb` in the merged mapping -- present, equal to what
            # the overlay asked for, and read by nobody, because the configuration reads
            # `orchestration.signal`. Comparing the overlay against the merged result cannot see
            # that; comparing against the *example* can.
            assert path[0] in example, (
                f"'{label}' overlay sets {'.'.join(path)}, but the example has no {path[0]!r} "
                f"section -- the override creates a branch nothing reads"
            )
            node = merged
            for step in path:
                assert isinstance(node, dict), f"'{label}' overlay sets {'.'.join(path)}, but {step!r} is not a section"
                assert step in node, (
                    f"'{label}' overlay sets {'.'.join(path)}, which does not exist in the merged "
                    f"configuration -- the key is at the wrong level and the override does nothing"
                )
                node = node[step]
            assert node == expected, (
                f"'{label}' overlay sets {'.'.join(path)} to {expected!r} but the merged configuration has {node!r}"
            )
