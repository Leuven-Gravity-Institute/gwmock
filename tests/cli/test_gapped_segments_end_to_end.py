"""The released frames themselves carry the gaps.

The assertion that matters for gapped segments is not what the simulator reports about its epochs:
it is what a consumer reads off the files. So this drives the real CLI and takes every epoch out of
the **written frame's own ``t0``**, not out of its file name and not out of the metadata -- a run
whose names and records agreed with each other while the frames were contiguous would pass every
other test in the suite.

The second half is the resume: a run interrupted mid-plan and restarted has to cross a gap boundary
and come back bit-identical, which is the one place the extra draws the gapped noise stream makes
could silently fall out of step.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import typer
import yaml
from gwpy.timeseries import TimeSeries as GWpyTimeSeries

from gwmock.cli.simulate import _simulate_impl

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_START = 1000000000.0
_DURATION = 8.0
_GAP = 4.0
_N_SEGMENTS = 4
_SAMPLING_FREQUENCY = 128.0
_SPAN = _N_SEGMENTS * _DURATION + (_N_SEGMENTS - 1) * _GAP

#: The epochs the run must write, stated rather than derived from the stride under test. On the
#: unfixed code these come out as 1000000000, ...008, ...016, ...024 -- contiguous.
EXPECTED_EPOCHS = [_START + index * (_DURATION + _GAP) for index in range(_N_SEGMENTS)]


def _write_config(path: Path, working_directory: Path) -> None:
    """Write a compact gapped noise-and-signal orchestration config."""
    config = {
        "globals": {
            "working-directory": str(working_directory),
            "output-directory": "output",
            "metadata-directory": "metadata",
            "simulator-arguments": {
                "sampling-frequency": _SAMPLING_FREQUENCY,
                "duration": _DURATION,
                "segment-gap": _GAP,
                "total-duration": _SPAN,
                "start-time": _START,
                "seed": 20260914,
            },
        },
        "orchestration": {
            "signal": {
                "source-type": "sgwb",
                "detectors": ["H1", "L1"],
                "minimum-frequency": 8,
                "parameters": {"omega_ref": 1.0e30, "spectral_index": 0.0, "reference_frequency": 25.0},
                "output": {
                    "output_directory": "signal",
                    "file_name": "sgwb-{{ start_time }}.hdf5",
                },
            },
            "noise": {
                # A PSD, like every shipped configuration: the backend without one is
                # `gwmock_noise`'s white-noise simulator, which reseeds from entropy after its
                # first chunk and so is not reproducible past it -- which would make the resume
                # comparison below assert nothing.
                "arguments": {"detectors": ["H1", "L1"], "psd_file": "ET_10_full_cryo_psd"},
                "output": {
                    "output_directory": "noise",
                    "file_name": "E-{{ detectors }}_NOISE-{{ start_time }}-{{ duration }}.gwf",
                    "arguments": {"channel": "{{ detectors }}:STRAIN_NOISE"},
                },
            },
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def _frame_epochs(noise_directory: Path, detector: str) -> list[float]:
    """Return the ``t0`` of every written frame for one detector, in time order.

    Read out of the frame, not parsed out of its name: the name is composed from the same
    ``start_time`` the epoch comes from, so a name-based check could not tell a correct epoch from
    a correctly-named frame holding the wrong span.
    """
    epochs = []
    for path in sorted(noise_directory.glob(f"E-{detector}_NOISE-*.gwf")):
        series = GWpyTimeSeries.read(str(path), channel=f"{detector}:STRAIN_NOISE")
        epochs.append(float(series.t0.value))
    return sorted(epochs)


def _run(tmp_path: Path) -> Path:
    """Run the gapped configuration end to end and return the working directory."""
    work = tmp_path
    work.mkdir(parents=True, exist_ok=True)
    config_path = work / "gapped.yaml"
    _write_config(config_path, work)
    _simulate_impl(str(config_path), overwrite=True, metadata=True)
    return work


class TestTheWrittenFramesAreDiscontiguous:
    """Read the layout back out of the data, the way a consumer of the release would."""

    def test_every_frame_epoch_comes_off_the_file_and_matches_the_gapped_layout(self, tmp_path):
        """The acceptance assertion: the epochs step by ``duration + gap``, taken from the frames."""
        work = _run(tmp_path / "run")
        noise = work / "output" / "noise"

        for detector in ("H1", "L1"):
            assert _frame_epochs(noise, detector) == EXPECTED_EPOCHS, detector

    def test_consecutive_frames_leave_the_gap_unwritten(self, tmp_path):
        """Not just "the epochs differ": each frame ends a whole gap before the next one starts."""
        work = _run(tmp_path / "run")
        path = sorted((work / "output" / "noise").glob("E-H1_NOISE-*.gwf"))

        spans = []
        for frame in path:
            series = GWpyTimeSeries.read(str(frame), channel="H1:STRAIN_NOISE")
            spans.append((float(series.t0.value), float(series.t0.value) + float(series.duration.value)))
        spans.sort()

        assert [round(start - end, 9) for (_, end), (start, _) in pairwise(spans)] == [
            _GAP,
            _GAP,
            _GAP,
        ]
        assert sum(end - start for start, end in spans) == _N_SEGMENTS * _DURATION
        assert spans[-1][1] - spans[0][0] == _SPAN

    def test_the_signal_frames_carry_the_same_epochs(self, tmp_path):
        """Both writers take their epoch from the same advancing ``start_time``; assert it."""
        work = _run(tmp_path / "run")
        names = sorted((work / "output" / "signal").glob("sgwb-*.hdf5"))

        assert [float(path.stem.split("-")[1]) for path in names] == EXPECTED_EPOCHS

    def test_the_metadata_epoch_is_the_frame_epoch(self, tmp_path):
        """Provenance that disagreed with the data would be worse than no provenance."""
        work = _run(tmp_path / "run")

        for index, expected in enumerate(EXPECTED_EPOCHS):
            record = json.loads((work / "metadata" / f"orchestration-{index}.metadata.json").read_text())
            layout = record["simulator_metadata"]["orchestration"]["segment_layout"]
            assert layout["segment_index"] == index
            assert layout["run_start_time"] + index * (layout["duration"] + layout["segment_gap"]) == expected
            assert layout["span"] == _SPAN
            assert layout["analysed_livetime"] == _N_SEGMENTS * _DURATION
            noise_output = next(entry for entry in record["outputs"] if entry["kind"] == "noise")
            assert noise_output["t0"] == expected

    def test_a_released_file_does_not_carry_the_gap_truth(self, tmp_path):
        """The embedded copy withholds what the gaps did to the signals, like the injections."""
        from gwmock.cli.utils.metadata import embeddable_metadata

        record = {
            "signal": {
                "injections": [{"event_id": 0}],
                "gap_excluded_injections": [{"event_id": 1, "parameters": {"coa_time": 1.0}}],
                "gap_discarded_injections": [{"event_id": 2, "parameters": {"coa_time": 2.0}}],
            }
        }

        embedded = embeddable_metadata(record)

        assert "gap_excluded_injections" not in embedded["signal"]
        assert "gap_discarded_injections" not in embedded["signal"]
        assert "injections" not in embedded["signal"]
        assert "gap_excluded_injections" in embeddable_metadata(record, include_injection_parameters=True)["signal"]


class TestResumingAcrossAGapBoundary:
    """A run interrupted between two segments separated by a gap resumes bit-identically.

    The gapped noise stream draws the gap's samples between one analysed chunk and the next, so a
    resumed run has to replay those draws as well when it realigns the stream to the batch it is
    restarting at. Getting that wrong shifts every frame after the resume point onto noise the
    uninterrupted run never wrote -- which no amount of looking at the frames would reveal, because
    the wrong noise looks exactly like the right noise.
    """

    @staticmethod
    def _samples(work: Path, detector: str) -> dict[float, np.ndarray]:
        """Every frame's samples for one detector, keyed by the epoch read off the file."""
        samples = {}
        for path in sorted((work / "output" / "noise").glob(f"E-{detector}_NOISE-*.gwf")):
            series = GWpyTimeSeries.read(str(path), channel=f"{detector}:STRAIN_NOISE")
            samples[float(series.t0.value)] = np.asarray(series.value, dtype=float)
        return samples

    def test_the_frames_after_the_resume_point_are_bit_identical(self, tmp_path, monkeypatch):
        """Interrupt after batch 1, restart, and compare against an uninterrupted run.

        Batch 2 is the batch after a gap, so the restart point is a gap boundary rather than an
        ordinary one.
        """
        from gwmock.cli import simulate_utils

        monkeypatch.chdir(_fresh(tmp_path / "cwd-a"))
        reference = self._samples(_run(tmp_path / "uninterrupted"), "H1")
        assert sorted(reference) == EXPECTED_EPOCHS

        interrupted = tmp_path / "interrupted"
        checkpoint_cwd = _fresh(tmp_path / "cwd-b")
        monkeypatch.chdir(checkpoint_cwd)
        real_process_batch = simulate_utils.process_batch

        def fail_from_batch_two(simulator, batch_data, batch, output_directory, overwrite):
            if batch.batch_index >= 2:
                raise RuntimeError("interrupted")
            return real_process_batch(simulator, batch_data, batch, output_directory, overwrite)

        monkeypatch.setattr(simulate_utils, "process_batch", fail_from_batch_two)
        with pytest.raises(typer.Exit):
            _run(interrupted)

        checkpoint = json.loads((checkpoint_cwd / ".gwmock_checkpoints" / "simulation.checkpoint.json").read_text())
        assert sorted(checkpoint["completed_batch_indices"]) == [0, 1], (
            "the run has to stop with two batches behind it, so the resume crosses the gap "
            "between segment 1 and segment 2"
        )

        monkeypatch.setattr(simulate_utils, "process_batch", real_process_batch)
        resumed = self._samples(_run(interrupted), "H1")

        assert sorted(resumed) == EXPECTED_EPOCHS
        for epoch in EXPECTED_EPOCHS:
            np.testing.assert_array_equal(
                resumed[epoch],
                reference[epoch],
                err_msg=f"frame at {epoch} differs after a resume across a gap boundary",
            )


def _fresh(path: Path) -> Path:
    """Return *path* as an existing empty directory, so each run gets its own checkpoint store."""
    path.mkdir(parents=True, exist_ok=True)
    return path
