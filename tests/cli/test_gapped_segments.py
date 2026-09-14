"""Segment epochs that advance by ``duration + segment-gap``, and everything that follows from it.

A run configured with ``globals.simulator-arguments.segment-gap`` writes frames that are
**discontiguous in GPS**: the seconds inside a gap appear in no frame at all. Four things had to
follow the epoch, and each has its own class here:

* the batch count, because ``total-duration`` is a span the segments no longer tile;
* the end of the run, because ``final_end_time`` is what drops cached content;
* spillover, because a waveform crossing a gap must lose only the part inside it;
* the stateful noise stream, whose ``psd_schedule`` is interpolated against samples *produced*
  rather than against GPS -- so the gap's samples are generated and discarded, and the test that
  says so is exact rather than statistical.

``tests/cli/test_gapped_segments_end_to_end.py`` is the counterpart that reads the epochs back off
written frames.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from astropy.units.quantity import Quantity

from gwmock.cli.utils.config import Config
from gwmock.data.time_series.time_series import TimeSeries
from gwmock.noise.adapter import NoiseAdapter

_POPULATION_CSV = Path(__file__).resolve().parents[2] / "examples" / "signal" / "bbh_population.csv"
_START = 1577491296.0
_SAMPLING_FREQUENCY = 256.0
_SEGMENT_DURATION = 16.0
_SEGMENT_GAP = 8.0

_COMPLETE_EVENT: dict[str, Any] = {
    "detector_frame_mass_1": 30.0,
    "detector_frame_mass_2": 25.0,
    "distance": 400.0,
    "right_ascension": 1.0,
    "declination": 0.5,
    "polarization_angle": 0.2,
    "inclination": 0.3,
}


def _config(
    working_directory: Path,
    *,
    duration: float = _SEGMENT_DURATION,
    gap: float = _SEGMENT_GAP,
    n_segments: int = 4,
) -> dict[str, Any]:
    """Return a signal-only config whose span holds exactly *n_segments* gapped segments."""
    return {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": _SAMPLING_FREQUENCY,
                "duration": duration,
                "segment-gap": gap,
                "total-duration": n_segments * duration + (n_segments - 1) * gap,
                "start-time": _START,
                "seed": 20260914,
            },
            "working-directory": str(working_directory),
            "output-directory": "output",
            "metadata-directory": "metadata",
        },
        "orchestration": {
            "population": {
                "backend": "FilePopulationLoader",
                "source-type": "bbh",
                "n-samples": 1,
                "arguments": {"path": str(_POPULATION_CSV)},
            },
            "signal": {
                "source-type": "bbh",
                "waveform-model": "IMRPhenomD",
                "minimum-frequency": 30,
                "detectors": ["ET-Triangle-Sardinia"],
                "output": {
                    "output_directory": "signal",
                    "file_name": "sig-{{ detectors }}.gwf",
                    "arguments": {"channel": "{{ detectors }}:STRAIN"},
                },
            },
        },
    }


def _orchestrator(working_directory: Path, **kwargs: Any):
    from gwmock.cli.adapter_orchestration import AdapterOrchestrator

    working_directory.mkdir(parents=True, exist_ok=True)
    config = Config.model_validate(_config(working_directory, **kwargs))
    return AdapterOrchestrator.from_config(
        config.orchestration,
        global_simulator_arguments=dict(config.globals.simulator_arguments),
    )


class _StubAdapter:
    """A signal adapter answering the placement queries with fixed lead and tail."""

    def __init__(self, lead: float | None = 0.0, tail: float | None = 0.0) -> None:
        self._lead = lead
        self._tail = tail
        self.detector_names = ("E1",)

    def pre_coalescence_duration(self, parameters: Mapping[str, Any], **_: Any) -> float | None:
        return self._lead

    def post_coalescence_duration(self, parameters: Mapping[str, Any], **_: Any) -> float | None:
        return self._tail


def _epoch(orchestrator) -> float:
    return float(getattr(orchestrator.start_time, "value", orchestrator.start_time))


class TestTheEpochAdvancesByDurationPlusGap:
    """The requirement itself, asserted as an exact epoch list."""

    def test_the_epoch_list_is_exact_for_the_motivating_layout(self, tmp_path):
        """Four 1024 s segments per block separated by 256 s gaps, eight blocks: 32 epochs.

        Spelled out rather than generated from the same stride the code walks, and on the unfixed
        code every one of these after the first is wrong -- the epochs come out contiguous at
        1024 s spacing.
        """
        orchestrator = _orchestrator(tmp_path, duration=1024.0, gap=256.0, n_segments=32)

        epochs = []
        for _ in range(32):
            epochs.append(_epoch(orchestrator))
            orchestrator.update_state()

        assert epochs[:6] == [
            _START,
            _START + 1280.0,
            _START + 2560.0,
            _START + 3840.0,
            _START + 5120.0,
            _START + 6400.0,
        ]
        assert epochs[-1] == _START + 31 * 1280.0
        assert epochs == [_START + index * 1280.0 for index in range(32)]

    def test_a_run_without_a_gap_still_advances_by_the_duration_alone(self, tmp_path):
        """The default is zero, so nothing configured before this feature moves."""
        orchestrator = _orchestrator(tmp_path, gap=0.0, n_segments=4)

        epochs = []
        for _ in range(4):
            epochs.append(_epoch(orchestrator))
            orchestrator.update_state()

        assert epochs == [_START + index * _SEGMENT_DURATION for index in range(4)]

    def test_the_batch_count_comes_from_the_span_and_not_from_the_livetime(self, tmp_path):
        """40704 s of span at 1024 s segments and 256 s gaps is 32 batches, not 39."""
        orchestrator = _orchestrator(tmp_path, duration=1024.0, gap=256.0, n_segments=32)

        assert orchestrator.max_samples == 32
        assert float(orchestrator.total_duration.value) == 40704.0
        assert float(orchestrator.analysed_livetime.value) == 32 * 1024.0

    def test_the_run_ends_where_the_last_segment_ends_and_stops_moving(self, tmp_path):
        """``final_end_time`` is measured from the configured start, not the current segment.

        Anchored to the advancing epoch it would slide forward a stride per segment, so "outside
        the total duration" would name a moving target instead of the end of the data.
        """
        orchestrator = _orchestrator(tmp_path, n_segments=4)
        expected = _START + (4 * _SEGMENT_DURATION + 3 * _SEGMENT_GAP)

        assert float(orchestrator.final_end_time.value) == expected
        for _ in range(3):
            orchestrator.update_state()
            assert float(orchestrator.final_end_time.value) == expected

        # And that end time is exactly the last segment's end.
        assert _epoch(orchestrator) + _SEGMENT_DURATION == expected

    def test_a_span_that_does_not_divide_is_refused_before_anything_is_generated(self, tmp_path):
        """Naming the three numbers, so the message says which of them to change."""
        from gwmock.cli.adapter_orchestration import AdapterOrchestrator

        tmp_path.mkdir(parents=True, exist_ok=True)
        config = Config.model_validate(_config(tmp_path))
        arguments = dict(config.globals.simulator_arguments)
        arguments["total-duration"] = 100.0

        with pytest.raises(ValueError, match="does not divide into whole segments") as error:
            AdapterOrchestrator.from_config(config.orchestration, global_simulator_arguments=arguments)

        message = str(error.value)
        assert "100" in message
        assert "16" in message
        assert "8" in message


class TestTheMetadataRecordsTheRealEpoch:
    """Provenance has to describe the frame that was written, not the layout it assumed."""

    def test_each_batch_records_its_own_epoch_and_the_layout_it_came_from(self, tmp_path):
        orchestrator = _orchestrator(tmp_path, n_segments=4)
        orchestrator.update_state()
        orchestrator.update_state()

        metadata = orchestrator.metadata
        assert float(metadata["time_series"]["arguments"]["start_time"].value) == _START + 2 * 24.0
        assert float(metadata["time_series"]["arguments"]["segment_gap"].value) == _SEGMENT_GAP

        layout = metadata["orchestration"]["segment_layout"]
        assert layout["run_start_time"] == _START
        assert layout["segment_index"] == 2
        assert layout["duration"] == _SEGMENT_DURATION
        assert layout["segment_gap"] == _SEGMENT_GAP
        assert layout["n_segments"] == 4
        assert layout["span"] == 4 * _SEGMENT_DURATION + 3 * _SEGMENT_GAP
        assert layout["analysed_livetime"] == 4 * _SEGMENT_DURATION

    def test_the_frame_name_follows_the_epoch(self, tmp_path):
        """Asserted rather than assumed: the template reads ``start_time`` off the simulator."""
        from gwmock.cli.utils.template import expand_template_variables

        orchestrator = _orchestrator(tmp_path, n_segments=4)
        names = []
        for _ in range(4):
            names.append(expand_template_variables("f-{{ start_time }}-{{ duration }}.gwf", orchestrator))
            orchestrator.update_state()

        assert [float(name.split("-")[1]) for name in names] == [_START + index * 24.0 for index in range(4)]


class TestASignalThatFallsInAGap:
    """A catalogue event with nowhere to be written, and one that survives the crossing."""

    def test_an_event_wholly_inside_a_gap_is_excluded_and_recorded(self, tmp_path, caplog):
        """Written to no frame -- and said so, rather than vanishing between two epochs."""
        orchestrator = _orchestrator(tmp_path, n_segments=4)
        orchestrator.signal_adapter = _StubAdapter(lead=1.0, tail=1.0)
        # The gap after segment 0 covers [start + 16, start + 24).
        event = {**_COMPLETE_EVENT, "coa_time": _START + 20.0}

        gap = orchestrator._gap_excluding_event(event)
        assert gap == (_START + 16.0, _START + 24.0)

        with caplog.at_level(logging.WARNING, logger="gwmock"):
            orchestrator._record_gap_excluded_event(7, event, gap)

        assert orchestrator.metadata["orchestration"]["signal"]["gap_excluded_injections"] == [
            {
                "event_id": 7,
                "parameters": event,
                "gap_start": _START + 16.0,
                "gap_end": _START + 24.0,
            }
        ]
        assert "entirely inside the configured segment gap" in caplog.text

    def test_an_event_whose_inspiral_reaches_back_into_a_segment_is_still_generated(self, tmp_path):
        """Only the *whole* event counts; a crossing one is claimed and cropped by injection."""
        orchestrator = _orchestrator(tmp_path, n_segments=4)
        orchestrator.signal_adapter = _StubAdapter(lead=6.0, tail=1.0)
        # Coalescence 2 s into the gap, but the waveform starts 4 s before the gap does.
        event = {**_COMPLETE_EVENT, "coa_time": _START + 18.0}

        assert orchestrator._gap_excluding_event(event) is None

    def test_an_event_whose_ringdown_reaches_the_next_segment_is_still_generated(self, tmp_path):
        orchestrator = _orchestrator(tmp_path, n_segments=4)
        orchestrator.signal_adapter = _StubAdapter(lead=0.5, tail=6.0)
        event = {**_COMPLETE_EVENT, "coa_time": _START + 22.0}

        assert orchestrator._gap_excluding_event(event) is None

    def test_an_unknown_tail_claims_the_event_rather_than_excluding_it(self, tmp_path):
        """Unknown is not zero: reading it as zero would exclude a whole backend's events."""
        orchestrator = _orchestrator(tmp_path, n_segments=4)
        orchestrator.signal_adapter = _StubAdapter(lead=1.0, tail=None)

        assert orchestrator._gap_excluding_event({**_COMPLETE_EVENT, "coa_time": _START + 20.0}) is None

    def test_a_contiguous_run_never_excludes_anything(self, tmp_path):
        """There is no gap to fall into, so the check must not invent one."""
        orchestrator = _orchestrator(tmp_path, gap=0.0, n_segments=4)
        orchestrator.signal_adapter = _StubAdapter(lead=1.0, tail=1.0)

        assert orchestrator._gap_excluding_event({**_COMPLETE_EVENT, "coa_time": _START + 20.0}) is None

    def test_an_event_after_the_run_is_not_reported_as_falling_in_a_gap(self, tmp_path):
        """Past the end of the run is not a hole in the data; it is outside what the run covers."""
        orchestrator = _orchestrator(tmp_path, n_segments=4)
        orchestrator.signal_adapter = _StubAdapter(lead=1.0, tail=1.0)

        assert orchestrator._gap_excluding_event({**_COMPLETE_EVENT, "coa_time": _START + 1000.0}) is None


class TestSpilloverAcrossAGap:
    """A chunk crossing a gap keeps its far-side content at its true GPS sample.

    Built the way a run builds it, in two steps: a chunk is injected into the segment before the
    gap, which hands back the tail cropped at that segment's end, and the tail is then injected
    into the segment after the gap. Injecting a raw chunk straight into the later segment would
    test a path no run takes, and would miss that the tail begins exactly on the gap's first
    instant -- which is what tells a gap loss apart from a placement failure.
    """

    SAMPLING_FREQUENCY = 16.0
    SEGMENT_DURATION = 8.0
    GAP = 8.0
    FIRST_EPOCH = 1000.0
    GAP_START = FIRST_EPOCH + SEGMENT_DURATION  # 1008.0
    GAP_END = GAP_START + GAP  # 1016.0

    @classmethod
    def _segment(cls, start_time: float) -> TimeSeries:
        return TimeSeries(
            data=np.zeros((1, round(cls.SEGMENT_DURATION * cls.SAMPLING_FREQUENCY))),
            start_time=Quantity(start_time, unit="s"),
            sampling_frequency=Quantity(cls.SAMPLING_FREQUENCY, unit="Hz"),
        )

    @classmethod
    def _crossing_chunk(cls) -> TimeSeries:
        """A 16 s ramp starting 4 s into the first segment, so every sample names its own offset.

        It covers [1004, 1020): 4 s inside the first segment, the whole 8 s gap, and 4 s inside the
        segment after it.
        """
        n_samples = round(16.0 * cls.SAMPLING_FREQUENCY)
        chunk = TimeSeries(
            data=np.arange(1, n_samples + 1, dtype=float).reshape(1, n_samples),
            start_time=Quantity(cls.FIRST_EPOCH + 4.0, unit="s"),
            sampling_frequency=Quantity(cls.SAMPLING_FREQUENCY, unit="Hz"),
        )
        chunk.metadata.update({"injection_parameters": {"coa_time": 1012.0}, "event_id": 3})
        return chunk

    @classmethod
    def _tail_across_the_gap(cls) -> TimeSeries:
        """The spillover the first segment hands on: the chunk cropped at the gap's first instant."""
        tail = cls._segment(cls.FIRST_EPOCH).inject(cls._crossing_chunk())
        assert tail is not None
        assert float(tail.start_time.value) == cls.GAP_START
        return tail

    def test_the_far_side_of_a_crossing_chunk_lands_at_its_true_gps_sample(self):
        """Not shifted forward to close the gap, which is the failure this exists to prevent."""
        segment = self._segment(self.GAP_END)

        segment.inject(self._tail_across_the_gap(), preceding_gap=(self.GAP_START, self.GAP_END))

        values = np.asarray(segment[0].value)
        # GPS 1016 is the ramp's 12 s mark, which is its sample number 12 * 16 + 1 = 193.
        assert values[0] == pytest.approx(193.0, abs=0.0)
        assert values[1] == pytest.approx(194.0, abs=0.0)
        # 4 s of ramp remain; the rest of the segment is silent.
        assert values[round(4.0 * self.SAMPLING_FREQUENCY) - 1] == pytest.approx(256.0, abs=0.0)
        assert np.all(values[round(4.0 * self.SAMPLING_FREQUENCY) :] == 0.0)

    def test_nothing_from_inside_the_gap_is_written(self):
        """The in-gap samples are the ramp's 65..192; none of them may appear in the segment."""
        segment = self._segment(self.GAP_END)

        segment.inject(self._tail_across_the_gap(), preceding_gap=(self.GAP_START, self.GAP_END))

        written = set(np.asarray(segment[0].value)[np.asarray(segment[0].value) > 0].tolist())
        assert not written & {float(value) for value in range(65, 193)}

    def test_a_gap_crossing_is_reported_as_a_gap_and_not_as_a_placement_failure(self, caplog):
        """The standing warning sends a reader after a cause that does not exist in a gapped run."""
        segment = self._segment(self.GAP_END)

        with caplog.at_level(logging.WARNING, logger="gwmock"):
            segment.inject(self._tail_across_the_gap(), preceding_gap=(self.GAP_START, self.GAP_END))

        assert "before this segment begins" not in caplog.text

    def test_the_same_loss_without_a_gap_is_still_reported_as_a_placement_failure(self, caplog):
        """The relabelling must not swallow the case the warning was written for."""
        segment = self._segment(self.GAP_END)

        with caplog.at_level(logging.WARNING, logger="gwmock"):
            segment.inject(self._tail_across_the_gap(), preceding_gap=None)

        assert "before this segment begins" in caplog.text

    def test_what_the_gap_swallowed_is_recorded_with_the_gap_that_swallowed_it(self):
        """The samples are gone, so this record is the only thing that says they existed."""
        from gwmock.mixin.time_series import _gap_discarded_records

        segment = self._segment(self.GAP_END)

        records = _gap_discarded_records(segment, [self._tail_across_the_gap()], (self.GAP_START, self.GAP_END))

        assert len(records) == 1
        assert records[0]["event_id"] == 3
        assert records[0]["parameters"] == {"coa_time": 1012.0}
        assert records[0]["gap_start"] == self.GAP_START
        assert records[0]["gap_end"] == self.GAP_END
        # The whole 8 s gap, and nothing else: the tail begins exactly where the gap does.
        assert records[0]["discarded_samples"] == round(self.GAP * self.SAMPLING_FREQUENCY)
        assert records[0]["discarded_seconds"] == pytest.approx(self.GAP, abs=0.0)
        assert 0.0 < records[0]["discarded_energy_fraction"] < 1.0

    def test_a_contiguous_run_records_nothing(self):
        from gwmock.mixin.time_series import _gap_discarded_records

        segment = self._segment(self.GAP_END)

        assert _gap_discarded_records(segment, [self._tail_across_the_gap()], None) == []


class TestTheNoiseStreamCrossesTheGap:
    """The stateful stream generates the gap's samples and throws them away.

    Documented at ``TimeSeriesMixin.segment_gap``, and the reason it is a choice at all is that
    ``gwmock_noise.ColoredNoiseSimulator`` interpolates its ``psd_schedule`` against the number of
    samples it has *produced*, not against GPS.
    """

    @staticmethod
    def _psd_file(path: Path, scale: float, sampling_frequency: float) -> Path:
        """Write a flat one-sided PSD of ``scale * 1e-46`` strain^2/Hz."""
        frequencies = np.linspace(0.0, sampling_frequency / 2.0, 1025)
        np.savetxt(path, np.column_stack([frequencies, np.full_like(frequencies, scale * 1e-46)]))
        return path

    def test_the_analysed_chunks_are_the_contiguous_stream_sampled_across_the_gaps(self, tmp_path):
        """The exact anchor, and an external one: a separate contiguous run of the same span.

        With ``duration == gap``, a gapped run draws chunk, gap, chunk, gap, ... from one stream,
        which is the same sequence of draws a contiguous run of the same GPS span makes. So the
        gapped run's analysed chunk *k* must be **bit-identical** to the contiguous run's chunk
        ``2k``. Under the rejected alternative -- skipping the gap rather than generating it --
        gapped chunk *k* would equal contiguous chunk *k* instead, so this single comparison
        decides which of the two is implemented.
        """
        psd = self._psd_file(tmp_path / "psd.txt", 1.0, 256.0)
        gapped = NoiseAdapter.from_backend().open_stream(
            chunk_duration=4.0,
            gap_duration=4.0,
            sampling_frequency=256.0,
            detectors=["H1"],
            seed=11,
            psd_file=str(psd),
        )
        contiguous = NoiseAdapter.from_backend().open_stream(
            chunk_duration=4.0,
            sampling_frequency=256.0,
            detectors=["H1"],
            seed=11,
            psd_file=str(psd),
        )

        gapped_chunks = [next(gapped)["H1"] for _ in range(4)]
        contiguous_chunks = [next(contiguous)["H1"] for _ in range(7)]

        for index, chunk in enumerate(gapped_chunks):
            assert np.array_equal(chunk, contiguous_chunks[2 * index]), (
                f"gapped chunk {index} must continue the stream across the gap, so it is the "
                f"contiguous run's chunk {2 * index}"
            )
        assert not np.array_equal(gapped_chunks[1], contiguous_chunks[1]), (
            "if it matched chunk 1 the stream would have skipped the gap, and the psd_schedule "
            "axis would track analysed livetime rather than GPS"
        )

    def test_a_zero_gap_opens_the_plain_upstream_stream(self, tmp_path):
        """No gap configured means no extra draws and the same stream as before this feature."""
        psd = self._psd_file(tmp_path / "psd.txt", 1.0, 256.0)
        with_zero = NoiseAdapter.from_backend().open_stream(
            chunk_duration=4.0, gap_duration=0.0, sampling_frequency=256.0, detectors=["H1"], seed=11, psd_file=str(psd)
        )
        without = NoiseAdapter.from_backend().open_stream(
            chunk_duration=4.0, sampling_frequency=256.0, detectors=["H1"], seed=11, psd_file=str(psd)
        )

        for _ in range(3):
            assert np.array_equal(next(with_zero)["H1"], next(without)["H1"])

    def test_a_negative_gap_is_refused(self, tmp_path):
        psd = self._psd_file(tmp_path / "psd.txt", 1.0, 256.0)
        with pytest.raises(ValueError, match="gap_duration must be non-negative"):
            NoiseAdapter.from_backend().open_stream(
                chunk_duration=4.0,
                gap_duration=-1.0,
                sampling_frequency=256.0,
                detectors=["H1"],
                seed=11,
                psd_file=str(psd),
            )

    def test_gaps_and_a_psd_drift_and_spectral_lines_compose(self, tmp_path):
        """One run with all three, with the drift anchored to a number computed independently.

        The schedule ramps one PSD curve from 0.9x to 1.1x. A PSD scales strain **squared**, so the
        broadband rms ratio between the two ends is ``sqrt(1.1 / 0.9) = 1.105542`` -- arithmetic,
        not a measurement, and the anchor this run is judged against.

        The two anchors are placed so the measurement lands exactly on those ends rather than
        somewhere along the ramp: the first anchor sits at the *end* of the first analysed segment
        and the second at the *start* of the last one, and the interpolation clamps outside its
        anchors. So every frame of the first segment sees exactly 0.9x and every frame of the last
        sees exactly 1.1x -- **on the GPS axis**. On the analysed-livetime axis the last segment
        would sit in the middle of the ramp instead, at a predicted ratio of 1.059, which is what
        makes this a test of the documented choice and not just of the arithmetic.

        Measured here: 1.10838, which is 0.26% above the anchor -- inside the 0.39% standard
        deviation of an rms estimated from 65536 samples, and 12 of those standard deviations away
        from the livetime prediction.
        """
        sampling_frequency = 512.0
        duration, gap, count = 128.0, 64.0, 4
        low = self._psd_file(tmp_path / "low.txt", 0.9, sampling_frequency)
        high = self._psd_file(tmp_path / "high.txt", 1.1, sampling_frequency)
        line_frequency = 60.0

        stream = NoiseAdapter.from_backend().open_stream(
            chunk_duration=duration,
            gap_duration=gap,
            sampling_frequency=sampling_frequency,
            detectors=["H1"],
            seed=2026,
            psd_schedule=[(duration, str(low)), ((count - 1) * (duration + gap), str(high))],
            spectral_lines=[{"frequency": line_frequency, "amplitude": 3e-24}],
        )
        chunks = [next(stream)["H1"] for _ in range(count)]

        def broadband_rms(samples: np.ndarray) -> float:
            """Root-mean-square with the line's bins notched out.

            The line is a fixed-amplitude sinusoid that the PSD ramp does not scale, so leaving it
            in would bias the ratio towards one by exactly the share of the energy it carries.
            """
            spectrum = np.fft.rfft(samples)
            frequencies = np.fft.rfftfreq(samples.size, 1.0 / sampling_frequency)
            spectrum[np.abs(frequencies - line_frequency) < 1.0] = 0.0
            return float(np.sqrt(np.sum(np.abs(spectrum) ** 2) * 2.0 / samples.size**2))

        measured = broadband_rms(chunks[-1]) / broadband_rms(chunks[0])
        gps_axis_anchor = np.sqrt(1.1 / 0.9)
        livetime_axis_prediction = np.sqrt(
            (0.9 * (1.1 / 0.9) ** ((3 * duration - duration) / ((count - 1) * (duration + gap) - duration))) / 0.9
        )

        assert float(f"{gps_axis_anchor:.7g}") == 1.105542
        assert measured == pytest.approx(gps_axis_anchor, rel=0.01)
        assert abs(measured - livetime_axis_prediction) > 10 * 0.0039 * measured

        # And the line survived the composition rather than being lost with the gaps.
        spectrum = np.abs(np.fft.rfft(chunks[0]))
        frequencies = np.fft.rfftfreq(chunks[0].size, 1.0 / sampling_frequency)
        line_bin = int(np.argmin(np.abs(frequencies - line_frequency)))
        off_line = np.median(spectrum[(frequencies > 100.0) & (frequencies < 200.0)])
        assert spectrum[line_bin] > 10.0 * off_line
