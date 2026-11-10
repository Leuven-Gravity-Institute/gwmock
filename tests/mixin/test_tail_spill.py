"""Signal tails spilled to disk place exactly what tails held in memory place.

A tail is the part of a signal past the end of the segment that generated it. With
``tail_spill_directory`` set, :class:`~gwmock.mixin.time_series.TimeSeriesMixin` writes each tail to a
file once and reads only the window of it each later segment needs. Nothing about the data may change:
every segment, every provenance record and every gap record has to be the one the in-memory path
produces. These tests run one simulator both ways and compare.
"""

from __future__ import annotations

import copy
import dataclasses
import json

import numpy as np
import pytest
from astropy.units import Quantity

from gwmock.data.serialize.decoder import Decoder
from gwmock.data.serialize.encoder import Encoder
from gwmock.data.time_series.spilled import SpilledTimeSeries
from gwmock.data.time_series.time_series import TimeSeries
from gwmock.data.time_series.time_series_list import TimeSeriesList
from gwmock.mixin.time_series import TimeSeriesMixin
from gwmock.simulator.base import Simulator

_RATE = 64.0
_DURATION = 4.0


def _chunk(start: float, samples: int, event_id: int, seed: int, channels: int = 2) -> TimeSeries:
    """A chunk of seeded noise standing in for a signal, carrying its provenance and channel names."""
    rng = np.random.default_rng(seed)
    chunk = TimeSeries(
        data=rng.standard_normal((channels, samples)) * 1e-21,
        start_time=Quantity(start, unit="s"),
        sampling_frequency=Quantity(_RATE, unit="Hz"),
    )
    chunk.metadata.update({"injection_parameters": {"coa_time": start + samples / _RATE}, "event_id": event_id})
    for index in range(channels):
        chunk[index] = dataclasses.replace(chunk[index], name=f"X{index}:STRAIN", unit="strain")
        chunk.channels[index] = f"X{index}:STRAIN"
    return chunk


class _LongSignals(TimeSeriesMixin, Simulator):
    """Each segment generates two signals lasting several segments; one starts between samples."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generated = 0

    def _simulate(self, *args, **kwargs) -> TimeSeriesList:
        start = float(self.start_time.value)
        chunks = TimeSeriesList(
            [
                # On the segment's grid, 3.4 segments long.
                _chunk(
                    start + 1.0, int(3.4 * _DURATION * _RATE), event_id=2 * int(self.counter), seed=int(self.counter)
                ),
                # A third of a sample off the grid, so the first segment resamples it.
                _chunk(
                    start + 2.0 + 1.0 / (3.0 * _RATE),
                    int(2.2 * _DURATION * _RATE),
                    event_id=2 * int(self.counter) + 1,
                    seed=1000 + int(self.counter),
                ),
            ]
        )
        self.generated += len(chunks)
        return chunks

    def save_batch(self, batch, file_name, overwrite=False, **kwargs):  # pragma: no cover - not written here
        raise NotImplementedError


def _run(spill_directory, segments: int = 6, segment_gap: float = 0.0, stop: int | None = None):
    """Build the first *stop* of *segments* segments; return each one's samples and records, and the simulator."""
    simulator = _LongSignals(
        start_time=1_000_000_000,
        duration=_DURATION,
        total_duration=segments * _DURATION + (segments - 1) * segment_gap,
        sampling_frequency=_RATE,
        num_of_channels=2,
        segment_gap=segment_gap,
    )
    simulator.tail_spill_directory = spill_directory
    results = []
    for counter in range(segments if stop is None else stop):
        simulator.counter = counter
        simulator.start_time = Quantity(1_000_000_000 + counter * (_DURATION + segment_gap), unit="s")
        segment = simulator.simulate()
        results.append(
            {
                "samples": np.asarray([np.asarray(segment[i].data) for i in range(segment.num_of_channels)]),
                "carried": copy.deepcopy(simulator.carried_injections),
                "gap": copy.deepcopy(simulator.gap_discarded),
                "spilled": sum(isinstance(chunk, SpilledTimeSeries) for chunk in simulator.cached_data_chunks),
            }
        )
    return results, simulator


class TestSpilledTailsPlaceTheSameData:
    """The run with tails on disk against the same run with tails in memory."""

    @pytest.mark.parametrize("segment_gap", [0.0, 1.0, 1.0 + 1.0 / (2.0 * _RATE)])
    def test_every_segment_is_bit_identical(self, tmp_path, segment_gap):
        """Contiguous, across whole-sample gaps, and across a gap that moves the grid by half a sample."""
        in_memory, _ = _run(None, segment_gap=segment_gap)
        spilled, _ = _run(tmp_path / "tails", segment_gap=segment_gap)

        assert all(result["spilled"] for result in spilled[:-1]), "a tail stayed in memory"
        for index, (want, got) in enumerate(zip(in_memory, spilled, strict=True)):
            assert np.any(want["samples"] != 0.0)
            np.testing.assert_array_equal(got["samples"], want["samples"], err_msg=f"segment {index}")
            assert got["carried"] == want["carried"], f"segment {index}"

    def test_gap_records_match_the_in_memory_ones(self, tmp_path):
        """The same events, spans and sample counts; the energy share to the last bits of a block sum."""
        in_memory, _ = _run(None, segment_gap=1.0)
        spilled, _ = _run(tmp_path / "tails", segment_gap=1.0)

        assert any(result["gap"] for result in in_memory), "no signal crossed a gap, so this proves nothing"
        for want, got in zip(in_memory, spilled, strict=True):
            assert len(got["gap"]) == len(want["gap"])
            for got_record, want_record in zip(got["gap"], want["gap"], strict=True):
                fraction = got_record.pop("discarded_energy_fraction")
                expected = want_record.pop("discarded_energy_fraction")
                assert fraction == pytest.approx(expected, rel=1e-12, abs=0.0)
                assert got_record == want_record

    def test_each_tail_is_written_once(self, tmp_path):
        """Consuming a tail reads further into the same file; it never writes the file again."""
        directory = tmp_path / "tails"
        _, simulator = _run(directory, stop=2)
        first = {path.name: path.stat().st_mtime_ns for path in directory.iterdir()}

        simulator.counter = 2
        simulator.start_time = Quantity(1_000_000_000 + 2 * _DURATION, unit="s")
        simulator.simulate()

        after = {path.name: path.stat().st_mtime_ns for path in directory.iterdir()}
        assert set(first) <= set(after)
        assert all(after[name] == first[name] for name in first)
        # Two signals per segment, each outliving the segment that generated it.
        assert len(after) == 2 * 3

    def test_without_a_directory_nothing_is_written(self, tmp_path, monkeypatch):
        """The default keeps tails in memory, as before."""
        monkeypatch.chdir(tmp_path)
        results, _ = _run(None)

        assert not any(result["spilled"] for result in results)
        assert list(tmp_path.iterdir()) == []


class TestTheSpilledSeries:
    """The disk-backed series on its own."""

    def test_it_equals_the_chunk_it_was_spilled_from(self, tmp_path):
        chunk = _chunk(1000.0, 300, event_id=7, seed=1)
        spilled = SpilledTimeSeries.spill(chunk, tmp_path)

        np.testing.assert_array_equal(np.asarray(spilled), np.asarray(chunk))
        assert spilled.start_time == chunk.start_time
        assert spilled.sampling_frequency == chunk.sampling_frequency
        assert spilled.metadata == chunk.metadata
        assert [spilled[i].name for i in range(2)] == ["X0:STRAIN", "X1:STRAIN"]
        assert [spilled[i].unit for i in range(2)] == ["strain", "strain"]
        assert spilled.channels == chunk.channels

    def test_views_split_it_at_a_time_without_copying_the_file(self, tmp_path):
        chunk = _chunk(1000.0, 300, event_id=7, seed=1)
        spilled = SpilledTimeSeries.spill(chunk, tmp_path)
        boundary = Quantity(1002.0, unit="s")

        head, rest = spilled.until(boundary), spilled.after(boundary)

        assert rest is not None
        assert head.spill_path == rest.spill_path == spilled.spill_path
        assert rest.start_time == boundary
        samples = np.asarray(chunk)
        np.testing.assert_array_equal(np.asarray(head), samples[:, :128])
        np.testing.assert_array_equal(np.asarray(rest), samples[:, 128:])
        assert spilled.after(Quantity(1010.0, unit="s")) is None
        assert len(list(tmp_path.iterdir())) == 1

    def test_a_checkpoint_records_the_file_not_the_samples(self, tmp_path):
        """A few hundred bytes for a tail of any length, read back to the same samples and provenance."""
        chunk = _chunk(1000.0, 100_000, event_id=7, seed=1)
        rest = SpilledTimeSeries.spill(chunk, tmp_path).after(Quantity(1001.0, unit="s"))

        text = json.dumps(TimeSeriesList([rest]), cls=Encoder)
        restored = json.loads(text, cls=Decoder)

        assert len(text) < 2_000
        assert isinstance(restored[0], SpilledTimeSeries)
        np.testing.assert_array_equal(np.asarray(restored[0]), np.asarray(chunk)[:, 64:])
        assert restored[0].start_time == rest.start_time
        assert restored[0].metadata == chunk.metadata
        assert restored[0].channels == chunk.channels

    def test_a_missing_file_is_named_when_the_checkpoint_is_read(self, tmp_path):
        spilled = SpilledTimeSeries.spill(_chunk(1000.0, 64, event_id=7, seed=1), tmp_path)
        text = json.dumps(spilled, cls=Encoder)
        spilled.spill_path.unlink()

        with pytest.raises(FileNotFoundError, match="no longer exists"):
            json.loads(text, cls=Decoder)

    def test_a_deep_copy_shares_the_file(self, tmp_path):
        """The retry snapshot copies the tails; it must not read them into memory."""
        spilled = SpilledTimeSeries.spill(_chunk(1000.0, 64, event_id=7, seed=1), tmp_path)

        copied = copy.deepcopy(spilled)

        assert isinstance(copied, SpilledTimeSeries)
        assert copied.spill_path == spilled.spill_path
        assert isinstance(copied[0].data, np.memmap)
        copied.metadata["event_id"] = 8
        assert spilled.metadata["event_id"] == 7

    def test_its_samples_cannot_be_changed(self, tmp_path):
        spilled = SpilledTimeSeries.spill(_chunk(1000.0, 64, event_id=7, seed=1), tmp_path)

        with pytest.raises(ValueError, match="read-only"):
            spilled[0].data[0] = 1.0

    def test_summed_squares_is_the_sum_of_every_sample_squared(self, tmp_path):
        chunk = _chunk(1000.0, 10_000, event_id=7, seed=1)
        spilled = SpilledTimeSeries.spill(chunk, tmp_path)

        expected = float(np.sum(np.square(np.asarray(chunk))))
        assert spilled.summed_squares(block_samples=999) == pytest.approx(expected, rel=1e-12, abs=0.0)
