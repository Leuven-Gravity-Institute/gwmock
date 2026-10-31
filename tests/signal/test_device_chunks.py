"""Converting batched device output into gwmock chunks.

The conversion is pure host-side arithmetic on placement and ordering, so almost all of it is tested
against a stub batch rather than a real device run: that keeps the guards covered in an installation
without the ``[jax]`` extra, and it lets the failure cases be constructed deliberately instead of
hoped for. One test at the end runs the real batched path to check the stub has the right shape.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from gwmock.signal.device_chunks import (
    BATCHED_PARAMETERS,
    batched_strain_to_chunks,
    require_batched_parameters_supported,
)

_SAMPLING_FREQUENCY = 1024.0
_EPOCH = 1577491296.0


class _Grid:
    """Stand-in for ``gwmock_signal.SamplingGrid`` with the two methods the conversion uses."""

    def __init__(self, epoch: float = _EPOCH, sampling_frequency: float = _SAMPLING_FREQUENCY) -> None:
        self.epoch = epoch
        self.sampling_frequency = sampling_frequency

    def time_of(self, index: int) -> float:
        return self.epoch + index / self.sampling_frequency

    def index_of(self, gps_time):
        return (np.asarray(gps_time, dtype=float) - self.epoch) * self.sampling_frequency

    def is_on_lattice(self, gps_time):
        exact = self.index_of(gps_time)
        return np.abs(exact - np.round(exact)) <= 1e-3


@dataclass
class _Batch:
    """Stand-in for ``BatchedDetectorStrain``, carrying only the fields the conversion reads."""

    strain: np.ndarray
    detector_names: tuple[str, ...]
    start_index: Any
    grid: Any
    sampling_frequency: float = _SAMPLING_FREQUENCY
    coa_time: Any = None
    epoch: float = 0.0


def _batch(n_events: int = 2, n_detectors: int = 3, n_samples: int = 64, start_indices=(10, 500)) -> _Batch:
    """Return a stub batch whose samples identify their own event and detector.

    Each row is filled with ``event + detector / 10``, so a chunk that ends up in the wrong place is
    identifiable from its contents rather than only from a length.
    """
    strain = np.zeros((n_events, n_detectors, n_samples))
    for event in range(n_events):
        for detector in range(n_detectors):
            strain[event, detector, :] = event + detector / 10.0
    return _Batch(
        strain=strain,
        detector_names=tuple(f"ET{index + 1}_SARD" for index in range(n_detectors)),
        start_index=np.asarray(start_indices[:n_events]),
        grid=_Grid(),
    )


class TestConversion:
    """What the chunks look like when everything is well formed."""

    def test_one_chunk_per_event_in_order(self):
        chunks = batched_strain_to_chunks(_batch())

        assert len(chunks) == 2
        assert np.asarray(chunks[0])[0, 0] == pytest.approx(0.0)
        assert np.asarray(chunks[1])[0, 0] == pytest.approx(1.0), "chunks are not in event order"

    def test_each_chunk_keeps_every_detector_row(self):
        """Channel *k* becomes detector *k* downstream, so the rows must survive intact."""
        chunks = batched_strain_to_chunks(_batch())

        first = np.asarray(chunks[0])
        assert first.shape == (3, 64)
        assert [first[row, 0] for row in range(3)] == pytest.approx([0.0, 0.1, 0.2])

    def test_start_times_come_from_the_grid(self):
        """Placement is the whole point: a wrong start time misplaces the signal in time."""
        chunks = batched_strain_to_chunks(_batch(start_indices=(10, 500)))

        assert float(chunks[0].start_time.value) == pytest.approx(_EPOCH + 10 / _SAMPLING_FREQUENCY)
        assert float(chunks[1].start_time.value) == pytest.approx(_EPOCH + 500 / _SAMPLING_FREQUENCY)

    def test_the_chunks_land_on_whole_samples_of_a_matching_segment(self):
        """The property that makes injection an integer-offset add rather than a resample.

        Asserted through the real injection path, since that is what consumes these chunks: an
        exactly-placed chunk is added verbatim, so its constant value survives.
        """
        from gwmock.data.time_series.time_series import TimeSeries

        chunks = batched_strain_to_chunks(_batch(start_indices=(10, 500)))
        segment = TimeSeries(
            data=np.zeros((3, 1024)),
            start_time=_EPOCH,
            sampling_frequency=_SAMPLING_FREQUENCY,
        )

        remaining = segment.inject_from_list(chunks)

        assert len(remaining) == 0, "no chunk should overflow this segment"
        values = np.asarray(segment)
        assert values[0, 10] == pytest.approx(0.0)
        assert values[1, 500] == pytest.approx(1.1), (
            "the second event's second detector was not placed verbatim at its grid index"
        )

    def test_each_chunk_is_declared_on_the_output_grid(self):
        """Built there, so the injection places it by whole samples; the grid it names is the output grid."""
        from gwmock_data.multichannel_series import ON_GRID_EPOCH_METADATA_KEY, ON_GRID_METADATA_KEY

        for chunk in batched_strain_to_chunks(_batch()):
            assert chunk.metadata[ON_GRID_METADATA_KEY] is True
            assert chunk.metadata[ON_GRID_EPOCH_METADATA_KEY] == _EPOCH

    def test_a_chunk_whose_start_rounds_off_the_grid_is_placed_verbatim(self):
        """At GPS 1577491296 and 4000 Hz, sample 10's time is 2.29e-4 samples off the grid: beyond LAL's 1e-4
        no-op shift, so undeclared the chunk would be resampled at the time that rounding gave it."""
        from gwmock.data.time_series.time_series import TimeSeries

        rate = 4000.0
        batch = _batch(n_events=1, start_indices=(10,))
        batch.grid = _Grid(sampling_frequency=rate)
        batch.sampling_frequency = rate
        ramp = np.arange(1.0, 65.0)
        batch.strain = np.stack([np.stack([ramp, 2 * ramp, 3 * ramp])])
        offset = (batch.grid.time_of(10) - _EPOCH) * rate
        assert abs(offset - 10) > 1e-4

        segment = TimeSeries(data=np.zeros((3, 1024)), start_time=_EPOCH, sampling_frequency=rate)
        segment.inject_from_list(batched_strain_to_chunks(batch))

        np.testing.assert_array_equal(np.asarray(segment)[:, 10:74], batch.strain[0])

    def test_a_scalar_start_index_is_accepted(self):
        """A one-event batch may carry a scalar rather than a length-one array."""
        batch = _batch(n_events=1, start_indices=(7,))
        batch.start_index = 7

        chunks = batched_strain_to_chunks(batch)

        assert len(chunks) == 1
        assert float(chunks[0].start_time.value) == pytest.approx(_EPOCH + 7 / _SAMPLING_FREQUENCY)


class TestGuards:
    """Each failure this refuses to let through quietly."""

    def test_a_batch_without_a_grid_is_refused(self):
        """Without a grid every chunk would be resampled onto the segment, losing accuracy.

        Separate from the missing-index case below: nulling both at once would be satisfied by an
        implementation that checked only one of them.
        """
        batch = _batch()
        batch.grid = None

        with pytest.raises(ValueError, match="without an output grid"):
            batched_strain_to_chunks(batch)

    def test_a_batch_without_start_indices_is_refused(self):
        """The other half of the same guard, on its own."""
        batch = _batch()
        batch.start_index = None

        with pytest.raises(ValueError, match="without an output grid"):
            batched_strain_to_chunks(batch)

    def test_disagreeing_sampling_frequencies_are_refused(self):
        """Start times come from the grid and samples from the batch; at two rates they disagree."""
        batch = _batch()
        batch.sampling_frequency = _SAMPLING_FREQUENCY * 2

        with pytest.raises(ValueError, match="timestamped at one rate and spaced at another"):
            batched_strain_to_chunks(batch)

    def test_a_fractional_start_index_is_refused(self):
        """`int()` truncates, so this would place the buffer early and look normal afterwards."""
        batch = _batch()
        batch.start_index = np.asarray([0.5, 8.0])

        with pytest.raises(ValueError, match="whole samples"):
            batched_strain_to_chunks(batch)

    def test_a_buffer_off_the_lattice_is_refused(self):
        """A half-sample start would be interpolated instead of placed, silently."""
        batch = _batch()

        class _OffByHalf(_Grid):
            def time_of(self, index: int) -> float:
                return self.epoch + (index + 0.5) / self.sampling_frequency

        batch.grid = _OffByHalf()

        with pytest.raises(ValueError, match="off the output grid") as raised:
            batched_strain_to_chunks(batch)
        message = str(raised.value)
        # Both named, because the two predicates can disagree and which one refused is what tells
        # the reader where to look.
        assert "on-lattice" in message
        assert "placeable" in message

    def test_a_buffer_the_producer_accepts_but_the_consumer_cannot_place_is_refused(self):
        """Agreeing with the producer is not the same as agreeing with the consumer.

        gwmock-signal's ``is_on_lattice`` answers whether the buffer sits on the grid it was
        generated for; ``TimeSeries.inject`` applies its own tolerance when placing it. Those can
        disagree, and when they do the chunk is silently interpolated after this module has promised
        it would be placed verbatim -- which is exactly what happened at GPS 1577491296, 4000 Hz,
        sample 10, where the 2.29e-4 sample representation error was on-lattice for one and not for
        the other. Checking only the producer is how that went unnoticed.
        """
        batch = _batch(n_events=1, start_indices=(10,))

        class _GenerousGrid(_Grid):
            """Accepts everything, as a producer with a looser tolerance would."""

            def is_on_lattice(self, gps_time):
                return np.ones(np.atleast_1d(np.asarray(gps_time, dtype=float)).shape, dtype=bool)

            def time_of(self, index: int) -> float:
                # A quarter of a sample out: within no sane tolerance, but this grid says fine.
                return self.epoch + (index + 0.25) / self.sampling_frequency

        batch.grid = _GenerousGrid()

        with pytest.raises(ValueError, match="off the output grid") as raised:
            batched_strain_to_chunks(batch)
        message = str(raised.value)
        assert "on-lattice: True" in message, "the producer accepted it; the message should say so"
        assert "placeable within" in message
        assert "False" in message

    def test_reordered_detectors_are_refused(self):
        """Channel order carries detector identity, so a permutation misattributes signals."""
        batch = _batch()

        with pytest.raises(ValueError, match="expects"):
            batched_strain_to_chunks(batch, expected_detector_names=("ET3_SARD", "ET2_SARD", "ET1_SARD"))

    def test_matching_detectors_are_accepted(self):
        """The complement, so the check above cannot pass by rejecting everything."""
        chunks = batched_strain_to_chunks(_batch(), expected_detector_names=("ET1_SARD", "ET2_SARD", "ET3_SARD"))

        assert len(chunks) == 2

    def test_a_start_index_per_event_is_required(self):
        batch = _batch()
        batch.start_index = np.asarray([10])

        with pytest.raises(ValueError, match="start indices for 2 events"):
            batched_strain_to_chunks(batch)

    def test_detector_rows_must_match_the_names(self):
        batch = _batch()
        batch.detector_names = ("ET1_SARD", "ET2_SARD")

        with pytest.raises(ValueError, match="detector rows but 2 detector names"):
            batched_strain_to_chunks(batch)

    def test_a_non_batched_shape_is_refused(self):
        batch = _batch()
        batch.strain = np.zeros((3, 64))

        with pytest.raises(ValueError, match="Expected strain of shape"):
            batched_strain_to_chunks(batch)


class TestAgainstTheRealDeviceOutput:
    """That the stub above matches what gwmock-signal actually returns."""

    def test_a_real_batch_converts_and_places(self):
        """Runs the batched path for real, so the stub cannot drift from the true contract.

        Without this the guards could all pass against a stub whose field names or shapes no longer
        match ``BatchedDetectorStrain``.
        """
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        from gwmock_signal import Network, SamplingGrid, simulate_cbc_batch

        from gwmock.data.time_series.time_series import TimeSeries

        detectors = list(Network.from_preset("ET-Triangle-Sardinia").detector_names)
        grid = SamplingGrid(_EPOCH, _SAMPLING_FREQUENCY)
        batch = simulate_cbc_batch(
            "IMRPhenomD",
            detectors,
            sampling_frequency=_SAMPLING_FREQUENCY,
            minimum_frequency=30.0,
            parameters={
                "detector_frame_mass_1": [30.0, 25.0],
                "detector_frame_mass_2": [28.0, 24.0],
                "luminosity_distance": [400.0, 500.0],
                "inclination": [0.4, 0.2],
                "coa_phase": [0.0, 0.1],
                "right_ascension": [1.0, 1.2],
                "declination": [0.5, 0.3],
                "polarization_angle": [0.2, 0.1],
                "coa_time": [_EPOCH + 4.0, _EPOCH + 9.0],
                "spin_1z": [0.0, 0.0],
                "spin_2z": [0.0, 0.0],
            },
            output_grid=grid,
        )

        chunks = batched_strain_to_chunks(batch, expected_detector_names=("ET1_SARD", "ET2_SARD", "ET3_SARD"))

        assert len(chunks) == 2
        segment = TimeSeries(
            data=np.zeros((3, int(16 * _SAMPLING_FREQUENCY))),
            start_time=_EPOCH,
            sampling_frequency=_SAMPLING_FREQUENCY,
        )
        segment.inject_from_list(chunks)
        values = np.asarray(segment)
        assert np.count_nonzero(values), "the converted chunks put no signal into the segment"
        assert np.all(np.isfinite(values))


class TestSpillOverAcrossSegments:
    """A device chunk longer than one segment, placed through the real assembly path.

    This is the normal case rather than an edge one: a BNS inspiral at 20 Hz runs ~160 s, far longer
    than a segment. gwmock carries such a chunk forward by injecting what fits and caching the
    remainder, so the property that matters is that the converted chunks partition exactly -- every
    sample placed once, none dropped, none double-counted.

    The conversion tests above only ever used a chunk that fits inside one segment, so nothing
    covered this until now.
    """

    SEGMENT_SAMPLES = 32
    CHUNK_SAMPLES = 80

    def _batch_with_long_buffer(self):
        """A one-event batch whose buffer spans two and a half segments, with distinct samples.

        Samples count upward so a dropped or duplicated region is identifiable by value rather than
        only by length.
        """
        strain = np.arange(1, self.CHUNK_SAMPLES + 1, dtype=float).reshape(1, 1, self.CHUNK_SAMPLES)
        return _Batch(
            strain=strain,
            detector_names=("ET1_SARD",),
            start_index=np.asarray([0]),
            grid=_Grid(),
        )

    def test_the_chunk_is_partitioned_across_segments_without_loss(self):
        from gwmock.data.time_series.time_series import TimeSeries

        chunks = batched_strain_to_chunks(self._batch_with_long_buffer())
        pending = chunks
        placed: list[float] = []

        for segment_index in range(3):
            segment = TimeSeries(
                data=np.zeros((1, self.SEGMENT_SAMPLES)),
                start_time=_EPOCH + segment_index * self.SEGMENT_SAMPLES / _SAMPLING_FREQUENCY,
                sampling_frequency=_SAMPLING_FREQUENCY,
            )
            pending = segment.inject_from_list(pending)
            placed.extend(np.asarray(segment)[0].tolist())

        occupied = [value for value in placed if value != 0.0]
        expected = list(np.arange(1, self.CHUNK_SAMPLES + 1, dtype=float))

        assert occupied == expected, (
            "the chunk was not partitioned exactly across segments: samples were dropped, duplicated, or reordered"
        )
        assert len(pending) == 0, "the chunk should be fully consumed after three segments"

    def test_nothing_is_double_counted_where_two_chunks_overlap(self):
        """Two events overlapping in time must sum, not replace or double-place.

        Superposition is the whole point of an injection, and it is the property most easily broken
        by a placement bug that looks correct for a single chunk.
        """
        from gwmock.data.time_series.time_series import TimeSeries

        batch = _Batch(
            strain=np.ones((2, 1, 16)),
            detector_names=("ET1_SARD",),
            start_index=np.asarray([4, 8]),
            grid=_Grid(),
        )
        chunks = batched_strain_to_chunks(batch)
        segment = TimeSeries(
            data=np.zeros((1, self.SEGMENT_SAMPLES)),
            start_time=_EPOCH,
            sampling_frequency=_SAMPLING_FREQUENCY,
        )

        segment.inject_from_list(chunks)

        # Chunk one covers samples 4..19, chunk two covers 8..23, so 8..19 is the overlap.
        values = np.asarray(segment)[0]
        assert values[3] == pytest.approx(0.0), "nothing reaches sample 3"
        assert values[4] == pytest.approx(1.0), "only the first chunk covers sample 4"
        assert values[8] == pytest.approx(2.0), "both chunks cover sample 8 and must sum"
        assert values[19] == pytest.approx(2.0), "sample 19 is the last of the overlap"
        assert values[20] == pytest.approx(1.0), "only the second chunk covers sample 20"
        assert values[23] == pytest.approx(1.0), "the second chunk ends at sample 23"
        assert values[24] == pytest.approx(0.0), "nothing reaches sample 24"


class TestTheParameterAllowList:
    """What the batched path claims to read, checked against what gwmock-signal actually reads.

    The list is written by hand from the library's contract, so the failure mode is drift. Two
    directions, and they fail differently: a parameter present but unread is a silent drop, and a
    parameter read but absent is a rejected configuration that should have worked. Both happened --
    ``f_ref`` was the first and the in-plane spins were the second -- because nothing checked.

    These tests read the library rather than a second hand-written copy of the same list, so a
    gwmock-signal bump that renames or adds a parameter fails here instead of in a user's run.
    """

    def test_the_precessing_spin_components_are_accepted(self):
        """Ripple's batch resolver reads all six for IMRPhenomPv2, XP and XPHM.

        Omitting the in-plane four rejected every precessing configuration, which is also what the
        bundled ET presets use.
        """
        spins = {f"spin_{body}{axis}": 0.1 for body in (1, 2) for axis in ("x", "y", "z")}

        require_batched_parameters_supported(spins)

    def test_every_in_plane_spin_ripple_reads_is_in_the_list(self):
        """Anchored on the library's own source rather than on this test's opinion."""
        ripple = pytest.importorskip("gwmock_signal.waveform.backends.ripple")
        source = inspect.getsource(ripple)

        read_by_ripple = set(re.findall(r'"(spin_[12][xyz])"', source))

        assert read_by_ripple, "the spin names could not be found; this test has stopped checking anything"
        assert read_by_ripple <= BATCHED_PARAMETERS, (
            f"ripple reads {sorted(read_by_ripple - BATCHED_PARAMETERS)} but the batched path "
            f"rejects them, so precessing configurations fail"
        )

    def test_f_ref_is_not_a_per_event_parameter(self):
        """It configures the backend. In the parameter mapping it would be read by nobody."""
        assert "f_ref" not in BATCHED_PARAMETERS

    def test_gwmock_signal_still_agrees_that_f_ref_belongs_to_the_backend(self):
        """The anchor for routing it to the constructor. If this changes, the routing is wrong.

        Asserted against gwmock-signal's own reserved-argument table rather than against a comment
        here, so a library that starts accepting ``f_ref`` per event fails this instead of silently
        making the forwarding redundant.
        """
        ripple = pytest.importorskip("gwmock_signal.waveform.backends.ripple")

        assert "f_ref" in ripple._RESERVED_WAVEFORM_ARGUMENTS

    def test_f_ref_is_still_accepted_from_the_configuration(self):
        """Rejecting it would be the opposite error: the per-event path takes it in this key."""
        require_batched_parameters_supported({"f_ref": 20.0})

    def test_an_unreadable_argument_is_still_refused(self):
        """Guards the widening: the list must not have become permissive."""
        with pytest.raises(ValueError, match="cannot apply these waveform-arguments"):
            require_batched_parameters_supported({"not_a_waveform_parameter": 1.0})
