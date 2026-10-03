"""Unit tests for the TimeSeries class."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.units import Quantity
from gwmock_signal.projection.resampling import resample_uniform_sinc

from gwmock.data.time_series.time_series import TimeSeries
from gwmock.data.time_series.time_series_list import TimeSeriesList


@pytest.fixture
def sample_timeseries() -> TimeSeries:
    """Fixture for a sample TimeSeries instance."""
    np.random.seed(42)
    data = np.random.rand(2, 1024)  # 2 channels, 1024 samples
    start_time = Quantity(1234567890, unit="s")
    sampling_frequency = Quantity(4096, unit="Hz")
    return TimeSeries(data=data, start_time=start_time, sampling_frequency=sampling_frequency)


@pytest.fixture
def small_timeseries() -> TimeSeries:
    """Fixture for a smaller TimeSeries for injection tests."""
    data = np.ones((2, 512))  # 2 channels, 512 samples
    start_time = Quantity(1234567890.1, unit="s")  # Overlaps with sample
    sampling_frequency = Quantity(4096, unit="Hz")
    return TimeSeries(data=data, start_time=start_time, sampling_frequency=sampling_frequency)


class TestTimeSeriesInitialization:
    """Test TimeSeries initialization."""

    def test_init_with_valid_data(self, sample_timeseries: TimeSeries):
        """Test initialization with valid 2D array."""
        expected_num_of_channels = 2
        assert sample_timeseries.num_of_channels == expected_num_of_channels
        assert sample_timeseries.dtype == np.float64
        expected_shape = (2, 1024)
        assert len(sample_timeseries) == expected_shape[0]

    def test_init_with_int_start_time(self):
        """Test initialization with int start_time."""
        data = np.random.rand(1, 100)
        ts = TimeSeries(data, start_time=1000, sampling_frequency=100)
        assert ts.start_time == Quantity(1000, unit="s")

    def test_init_with_float_sampling_freq(self):
        """Test initialization with float sampling_frequency."""
        data = np.random.rand(1, 100)
        ts = TimeSeries(data, start_time=1000, sampling_frequency=100.0)
        assert ts.sampling_frequency == Quantity(100.0, unit="Hz")

    def test_init_raises_for_1d_data(self):
        """Test that 1D data raises ValueError."""
        data = np.random.rand(100)
        with pytest.raises(ValueError, match="Data must be a 2D"):
            TimeSeries(data, start_time=1000, sampling_frequency=100)


class TestTimeSeriesProperties:
    """Test TimeSeries properties."""

    def test_start_time_property(self, sample_timeseries: TimeSeries):
        """Test start_time property."""
        assert sample_timeseries.start_time == Quantity(1234567890, unit="s")

    def test_duration_property(self, sample_timeseries: TimeSeries):
        """Test duration property."""
        expected_duration = Quantity(1024 / 4096, unit="s")  # samples / freq
        assert sample_timeseries.duration == expected_duration

    def test_end_time_property(self, sample_timeseries: TimeSeries):
        """Test end_time property."""
        assert sample_timeseries.end_time == sample_timeseries.start_time + sample_timeseries.duration

    def test_sampling_frequency_property(self, sample_timeseries: TimeSeries):
        """Test sampling_frequency property."""
        assert sample_timeseries.sampling_frequency == Quantity(4096, unit="Hz")

    def test_time_array_property(self, sample_timeseries: TimeSeries):
        """Test time_array property."""
        times = sample_timeseries.time_array
        expected_len = 1024
        assert len(times) == expected_len
        assert times[0] == sample_timeseries.start_time


class TestTimeSeriesIndexing:
    """Test TimeSeries indexing and iteration."""

    def test_getitem(self, sample_timeseries: TimeSeries):
        """Test __getitem__."""
        channel = sample_timeseries[0]
        assert hasattr(channel, "value")  # GWpy TimeSeries

    def test_len(self, sample_timeseries: TimeSeries):
        """Test __len__."""
        expected_len = 2
        assert len(sample_timeseries) == expected_len

    def test_iter(self, sample_timeseries: TimeSeries):
        """Test __iter__."""
        channels = list(sample_timeseries)
        expected_num_of_channels = 2
        assert len(channels) == expected_num_of_channels


class TestTimeSeriesCrop:
    """Test TimeSeries crop method."""

    def test_crop_with_start_end(self, sample_timeseries: TimeSeries):
        """Test cropping with start and end times."""
        original_start = sample_timeseries.start_time
        original_duration = sample_timeseries.duration
        cropped = sample_timeseries.crop(
            start_time=original_start + Quantity(0.1, unit="s"), end_time=original_start + Quantity(0.2, unit="s")
        )
        assert cropped.start_time > original_start
        assert cropped.duration < original_duration

    def test_crop_returns_self(self, sample_timeseries: TimeSeries):
        """Test that crop returns self."""
        result = sample_timeseries.crop()
        assert result is sample_timeseries


class TestTimeSeriesInject:
    """Test TimeSeries inject method."""

    def test_inject_overlapping(self, sample_timeseries: TimeSeries, small_timeseries: TimeSeries):
        """Test injecting an overlapping TimeSeries."""
        original_value = sample_timeseries[0].value[512]  # Middle sample
        sample_timeseries.inject(small_timeseries)

        # Check that injection modified the data (assuming small_timeseries has ones)
        assert sample_timeseries[0].value[512] != original_value

    def test_inject_mismatched_channels_raises(self, sample_timeseries: TimeSeries):
        """Test that mismatched channel count raises ValueError."""
        wrong_channels = TimeSeries(
            np.ones((3, 100)),  # 3 channels vs 2
            start_time=sample_timeseries.start_time,
            sampling_frequency=sample_timeseries.sampling_frequency,
        )
        with pytest.raises(ValueError, match="Number of channels"):
            sample_timeseries.inject(wrong_channels)

    def test_inject_extending_beyond(self, sample_timeseries: TimeSeries):
        """Test injecting a TimeSeries that extends beyond the end."""
        extending_ts = TimeSeries(
            np.ones((2, 100)),
            start_time=sample_timeseries.end_time - Quantity(50 / 4096, unit="s"),  # Overlaps end
            sampling_frequency=sample_timeseries.sampling_frequency,
        )
        remaining = sample_timeseries.inject(extending_ts)
        assert remaining is not None
        assert isinstance(remaining, TimeSeries)


class TestTimeSeriesInjectFromList:
    """Test TimeSeries inject_from_list method."""

    def test_inject_from_list(self, sample_timeseries: TimeSeries, small_timeseries: TimeSeries):
        """Test injecting from a TimeSeriesList."""
        ts_list = TimeSeriesList([small_timeseries])
        remaining_list = sample_timeseries.inject_from_list(ts_list)
        assert isinstance(remaining_list, TimeSeriesList)


class TestTimeSeriesSerialization:
    """Test TimeSeries serialization."""

    def test_to_json_dict(self, sample_timeseries: TimeSeries):
        """Test to_json_dict produces correct structure."""
        data = sample_timeseries.to_json_dict()
        assert data["__type__"] == "TimeSeries"
        assert "data" in data
        assert "start_time" in data
        assert "start_time_unit" in data
        assert "sampling_frequency" in data
        assert "sampling_frequency_unit" in data
        # An ndarray, not a list of lists: the encoder base64s an array and writes JSON text
        # numbers for a list, 10.7 bytes per sample against 35.9. Spillover chunks now go into
        # checkpoints, where measured on a 1000 s three-detector tail that is 131 MB and 1.1 s
        # against 44.1 MB and 9.02 s per 100 s of it. What matters to a consumer is the shape.
        assert isinstance(data["data"], np.ndarray)
        assert data["data"].shape[0] == sample_timeseries.num_of_channels

    def test_from_json_dict_round_trip(self, sample_timeseries: TimeSeries):
        """Test round-trip serialization."""
        json_data = sample_timeseries.to_json_dict()
        reconstructed = TimeSeries.from_json_dict(json_data)
        assert reconstructed.num_of_channels == sample_timeseries.num_of_channels
        assert reconstructed.start_time == sample_timeseries.start_time
        assert reconstructed.sampling_frequency == sample_timeseries.sampling_frequency
        np.testing.assert_array_equal(reconstructed[0].value, sample_timeseries[0].value)


class TestInjectOffLatticeResampling:
    """An off-lattice chunk injected through the wrapper is resampled with the windowed-sinc kernel.

    The wrapper used to pre-resample such a chunk onto the segment grid by linear interpolation,
    so the per-channel ``inject`` only ever saw an aligned chunk and never reached the kernel
    gwmock-signal's ``inject_strain`` uses. At 0.8 x Nyquist that left an interior error of ~0.63
    of peak against the analytic signal.
    """

    def test_matches_the_shared_kernel_and_the_analytic_signal(self):
        """Pinned to the kernel itself and to the analytic signal, on every channel."""
        sampling_frequency, start = 4096.0, 1e9
        # 1000.375 samples is 8003 * 2**-15 s, exactly representable beside a GPS epoch of 1e9, so
        # the offset `inject` measures is the nominal one and both checks below can be tight.
        length, shift, frequency = 2048, 1000.375, 0.8 * sampling_frequency / 2
        envelope = np.hanning(length)
        values = envelope * np.sin(2 * np.pi * frequency * np.arange(length) / sampling_frequency)
        segment = TimeSeries(data=np.zeros((2, 4096)), start_time=start, sampling_frequency=sampling_frequency)
        chunk = TimeSeries(
            data=np.stack([values, -2.0 * values]),
            start_time=start + shift / sampling_frequency,
            sampling_frequency=sampling_frequency,
        )

        assert segment.inject(chunk) is None

        covered = np.arange(1001, 3048)
        positions = covered - shift
        expected = resample_uniform_sinc(values, positions)
        interior = (positions > 200) & (positions < length - 200)
        analytic = (0.5 - 0.5 * np.cos(2 * np.pi * positions / (length - 1))) * np.sin(
            2 * np.pi * frequency * positions / sampling_frequency
        )
        for channel, scale in enumerate((1.0, -2.0)):
            written = np.asarray(segment[channel])
            np.testing.assert_allclose(written[covered], scale * expected, rtol=0.0, atol=1e-15)
            assert np.max(np.abs(written[covered] - scale * analytic)[interior]) < 1e-9


class TestInjectBoundaryOverflow:
    """A chunk crossing the segment boundary must hand its tail back, aligned or not.

    ``TimeSeriesMixin.simulate`` carries a signal across segments by caching whatever
    ``inject``/``inject_from_list`` returns. The interpolation path rebinds ``other`` to samples
    drawn from the segment's own time array, which cannot extend past the segment end, so the
    overflow check compared against the wrong series and the tail was dropped -- silently
    truncating any misaligned signal at a segment boundary.

    This was latent while the alignment test used a relative tolerance, because long segments always
    took the aligned branch. Fixing that tolerance made the interpolation branch reachable and the
    bug live, which is why it belongs with that change.
    """

    SAMPLING_FREQUENCY = 4096.0
    START = 1e9
    SEGMENT_SAMPLES = 1000
    CHUNK_SAMPLES = 400
    CHUNK_START_SAMPLE = 900

    def _segment(self):
        return TimeSeries(
            data=np.zeros((1, self.SEGMENT_SAMPLES)),
            start_time=self.START,
            sampling_frequency=self.SAMPLING_FREQUENCY,
        )

    def _chunk(self, offset_samples: float):
        return TimeSeries(
            data=np.ones((1, self.CHUNK_SAMPLES)),
            start_time=self.START + offset_samples / self.SAMPLING_FREQUENCY,
            sampling_frequency=self.SAMPLING_FREQUENCY,
        )

    @pytest.mark.parametrize(
        ("offset_samples", "description", "expected_tail"),
        [
            (900.0, "aligned", 300),
            # Resampled onto the segment lattice first: the chunk covers grid points 901..1299, and
            # the tail is the 300 of those from the boundary at 1000 onward.
            (900.5, "half a sample misaligned", 300),
        ],
    )
    def test_the_overflow_is_returned(self, offset_samples: float, description: str, expected_tail: int):
        """Both branches must return the part of the chunk beyond the segment."""
        segment = self._segment()

        remaining = segment.inject(self._chunk(offset_samples))

        assert remaining is not None, (
            f"a {description} chunk crossing the segment boundary returned no remainder, so its "
            f"tail was discarded instead of being carried into the next segment"
        )
        assert len(np.asarray(remaining)[0]) == expected_tail

    def test_a_misaligned_tail_is_returned_on_the_segment_lattice(self):
        """The tail must start exactly where the next contiguous segment does.

        The chunk is resampled once, as a whole, onto the segment's lattice; the tail is the part of
        that past the boundary. Starting it anywhere else would send it through the kernel a second
        time in the next segment, with no context on its left.
        """
        segment = self._segment()

        remaining = segment.inject(self._chunk(900.5))

        assert remaining is not None
        offset = (float(remaining.start_time.value) - float(segment.end_time.value)) * self.SAMPLING_FREQUENCY
        assert abs(offset) < 1e-3, f"the tail starts {offset} samples off the next segment's grid"

    def test_a_misaligned_signal_is_continuous_across_the_boundary(self):
        """A signal crossing segments must be resampled as accurately at the boundary as inside one.

        The windowed-sinc kernel needs about half its taps of context on either side. Resampling the
        chunk segment by segment cuts that context at every boundary, and the error there was
        measured at the signal's own peak; resampling the whole chunk once leaves none.
        """
        sampling_frequency = self.SAMPLING_FREQUENCY
        segment_samples, length, frequency = 4096, 6000, 50.0
        # 1000.375 samples is 8003 * 2**-15 s, exactly representable beside the epoch.
        shift = 1000.375
        values = np.hanning(length) * np.sin(2 * np.pi * frequency * np.arange(length) / sampling_frequency)
        segments = [
            TimeSeries(
                data=np.zeros((1, segment_samples)),
                start_time=self.START + k * segment_samples / sampling_frequency,
                sampling_frequency=sampling_frequency,
            )
            for k in range(2)
        ]
        chunk = TimeSeries(
            data=values[None, :],
            start_time=self.START + shift / sampling_frequency,
            sampling_frequency=sampling_frequency,
        )

        for segment in segments:
            chunk = segment.inject(chunk)
        assert chunk is None

        written = np.concatenate([np.asarray(segment[0]) for segment in segments])
        positions = np.arange(len(written)) - shift
        near_boundary = slice(segment_samples - 64, segment_samples + 64)
        analytic = (0.5 - 0.5 * np.cos(2 * np.pi * positions / (length - 1))) * np.sin(
            2 * np.pi * frequency * positions / sampling_frequency
        )
        assert np.max(np.abs(written - analytic)[near_boundary]) < 1e-9

    @pytest.mark.parametrize("offset_samples", [900.0, 900.5])
    def test_the_caller_s_chunk_is_not_modified(self, offset_samples: float):
        """Injection must not truncate the chunk it was handed.

        ``crop`` rewrites ``_data`` in place and returns ``self``, so cropping the supplied chunk to
        produce the remainder would hand the caller its own object back, shortened.
        ``inject_from_list`` walks a caller-provided list, so that would mutate every element that
        overflows.
        """
        segment = self._segment()
        chunk = self._chunk(offset_samples)
        original_length = len(np.asarray(chunk)[0])
        original_start = float(chunk.start_time.value)

        remaining = segment.inject(chunk)

        assert len(np.asarray(chunk)[0]) == original_length, "the caller's chunk was truncated"
        assert float(chunk.start_time.value) == original_start, "the caller's chunk was moved"
        assert remaining is not chunk, "the remainder must not be the caller's own object"

    def test_the_tail_keeps_the_chunk_s_metadata_and_channel_identity(self):
        """A tail is the same signal continuing, so it must arrive described the same way.

        The remainder is rebuilt as a new TimeSeries to avoid mutating the caller's chunk. Building
        it from data, start time and rate alone silently drops ``metadata`` -- which carries
        ``injection_parameters`` -- along with each channel's name and unit. Any injection long
        enough to cross a segment boundary would then lose its provenance in the next segment, and a
        long inspiral is exactly the case that crosses one.
        """
        segment = self._segment()
        chunk = self._chunk(900.0)
        chunk.metadata.update({"injection_parameters": {"coa_time": 1234.5}})
        chunk[0].name = "H1:STRAIN"

        remaining = segment.inject(chunk)

        assert remaining is not None
        assert remaining.metadata.get("injection_parameters") == {"coa_time": 1234.5}, (
            "the tail lost the injection provenance it was carrying"
        )
        assert remaining[0].name == "H1:STRAIN", "the tail lost its channel identity"

    def test_a_chunk_inside_the_segment_returns_nothing(self):
        """The complement, so the test above cannot pass by always returning a remainder."""
        segment = self._segment()

        assert segment.inject(self._chunk(100.5)) is None
