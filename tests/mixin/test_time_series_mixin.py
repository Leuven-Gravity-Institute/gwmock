"""Unit tests for TimeSeriesMixin."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from astropy.units import Quantity

try:
    from gwpy.timeseries import TimeSeries as GWPyTimeSeries
except ModuleNotFoundError as exc:  # Python 3.15: GWpy does not install yet; its tests are skipped
    if exc.name != "gwpy":
        raise
    GWPyTimeSeries = None

from gwmock.data.time_series.time_series import TimeSeries
from gwmock.data.time_series.time_series_list import TimeSeriesList
from gwmock.mixin.time_series import TimeSeriesMixin
from gwmock.simulator.base import Simulator


class MockTimeSeriesSimulator(TimeSeriesMixin, Simulator):
    """Mock simulator for testing TimeSeriesMixin."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generated_chunks = TimeSeriesList()
        self.saved_batches = []

    def _simulate(self, *args, **kwargs) -> TimeSeriesList:
        """Generate dummy time series data."""
        # Generate random data
        data = np.random.randn(self.num_of_channels, int(self.duration * self.sampling_frequency))
        ts = TimeSeries(data, self.start_time, self.sampling_frequency)
        self.generated_chunks.append(ts)
        return TimeSeriesList([ts])

    def save_batch(self, batch, file_name, overwrite=False, **kwargs):
        """Mock save batch method."""
        self.saved_batches.append((batch, file_name, kwargs))

    @property
    def metadata(self) -> dict:
        """Mock metadata method."""
        return super().metadata


class TestTimeSeriesMixin:
    """Test suite for TimeSeriesMixin."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        simulator = MockTimeSeriesSimulator()
        assert simulator.start_time == Quantity(0, unit="s")
        assert simulator.duration == Quantity(4, unit="s")
        assert simulator.sampling_frequency == Quantity(4096, unit="Hz")
        assert simulator.num_of_channels == 1
        assert simulator.dtype == np.float64

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        num_of_channels = 2
        simulator = MockTimeSeriesSimulator(
            start_time=100, duration=8, sampling_frequency=2048, num_of_channels=num_of_channels, dtype=np.float32
        )
        assert simulator.start_time == Quantity(100, unit="s")
        assert simulator.duration == Quantity(8, unit="s")
        assert simulator.sampling_frequency == Quantity(2048, unit="Hz")
        assert simulator.num_of_channels == num_of_channels
        assert simulator.dtype == np.float32

    def test_total_duration_derived_message_is_not_misleading(self, caplog):
        """Deriving total_duration from max_samples must not claim it was 'not set'.

        Regression for #133: orchestration resolves a config total_duration into
        max_samples and forwards only that, so this simulator receives no
        total_duration and derives it here. The log must report the derived value
        factually rather than asserting the user never set it.
        """
        with caplog.at_level(logging.INFO, logger="gwmock"):
            simulator = MockTimeSeriesSimulator(max_samples=3, duration=2, sampling_frequency=100)

        assert simulator.total_duration == Quantity(6, unit="s")
        assert "total_duration not set" not in caplog.text
        assert "Resolved total_duration to 6" in caplog.text

    def test_total_duration_explicit_reports_set(self, caplog):
        """An explicitly provided total_duration is reported as set and pins max_samples."""
        with caplog.at_level(logging.INFO, logger="gwmock"):
            simulator = MockTimeSeriesSimulator(duration=2, total_duration=8, sampling_frequency=100)

        assert simulator.total_duration == Quantity(8, unit="s")
        assert "Total duration set to" in caplog.text
        assert "total_duration not set" not in caplog.text

    def test_init_with_detectors(self):
        """Test initialization with detectors list."""
        detectors = ["H1", "L1"]
        simulator = MockTimeSeriesSimulator(detectors=detectors)
        assert simulator.num_of_channels == len(detectors)

    def test_init_detectors_mismatch(self):
        """Test initialization with mismatched detectors and num_of_channels."""
        with pytest.raises(ValueError, match="Number of detectors does not match num_of_channels"):
            MockTimeSeriesSimulator(num_of_channels=3, detectors=["H1", "L1"])

    def test_simulate_single_channel(self):
        """Test simulate method with single channel."""
        duration = 1
        sampling_frequency = 100
        num_of_data_points = duration * sampling_frequency
        simulator = MockTimeSeriesSimulator(duration=duration, sampling_frequency=sampling_frequency)
        result = simulator.simulate()

        assert isinstance(result, TimeSeries)
        assert len(result._data) == 1  # 1 Channel
        assert result._data[0].n_samples == num_of_data_points  # 100 samples
        assert result.start_time == 0
        assert result.sampling_frequency.value == sampling_frequency

    def test_simulate_multi_channel(self):
        """Test simulate method with multiple channels."""
        num_of_channels = 3
        duration = 1
        sampling_frequency = 100
        num_of_data_points = duration * sampling_frequency
        simulator = MockTimeSeriesSimulator(
            num_of_channels=num_of_channels, duration=duration, sampling_frequency=sampling_frequency
        )
        result = simulator.simulate()

        assert isinstance(result, TimeSeries)
        assert len(result._data) == num_of_channels  # 3 Channels
        assert result._data[0].n_samples == num_of_data_points  # 100 samples
        assert result._data[1].n_samples == num_of_data_points  # 100 samples
        assert result._data[2].n_samples == num_of_data_points  # 100 samples

    def test_simulate_continuity(self):
        """Test that multiple simulate calls maintain continuity via caching."""
        simulator = MockTimeSeriesSimulator(duration=1, sampling_frequency=100)

        # First simulate
        _result1 = simulator.simulate()
        assert len(simulator.cached_data_chunks) == 0  # No remaining chunks

        # Second simulate - should continue from where first left off
        simulator.start_time = 1  # Advance time
        result2 = simulator.simulate()

        # Check that times are sequential
        assert result2.start_time.value == 1
        # Since _simulate generates full duration, and inject handles overlap,
        # cached_data_chunks should be empty if no overlap

    def test_simulate_with_cached_data(self):
        """Test simulate with pre-existing cached data."""
        duration = 1
        sampling_frequency = 100
        simulator = MockTimeSeriesSimulator(duration=duration, sampling_frequency=sampling_frequency)

        # Add some cached data
        cached_data = np.ones((1, 50))
        cached_ts = TimeSeries(cached_data, 0, 100)
        simulator.cached_data_chunks = TimeSeriesList([cached_ts])

        result = simulator.simulate()

        # The cached data should be injected into the result
        # Since cached_ts starts at 0 and result at 0, it should be included
        assert len(result._data) == 1
        assert result._data[0].n_samples == duration * sampling_frequency

    def test_metadata(self):
        """Test metadata property."""
        simulator = MockTimeSeriesSimulator(duration=2, sampling_frequency=512, dtype=np.float32)
        metadata = simulator.metadata

        assert metadata["time_series"]["arguments"]["duration"] == Quantity(2, unit="s")
        assert metadata["time_series"]["arguments"]["sampling_frequency"] == Quantity(512, unit="Hz")
        assert metadata["time_series"]["arguments"]["dtype"] == str(np.float32)

    def test_iteration_protocol(self):
        """Test that the simulator works with iteration protocol."""
        simulator = MockTimeSeriesSimulator(max_samples=2, duration=1, sampling_frequency=100)

        batches = list(simulator)
        expected_batches = 2  # max_samples=2
        assert len(batches) == expected_batches
        assert all(isinstance(batch, TimeSeries) for batch in batches)
        expected_counter = 2
        assert simulator.counter == expected_counter

    def test_save_batch_can_be_called_on_generated_batch(self):
        """Test that save_batch can be called on a generated batch from iteration."""
        simulator = MockTimeSeriesSimulator(max_samples=1, duration=1, sampling_frequency=100)

        # Generate a batch via iteration (what the simulator does)
        batches = list(simulator)
        batch = batches[0]

        # Now test saving it (this is what the CLI would do)
        with patch.object(simulator, "save_batch") as mock_save:
            simulator.save_batch(batch, "test.gwf", overwrite=False)
            mock_save.assert_called_once_with(batch, "test.gwf", overwrite=False)

    def test_cli_compatibility(self):
        """Test compatibility with CLI expectations."""
        simulator = MockTimeSeriesSimulator(duration=1, sampling_frequency=100)

        # Simulate what CLI does: call simulate, then save_batch
        batch = simulator.simulate()
        assert isinstance(batch, TimeSeries)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.gwf"
            simulator.save_batch(batch, file_path, channel="TEST:STRAIN")
            assert len(simulator.saved_batches) == 1
            saved_batch, saved_path, saved_kwargs = simulator.saved_batches[0]
            assert saved_batch is batch
            assert saved_path == file_path
            assert saved_kwargs == {"channel": "TEST:STRAIN"}

    def test_state_persistence(self):
        """Test that simulator state is maintained correctly."""
        simulator = MockTimeSeriesSimulator(max_samples=3)

        # Simulate some iterations
        next(simulator)
        assert simulator.counter == 1

        next(simulator)
        expected_counter = 2
        assert simulator.counter == expected_counter

        # Check that TimeSeriesMixin state is separate from base state
        state = simulator.state
        assert "counter" in state
        # cached_data_chunks is not in state, as it's not a StateAttribute

    @pytest.mark.requires_gwpy
    @patch("gwpy.timeseries.TimeSeries.write")
    def test_save_data_gwf_success(self, mock_write):
        """Test save_data with valid custom TimeSeries and .gwf file."""
        simulator = MockTimeSeriesSimulator(duration=1, sampling_frequency=100, num_of_channels=1)

        # Create our custom TimeSeries with dummy data
        data_array = np.array([[1.0, 2.0, 3.0]])
        data = TimeSeries(data_array, start_time=0, sampling_frequency=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.gwf"
            simulator.save_data(data, file_path)

            # Check that GWPy write was called (internally via _save_gwf_data)
            mock_write.assert_called_once_with(str(file_path))

    @pytest.mark.requires_gwpy
    @patch("gwpy.timeseries.TimeSeries.write")
    def test_save_data_gwf_with_channel(self, mock_write):
        """Test save_data with custom TimeSeries converts and saves correctly."""
        simulator = MockTimeSeriesSimulator(duration=1, sampling_frequency=100, num_of_channels=1)

        # Create our custom TimeSeries with dummy data
        data_array = np.array([[1.0, 2.0, 3.0]])
        data = TimeSeries(data_array, start_time=0, sampling_frequency=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.gwf"
            simulator.save_data(data, file_path, channel="H1:STRAIN")

            # Check that GWPy write was called (internally via _save_gwf_data)
            mock_write.assert_called_once_with(str(file_path))

    @pytest.mark.requires_gwpy
    def test_save_data_invalid_data_type(self):
        """Test save_data raises error for invalid data type."""
        simulator = MockTimeSeriesSimulator()

        # Use GWPy TimeSeries directly, which is not accepted by save_data
        data = GWPyTimeSeries([1, 2, 3], sample_rate=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.gwf"
            with pytest.raises(AttributeError):
                # GWPy TimeSeries doesn't have num_of_channels attribute
                simulator.save_data(data, file_path)
