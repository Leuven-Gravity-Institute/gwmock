"""Unit tests for TimeSeriesList class."""

from __future__ import annotations

import json

import numpy as np
import pytest
from astropy.units import Quantity

from gwmock.data.serialize.decoder import Decoder
from gwmock.data.serialize.encoder import Encoder
from gwmock.data.time_series.time_series import TimeSeries
from gwmock.data.time_series.time_series_list import TimeSeriesList


@pytest.fixture
def sample_ts():
    """Create a sample TimeSeries for testing."""
    data = np.random.randn(2, 100)  # 2 channels, 100 samples
    start_time = 0
    sampling_frequency = 100.0
    return TimeSeries(data, start_time, sampling_frequency)


@pytest.fixture
def sample_ts2():
    """Create another sample TimeSeries for testing."""
    data = np.random.randn(1, 50)  # 1 channel, 50 samples
    start_time = Quantity(10, unit="s")
    sampling_frequency = Quantity(200.0, unit="Hz")
    return TimeSeries(data, start_time, sampling_frequency)


@pytest.fixture
def sample_ts3():
    """Create a third sample TimeSeries for testing."""
    data = np.random.randn(3, 75)  # 3 channels, 75 samples
    start_time = 20
    sampling_frequency = 150.0
    return TimeSeries(data, start_time, sampling_frequency)


class TestTimeSeriesList:
    """Test suite for TimeSeriesList class."""

    def test_init_empty(self):
        """Test initialization with no arguments."""
        ts_list = TimeSeriesList()
        assert len(ts_list) == 0

    def test_init_with_list(self, sample_ts):
        """Test initialization with a list of TimeSeries."""
        ts_list = TimeSeriesList([sample_ts])
        assert len(ts_list) == 1
        assert ts_list[0] is sample_ts

    def test_init_with_invalid_items(self):
        """Test initialization with invalid items raises TypeError."""
        with pytest.raises(TypeError, match="All items must be MultichannelTimeSeries instances"):
            TimeSeriesList([1, 2, 3])

    def test_validate_items_valid(self, sample_ts, sample_ts2):
        """Test _validate_items with valid TimeSeries objects."""
        ts_list = TimeSeriesList()
        ts_list._validate_items([sample_ts, sample_ts2])  # Should not raise

    def test_validate_items_invalid(self):
        """Test _validate_items with invalid items raises TypeError."""
        ts_list = TimeSeriesList()
        with pytest.raises(TypeError, match="All items must be MultichannelTimeSeries instances"):
            ts_list._validate_items([1])

    def test_setitem_int(self, sample_ts, sample_ts2):
        """Test setting item by integer index."""
        ts_list = TimeSeriesList([sample_ts])
        ts_list[0] = sample_ts2
        assert ts_list[0] is sample_ts2

    def test_setitem_int_invalid(self, sample_ts):
        """Test setting item by integer index with invalid value raises TypeError."""
        ts_list = TimeSeriesList([sample_ts])
        with pytest.raises(TypeError, match="Value must be a MultichannelTimeSeries instance"):
            ts_list[0] = 1

    def test_setitem_slice(self, sample_ts, sample_ts2, sample_ts3):
        """Test setting items by slice."""
        ts_list = TimeSeriesList([sample_ts, sample_ts2])
        ts_list[0:2] = [sample_ts3, sample_ts]
        assert ts_list[0] is sample_ts3
        assert ts_list[1] is sample_ts

    def test_setitem_slice_invalid(self, sample_ts):
        """Test setting items by slice with invalid values raises TypeError."""
        ts_list = TimeSeriesList([sample_ts])
        with pytest.raises(TypeError, match="All items must be MultichannelTimeSeries instances"):
            ts_list[0:1] = [1]

    def test_setitem_invalid_index_type(self, sample_ts):
        """Test setting item with invalid index type raises TypeError."""
        ts_list = TimeSeriesList([sample_ts])
        with pytest.raises(TypeError, match="Index must be an int or slice"):
            ts_list["invalid"] = sample_ts

    def test_getitem_int(self, sample_ts):
        """Test getting item by integer index."""
        ts_list = TimeSeriesList([sample_ts])
        assert ts_list[0] is sample_ts

    def test_getitem_slice(self, sample_ts, sample_ts2):
        """Test getting items by slice."""
        ts_list = TimeSeriesList([sample_ts, sample_ts2])
        result = ts_list[0:1]
        assert isinstance(result, list)
        assert result[0] is sample_ts

    def test_len(self, sample_ts, sample_ts2):
        """Test length of TimeSeriesList."""
        ts_list = TimeSeriesList([sample_ts, sample_ts2])
        expected_length = 2
        assert len(ts_list) == expected_length

    def test_iter(self, sample_ts, sample_ts2):
        """Test iteration over TimeSeriesList."""
        ts_list = TimeSeriesList([sample_ts, sample_ts2])
        items = list(ts_list)
        assert items == [sample_ts, sample_ts2]

    def test_append(self, sample_ts, sample_ts2):
        """Test appending a TimeSeries."""
        ts_list = TimeSeriesList([sample_ts])
        ts_list.append(sample_ts2)
        expected_length = 2
        assert len(ts_list) == expected_length
        assert ts_list[1] is sample_ts2

    def test_append_invalid(self, sample_ts):
        """Test appending invalid value raises TypeError."""
        ts_list = TimeSeriesList([sample_ts])
        with pytest.raises(TypeError, match="Value must be a MultichannelTimeSeries instance"):
            ts_list.append(1)

    def test_extend(self, sample_ts, sample_ts2, sample_ts3):
        """Test extending with iterable of TimeSeries."""
        ts_list = TimeSeriesList([sample_ts])
        ts_list.extend([sample_ts2, sample_ts3])
        expected_length = 3
        assert len(ts_list) == expected_length
        assert ts_list[1] is sample_ts2
        assert ts_list[2] is sample_ts3

    def test_extend_invalid(self, sample_ts):
        """Test extending with invalid values raises TypeError."""
        ts_list = TimeSeriesList([sample_ts])
        with pytest.raises(TypeError, match="All items must be MultichannelTimeSeries instances"):
            ts_list.extend([1])

    def test_insert(self, sample_ts, sample_ts2):
        """Test inserting a TimeSeries at index."""
        ts_list = TimeSeriesList([sample_ts])
        ts_list.insert(0, sample_ts2)
        expected_length = 2
        assert len(ts_list) == expected_length
        assert ts_list[0] is sample_ts2
        assert ts_list[1] is sample_ts

    def test_insert_invalid(self, sample_ts):
        """Test inserting invalid value raises TypeError."""
        ts_list = TimeSeriesList([sample_ts])
        with pytest.raises(TypeError, match="Value must be a MultichannelTimeSeries instance"):
            ts_list.insert(0, 1)

    def test_pop(self, sample_ts, sample_ts2):
        """Test popping last item."""
        ts_list = TimeSeriesList([sample_ts, sample_ts2])
        popped = ts_list.pop()
        assert popped is sample_ts2
        assert len(ts_list) == 1

    def test_pop_index(self, sample_ts, sample_ts2):
        """Test popping item at specific index."""
        ts_list = TimeSeriesList([sample_ts, sample_ts2])
        popped = ts_list.pop(0)
        assert popped is sample_ts
        assert len(ts_list) == 1

    def test_to_json_dict(self, sample_ts):
        """Test converting to JSON dict."""
        ts_list = TimeSeriesList([sample_ts])
        data = ts_list.to_json_dict()
        assert data["__type__"] == "TimeSeriesList"
        assert "data" in data
        assert data["data"] == [sample_ts]

    def test_from_json_dict(self, sample_ts):
        """Test reconstructing from JSON dict."""
        data = {"__type__": "TimeSeriesList", "data": [sample_ts]}
        ts_list = TimeSeriesList.from_json_dict(data)
        assert len(ts_list) == 1
        assert ts_list[0] is sample_ts

    def test_from_json_dict_missing_key(self):
        """Test from_json_dict with missing key raises ValueError."""
        with pytest.raises(ValueError, match="Missing required key in JSON dict"):
            TimeSeriesList.from_json_dict({})

    def test_from_json_dict_invalid_item(self):
        """Test from_json_dict with invalid item raises TypeError."""
        data = {"__type__": "TimeSeriesList", "data": [1]}
        with pytest.raises(TypeError, match="Invalid item in MultichannelTimeSeriesList JSON data"):
            TimeSeriesList.from_json_dict(data)

    def test_repr(self, sample_ts):
        """Test string representation."""
        ts_list = TimeSeriesList([sample_ts])
        repr_str = repr(ts_list)
        assert "TimeSeriesList" in repr_str
        assert str(sample_ts) in repr_str

    def test_json_serialization_round_trip(self, sample_ts, sample_ts2):
        """Test full JSON serialization round-trip using Encoder and Decoder."""
        original_ts_list = TimeSeriesList([sample_ts, sample_ts2])

        # Serialize to JSON string
        json_str = json.dumps(original_ts_list, cls=Encoder)

        # Deserialize back to TimeSeriesList
        deserialized_ts_list = json.loads(json_str, cls=Decoder)

        # Assert that the deserialized object is equal to the original
        assert len(deserialized_ts_list) == len(original_ts_list)
        for original, deserialized in zip(original_ts_list, deserialized_ts_list, strict=False):
            assert original == deserialized  # Assuming TimeSeries has __eq__ method

    def test_json_serialization_includes_type(self, sample_ts):
        """Test that JSON serialization includes the __type__ key."""
        ts_list = TimeSeriesList([sample_ts])
        json_str = json.dumps(ts_list, cls=Encoder)
        json_dict = json.loads(json_str)
        assert "__type__" in json_dict
        assert json_dict["__type__"] == "TimeSeriesList"
