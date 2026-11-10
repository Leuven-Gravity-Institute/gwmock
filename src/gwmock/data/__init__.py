"""Initialization code for the gwmock.data package."""

from __future__ import annotations

from gwmock.data.time_series.spilled import SpilledTimeSeries
from gwmock.data.time_series.time_series import TimeSeries
from gwmock.data.time_series.time_series_list import TimeSeriesList

__all__ = ["SpilledTimeSeries", "TimeSeries", "TimeSeriesList"]
