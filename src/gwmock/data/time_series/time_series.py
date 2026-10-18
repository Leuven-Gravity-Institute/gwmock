"""Time series data for several channels on one sample grid.

:class:`TimeSeries` is :class:`gwmock_data.MultichannelTimeSeries`, shared by the gwmock packages and
re-exported here under the name gwmock has always used, so ``gwmock.data.TimeSeries`` and the
checkpoint records that name it (``"__type__": "TimeSeries"``) keep working.
"""

from __future__ import annotations

from gwmock_data.multichannel_series import MultichannelTimeSeries as TimeSeries

__all__ = ["TimeSeries"]
