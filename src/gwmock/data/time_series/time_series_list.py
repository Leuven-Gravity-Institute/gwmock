"""An ordered list of multichannel time series.

:class:`TimeSeriesList` is :class:`gwmock_data.MultichannelTimeSeriesList`, shared by the gwmock packages
and re-exported here under the name gwmock has always used, so ``gwmock.data.TimeSeriesList`` and the
checkpoint records that name it (``"__type__": "TimeSeriesList"``) keep working.
"""

from __future__ import annotations

from gwmock_data.multichannel_series import MultichannelTimeSeriesList as TimeSeriesList

__all__ = ["TimeSeriesList"]
