"""Injection of one time series into another, with support for sub-sample time offsets.

These are :mod:`gwmock_data.injection`'s, shared by the gwmock packages and re-exported here.
"""

from __future__ import annotations

from gwmock_data.injection import (
    alignment_tolerance,
    inject,
    is_aligned,
    measure_content_before,
    units_are_compatible,
)

__all__ = ["alignment_tolerance", "inject", "is_aligned", "measure_content_before", "units_are_compatible"]
