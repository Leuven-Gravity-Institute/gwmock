"""Module to handle injection of one TimeSeries into another, with support for time offsets."""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any

import numpy as np
from gwmock_data import TimeSeries
from gwmock_signal.projection.resampling import resample_uniform_sinc

logger = logging.getLogger("gwmock")

#: Multiple of the float64 spacing allowed as alignment error, in ULPs.
#:
#: 8, to match ``SamplingGrid.lattice_tolerance_samples`` in gwmock-signal, which reached the same
#: ULP-scaling design independently and with the same floor and ceiling. This is one decision about
#: numerical resolution encoded in two packages, and gwmock-signal's value is the older of the two,
#: so this defers to it rather than leaving the pair to disagree by a factor of two.
#:
#: It is comfortably above the largest error measured for a genuinely aligned chunk here (4.0e-4
#: samples at GPS 1.577e9, 4000 Hz) and, via the ceiling below, still far under the half sample that
#: would make everything look aligned.
_ALIGNMENT_SPACING_MARGIN = 8.0

#: Smallest tolerance ever used, for epochs near zero where the spacing is meaningless.
_MINIMUM_ALIGNMENT_TOLERANCE = 1e-9

#: Largest tolerance ever used, in samples.
#:
#: A fractional offset cannot exceed 0.5 samples, so a tolerance approaching that would classify
#: everything as aligned -- reintroducing the bug this machinery exists to prevent. 0.01 keeps a
#: hundredth of a sample detectable. Reaching this ceiling means float64 can no longer resolve
#: sample alignment at the given epoch and rate, which is worth saying out loud rather than
#: silently degrading.
_MAXIMUM_ALIGNMENT_TOLERANCE = 1e-2


def alignment_tolerance(
    reference_time: float,
    sampling_frequency: float,
    gps_times: object = None,
) -> float:
    """Return how far from a whole sample an offset may sit and still count as aligned.

    Derived from the floating-point spacing at *reference_time* rather than fixed. The offset is
    computed as ``(other_start - self_start) * sampling_frequency`` from two GPS-scale times, so an
    exactly-aligned chunk still carries error of order ``spacing(epoch) * sampling_frequency``: at
    GPS 1.6e9 and 4096 Hz that is ~1e-3 samples, and it grows with both the epoch and the rate.

    A fixed tolerance therefore cannot be right everywhere. 1e-3 covers GPS 1e9-1.8e9 at 512-16384
    Hz, but at GPS 1e10 and 2000 Hz the real error reaches ~1.7e-3 -- and a tolerance below the
    error means genuinely aligned chunks get interpolated, which is worse than the misclassification
    this whole check exists to avoid. Scaling with the spacing removes the domain assumption.

    Args:
        reference_time: Epoch the offset is measured from, in seconds.
        sampling_frequency: Sample rate in Hz.
        gps_times: Optional timestamps taking part in the comparison. The spacing is taken at the
            largest magnitude involved, not at the epoch alone, since a time later than the epoch
            has coarser resolution and it is the coarsest that bounds the error. Mirrors
            ``SamplingGrid.lattice_tolerance_samples`` in gwmock-signal.

    Returns:
        Tolerance in samples, clamped to :data:`_MINIMUM_ALIGNMENT_TOLERANCE` and
        :data:`_MAXIMUM_ALIGNMENT_TOLERANCE`.
    """
    magnitude = abs(float(reference_time))
    if gps_times is not None:
        times = np.asarray(gps_times, dtype=float)
        if times.size:
            magnitude = max(magnitude, float(np.max(np.abs(times))))
    derived = _ALIGNMENT_SPACING_MARGIN * float(np.spacing(magnitude)) * float(sampling_frequency)
    if derived > _MAXIMUM_ALIGNMENT_TOLERANCE:
        logger.warning(
            "Sample alignment cannot be resolved at epoch %s and %s Hz: the floating-point spacing "
            "implies an uncertainty of %.3g samples, so alignment decisions there are unreliable. "
            "Capping the tolerance at %s.",
            reference_time,
            sampling_frequency,
            derived,
            _MAXIMUM_ALIGNMENT_TOLERANCE,
        )
    return min(_MAXIMUM_ALIGNMENT_TOLERANCE, max(_MINIMUM_ALIGNMENT_TOLERANCE, derived))


def is_aligned(offset_samples: float, tolerance: float) -> bool:
    """Return whether *offset_samples* is a whole number of samples, within *tolerance*.

    Absolute, not relative. ``np.isclose``'s default tolerance is relative, and since a fractional
    offset can never exceed 0.5 samples, a relative test passes unconditionally once
    ``rtol * offset >= 0.5`` -- about 50,000 samples, or 12 s into a segment at 4096 Hz. Beyond that
    the interpolation branch was unreachable and every chunk was snapped to the nearest sample,
    displacing signals by up to half a sample.

    Args:
        offset_samples: Offset between two series' start times, in samples.
        tolerance: Permitted deviation in samples, from :func:`alignment_tolerance`.

    Returns:
        Whether the offset counts as aligned.
    """
    return bool(abs(offset_samples - round(offset_samples)) <= tolerance)


def measure_content_before(
    segment_start_time: float, sampling_frequency: float, chunk: Any
) -> tuple[int, float, float]:
    """Measure the part of *chunk* lying before *segment_start_time*, which injection discards.

    :func:`inject` crops a chunk to the target's span, so anything earlier is dropped. Forward
    overflow is different: it is returned as a tail and carried into the next segment. Backward
    overflow has nowhere to go, because the segments it belongs to have already been written.

    That mattered for every compact binary landing near a boundary while segments claimed an event by
    ``coa_time``, whose inspiral *precedes* it. Segments now claim by where the waveform starts, so
    what this measures is the residue: a waveform beginning before the run itself, or a placement
    that fell back to ``coa_time`` because the pre-coalescence duration could not be established.

    Args:
        segment_start_time: Start of the segment being injected into, in the chunk's time unit.
        sampling_frequency: Sample rate of the segment, in Hz.
        chunk: The time series about to be injected.

    Returns:
        A ``(samples, seconds, energy_fraction)`` triple describing what lies before the segment.
        ``seconds`` is the span of those samples, so it always agrees with ``samples``.
        ``energy_fraction`` is the share of the chunk's summed squares that is dropped -- unweighted
        strain-squared energy, ``0.0`` for a silent chunk. It is a **proxy, not an SNR loss**: a
        matched-filter figure needs a detector PSD and frequency-domain weighting, neither of which
        is available here. It is reported in preference to the sample fraction only because the two
        differ by more than a factor of two in ordinary cases, and the sample fraction is the more
        misleading of the pair.
    """
    times = np.asarray(chunk.time_array.value, dtype=float)
    if times.size == 0:
        return 0, 0.0, 0.0

    # Half a sample of slack. A tail carried from the previous segment starts exactly on this
    # boundary, and at GPS epochs the float64 spacing (~2.4e-7 s at 1.6e9) can put it a hair below
    # -- which would otherwise be reported as a whole dropped sample every single segment.
    #
    # Half a sample, rather than the ULP-scaled `alignment_tolerance` used for grid alignment,
    # because the question here is different: a sample nearer the boundary than half a sample period
    # rounds *to* the boundary, so counting it as dropped would be wrong however exact the arithmetic
    # was. The cost is that up to one genuinely discarded sample can go unreported, which is bounded
    # and negligible beside the multi-second losses this exists to surface.
    #
    # Measured margin: spacing(1.577e9) = 2.384e-07 s is 2.4e-4 samples at 1024 Hz and 3.9e-3 at
    # 16384 Hz -- 2048x and 128x under the threshold. It would take a sampling frequency of 2.1 MHz
    # for float64 spacing at a GPS epoch to reach half a sample, so the slack holds across every rate
    # this package supports.
    # `side="left"`, so a sample sitting exactly on the threshold is *not* counted as dropped.
    # That case is a rounding tie by construction -- exactly half a sample from the boundary --
    # and resolving it towards the boundary matches what the slack is for.
    n_before = int(np.searchsorted(times, segment_start_time - 0.5 / sampling_frequency, side="left"))
    if n_before <= 0:
        return 0, 0.0, 0.0

    data = np.atleast_2d(np.asarray(chunk, dtype=float))
    total = float(np.sum(np.square(data)))
    dropped = float(np.sum(np.square(data[:, :n_before])))
    fraction = dropped / total if total > 0.0 else 0.0
    # The span of the dropped samples, not the gap from the chunk's start to the boundary. The
    # two differ whenever the chunk ends before the boundary: 1024 samples sitting 10 s early
    # span 1 s, and the warning pairs this figure with the sample count as though they agreed.
    # Deriving it from the count also avoids subtracting two close GPS-scale times.
    return n_before, n_before / sampling_frequency, fraction


#: Units that carry no dimension: strain is a ratio of lengths, so it adds to a unit-less series.
_DIMENSIONLESS = frozenset({None, "", "dimensionless", "strain"})


def units_are_compatible(target_unit: str | None, other_unit: str | None) -> bool:
    """Whether samples in *other_unit* may be added to samples in *target_unit*.

    Equal units are compatible, and so are any two of: no unit, ``"dimensionless"`` and ``"strain"``
    -- gwmock's segments carry no unit and the chunks injected into them carry ``"strain"``. Any other
    pair is refused, as GWpy refused it, rather than summed under the target's unit: a missing unit is
    not a wildcard for a dimensional one.

    Args:
        target_unit: The unit of the series added into, or ``None``.
        other_unit: The unit of the series added, or ``None``.

    Returns:
        Whether the two may be summed.
    """
    if target_unit == other_unit:
        return True
    return target_unit in _DIMENSIONLESS and other_unit in _DIMENSIONLESS


def inject(timeseries: TimeSeries, other: TimeSeries, interpolate_if_offset: bool = True) -> TimeSeries:
    """Add *other* into *timeseries* where the two overlap, and return the sum.

    Placement is by sample index on *timeseries*' grid. The offset of *other*'s first sample is
    ``(other.t0 - timeseries.t0) * sample_rate`` samples; when that is a whole number to within
    :func:`alignment_tolerance`, the overlapping samples are added one for one. Samples of *other*
    outside *timeseries*' span are not placed: those before it are lost (the caller reports them,
    see :func:`measure_content_before`), and those after it are the caller's to carry forward.

    When the offset is fractional, *other* is resampled onto *timeseries*' grid at the grid points
    inside its own sample span, with the band-limited windowed-sinc kernel gwmock-signal projects
    with (``resample_uniform_sinc``), and those are added. Positions come from integer grid indices
    minus one offset rather than from per-sample GPS times, each of which carries float64 rounding
    of a sizeable fraction of a sample. The whole of *other* is the kernel's input, so samples just
    outside *timeseries* still give context at its edges.

    Args:
        timeseries: The series to add into. It is not changed.
        other: The series to add. Same sample rate as *timeseries*.
        interpolate_if_offset: Whether to resample a fractional offset; if not, *timeseries* is
            returned unchanged.

    Returns:
        A new series with *timeseries*' timing, name and unit, holding the sum.

    Raises:
        ValueError: If the sample rates or the units are incompatible (see
            :func:`units_are_compatible`), or either series is not one-dimensional.
    """
    if not units_are_compatible(timeseries.unit, other.unit):
        raise ValueError(f"Cannot inject a series in {other.unit!r} into one in {timeseries.unit!r}.")
    if other.sample_rate != timeseries.sample_rate:
        raise ValueError(
            f"Cannot inject a series sampled at {other.sample_rate} Hz into one sampled at {timeseries.sample_rate} Hz."
        )
    if timeseries.data.ndim != 1 or other.data.ndim != 1:
        raise ValueError("inject works on one channel at a time: both series must be one-dimensional.")

    n_target = timeseries.n_samples
    n_other = other.n_samples
    if n_target == 0 or n_other == 0:
        logger.debug("Nothing to inject: one of the series has no samples; returning the target unchanged")
        return timeseries

    sample_spacing = 1.0 / timeseries.sample_rate
    offset = (other.t0 - timeseries.t0) / sample_spacing
    tolerance = alignment_tolerance(
        timeseries.t0,
        timeseries.sample_rate,
        gps_times=(other.t0, other.t0 + (n_other - 1) * sample_spacing),
    )
    injected_data = np.array(timeseries.data, dtype=np.result_type(timeseries.data, other.data), copy=True)

    if is_aligned(offset, tolerance):
        shift = round(offset)
        logger.debug("Injecting with aligned grids (offset: %s samples)", shift)
        first = max(shift, 0)
        stop = min(shift + n_other, n_target)
        if first >= stop:
            logger.debug("No overlap between the series; returning the target unchanged")
            return timeseries
        injected_data[first:stop] += np.asarray(other.data)[first - shift : stop - shift]
        return dataclasses.replace(timeseries, data=injected_data)

    if not interpolate_if_offset:
        logger.debug("Non-integer offset of %s samples; not interpolating, returning original timeseries", offset)
        return timeseries

    logger.debug("Injecting with interpolation due to non-integer offset of %s samples", offset)
    # Grid points of the target inside the other's sample span [offset, offset + n_other - 1].
    first = max(math.ceil(offset), 0)
    last = min(math.floor(offset + n_other - 1), n_target - 1)
    if first > last:
        logger.debug("No grid point of the target lies within the other's span; returning the target unchanged")
        return timeseries
    positions = np.arange(first, last + 1) - offset
    injected_data[first : last + 1] += resample_uniform_sinc(np.asarray(other.data), positions)
    return dataclasses.replace(timeseries, data=injected_data)
