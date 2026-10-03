"""Module to handle injection of one TimeSeries into another, with support for time offsets."""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from astropy.units import second  # pylint: disable=no-name-in-module
from gwmock_signal.projection.resampling import resample_uniform_sinc
from gwpy.timeseries import TimeSeries

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


def inject(timeseries: TimeSeries, other: TimeSeries, interpolate_if_offset: bool = True) -> TimeSeries:
    """Inject one TimeSeries into another, handling time offsets.

    Args:
        timeseries: The target TimeSeries to inject into.
        other: The TimeSeries to be injected.
        interpolate_if_offset: Whether to interpolate if there is a non-integer sample offset.

    Returns:
        TimeSeries: The resulting TimeSeries after injection.
    """
    # Check whether timeseries is compatible with other
    timeseries.is_compatible(other)

    # crop to fit
    if (timeseries.xunit == second) and (other.xspan[0] < timeseries.xspan[0]):
        other = cast(TimeSeries, other.crop(start=timeseries.xspan[0]))
    if (timeseries.xunit == second) and (other.xspan[1] > timeseries.xspan[1]):
        other = cast(TimeSeries, other.crop(end=timeseries.xspan[1]))

    # Check if other is empty after cropping
    if len(other.times) == 0:
        logger.debug("Other TimeSeries is empty after cropping to fit; returning original timeseries")
        return timeseries

    target_times = timeseries.times.value
    other_times = other.times.value
    sample_spacing = float(timeseries.dt.value)

    # Calculate offset between start times
    offset = (other_times[0] - target_times[0]) / sample_spacing

    # Check if offset is aligned (integer number of samples)
    tolerance = alignment_tolerance(target_times[0], 1.0 / sample_spacing, gps_times=other_times)
    if not is_aligned(offset, tolerance):
        if not interpolate_if_offset:
            logger.debug("Non-integer offset of %s samples; not interpolating, returning original timeseries", offset)
            return timeseries

        # Interpolate to align grids
        logger.debug("Injecting with interpolation due to non-integer offset of %s samples", offset)

        # Determine overlap range in target time grid
        start_idx = int(np.searchsorted(target_times, other_times[0], side="left"))
        end_idx = int(np.searchsorted(target_times, other_times[-1], side="right")) - 1

        if start_idx >= len(target_times) or end_idx < 0 or start_idx > end_idx:
            logger.debug("No overlap between timeseries and other after searching; returning original timeseries")
            return timeseries

        # The band-limited windowed-sinc kernel gwmock-signal's `inject_strain` resamples with, so the
        # two packages agree on the same operation. Positions come from integer target indices minus
        # one offset, as there, rather than from per-sample GPS timestamps, each of which carries
        # float64 rounding of a sizeable fraction of a sample.
        positions = np.arange(start_idx, end_idx + 1) - offset
        resampled = resample_uniform_sinc(other.value, positions)

        # Create a new TimeSeries with explicit parameters to avoid floating-point precision issues
        injected_data = timeseries.value.copy()
        injected_data[start_idx : end_idx + 1] += resampled
        injected = TimeSeries(
            injected_data,
            t0=timeseries.t0,
            dt=timeseries.dt,
            unit=timeseries.unit,
        )
        return injected

    # Aligned case: offset is integer
    logger.debug("Injecting with aligned grids (offset: %s samples)", round(offset))
    start_idx = round(offset)
    end_idx = start_idx + len(other.value) - 1

    # Bounds check
    if start_idx < 0 or end_idx >= len(target_times) or start_idx >= len(target_times):
        logger.warning(
            "Injection range [%s:%s] out of bounds for timeseries of length %s; skipping injection",
            start_idx,
            end_idx,
            len(target_times),
        )
        return timeseries

    # Create a new TimeSeries with explicit parameters to avoid floating-point precision issues
    injected_data = timeseries.value.copy()
    inject_len = min(len(other.value), end_idx - start_idx + 1)
    injected_data[start_idx : start_idx + inject_len] += other.value[:inject_len]
    injected = TimeSeries(
        injected_data,
        t0=timeseries.t0,
        dt=timeseries.dt,
        unit=timeseries.unit,
    )
    return injected
