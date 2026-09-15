"""Configuration resolution utilities for pre-instantiation parameter calculations."""

from __future__ import annotations

import logging
import math
from typing import Any

from gwmock.cli.utils.segment_layout import resolve_segment_count
from gwmock.utils.datetime_parser import parse_duration_to_seconds

logger = logging.getLogger("gwmock")


def parse_seconds(value: float | int | str, name: str) -> float:
    """Return *value* in seconds, accepting the same spellings ``total-duration`` accepts.

    Args:
        value: A number of seconds, or a string such as ``"1 day"`` or ``"30 min"``.
        name: The setting's name, used in the error message.

    Returns:
        The value in seconds.

    Raises:
        ValueError: If the value is neither a number nor a parsable duration string.
    """
    if isinstance(value, bool) or not isinstance(value, (float, int, str)):
        raise ValueError(f"{name} must be a float, int, or str representing a duration; got {value!r}.")
    seconds = float(parse_duration_to_seconds(value)) if isinstance(value, str) else float(value)
    # NaN and infinity are rejected here rather than left to fail later. Every comparison against a
    # NaN is False, so one slips past a `< 0` guard, through the layout, and surfaces seconds or
    # minutes downstream as "cannot convert float NaN to integer" from inside a sample-count
    # rounding -- a message that names neither the setting nor the value that was wrong.
    if not math.isfinite(seconds):
        raise ValueError(f"{name} must be a finite number of seconds; got {value!r}.")
    return seconds


def resolve_segment_gap(
    simulator_args: dict[str, Any],
    global_args: dict[str, Any],
) -> float:
    """Resolve the configured gap between consecutive analysed segments, in seconds.

    Args:
        simulator_args: Simulator-specific arguments (normalized with underscores).
        global_args: Global simulator arguments (normalized with underscores).

    Returns:
        The gap in seconds, ``0.0`` when none is configured.

    Raises:
        ValueError: If the configured gap is negative or cannot be parsed.
    """
    raw = simulator_args.get("segment_gap")
    if raw is None:
        raw = global_args.get("segment_gap")
    if raw is None:
        return 0.0
    gap = parse_seconds(raw, "segment-gap")
    if gap < 0:
        raise ValueError(f"segment-gap must be non-negative; got {gap:g} s.")
    return gap


def resolve_max_samples(
    simulator_args: dict[str, Any],
    global_args: dict[str, Any],
) -> int:
    """Resolve max_samples from simulator and global arguments.

    This function replicates the logic in TimeSeriesMixin to compute max_samples
    from total_duration and duration, allowing plan creation to determine batch counts
    without instantiating simulators.

    ``total_duration`` is the run's GPS **span**, so with a non-zero ``segment_gap`` the batch
    count is no longer ``total_duration / duration``: the gaps take up part of the span. See
    :func:`~gwmock.cli.utils.segment_layout.resolve_segment_count`, which owns that arithmetic and
    is where a span that does not divide is refused.

    Priority order:
    1. Computed from total_duration / duration (simulator or global)
    2. Explicit max_samples in simulator_args
    3. Explicit max_samples in global_args
    4. Default to 1

    Args:
        simulator_args: Simulator-specific arguments (normalized with underscores)
        global_args: Global simulator arguments (normalized with underscores)

    Returns:
        Resolved max_samples value

    Raises:
        ValueError: If a gapped configuration's span does not divide into whole segments.

    Example:
        >>> resolve_max_samples({"total_duration": "1 hour", "duration": 4}, {"max_samples": 10})
        900
    """
    # Try to compute from total_duration and duration (highest priority)
    total_duration = simulator_args.get("total_duration") or global_args.get("total_duration")
    duration = simulator_args.get("duration") or global_args.get("duration", 4)
    segment_gap = resolve_segment_gap(simulator_args, global_args)

    if total_duration is not None:
        total_duration_seconds = parse_seconds(total_duration, "total-duration")
        duration_seconds = float(duration)

        if total_duration_seconds < duration_seconds:
            logger.warning(
                "total_duration (%s) < duration (%s), setting max_samples=1",
                total_duration_seconds,
                duration_seconds,
            )
            return 1

        computed_samples = resolve_segment_count(total_duration_seconds, duration_seconds, segment_gap)
        logger.debug(
            "Computed max_samples=%d from total_duration=%s, duration=%s, segment_gap=%s",
            computed_samples,
            total_duration_seconds,
            duration_seconds,
            segment_gap,
        )
        return computed_samples

    # Fall back to explicit max_samples in simulator args
    if "max_samples" in simulator_args:
        return int(simulator_args["max_samples"])

    # Fall back to explicit max_samples in global args
    if "max_samples" in global_args:
        return int(global_args["max_samples"])

    # Default
    logger.debug("No max_samples or total_duration specified, defaulting to 1")
    return 1
