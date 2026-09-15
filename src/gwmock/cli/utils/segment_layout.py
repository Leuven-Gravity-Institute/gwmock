"""Where a run's analysed segments sit in GPS time, and how many there are.

One object owns the answer to "which GPS times does this run write, and which does it skip",
so that the batch count, the epoch of each frame, the end of the run and the question "is this
instant inside a gap" cannot disagree with each other. Everything that used to assume segments
tile their span contiguously asks here instead.

The layout is regular: *count* analysed segments of *duration* seconds, each separated from the
next by *gap* seconds of GPS time that the run does not write. ``gap = 0`` is the contiguous
layout, and reproduces the arithmetic that preceded this module exactly.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger("gwmock")

#: Relative slack allowed when deciding whether a span divides into whole segments.
#:
#: The three numbers arrive as float64 seconds, often from a parsed string such as ``"1 day"``, so
#: an exact remainder test would refuse layouts that are exact in the arithmetic a reader does. The
#: slack is relative to the span because that is the largest quantity in the comparison, and a
#: fixed absolute tolerance would be meaninglessly tight for a day-long run and meaninglessly loose
#: for a four-second one.
_DIVISION_TOLERANCE = 1e-9


@dataclass(frozen=True)
class SegmentLayout:
    """The GPS epochs a run writes, and the gaps between them.

    Attributes:
        start_time: GPS time of the first analysed segment's first sample.
        duration: Length of each analysed segment, in seconds.
        gap: Seconds of GPS time between the end of one analysed segment and the start of the
            next. Zero for a contiguous run.
        count: Number of analysed segments.
    """

    start_time: float
    duration: float
    gap: float
    count: int

    def __post_init__(self) -> None:
        """Reject a layout that cannot describe any run.

        The finiteness checks come first, and they are not decoration: every comparison against a
        NaN is False, so a NaN duration or gap passes both range checks below and then makes every
        predicate on this layout answer silently -- ``contains`` returns ``False`` for every
        instant of the run rather than raising.

        Raises:
            ValueError: If any of the three is not finite, the duration is not positive, the gap is
                negative, or the count is negative.
        """
        for name, value in (("start_time", self.start_time), ("duration", self.duration), ("gap", self.gap)):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite; got {value}.")
        if self.duration <= 0:
            raise ValueError(f"duration must be positive; got {self.duration}.")
        if self.gap < 0:
            raise ValueError(f"segment-gap must be non-negative; got {self.gap}.")
        if self.count < 0:
            raise ValueError(f"count must be non-negative; got {self.count}.")

    @property
    def stride(self) -> float:
        """Seconds from one segment's start to the next one's start."""
        return self.duration + self.gap

    @property
    def span(self) -> float:
        """Seconds from the first segment's start to the last segment's end.

        This is what ``total-duration`` means: the run's GPS extent, gaps included and **no
        trailing gap** -- a run stops at the end of data, not after a gap nobody wrote. So
        ``span == count * duration + (count - 1) * gap``, which collapses to ``count * duration``
        for a contiguous run.
        """
        if self.count <= 0:
            return 0.0
        return self.count * self.duration + (self.count - 1) * self.gap

    @property
    def livetime(self) -> float:
        """Seconds of analysed data the run writes: ``count * duration``.

        Equal to :attr:`span` only when the run is contiguous. The pair is what makes
        ``total-duration`` ambiguous once gaps exist, which is why this package names both.
        """
        return self.count * self.duration

    @property
    def end_time(self) -> float:
        """GPS time one sample interval past the last analysed segment's last sample."""
        return self.start_time + self.span

    def epoch(self, index: int) -> float:
        """Return the GPS start time of the analysed segment at *index*.

        Args:
            index: Zero-based segment index. Not range-checked, so a caller building the epoch of
                a segment beyond the run gets the epoch that segment would have had.

        Returns:
            The segment's GPS start time.
        """
        return self.start_time + index * self.stride

    def epochs(self) -> list[float]:
        """Return the GPS start time of every analysed segment, in order."""
        return [self.epoch(index) for index in range(self.count)]

    def gap_before(self, index: int) -> tuple[float, float] | None:
        """Return the gap immediately preceding the segment at *index*.

        Args:
            index: Zero-based segment index.

        Returns:
            ``(gap_start, gap_end)`` in GPS seconds, or ``None`` when no gap precedes that
            segment -- which is the case for the first segment and for every segment of a
            contiguous run.
        """
        if self.gap <= 0 or index <= 0:
            return None
        gap_end = self.epoch(index)
        return gap_end - self.gap, gap_end

    def contains(self, gps_time: float) -> bool:
        """Whether *gps_time* falls inside an analysed segment.

        The run's own span bounds the answer, so a time before the first segment or at or after
        the end of the last one is outside regardless of where it sits modulo the stride.

        Args:
            gps_time: The instant to classify, in GPS seconds.

        Returns:
            Whether the run writes a sample covering that instant.
        """
        if self.count <= 0:
            return False
        offset = gps_time - self.start_time
        if offset < 0 or offset >= self.span:
            return False
        if self.gap <= 0:
            return True
        return (offset % self.stride) < self.duration

    def gap_containing(self, gps_time: float) -> tuple[float, float] | None:
        """Return the gap *gps_time* falls in, if it falls in one.

        Args:
            gps_time: The instant to classify, in GPS seconds.

        Returns:
            ``(gap_start, gap_end)`` for the enclosing gap, or ``None`` when the instant is inside
            an analysed segment or outside the run altogether. Times before the run and after it
            are deliberately not gaps: they are not holes punched in the released data, they are
            outside what the run claims to cover at all.
        """
        if self.gap <= 0 or self.count <= 0:
            return None
        offset = gps_time - self.start_time
        if offset < 0 or offset >= self.span:
            return None
        index = int(offset // self.stride)
        within = offset - index * self.stride
        if within < self.duration:
            return None
        gap_start = self.start_time + index * self.stride + self.duration
        return gap_start, gap_start + self.gap

    def intersects(self, start: float, end: float) -> bool:
        """Whether the half-open interval ``[start, end)`` reaches any analysed segment.

        Used to decide whether a waveform has anywhere to be written. A signal answering ``False``
        here is one the run's frames cannot carry: it lies wholly inside a gap, wholly before the
        run, or wholly after it.

        Args:
            start: Interval start, in GPS seconds.
            end: Interval end, in GPS seconds, exclusive.

        Returns:
            Whether the interval overlaps at least one analysed segment.
        """
        if self.count <= 0 or end <= start:
            return False
        # Clip to the run first, so the modular walk below only has to consider segments that
        # exist. An interval reaching past either end still counts if the clipped part survives.
        lower = max(start, self.start_time)
        upper = min(end, self.end_time)
        if upper <= lower:
            return False
        if self.gap <= 0:
            return True
        first = int((lower - self.start_time) // self.stride)
        last = int((upper - self.start_time) // self.stride)
        for index in range(first, min(last, self.count - 1) + 1):
            segment_start = self.epoch(index)
            if lower < segment_start + self.duration and upper > segment_start:
                return True
        return False


def resolve_segment_count(total_duration: float, duration: float, gap: float) -> int:
    """Return how many analysed segments a span of *total_duration* seconds holds.

    Inverts :attr:`SegmentLayout.span`: with ``count`` segments of ``duration`` separated by
    ``count - 1`` gaps, ``total_duration = count * (duration + gap) - gap``, so

        ``count = (total_duration + gap) / (duration + gap)``.

    **A gapped span that does not divide is refused**, naming the three numbers, because rounding
    it would silently move the run's last epoch and produce frames whose layout does not match the
    configuration that claims to describe them.

    **A contiguous span that does not divide is rounded, with a warning.** The asymmetry is
    deliberate and is a compatibility decision, not a physical one: rounding is what every release
    before gaps existed did, and the shipped example configurations rely on it (``1 day`` of
    ``4096`` s segments is 21.09 segments). Turning that into a hard error would reject
    configurations that have always been accepted. The warning is what stops it being *silent* --
    it names the same three numbers and the span actually used.

    Args:
        total_duration: The run's GPS span in seconds, gaps included.
        duration: Length of one analysed segment in seconds.
        gap: Seconds between consecutive analysed segments.

    Returns:
        The number of analysed segments, at least 1.

    Raises:
        ValueError: If any argument is not finite, or a gapped configuration's span does not divide
            into whole segments.
    """
    for name, value in (("total-duration", total_duration), ("duration", duration), ("segment-gap", gap)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number of seconds; got {value}.")
    stride = duration + gap
    exact = (total_duration + gap) / stride
    count = max(1, round(exact))
    realized = count * stride - gap
    if abs(realized - total_duration) <= _DIVISION_TOLERANCE * max(abs(total_duration), stride):
        return count

    if gap > 0:
        lower = max(1, int((total_duration + gap) // stride))
        raise ValueError(
            f"total-duration {total_duration:g} s does not divide into whole segments of "
            f"duration {duration:g} s separated by segment-gap {gap:g} s: it implies "
            f"{exact:g} segments. A gapped run spans "
            f"count * duration + (count - 1) * segment-gap, so the nearest spans that do divide "
            f"are {lower * stride - gap:g} s ({lower} segments) and "
            f"{(lower + 1) * stride - gap:g} s ({lower + 1} segments). "
            "Set total-duration to one of those, or change duration or segment-gap."
        )

    logger.warning(
        "total-duration %g s does not divide into whole segments of duration %g s "
        "(segment-gap %g s): it implies %g segments. Rounding to %d segments, so the run actually "
        "spans %g s. This rounding is kept for compatibility with contiguous configurations; a run "
        "with a non-zero segment-gap refuses instead. Set total-duration to a multiple of duration "
        "to say what you mean.",
        total_duration,
        duration,
        gap,
        exact,
        count,
        realized,
    )
    return count
