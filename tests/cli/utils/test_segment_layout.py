"""The GPS layout of a run's analysed segments, and what a span is allowed to mean.

These are the arithmetic tests. ``tests/cli/test_gapped_segments.py`` is where the layout is
checked against what a simulator and a written frame set actually do.
"""

from __future__ import annotations

import logging

import pytest

from gwmock.cli.utils.config_resolution import parse_seconds, resolve_max_samples, resolve_segment_gap
from gwmock.cli.utils.segment_layout import SegmentLayout, resolve_segment_count

#: The motivating layout, stated once: four 1024 s analysed segments separated by 256 s gaps per
#: block, eight blocks. It is a plain regular layout -- the block structure is how a user thinks
#: about it, not something the run has to know -- so it is 32 segments of 1024 s at a stride of
#: 1280 s.
SEGMENTS_PER_BLOCK = 4
BLOCKS = 8
SEGMENT_DURATION = 1024.0
SEGMENT_GAP = 256.0
N_SEGMENTS = SEGMENTS_PER_BLOCK * BLOCKS


class TestSpanAndLivetime:
    """``total-duration`` is the span; the livetime is the other number, and they differ."""

    def test_a_contiguous_layout_spans_exactly_its_livetime(self):
        """The pre-gap identity has to keep holding, or every existing configuration moves."""
        layout = SegmentLayout(start_time=1000.0, duration=8.0, gap=0.0, count=5)

        assert layout.span == 40.0
        assert layout.livetime == 40.0

    def test_a_gapped_layout_spans_its_livetime_plus_the_interior_gaps(self):
        """No trailing gap: a run stops at the end of data, not after a hole nobody wrote."""
        layout = SegmentLayout(start_time=1000.0, duration=1024.0, gap=256.0, count=32)

        assert layout.livetime == 32 * 1024.0
        assert layout.span == 32 * 1024.0 + 31 * 256.0
        assert layout.end_time == 1000.0 + layout.span

    def test_the_epoch_list_is_the_stride_walked_from_the_start(self):
        """The exact list, spelled out rather than recomputed from the same formula under test."""
        layout = SegmentLayout(start_time=1_000_000_000.0, duration=1024.0, gap=256.0, count=6)

        assert layout.epochs() == [
            1_000_000_000.0,
            1_000_001_280.0,
            1_000_002_560.0,
            1_000_003_840.0,
            1_000_005_120.0,
            1_000_006_400.0,
        ]

    def test_a_single_segment_has_no_gap_in_it_at_any_configured_gap(self):
        """One segment means no interior gap, so the span cannot pick one up from the setting."""
        layout = SegmentLayout(start_time=0.0, duration=64.0, gap=1024.0, count=1)

        assert layout.span == 64.0
        assert layout.gap_before(0) is None

    @pytest.mark.parametrize(
        ("duration", "gap", "count"),
        [(8.0, 0.0, 3), (1024.0, 256.0, 32), (4.0, 4.0, 2), (0.5, 0.25, 7)],
    )
    def test_the_count_round_trips_through_the_span(self, duration, gap, count):
        """The inverse the batch count is taken with has to return the count the span came from."""
        layout = SegmentLayout(start_time=0.0, duration=duration, gap=gap, count=count)

        assert resolve_segment_count(layout.span, duration, gap) == count

    def test_a_layout_with_a_non_positive_duration_is_refused(self):
        with pytest.raises(ValueError, match="duration must be positive"):
            SegmentLayout(start_time=0.0, duration=0.0, gap=1.0, count=2)

    def test_a_layout_with_a_negative_gap_is_refused(self):
        with pytest.raises(ValueError, match="segment-gap must be non-negative"):
            SegmentLayout(start_time=0.0, duration=4.0, gap=-1.0, count=2)


class TestClassifyingAnInstant:
    """Which GPS times a run writes, and which fall in the holes."""

    LAYOUT = SegmentLayout(start_time=100.0, duration=10.0, gap=5.0, count=3)
    # Segments cover [100, 110), [115, 125), [130, 140); gaps are [110, 115) and [125, 130).

    def test_an_instant_inside_a_segment_is_written(self):
        assert self.LAYOUT.contains(100.0)
        assert self.LAYOUT.contains(109.999)
        assert self.LAYOUT.contains(139.999)

    def test_an_instant_inside_a_gap_is_not_written_and_names_its_gap(self):
        assert not self.LAYOUT.contains(112.0)
        assert self.LAYOUT.gap_containing(112.0) == (110.0, 115.0)
        assert self.LAYOUT.gap_containing(127.5) == (125.0, 130.0)

    def test_the_segment_boundary_belongs_to_the_gap_that_follows_it(self):
        """A segment's span is half-open, so its end time is the gap's first instant."""
        assert not self.LAYOUT.contains(110.0)
        assert self.LAYOUT.gap_containing(110.0) == (110.0, 115.0)
        assert self.LAYOUT.contains(115.0)
        assert self.LAYOUT.gap_containing(115.0) is None

    def test_an_instant_outside_the_run_is_not_a_gap(self):
        """Before the run and after it are not holes punched in the data; they are not covered."""
        assert self.LAYOUT.gap_containing(99.0) is None
        assert self.LAYOUT.gap_containing(140.0) is None
        assert not self.LAYOUT.contains(99.0)
        assert not self.LAYOUT.contains(140.0)

    def test_an_interval_reaching_a_segment_intersects_even_if_it_starts_in_a_gap(self):
        """The spillover case: a waveform starting in a gap whose tail reaches the next segment."""
        assert self.LAYOUT.intersects(112.0, 116.0)
        assert self.LAYOUT.intersects(108.0, 112.0)

    def test_an_interval_wholly_inside_a_gap_intersects_nothing(self):
        assert not self.LAYOUT.intersects(111.0, 114.0)
        assert not self.LAYOUT.intersects(110.0, 115.0)

    def test_an_interval_spanning_a_whole_gap_and_both_neighbours_intersects(self):
        assert self.LAYOUT.intersects(105.0, 120.0)

    def test_a_contiguous_layout_has_no_gap_for_an_interval_to_hide_in(self):
        contiguous = SegmentLayout(start_time=100.0, duration=10.0, gap=0.0, count=3)

        assert contiguous.intersects(105.0, 106.0)
        assert contiguous.gap_containing(105.0) is None
        assert contiguous.gap_before(2) is None


class TestRefusingASpanThatDoesNotDivide:
    """A gapped span that does not divide is an error; a contiguous one is a loud rounding."""

    def test_a_gapped_span_that_does_not_divide_is_refused_naming_the_three_numbers(self):
        """Not rounded: rounding moves the run's last epoch away from what the config claims."""
        with pytest.raises(ValueError, match="does not divide into whole segments") as error:
            resolve_segment_count(40000.0, SEGMENT_DURATION, SEGMENT_GAP)

        message = str(error.value)
        assert "40000" in message
        assert "1024" in message
        assert "256" in message
        # The two spans that do divide, so the message says what to write instead.
        assert "39424" in message
        assert "40704" in message

    def test_the_motivating_layout_divides_exactly(self):
        """32 segments of 1024 s at a 256 s gap span 40704 s, and that is the whole requirement."""
        span = N_SEGMENTS * SEGMENT_DURATION + (N_SEGMENTS - 1) * SEGMENT_GAP

        assert span == 40704.0
        assert resolve_segment_count(span, SEGMENT_DURATION, SEGMENT_GAP) == N_SEGMENTS

    def test_a_contiguous_span_that_does_not_divide_is_rounded_but_says_so(self, caplog):
        """Every shipped example is ``1 day`` of 4096 s segments, which is 21.09 of them.

        Refusing that would reject configurations this package has always accepted, so the
        contiguous path keeps rounding -- and warns, naming the same three numbers and the span it
        actually used, so the rounding is not silent.
        """
        with caplog.at_level(logging.WARNING, logger="gwmock"):
            count = resolve_segment_count(86400.0, 4096.0, 0.0)

        assert count == 21
        message = caplog.text
        assert "86400" in message
        assert "4096" in message
        assert "86016" in message

    def test_a_gapped_span_one_part_in_ten_billion_off_still_divides(self):
        """Spans arrive as float64 seconds, often parsed from a string, so an exact test is wrong."""
        span = 40704.0 * (1 + 1e-11)

        assert resolve_segment_count(span, SEGMENT_DURATION, SEGMENT_GAP) == N_SEGMENTS


class TestResolvingTheBatchCountFromAConfiguration:
    """What ``resolve_max_samples`` does once ``segment-gap`` is in the arguments."""

    def test_a_gapless_configuration_is_unchanged(self):
        """The regression guard: the gap defaults to zero and the old answer has to survive."""
        assert resolve_max_samples({}, {"total_duration": 3600.0, "duration": 4.0}) == 900

    def test_the_gap_is_taken_out_of_the_span_before_dividing(self):
        """``total_duration / duration`` would give 39 here; the span holds 32 segments."""
        assert resolve_max_samples({}, {"total_duration": 40704.0, "duration": 1024.0, "segment_gap": 256.0}) == 32

    def test_a_simulator_argument_overrides_the_global_gap(self):
        assert resolve_segment_gap({"segment_gap": 8.0}, {"segment_gap": 256.0}) == 8.0

    def test_the_gap_accepts_the_duration_strings_total_duration_accepts(self):
        assert resolve_segment_gap({}, {"segment_gap": "4 minute"}) == 240.0

    def test_a_negative_gap_is_refused(self):
        with pytest.raises(ValueError, match="segment-gap must be non-negative"):
            resolve_segment_gap({}, {"segment_gap": -1.0})

    def test_no_gap_configured_means_no_gap(self):
        assert resolve_segment_gap({}, {}) == 0.0


class TestRejectingNonFiniteDurations:
    """NaN and infinity are refused where they enter, not left to fail somewhere downstream.

    Every comparison against a NaN is False, so one passes a bare ``< 0`` guard untouched. It then
    travels through the layout -- where it makes every predicate answer ``False`` silently, so a
    run reports that no instant of it is inside a segment -- and finally surfaces from a sample
    count rounding as "cannot convert float NaN to integer", which names neither the setting nor
    the value that was wrong. Infinity takes the same route to an OverflowError.
    """

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_gap_is_refused_when_it_is_parsed(self, bad):
        with pytest.raises(ValueError, match="finite"):
            resolve_segment_gap({}, {"segment_gap": bad})

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_a_non_finite_total_duration_is_refused(self, bad):
        with pytest.raises(ValueError, match="finite"):
            parse_seconds(bad, "total-duration")

    @pytest.mark.parametrize("field", ["start_time", "duration", "gap"])
    def test_a_layout_built_on_a_non_finite_number_is_refused(self, field):
        arguments = {"start_time": 0.0, "duration": 4.0, "gap": 1.0, "count": 3} | {field: float("nan")}

        with pytest.raises(ValueError, match="must be finite"):
            SegmentLayout(**arguments)

    def test_a_non_finite_layout_would_otherwise_answer_silently(self):
        """Why the guard rejects rather than warns, demonstrated rather than argued.

        The NaN is injected *past* the constructor, which is the only way to get one into a layout
        now, so this shows what the guard is buying. The two failure shapes are both bad and they
        are bad differently, which is why neither is acceptable:

        * ``contains`` answers **silently and wrongly** -- the layout reports that no instant of
          its own run is inside a segment, which reads exactly like a correctly configured run
          whose signals all happen to miss;
        * ``gap_containing`` and ``intersects`` raise, but from an ``int()`` of a NaN inside the
          modular walk, with a message naming neither the setting nor the value that was wrong.

        Which method does which was measured rather than reasoned about: the first draft of this
        test asserted that all three were silent, and two of them were not.
        """
        layout = SegmentLayout(start_time=0.0, duration=4.0, gap=1.0, count=3)
        assert layout.contains(1.0)

        object.__setattr__(layout, "duration", float("nan"))

        assert not layout.contains(1.0)
        for query in (lambda: layout.gap_containing(4.5), lambda: layout.intersects(0.0, 100.0)):
            with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
                query()

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_a_non_finite_span_is_refused_by_the_count_resolver(self, bad):
        with pytest.raises(ValueError, match="finite"):
            resolve_segment_count(bad, 1024.0, 256.0)
