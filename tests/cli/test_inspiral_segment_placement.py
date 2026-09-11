"""Which segment claims a compact binary, when its waveform starts before its coalescence.

A compact binary's buffer begins seconds before ``coa_time``. A segment chosen from ``coa_time``
alone therefore starts *after* the waveform does, and injection crops the difference away with
nowhere to put it: the earlier segments are already written. The loss is not marginal -- for a 30+25
solar-mass binary at 1024 Hz the lead is 3.6 s, so an event landing 1 s into a 16-second segment
loses 2.6 s of inspiral.

The rule these tests pin is that a segment claims the events whose *waveform* starts before it ends,
taken from :meth:`SignalAdapter.pre_coalescence_duration`. Two loops apply it -- per-event and
batched -- and both are checked, because ``population_index`` is checkpointed state and a run
switched between modes must not skip or repeat an event because the two disagreed.

The rule alone is not enough, which is what ``TestConsumptionOrderDoesNotStrandEvents`` covers: both
loops stop at the first event that does not belong, so the catalogue has to be consumed in
waveform-start order rather than in ``coa_time`` order, or a long-lead event sitting behind a
short-lead one is never reached and is cropped exactly as before.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gwmock.cli.utils.config import Config

_POPULATION_CSV = Path(__file__).resolve().parents[2] / "examples" / "signal" / "bbh_population.csv"
_START = 1577491296.0
_SAMPLING_FREQUENCY = 1024.0
_SEGMENT_DURATION = 16.0

#: A complete BBH parameter set, so the query can actually answer. Underspecified events are the
#: subject of their own test below rather than an accident of this one.
_COMPLETE_EVENT: dict[str, Any] = {
    "detector_frame_mass_1": 30.0,
    "detector_frame_mass_2": 25.0,
    "distance": 400.0,
    "right_ascension": 1.0,
    "declination": 0.5,
    "polarization_angle": 0.2,
    "inclination": 0.3,
}


def _config(working_directory: Path, execution: str, total_duration: float = _SEGMENT_DURATION) -> dict[str, Any]:
    """Return a one-segment BBH config in the given execution mode.

    No ``waveform-backend`` key, so the default LAL library is used: naming ripple would make
    *constructing* the orchestrator instantiate ``RippleBackend``, which needs the ``[jax]`` extra
    long before any generation happens, and these tests are about placement rather than a library.
    """
    return {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": _SAMPLING_FREQUENCY,
                "duration": _SEGMENT_DURATION,
                "total-duration": total_duration,
                "start-time": _START,
                "seed": 20260804,
            },
            "working-directory": str(working_directory),
            "output-directory": "output",
            "metadata-directory": "metadata",
        },
        "orchestration": {
            "population": {
                "backend": "FilePopulationLoader",
                "source-type": "bbh",
                "n-samples": 1,
                "arguments": {"path": str(_POPULATION_CSV)},
            },
            "signal": {
                "source-type": "bbh",
                "waveform-model": "IMRPhenomD",
                "execution": execution,
                "minimum-frequency": 30,
                "detectors": ["ET-Triangle-Sardinia"],
                "output": {
                    "output_directory": "signal",
                    "file_name": "sig-{{ detectors }}.gwf",
                    "arguments": {"channel": "{{ detectors }}:STRAIN"},
                },
            },
        },
    }


def _orchestrator(working_directory: Path, execution: str = "per-event", total_duration: float = _SEGMENT_DURATION):
    from gwmock.cli.adapter_orchestration import AdapterOrchestrator

    working_directory.mkdir(parents=True, exist_ok=True)
    config = Config.model_validate(_config(working_directory, execution, total_duration))
    return AdapterOrchestrator.from_config(
        config.orchestration,
        global_simulator_arguments=dict(config.globals.simulator_arguments),
    )


class _StubAdapter:
    """A signal adapter that answers the placement query however the test needs.

    Stands in for the real adapter so the boundary rule can be tested without generating a waveform
    per case, and so *unknown* and *failing* answers can be produced at all -- neither is reachable
    from the bundled backends with complete parameters.
    """

    def __init__(
        self,
        lead: float | None = None,
        error: Exception | None = None,
        leads_by_coa_time: Mapping[float, float] | None = None,
        errors_by_coa_time: Mapping[float, Exception] | None = None,
    ) -> None:
        self._lead = lead
        self._error = error
        # Per-event answers, so a catalogue can be given leads that vary the way real ones do -- a
        # heavy binary black hole against a binary neutron star differ by an order of magnitude.
        self._leads_by_coa_time = dict(leads_by_coa_time or {})
        self._errors_by_coa_time = dict(errors_by_coa_time or {})
        self.detector_names = ("E1",)

    def pre_coalescence_duration(self, parameters: Mapping[str, Any], **_: Any) -> float | None:
        coa_time = parameters.get("coa_time")
        per_event_error = self._errors_by_coa_time.get(coa_time)
        if per_event_error is not None:
            raise per_event_error
        if self._error is not None and not self._errors_by_coa_time:
            raise self._error
        if coa_time in self._leads_by_coa_time:
            return self._leads_by_coa_time[coa_time]
        return self._lead


def _end_time(orchestrator) -> float:
    return float(getattr(orchestrator.end_time, "value", orchestrator.end_time))


class TestTheBoundaryRule:
    """The rule itself, with the lead supplied rather than generated."""

    def test_an_event_coalescing_after_the_segment_is_claimed_when_its_waveform_starts_inside(self, tmp_path):
        """This is the defect. Its ``coa_time`` is past the boundary; 3.6 s of its inspiral is not.

        Under the previous ``coa_time`` rule this event was left to the next segment, whose start is
        after the waveform's -- so the lead was cropped and could never be placed, because by then
        this segment had been written.
        """
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(lead=3.6)
        end_time = _end_time(orchestrator)

        assert orchestrator._event_starts_before_segment_end({**_COMPLETE_EVENT, "coa_time": end_time + 1.0}), (
            "an event whose waveform begins 2.6 s before the boundary belongs to this segment"
        )

    def test_an_event_whose_waveform_also_starts_later_is_left_for_the_next_segment(self, tmp_path):
        """The rule must still end the segment, or every remaining event would be pulled forward."""
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(lead=3.6)
        end_time = _end_time(orchestrator)

        assert not orchestrator._event_starts_before_segment_end({**_COMPLETE_EVENT, "coa_time": end_time + 10.0})

    def test_the_boundary_is_the_waveform_start_and_not_the_coalescence(self, tmp_path):
        """Pins which quantity is compared, by making the two answers differ.

        With a lead of 5 s an event coalescing 4 s past the boundary is claimed and one coalescing
        6 s past it is not, so the comparison cannot be against ``coa_time``.
        """
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(lead=5.0)
        end_time = _end_time(orchestrator)

        assert orchestrator._event_starts_before_segment_end({**_COMPLETE_EVENT, "coa_time": end_time + 4.0})
        assert not orchestrator._event_starts_before_segment_end({**_COMPLETE_EVENT, "coa_time": end_time + 6.0})

    def test_a_waveform_starting_exactly_on_the_boundary_goes_to_the_next_segment(self, tmp_path):
        """The comparison is strict, and which way it resolves matters rather than being arbitrary.

        A waveform starting exactly where the next segment starts loses nothing by being claimed
        there: that segment's start *is* the waveform's start, so injection crops zero samples.
        Claiming it here would generate it a segment early for no gain.
        """
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(lead=4.0)
        end_time = _end_time(orchestrator)

        assert not orchestrator._event_starts_before_segment_end({**_COMPLETE_EVENT, "coa_time": end_time + 4.0})

    def test_an_event_without_a_coalescence_time_always_belongs(self, tmp_path):
        """Non-coalescing sources reach the loop unchanged; there is nothing to place them by."""
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(lead=3.6)

        assert orchestrator._event_starts_before_segment_end({"frequency": 100.0})


class TestWhenTheLeadIsNotAvailable:
    """Unknown must mean unknown. Reading it as zero would drop the whole inspiral silently."""

    def test_an_unknown_lead_falls_back_to_the_coalescence_time(self, tmp_path):
        """``None`` restores the previous behaviour rather than claiming the waveform starts at tc.

        A backend that cannot say (PyCBC) and an installed gwmock-signal predating the query both
        arrive here. Treating ``None`` as ``0.0`` would look like an answer and behave like the bug.
        """
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(lead=None)
        end_time = _end_time(orchestrator)

        assert orchestrator._event_starts_before_segment_end({**_COMPLETE_EVENT, "coa_time": end_time - 0.5})
        assert not orchestrator._event_starts_before_segment_end({**_COMPLETE_EVENT, "coa_time": end_time + 0.5}), (
            "without a lead the only rule available is coa_time, which is where this started"
        )

    def test_a_query_that_raises_does_not_fail_the_run(self, tmp_path, caplog):
        """Placement must not be the thing that turns an odd catalogue into a crash.

        The same parameters go to generation immediately afterwards, which raises whatever this
        raised -- with the context of the event it was generating rather than of a boundary test.
        """
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(error=ValueError("Missing required parameter: 'distance'"))
        end_time = _end_time(orchestrator)

        with caplog.at_level(logging.WARNING, logger="gwmock"):
            claimed = orchestrator._event_starts_before_segment_end({"coa_time": end_time - 0.5})

        assert claimed, "the fallback is the coa_time rule, not a refusal to place the event"
        assert "Missing required parameter" in caplog.text, (
            "the swallowed reason must appear, or a run silently reverts to the cropping behaviour"
        )

    def test_the_fallback_warns_once_for_a_repeated_reason(self, tmp_path, caplog):
        """A catalogue the query cannot read fails identically for every row; say so once."""
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(error=ValueError("nope"))
        end_time = _end_time(orchestrator)

        with caplog.at_level(logging.WARNING, logger="gwmock"):
            for offset in range(5):
                orchestrator._event_starts_before_segment_end({"coa_time": end_time - offset - 1.0})

        assert caplog.text.count("Cannot establish how long before coalescence") == 1

    def test_a_second_distinct_reason_is_also_reported(self, tmp_path, caplog):
        """Deduplicating by a flag rather than by reason would name the wrong cause.

        One malformed row fails only its own query. Under a single flag, every later event failing for
        a *different* reason falls back silently beneath a warning describing the first one -- so the
        operator reads one cause and has no way to see the others.
        """
        orchestrator = _orchestrator(tmp_path)
        end_time = _end_time(orchestrator)
        first, second = end_time - 1.0, end_time - 2.0
        orchestrator.signal_adapter = _StubAdapter(
            errors_by_coa_time={
                first: ValueError("Missing required parameter: 'distance'"),
                second: TypeError("unsupported operand type(s)"),
            }
        )

        with caplog.at_level(logging.WARNING, logger="gwmock"):
            orchestrator._event_starts_before_segment_end({**_COMPLETE_EVENT, "coa_time": first})
            orchestrator._event_starts_before_segment_end({**_COMPLETE_EVENT, "coa_time": second})

        assert caplog.text.count("Cannot establish how long before coalescence") == 2
        assert "Missing required parameter" in caplog.text
        assert "unsupported operand" in caplog.text

    def test_a_run_without_a_signal_adapter_still_walks_the_catalogue(self, tmp_path):
        """Noise-only orchestration reaches this helper through shared code paths."""
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = None
        end_time = _end_time(orchestrator)

        assert orchestrator._event_starts_before_segment_end({**_COMPLETE_EVENT, "coa_time": end_time - 0.5})
        assert not orchestrator._event_starts_before_segment_end({**_COMPLETE_EVENT, "coa_time": end_time + 0.5})


class TestConsumptionOrderDoesNotStrandEvents:
    """A long-lead event behind a short-lead one must still be claimed by the segment it starts in.

    Both loops walk the catalogue and stop at the first event that does not belong, which is sound
    only if "belongs" is a prefix property. It was, under the ``coa_time`` rule. Under the
    waveform-start rule it is not for a ``coa_time``-sorted catalogue, because the lead varies with
    the source -- roughly 3 s for a heavy binary black hole against ~100 s for a binary neutron star.
    Consumption is therefore ordered by the placement key instead.

    Without that ordering the third event below is never reached, is generated a segment late, and is
    cropped -- the exact loss this rule exists to prevent, reintroduced by catalogue order alone.
    """

    def _catalogue(self, end_time: float):
        """Return a catalogue whose waveform-start order differs from its coa_time order."""
        return (
            {**_COMPLETE_EVENT, "coa_time": end_time + 1.0},  # start end-9   belongs
            {**_COMPLETE_EVENT, "coa_time": end_time + 2.0},  # start end+1   does not
            {**_COMPLETE_EVENT, "coa_time": end_time + 3.0},  # start end-17  belongs, sorts last
        )

    def _leads(self, end_time: float) -> dict[float, float]:
        return {end_time + 1.0: 10.0, end_time + 2.0: 1.0, end_time + 3.0: 20.0}

    def test_the_batched_walk_reaches_the_long_lead_event_behind_a_short_lead_one(self, tmp_path):
        orchestrator = _orchestrator(tmp_path, "batched")
        end_time = _end_time(orchestrator)
        orchestrator.signal_adapter = _StubAdapter(leads_by_coa_time=self._leads(end_time))
        orchestrator._population_events = self._catalogue(end_time)

        event_ids, events = orchestrator._events_for_this_segment()

        assert sorted(event_ids) == [0, 2], (
            "event 2 starts 17 s before the boundary; stopping at event 1 would strand it and crop it"
        )
        assert len(events) == 2
        assert 1 not in event_ids, "event 1's waveform starts after this segment ends"

    def test_the_per_event_loop_reaches_it_too(self, tmp_path):
        """The two loops must agree, or a mode switch mid-run changes which events exist where."""
        orchestrator = _orchestrator(tmp_path, "per-event")
        end_time = _end_time(orchestrator)
        orchestrator.signal_adapter = _StubAdapter(leads_by_coa_time=self._leads(end_time))
        orchestrator._population_events = self._catalogue(end_time)

        order = orchestrator._placement_order()
        claimed = []
        for position in order:
            if not orchestrator._event_starts_before_segment_end(orchestrator._population_events[position]):
                break
            claimed.append(position)

        assert sorted(claimed) == [0, 2]

    def test_the_deferred_event_is_picked_up_by_the_next_segment(self, tmp_path):
        """Consuming out of catalogue order must not skip or repeat the event left behind.

        ``population_index`` counts events consumed rather than naming a catalogue position, so this
        is the assertion that the count and the order stay consistent with each other.
        """
        orchestrator = _orchestrator(tmp_path, "batched")
        end_time = _end_time(orchestrator)
        orchestrator.signal_adapter = _StubAdapter(leads_by_coa_time=self._leads(end_time))
        orchestrator._population_events = self._catalogue(end_time)

        first_ids, _ = orchestrator._events_for_this_segment()
        orchestrator._commit_consumed_events(first_ids)

        assert int(orchestrator.population_index) == 2, "two events consumed, whatever their positions"

        orchestrator.update_state()
        second_ids, _ = orchestrator._events_for_this_segment()

        assert second_ids == [1], "the deferred event, exactly once"
        orchestrator._commit_consumed_events(second_ids)
        assert int(orchestrator.population_index) == 3
        assert sorted(first_ids + second_ids) == [0, 1, 2], "every event claimed exactly once"

    def test_events_without_a_coalescence_time_are_never_left_behind(self, tmp_path):
        """They belong to every segment, so they must not sit behind an event that ends the walk."""
        orchestrator = _orchestrator(tmp_path, "batched")
        end_time = _end_time(orchestrator)
        orchestrator.signal_adapter = _StubAdapter(lead=1.0)
        orchestrator._population_events = (
            {**_COMPLETE_EVENT, "coa_time": end_time + 100.0},
            {"frequency": 100.0},
        )

        event_ids, _ = orchestrator._events_for_this_segment()

        assert event_ids == [1]

    def test_the_order_is_stable_for_events_sharing_a_start(self, tmp_path):
        """Equal keys keep catalogue order, so provenance and injection order stay reproducible."""
        orchestrator = _orchestrator(tmp_path, "batched")
        end_time = _end_time(orchestrator)
        orchestrator.signal_adapter = _StubAdapter(lead=2.0)
        orchestrator._population_events = tuple({**_COMPLETE_EVENT, "coa_time": end_time - 5.0} for _ in range(4))

        assert orchestrator._placement_order() == (0, 1, 2, 3)


class TestBothLoopsAgree:
    """``population_index`` is checkpointed, so the two execution modes must claim the same events."""

    def test_the_batched_walk_claims_the_same_events_as_the_rule(self, tmp_path):
        """``_events_for_this_segment`` must apply the boundary rule, not its own copy of one."""
        orchestrator = _orchestrator(tmp_path, "batched")
        orchestrator.signal_adapter = _StubAdapter(lead=3.6)
        end_time = _end_time(orchestrator)
        orchestrator._population_events = (
            {**_COMPLETE_EVENT, "coa_time": end_time - 1.0},
            {**_COMPLETE_EVENT, "coa_time": end_time + 1.0},  # coalesces later, starts inside
            {**_COMPLETE_EVENT, "coa_time": end_time + 10.0},
        )

        event_ids, events = orchestrator._events_for_this_segment()

        assert event_ids == [0, 1], "the event coalescing just past the boundary starts inside it"
        assert len(events) == 2
        assert int(orchestrator.population_index) == 0, "reading must not advance the checkpointed index"

    def test_the_per_event_loop_claims_the_same_events(self, tmp_path):
        """The same catalogue through the other loop, so a mode switch cannot skip or repeat."""
        orchestrator = _orchestrator(tmp_path, "batched")
        orchestrator.signal_adapter = _StubAdapter(lead=3.6)
        end_time = _end_time(orchestrator)
        events = (
            {**_COMPLETE_EVENT, "coa_time": end_time - 1.0},
            {**_COMPLETE_EVENT, "coa_time": end_time + 1.0},
            {**_COMPLETE_EVENT, "coa_time": end_time + 10.0},
        )
        orchestrator._population_events = events
        batched_ids, _ = orchestrator._events_for_this_segment()

        per_event = _orchestrator(tmp_path / "per", "per-event")
        per_event.signal_adapter = _StubAdapter(lead=3.6)
        per_event._population_events = events
        claimed = [
            index for index, parameters in enumerate(events) if per_event._event_starts_before_segment_end(parameters)
        ]

        assert claimed == batched_ids, (
            "the two modes must agree on the boundary; population_index is checkpointed across them"
        )


class TestNothingIsDiscarded:
    """The point of the rule, measured on assembled segments rather than argued from the boundary."""

    def _run_two_segments(self, orchestrator, coa_time: float):
        """Assemble two consecutive segments containing one event, and return them."""
        orchestrator._population_events = ({**_COMPLETE_EVENT, "coa_time": coa_time},)
        first = orchestrator.simulate().signal_segment
        orchestrator.update_state()
        second = orchestrator.simulate().signal_segment
        return first, second

    def test_a_waveform_straddling_the_boundary_keeps_all_of_its_energy(self, tmp_path, caplog):
        """No sample is dropped, and the check is the strain itself rather than the absence of a log.

        The event coalesces 1 s into the *second* segment, so its buffer starts 2.6 s before that
        segment begins. Claimed by ``coa_time`` it would be generated during the second segment, and
        injection would crop those 2.6 s -- 65% of its samples -- because the first segment is
        already written by then.

        This offset is deliberately the *mild* case. **At this module's 30 Hz cutoff, on the
        ET-Triangle-Sardinia network it configures**, the unweighted strain-squared energy lost runs
        0.34% at 1 s past the boundary, 11.5% at 0.25 s, 50.3% at 0.1 s and 99.8% at 1 ms, because a
        compact binary's energy is concentrated in the last fraction of a second before merger. Testing the mild case makes the assertion depend on placement rather
        than on that concentration: 0.34% is a loss no energy-fraction tolerance would forgive, and a
        test pinned at the catastrophic end would still pass if placement were only approximately
        right.

        The cutoff has to be stated, because it moves these numbers by more than an order of
        magnitude while leaving the dropped *span* untouched. The same binary and network at a 20 Hz
        cutoff -- the one a real ET configuration uses -- loses 32.9% at 0.5 s past the boundary
        against 0.92% here at 30 Hz, since LAL's conditioning rounds both to the same 4.000 s buffer
        but only the lower cutoff puts real signal in the early samples. That rounding coincidence
        holds for this chirp-time bin on this backend and not generally -- a 10+10 binary gets 16 s at
        20 Hz against 8 s at 30 Hz -- so it is not a rule to carry elsewhere. See
        ``WaveformBackend.pre_coalescence_duration`` in gwmock-signal for the figures across cutoffs,
        offsets, masses and backends.

        Summed squares across the two assembled segments are compared against the same waveform
        generated on its own, which is the quantity that must survive placement.
        """
        orchestrator = _orchestrator(tmp_path, "per-event", total_duration=2 * _SEGMENT_DURATION)
        adapter = orchestrator.signal_adapter
        assert adapter is not None
        coa_time = _START + _SEGMENT_DURATION + 1.0
        standalone = adapter.simulate(
            {**_COMPLETE_EVENT, "coa_time": coa_time},
            sampling_frequency=_SAMPLING_FREQUENCY,
            minimum_frequency=30.0,
        )
        expected_energy = float(np.sum(np.square(np.atleast_2d(np.asarray(standalone, dtype=float)))))

        with caplog.at_level(logging.WARNING, logger="gwmock"):
            first, second = self._run_two_segments(orchestrator, coa_time)

        placed_energy = sum(
            float(np.sum(np.square(np.atleast_2d(np.asarray(segment, dtype=float))))) for segment in (first, second)
        )

        assert "Discarding" not in caplog.text, caplog.text
        # atol=0.0 because strain-squared energies are ~1e-45: the default atol would make any two
        # such numbers compare equal and the assertion could not fail.
        assert placed_energy == pytest.approx(expected_energy, rel=1e-9, abs=0.0), (
            f"placed {placed_energy} of {expected_energy}; the boundary lost "
            f"{100.0 * (1.0 - placed_energy / expected_energy):.2f}% of the waveform"
        )

    def test_the_event_is_attributed_to_the_segment_its_waveform_starts_in(self, tmp_path):
        """Provenance follows placement: the frame listing the event is the one its data begins in.

        Stated rather than assumed, because it is a change -- an event used to be listed against the
        frame containing its coalescence, and a reader of the metadata needs to know which.

        About ``_batch_injections``, which is what each batch *generated*. That is deliberately not
        what reaches the metadata: a signal crossing a boundary is present in both frames, so
        ``_segment_injections`` cross-lists it, and
        ``test_provenance_across_segments.py`` pins that. The two used to be the same list, and
        conflating them is what made `gwmock find-signal` name one frame out of three.
        """
        orchestrator = _orchestrator(tmp_path, "per-event", total_duration=2 * _SEGMENT_DURATION)
        coa_time = _START + _SEGMENT_DURATION + 1.0

        orchestrator._population_events = ({**_COMPLETE_EVENT, "coa_time": coa_time},)
        orchestrator.simulate()
        first_segment_injections = list(orchestrator._batch_injections)
        orchestrator.update_state()
        orchestrator.simulate()
        second_segment_injections = list(orchestrator._batch_injections)

        assert [entry["event_id"] for entry in first_segment_injections] == [0], (
            "the event belongs to the first segment, where its waveform starts"
        )
        assert second_segment_injections == [], (
            "and the second segment generated nothing -- `_batch_injections` is what this batch "
            "produced, which is the attribution #311 changed"
        )


class TestWhatAConsumerSees:
    """The attribution change reaches serialized metadata and the lookup, so it is checked there."""

    def test_a_signal_crossing_a_boundary_is_recorded_against_both_segments(self, tmp_path):
        """The metadata a consumer reads must name every frame the signal is in, not only the first.

        Goes through the orchestrator rather than the helpers because the defect was in the *wiring*:
        `_contributing_injections` can be perfectly correct while nothing calls it before the chunks
        are injected, and after injection the attribution is gone -- chunks are summed into shared
        channels. Reproduced end to end before the fix, this second segment recorded ``injections:
        []`` while holding the merger.
        """
        orchestrator = _orchestrator(tmp_path, "per-event", total_duration=2 * _SEGMENT_DURATION)
        coa_time = _START + _SEGMENT_DURATION + 1.0

        orchestrator._population_events = ({**_COMPLETE_EVENT, "coa_time": coa_time},)
        orchestrator.simulate()
        first = [entry["event_id"] for entry in orchestrator._segment_injections()]
        orchestrator.update_state()
        orchestrator.simulate()
        second = [entry["event_id"] for entry in orchestrator._segment_injections()]

        assert first == [0], "the segment its waveform starts in must record it"
        assert second == [0], (
            "the segment holding the rest of the signal -- including the coalescence -- recorded "
            "nothing, so `find-signal` names one frame out of two"
        )

    def test_the_schema_version_records_that_the_attribution_changed(self):
        """No field was added or removed, so nothing but the version tells the conventions apart.

        Three changes now ride on this version, and the first two are invisible in the shape of a
        record. 1.4.0: ``injections`` lists an event against the frame its waveform *starts* in
        rather than the frame holding its coalescence. 1.5.0: ``injections`` lists every event
        *present* in the frame, including one generated for an earlier segment, and
        ``signal_index.yaml`` stores contributions per batch so one event can name the frames of
        several. 1.6.0: ``noise.glitch_injections`` records the per-event glitch truth, which an
        older record simply lacks -- and lacking the key is not the same fact as having injected no
        glitches.

        A consumer reading an old record and a new one sees the same shape while ``injections``
        means something different in each. The version is the only signal available, so it has to
        move.
        """
        from gwmock.cli.utils.metadata import SCHEMA_VERSION

        assert SCHEMA_VERSION == "1.6.0"

    def test_an_older_record_still_loads(self):
        """Bumping the minor must not orphan archived runs: the major is what gates parsing."""
        from gwmock.cli.utils.metadata import MetadataRecord

        record = MetadataRecord.model_validate(
            {
                "schema_version": "1.3.0",
                "subpackage_versions": {},
                "config": {},
                "config_sha256": "0" * 64,
                "host": {"platform": "linux", "python": "3.12", "cpu": "x86_64"},
            }
        )

        assert record.schema_version == "1.3.0"

    def test_the_lookup_reports_the_frame_the_injection_was_recorded_against(self, tmp_path):
        """``find_signal`` reads ``signal.injections`` generically, so it follows placement.

        Pinned because it is the user-visible consequence: an event is reported against the frame its
        waveform starts in, which for a large lead can be a frame holding mostly quiet lead-in rather
        than the merger. The lookup itself needs no change; this is here so that stays true.
        """
        import json

        from gwmock.cli.utils.signal_lookup import find_signals

        metadata_directory = tmp_path / "metadata"
        metadata_directory.mkdir()
        (metadata_directory / "orchestration-0.metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.5.0",
                    "subpackage_versions": {},
                    "config": {},
                    "config_sha256": "0" * 64,
                    "host": {"platform": "linux", "python": "3.12", "cpu": "x86_64"},
                    "signal": {
                        "backend": "stub",
                        "injections": [{"event_id": 7, "parameters": {"coa_time": 1000.0}}],
                    },
                    "outputs": [{"kind": "signal", "path": "signal/sig-0.gwf"}],
                }
            )
        )

        results = find_signals(metadata_directory, param_filters=[("coa_time", "==", 1000.0)])

        assert [entry["event_id"] for entry in results] == [7]
        assert results[0]["frames"] == ["signal/sig-0.gwf"], (
            "the frame listed is the one the injection was recorded against, which is now the frame "
            "the waveform starts in"
        )


class TestAgainstRealGeneration:
    """The predicted lead has to be the lead generation actually produces, or placement is wrong."""

    def test_the_predicted_lead_matches_the_generated_buffer(self, tmp_path):
        """Measured against the produced ``TimeSeries``, not against another prediction.

        This is what makes the rule safe: if the query said less than generation produces, a segment
        chosen from it would still crop the start.
        """
        orchestrator = _orchestrator(tmp_path)
        adapter = orchestrator.signal_adapter
        assert adapter is not None
        coa_time = _START + 8.0
        parameters = {**_COMPLETE_EVENT, "coa_time": coa_time}

        predicted = adapter.pre_coalescence_duration(
            parameters,
            sampling_frequency=_SAMPLING_FREQUENCY,
            minimum_frequency=30.0,
        )
        strain = adapter.simulate(parameters, sampling_frequency=_SAMPLING_FREQUENCY, minimum_frequency=30.0)
        measured = coa_time - float(strain.start_time.value)

        assert predicted is not None, "LAL can answer this; a None here would disable the fix silently"
        assert predicted == pytest.approx(measured, abs=0.5 / _SAMPLING_FREQUENCY), (
            f"predicted lead {predicted} s but generation starts {measured} s before coalescence"
        )

    def test_an_event_just_past_the_boundary_is_claimed_using_the_real_query(self, tmp_path):
        """End to end through the real adapter: no stub, no supplied lead."""
        orchestrator = _orchestrator(tmp_path)
        end_time = _end_time(orchestrator)

        assert orchestrator._event_starts_before_segment_end({**_COMPLETE_EVENT, "coa_time": end_time + 1.0}), (
            "a 30+25 binary at 1024 Hz leads coalescence by 3.6 s, so this event starts in this segment"
        )
        assert not orchestrator._event_starts_before_segment_end({**_COMPLETE_EVENT, "coa_time": end_time + 5.0})
