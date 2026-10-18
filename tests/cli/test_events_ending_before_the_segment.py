"""Events whose waveform has finished before the segment starts, and the batch they inflate.

The companion to :mod:`tests.cli.test_inspiral_segment_placement`, from the other side. That
module pins the *upper* bound: a segment claims an event whose waveform starts before the segment
ends, so an inspiral is not cropped. The rule has no lower bound, and the omission is invisible in
a sequential run -- earlier segments have already consumed the earlier events, so the walk starts
past them.

It is not invisible in a run whose ``start-time`` is later than its population's first event.
Nothing has consumed those events, every one of them satisfies "starts before this segment ends",
and the first segment claims the entire back catalogue. **Measured on the shipped ET config:** a
run starting 53 ks into the catalogue batched 126 events where the segment contains 13, estimated
12.5 GiB, and was refused by the preflight; at 41 ks it batched 98 for 16; at 4 ks, 19 for 8. The
batch tracks the offset rather than the local event density. Their samples are then cropped as
out-of-segment, so the work is generated and discarded.

Resuming a campaign mid-catalogue, or slicing a long population into per-day jobs, is exactly this.

**Why the fix cannot be another clause in the same predicate.** Both loops ``break`` on the first
event that does not belong, and ``population_index`` -- checkpointed state meaning "every event
before this one is consumed" -- advances by the count of events consumed. An early event answering
"does not belong" would therefore stop the walk at position 0 and consume nothing, and it would do
so again on every following segment: the run would stall permanently rather than over-batch. An
event that has finished must be *skipped and consumed*, which is a different operation from the
one the upper bound performs, and ``test_a_finished_event_does_not_stop_the_walk`` is what pins
the difference.

``None`` from the query means unknown, and unknown must claim. The query exists to let a caller
*prove* an event is finished; on no answer the conservative branch is the one that generates the
event and lets injection crop it, because the alternative silently deletes signal.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from gwmock.cli.utils.config import Config

_POPULATION_CSV = Path(__file__).resolve().parents[2] / "examples" / "signal" / "bbh_population.csv"
_START = 1577491296.0
_SAMPLING_FREQUENCY = 1024.0
_SEGMENT_DURATION = 16.0

#: A complete BBH parameter set, so a real query could answer; these tests supply the answer
#: through the stub, but an underspecified event would exercise the unknown path by accident.
_COMPLETE_EVENT: dict[str, Any] = {
    "detector_frame_mass_1": 30.0,
    "detector_frame_mass_2": 25.0,
    "distance": 400.0,
    "right_ascension": 1.0,
    "declination": 0.5,
    "polarization_angle": 0.2,
    "inclination": 0.3,
}


def _config(working_directory: Path, execution: str) -> dict[str, Any]:
    """Return a one-segment BBH config, matching the placement module's fixture.

    No ``waveform-backend`` key, so constructing the orchestrator does not instantiate a backend
    needing the ``[jax]`` extra.
    """
    return {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": _SAMPLING_FREQUENCY,
                "duration": _SEGMENT_DURATION,
                "total-duration": _SEGMENT_DURATION,
                "start-time": _START,
                "seed": 20260807,
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


def _orchestrator(working_directory: Path, execution: str = "batched"):
    from gwmock.cli.adapter_orchestration import AdapterOrchestrator

    working_directory.mkdir(parents=True, exist_ok=True)
    config = Config.model_validate(_config(working_directory, execution))
    return AdapterOrchestrator.from_config(
        config.orchestration,
        global_simulator_arguments=dict(config.globals.simulator_arguments),
    )


class _StubAdapter:
    """A signal adapter answering both placement queries however the test needs.

    Both sides are supplied, because a test that fixed only the tail would let the lead default to
    something the sort key also depends on, and the two together decide which events the walk even
    reaches.
    """

    def __init__(self, lead: float | None = 0.0, tail: float | None = 0.0) -> None:
        self._lead = lead
        self._tail = tail
        self.detector_names = ("E1",)
        #: Every event handed to :meth:`simulate`, so a test can assert what was *generated* rather
        #: than what a predicate said. The two are not the same thing, which is the whole reason
        #: this list exists -- see ``test_the_per_event_loop_steps_over_finished_events``.
        self.simulated: list[float] = []

    def simulate(self, parameters: Mapping[str, Any], **_: Any):
        """Return a one-sample segment-length strain, recording that the event was generated."""
        import numpy as np

        from gwmock.data.time_series.time_series import TimeSeries

        self.simulated.append(float(parameters["coa_time"]))
        return TimeSeries(
            np.zeros((1, int(_SAMPLING_FREQUENCY * _SEGMENT_DURATION))),
            start_time=float(parameters["coa_time"]),
            sampling_frequency=_SAMPLING_FREQUENCY,
        )

    def pre_coalescence_duration(self, parameters: Mapping[str, Any], **_: Any) -> float | None:
        del parameters
        return self._lead

    def post_coalescence_duration(self, parameters: Mapping[str, Any], **_: Any) -> float | None:
        del parameters
        return self._tail


def _install_events(orchestrator, coa_times: list[float]) -> None:
    """Give the orchestrator a synthetic catalogue and drop the cached placement order.

    The order is cached on first use and keyed by the events present when it was built, so a test
    replacing the catalogue afterwards would otherwise walk positions that no longer exist.
    """
    orchestrator._population_events = [{**_COMPLETE_EVENT, "coa_time": t} for t in coa_times]
    orchestrator._placement_order_cache = None
    orchestrator.population_index = 0


def _start_time(orchestrator) -> float:
    return float(getattr(orchestrator.start_time, "value", orchestrator.start_time))


class TestTheBatchDoesNotGrowWithTheOffset:
    """The defect: a run starting after its population's first event batches all of them."""

    @pytest.mark.requires_lalsuite
    def test_the_batch_holds_the_events_in_the_window_not_the_ones_before_it(self, tmp_path):
        """The measured failure, reduced to its shape.

        Ten events sit wholly before the segment -- each finishing, with its 0.4 s tail, well
        before the segment starts -- and three fall inside it. A run whose ``start-time`` skipped
        the first ten claims all thirteen, generates ten waveforms whose samples are then cropped
        as out-of-segment, and sizes its batch from the offset rather than from the segment.
        """
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(lead=3.6, tail=0.4)
        start = _start_time(orchestrator)
        before = [start - 3600.0 + 60.0 * i for i in range(10)]
        inside = [start + 1.0, start + 5.0, start + 9.0]
        _install_events(orchestrator, before + inside)

        event_ids, events = orchestrator._events_for_this_segment()

        assert len(events) == 3, f"batched {len(events)} events for a segment containing 3"
        assert sorted(event_ids) == [10, 11, 12]

    @pytest.mark.requires_lalsuite
    def test_the_batch_tracks_the_segment_not_the_distance_from_the_catalogue_start(self, tmp_path):
        """Doubling the offset must not change the batch.

        This is the property the measurement actually showed violated -- 126, then 98, then 19 for
        the same kind of segment -- and it is the one a caller relies on when slicing a population
        into per-day jobs. A test fixing a single offset would pass against a fix that merely
        shifted the over-claiming.
        """
        counts = []
        for offset in (600.0, 1200.0, 2400.0):
            orchestrator = _orchestrator(tmp_path / f"offset_{int(offset)}")
            orchestrator.signal_adapter = _StubAdapter(lead=3.6, tail=0.4)
            start = _start_time(orchestrator)
            n_before = int(offset / 60.0)
            before = [start - offset + 60.0 * i for i in range(n_before)]
            _install_events(orchestrator, [*before, start + 1.0, start + 5.0])
            counts.append(len(orchestrator._events_for_this_segment()[1]))

        assert counts == [2, 2, 2], f"batch grew with the offset: {counts}"

    @pytest.mark.requires_lalsuite
    def test_a_finished_event_does_not_stop_the_walk(self, tmp_path):
        """Skipped, not refused -- the distinction the loops' ``break`` makes load-bearing.

        If a finished event answered the boundary question with "does not belong", the walk would
        stop at position 0 and consume nothing, on this segment and on every one after it. The run
        would stall rather than over-batch, which is a worse failure than the one being fixed. So
        the events *behind* the finished one must still be claimed.
        """
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(lead=3.6, tail=0.4)
        start = _start_time(orchestrator)
        _install_events(orchestrator, [start - 1000.0, start + 2.0, start + 6.0])

        event_ids, events = orchestrator._events_for_this_segment()

        assert len(events) == 2, "the walk stopped at the finished event instead of stepping over it"
        assert sorted(event_ids) == [1, 2]

    @pytest.mark.requires_lalsuite
    def test_a_skipped_event_is_consumed_so_the_next_segment_does_not_reconsider_it(self, tmp_path):
        """``population_index`` means every event before it is consumed; skipping must keep that.

        Advancing only by the events *generated* would leave the finished ones in front of the
        index forever: every later segment would re-examine and re-skip them, and a resumed run
        would restart inside a prefix it had already dealt with. The count is what resume depends
        on, so it is asserted rather than the ids.
        """
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(lead=3.6, tail=0.4)
        start = _start_time(orchestrator)
        _install_events(orchestrator, [start - 900.0, start - 800.0, start + 2.0])

        event_ids, _ = orchestrator._events_for_this_segment()
        orchestrator._commit_consumed_events(event_ids)

        assert orchestrator.population_index == 3, (
            f"index advanced to {orchestrator.population_index}; the two skipped events are still "
            "in front of it and will be reconsidered by the next segment"
        )


class TestTheRealAdapterAnswers:
    """No stub. The one test here that would have caught the fix being inert in production.

    Every other test in this module supplies a ``_StubAdapter`` that defines
    ``post_coalescence_duration``, so all of them passed while the *real* ``SignalAdapter`` had no
    such method at all: the orchestrator's ``getattr`` guard returned ``None`` for every event, the
    tail was always unknown, and the lower bound never fired in any real run. A reviewer found it
    by instantiating an ordinary orchestrator and looking; the suite could not, because the suite
    only ever asked a stand-in that was built to answer.
    """

    @pytest.mark.requires_lalsuite
    def test_the_orchestrators_own_adapter_exposes_the_query(self, tmp_path):
        """The wiring, asserted on the object a run actually holds."""
        orchestrator = _orchestrator(tmp_path)

        assert hasattr(orchestrator.signal_adapter, "post_coalescence_duration"), (
            "the adapter a real run holds cannot answer the tail query, so the lower bound is dead "
            "code behind the orchestrator's getattr guard"
        )

    @pytest.mark.requires_lalsuite
    def test_the_real_adapter_returns_a_usable_tail(self, tmp_path):
        """Present is not the same as answering, so the value is checked too.

        A method returning ``None`` for a complete binary-black-hole event would satisfy the
        `hasattr` test above while leaving the behaviour exactly as broken. The magnitude is pinned
        loosely -- the point is that it is a real positive number from the installed backend, not
        that it equals a figure recomputed here.
        """
        orchestrator = _orchestrator(tmp_path)

        tail = orchestrator.signal_adapter.post_coalescence_duration(
            {**_COMPLETE_EVENT, "coa_time": _START + 4.0},
            sampling_frequency=_SAMPLING_FREQUENCY,
            minimum_frequency=30.0,
        )

        assert tail is not None, "the installed gwmock-signal did not answer the tail query"
        assert 0.0 < tail < 10.0, f"implausible tail for a 30+25 binary at 30 Hz: {tail} s"

    def test_the_orchestrator_reaches_the_query_through_its_own_adapter(self, tmp_path):
        """End to end through the private helper, with nothing substituted.

        Closes the last gap between "the adapter has the method" and "the orchestrator calls it":
        the two are wired by name, and a rename on either side would leave both tests above
        passing.

        Needs ripple, and that is the point rather than an inconvenience: this fixture is a *batched*
        config, and a batched run whose generating library is missing now answers the tail as
        unknown on purpose, so that nothing is skipped on a number describing a different library's
        buffer. Without the extra there is no answer to reach.
        """
        pytest.importorskip("ripplegw", reason="a batched run cannot answer the tail without it")
        orchestrator = _orchestrator(tmp_path)

        tail = orchestrator._post_coalescence_duration({**_COMPLETE_EVENT, "coa_time": _START + 4.0})

        assert tail is not None
        assert tail > 0.0


class TestAgainstRealGeneration:
    """The predicted tail has to be the tail generation actually produces.

    The post-side mirror of ``test_the_predicted_lead_matches_the_generated_buffer``, and named by a
    reviewer as the single most valuable test missing from this change. Every other test here asks
    whether the query *answers*; this one asks whether the answer is *true*. The direction matters:
    a tail shorter than generation produces means an event is skipped while it is still contributing
    signal to the segment -- deleting real data, which is the one outcome strictly worse than the
    wasted work this change removes. It held when the reviewer probed it by hand; nothing pinned it,
    so a regression in the adapter's parameter merge or the backend's arithmetic would pass the
    whole suite.
    """

    @pytest.mark.requires_lalsuite
    def test_the_predicted_tail_matches_the_generated_buffer(self, tmp_path):
        """Measured on the produced ``TimeSeries``, not recomputed from the same formula."""
        orchestrator = _orchestrator(tmp_path)
        adapter = orchestrator.signal_adapter
        assert adapter is not None
        coa_time = _START + 8.0
        parameters = {**_COMPLETE_EVENT, "coa_time": coa_time}

        predicted = adapter.post_coalescence_duration(
            parameters,
            sampling_frequency=_SAMPLING_FREQUENCY,
            minimum_frequency=30.0,
        )
        strain = adapter.simulate(parameters, sampling_frequency=_SAMPLING_FREQUENCY, minimum_frequency=30.0)
        # `end_time` is the series' own accessor rather than arithmetic on the array, so the test
        # cannot disagree with the object about where the buffer ends.
        measured = float(strain.end_time.value) - coa_time

        assert predicted is not None, "LAL can answer this; a None here would disable the fix silently"
        assert predicted == pytest.approx(measured, abs=0.5 / _SAMPLING_FREQUENCY), (
            f"predicted tail {predicted} s but generation ends {measured} s after coalescence"
        )

    @pytest.mark.requires_lalsuite
    def test_the_two_sides_account_for_the_whole_generated_buffer(self, tmp_path):
        """Lead plus tail is the buffer, checked against the buffer that was generated.

        Catches a drift that splits the buffer wrongly while keeping each side self-consistent --
        the failure the two separate agreement tests can each miss on their own.
        """
        orchestrator = _orchestrator(tmp_path)
        adapter = orchestrator.signal_adapter
        assert adapter is not None
        parameters = {**_COMPLETE_EVENT, "coa_time": _START + 8.0}
        kwargs = {"sampling_frequency": _SAMPLING_FREQUENCY, "minimum_frequency": 30.0}

        lead = adapter.pre_coalescence_duration(parameters, **kwargs)
        tail = adapter.post_coalescence_duration(parameters, **kwargs)
        strain = adapter.simulate(parameters, **kwargs)
        generated = float(strain.duration.value)

        assert lead is not None
        assert tail is not None
        assert lead + tail == pytest.approx(generated, abs=1.0 / _SAMPLING_FREQUENCY)


class TestADegradedAdapterIsReported:
    """An adapter that cannot answer must say so once, not disappear into a silent fallback."""

    @pytest.mark.requires_lalsuite
    def test_an_adapter_without_the_query_warns_rather_than_going_quiet(self, tmp_path, caplog):
        """Why the query is called directly instead of behind a ``getattr`` guard.

        The guard was how this change shipped as a no-op: the real ``SignalAdapter`` had no tail
        method, the guard returned ``None`` for every event, and nothing in a run said anything.
        Calling directly puts the ``AttributeError`` into the same handler the pre side uses, which
        warns once and falls back -- the codebase's own rule that a fallback's cost is reported
        rather than silent. Restoring the guard is invisible to every other test in this module,
        because with the adapter fixed the two are behaviourally identical; this is the one test
        that can tell them apart.
        """

        class _AdapterWithoutTheQuery:
            detector_names = ("E1",)

            def pre_coalescence_duration(self, parameters, **_):
                del parameters
                return 3.6

        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _AdapterWithoutTheQuery()

        with caplog.at_level("WARNING"):
            tail = orchestrator._post_coalescence_duration({**_COMPLETE_EVENT, "coa_time": _START})

        assert tail is None
        assert any("after coalescence" in record.message for record in caplog.records), (
            "the adapter could not answer and the run was told nothing"
        )

    @pytest.mark.requires_lalsuite
    def test_the_warning_is_not_repeated_once_per_event(self, tmp_path, caplog):
        """Deduplicated by reason across a whole walk, not merely on a single call.

        A catalogue the query cannot read fails identically for every row, and one message is right
        for it -- a per-event warning would bury the run's real output. Asserted over a walk rather
        than two direct calls, because the walk is where the repetition would actually happen.
        """

        class _AdapterWithoutTheQuery:
            detector_names = ("E1",)

            def pre_coalescence_duration(self, parameters, **_):
                del parameters
                return 3.6

        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _AdapterWithoutTheQuery()
        start = _start_time(orchestrator)
        _install_events(orchestrator, [start - 900.0, start - 800.0, start + 2.0, start + 6.0])

        with caplog.at_level("WARNING"):
            _, events = orchestrator._events_for_this_segment()

        warnings = [r for r in caplog.records if "after coalescence" in r.message]
        assert len(warnings) == 1, f"warned {len(warnings)} times across a {len(events)}-event walk"
        # And the degraded query must not have dropped anything: unknown claims.
        assert len(events) == 4


class TestTheIndexSurvivesInterleavedSkips:
    """Skipped and claimed events interleave, so the index cannot be advanced by either count alone.

    Consumption is ordered by waveform *start*; skipping is decided by waveform *end*. A long event
    starting early and overlapping the segment can therefore be followed by a short one that starts
    later and has already finished. Found by a reviewer, who built the case below by hand.
    """

    @pytest.mark.requires_lalsuite
    def test_a_claimed_event_is_not_left_behind_the_index_when_generation_fails(self, tmp_path):
        """The one that silently drops a real event.

        Event A starts early, ends inside the segment, and must be generated. Event B starts later
        and is already finished, so it is skipped. Advancing by the skip count alone moves the index
        past A -- and if generation then fails and a direct library caller retries on the same
        orchestrator, A is behind the index and is never generated at all. Wasted work is the bug
        being fixed here; losing a real event would be a worse one introduced by the fix.
        """
        orchestrator = _orchestrator(tmp_path)
        start = _start_time(orchestrator)
        orchestrator.signal_adapter = _StubAdapter()
        # The leads must differ, or the case does not exist. Consumption is ordered by waveform
        # start, so giving both events the same lead puts the finished one first and the skips are
        # a prefix after all -- the first version of this test did exactly that and passed against
        # the unsafe code. A starts earliest *and* overlaps; B starts later and has already ended.
        #   A: coa S+20, lead 100 -> starts S-80, ends S+520   claimed, walked first
        #   B: coa S-40, lead  10 -> starts S-50, ends S-39    skipped, walked second
        leads = {start + 20.0: 100.0, start - 40.0: 10.0}
        tails = {start + 20.0: 500.0, start - 40.0: 1.0}
        orchestrator.signal_adapter.pre_coalescence_duration = lambda parameters, **_: leads[
            float(parameters["coa_time"])
        ]
        orchestrator.signal_adapter.post_coalescence_duration = lambda parameters, **_: tails[
            float(parameters["coa_time"])
        ]
        orchestrator._population_events = [
            {**_COMPLETE_EVENT, "coa_time": start + 20.0},
            {**_COMPLETE_EVENT, "coa_time": start - 40.0},
        ]
        orchestrator._placement_order_cache = None
        orchestrator.population_index = 0

        # The order the walk will take, asserted rather than assumed: if a future change to the
        # placement key made the skipped event sort first again, this test would silently stop
        # covering the interleaving it exists for.
        order = orchestrator._placement_order()
        assert order == (0, 1), f"the interleaving this test needs does not exist in order {order}"

        event_ids, events = orchestrator._events_for_this_segment()

        assert len(events) == 1, "expected only the overlapping event to be claimed"
        # Generation has NOT happened, so nothing may have been committed for the claimed event.
        # Only the walk's own bookkeeping may have moved, and it must not have stepped past A.
        remaining = orchestrator._placement_order()[int(orchestrator.population_index) :]
        assert event_ids[0] in remaining, (
            "the claimed event is behind population_index before its generation succeeded; a "
            "retry would never generate it"
        )

    @pytest.mark.requires_lalsuite
    def test_an_all_skipped_batch_advances_through_the_real_entry_point(self, tmp_path):
        """The case that justifies advancing inside the walk, exercised through the caller.

        The other index test drives `_events_for_this_segment` and `_commit_consumed_events`
        directly, which is exactly the path that cannot show this: `_simulate_batched_segment`
        returns early on an empty batch and never commits, so if the walk did not advance, the next
        segment would reconsider the same finished events -- and so would every segment after it.
        """
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(lead=3.6, tail=0.4)
        start = _start_time(orchestrator)
        _install_events(orchestrator, [start - 900.0, start - 800.0, start - 700.0])

        chunks = orchestrator._simulate_batched_segment()

        assert len(chunks) == 0
        assert orchestrator.population_index == 3, (
            f"index is {orchestrator.population_index}; the finished events would be walked again "
            "by every following segment"
        )


class TestUnknownStillClaims:
    """Unknown is not zero, and on this side reading it as zero deletes signal."""

    @pytest.mark.requires_lalsuite
    def test_an_event_whose_tail_is_unknown_is_claimed(self, tmp_path):
        """A backend that cannot say where its buffer ends must not have events dropped for it.

        PyCBC answers ``None`` by design, as does any custom-registered model. Reading that as a
        zero-length tail would conclude every such event finished at ``coa_time`` and discard the
        ones sitting before the segment -- silently, and for a whole backend at a time.
        """
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(lead=3.6, tail=None)
        start = _start_time(orchestrator)
        _install_events(orchestrator, [start - 1000.0, start + 2.0])

        _, events = orchestrator._events_for_this_segment()

        assert len(events) == 2, "an event with an unknown tail was dropped; unknown is not zero"

    @pytest.mark.requires_lalsuite
    def test_an_event_still_ringing_into_the_segment_is_claimed(self, tmp_path):
        """The tail is a fixed fraction of the buffer, so for a BNS it is tens of seconds.

        Measured on the shipped backends: 0.4 s for a 4 s BBH buffer, **25.6 s** for a 256 s BNS
        buffer. An event coalescing 10 s before the segment is therefore still producing signal
        inside it, and a lower bound written against a millisecond intuition would delete it.
        """
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(lead=100.0, tail=25.6)
        start = _start_time(orchestrator)
        _install_events(orchestrator, [start - 10.0, start + 2.0])

        _, events = orchestrator._events_for_this_segment()

        assert len(events) == 2, "an event whose ringdown reaches into the segment was dropped"

    @pytest.mark.requires_lalsuite
    def test_the_boundary_is_inclusive_the_way_the_upper_one_is(self, tmp_path):
        """An event ending exactly at the segment start contributes its final sample.

        Pinned because the off-by-one here is silent: it drops one event at one boundary, in a run
        that otherwise looks correct.
        """
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(lead=3.6, tail=2.0)
        start = _start_time(orchestrator)
        _install_events(orchestrator, [start - 2.0, start + 2.0])

        _, events = orchestrator._events_for_this_segment()

        assert len(events) == 2, "the event ending exactly at the segment start was dropped"


class TestBothLoopsAgree:
    """Per-event and batched must place identically, or resume skips or repeats events."""

    @pytest.mark.requires_lalsuite
    @pytest.mark.parametrize("execution", ["per-event", "batched"])
    def test_the_two_modes_claim_the_same_events(self, tmp_path, execution):
        """``population_index`` is checkpointed, so a run switched between modes must not disagree.

        Asserted through the shared predicate rather than by generating, because the per-event loop
        generates as it walks and this is a placement question.
        """
        orchestrator = _orchestrator(tmp_path / execution, execution=execution)
        orchestrator.signal_adapter = _StubAdapter(lead=3.6, tail=0.4)
        start = _start_time(orchestrator)
        _install_events(orchestrator, [start - 1000.0, start - 500.0, start + 2.0, start + 6.0])

        finished = [
            orchestrator._event_ended_before_segment_start(dict(orchestrator._population_events[i])) for i in range(4)
        ]

        assert finished == [True, True, False, False]

    @pytest.mark.requires_lalsuite
    def test_the_per_event_loop_steps_over_finished_events(self, tmp_path):
        """Asserted on what the loop *generated*, not on what the predicate returned.

        This test exists because the first version of this module did not have it, and a mutation
        deleting the per-event loop's lower bound outright survived the whole suite: every other
        test here reaches the rule through ``_events_for_this_segment`` or by calling the predicate
        directly, so the per-event walk -- the default execution mode -- was unpinned while looking
        covered. Both loops share the predicate but apply it separately, and ``population_index``
        is checkpointed state, so a mode that keeps generating finished events also breaks resume
        for a run switched between modes.
        """
        orchestrator = _orchestrator(tmp_path, execution="per-event")
        adapter = _StubAdapter(lead=3.6, tail=0.4)
        orchestrator.signal_adapter = adapter
        start = _start_time(orchestrator)
        _install_events(orchestrator, [start - 1000.0, start - 500.0, start + 2.0])

        orchestrator._simulate()

        assert adapter.simulated == [start + 2.0], (
            f"generated {len(adapter.simulated)} event(s); the finished ones were not stepped over"
        )
        assert orchestrator.population_index == 3
