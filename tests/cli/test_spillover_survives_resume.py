#
# Copyright (C) 2026 Leuven Gravity Institute
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
"""A resumed run must place the tail of a signal that crosses the resume point.

Measured on ``main`` with a 1.6+1.4 binary from 30 Hz across 32 s segments: interrupt after the
first batch checkpoints, re-run the same config, and the third frame -- the one holding the
**merger** -- comes back at a peak of exactly ``0.0`` against ``7.280e-23`` uninterrupted. The two
frames before it are bit-identical, so nothing about the run looks wrong; the loudest part of the
signal is simply absent.

The cause is that spillover lived only in memory. ``cached_data_chunks`` holds the part of a chunk
extending past the segment being built, and it was neither a ``StateAttribute`` nor saved anywhere
else, so a resumed process started with none of it.

Two things had to be right, and they fail independently:

* the checkpoint has to carry the chunks at all, and carry their ``metadata`` with them -- restoring
  the samples while dropping ``injection_parameters`` and ``event_id`` would look like a fix while
  losing the provenance the frames are indexed by;
* ``restore_batch_state`` has to put them back on the simulator.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from astropy.units.quantity import Quantity

from gwmock.cli.utils.checkpoint import CheckpointManager
from gwmock.data.time_series.time_series import TimeSeries
from gwmock.data.time_series.time_series_list import TimeSeriesList


def _chunk(start: float = 100.0, samples: int = 8, event_id: int = 3) -> TimeSeries:
    chunk = TimeSeries(
        data=np.arange(1.0, samples + 1.0).reshape(1, samples),
        start_time=Quantity(start, unit="s"),
        sampling_frequency=Quantity(8.0, unit="Hz"),
    )
    chunk.metadata.update({"injection_parameters": {"coa_time": start + 1.0}, "event_id": event_id})
    chunk[0] = dataclasses.replace(chunk[0], name="H1:STRAIN")
    chunk.channels[0] = "H1:STRAIN"
    return chunk


class TestTheCheckpointCarriesSpillover:
    """The serialization half."""

    def test_samples_and_provenance_both_survive(self, tmp_path):
        """Both, because restoring one without the other is the failure that looks like success.

        ``to_json_dict`` carried neither ``metadata`` nor the channel identity, so a fix that only
        made the chunks reachable would have resumed with data that no longer says which signal it
        is -- and ``inject`` copies the channel name onto a tail deliberately, for the same reason.
        """
        manager = CheckpointManager(tmp_path)
        manager.save_checkpoint([0], "orchestration", 0, {"counter": 1}, TimeSeriesList([_chunk()]))

        restored = manager.get_last_simulator_spillover()

        assert len(restored) == 1
        np.testing.assert_array_equal(np.asarray(restored[0]).ravel(), np.arange(1.0, 9.0))
        assert restored[0].metadata["event_id"] == 3
        assert restored[0].metadata["injection_parameters"] == {"coa_time": 101.0}
        # `restored[0]` is the first *series*; its channels are indexed one level in.
        assert restored[0][0].name == "H1:STRAIN"
        assert restored[0].channels[0] == "H1:STRAIN"
        assert float(restored[0].start_time.value) == 100.0

    def test_a_checkpoint_written_before_this_field_reads_as_no_spillover(self, tmp_path):
        """An upgrade mid-run must not turn a resumable checkpoint into a crash.

        ``None`` covers both "written by an older gwmock" and "that segment had no spillover", which
        are the same thing to the caller: there is nothing to carry in.
        """
        import json

        (tmp_path / "simulation.checkpoint.json").write_text(
            json.dumps(
                {
                    "completed_batch_indices": [0],
                    "last_simulator_name": "orchestration",
                    "last_completed_batch_index": 0,
                    "last_simulator_state": {"counter": 1},
                }
            )
        )

        assert CheckpointManager(tmp_path).get_last_simulator_spillover() is None


class TestSpilloverIsScopedToItsOwnBatch:
    """One checkpoint holds one tail, and handing it to the wrong consumer is worse than losing it.

    A plan can execute several simulators, and a lost tail is a hole -- visible, once you look for
    it. A *misplaced* tail is real strain of the right shape at the wrong time, in the wrong
    simulator's segment, and nothing downstream would flag it.
    """

    def test_another_simulators_spillover_is_refused(self, tmp_path):
        manager = CheckpointManager(tmp_path)
        manager.save_checkpoint([0], "orchestration", 0, {"counter": 1}, TimeSeriesList([_chunk()]))

        assert manager.get_last_simulator_spillover("noise", 1) is None
        assert manager.get_last_simulator_spillover("orchestration", 1) is not None

    def test_spillover_is_refused_for_any_batch_but_the_next_one(self, tmp_path):
        """It belongs to the batch immediately after the one that produced it, and nowhere else."""
        manager = CheckpointManager(tmp_path)
        manager.save_checkpoint([0], "orchestration", 0, {"counter": 1}, TimeSeriesList([_chunk()]))

        assert manager.get_last_simulator_spillover("orchestration", 1) is not None
        assert manager.get_last_simulator_spillover("orchestration", 0) is None, "the batch that produced it"
        assert manager.get_last_simulator_spillover("orchestration", 2) is None, "a later batch"


class TestRestorePutsSpilloverBack:
    """The wiring half, which the serialization test above cannot reach."""

    @staticmethod
    def _simulator_and_batch():
        from gwmock.cli.utils.simulation_plan import SimulationBatch

        class _Simulator:
            def __init__(self):
                self.cached_data_chunks = TimeSeriesList()
                self.counter = 1

            @property
            def state(self):
                return {"counter": self.counter}

            @state.setter
            def state(self, value):
                self.counter = value.get("counter", self.counter)

        batch = SimulationBatch.__new__(SimulationBatch)
        object.__setattr__(batch, "batch_index", 1)
        object.__setattr__(batch, "simulator_name", "orchestration")
        object.__setattr__(batch, "metadata", None)
        return _Simulator(), batch

    def test_the_simulator_gets_the_chunks(self):
        """Without this the checkpoint holds the tail and nothing ever reads it back."""
        from gwmock.cli.simulate_utils import restore_batch_state

        simulator, batch = self._simulator_and_batch()
        spillover = TimeSeriesList([_chunk()])

        restore_batch_state(simulator, batch, {"counter": 1}, spillover)

        assert len(simulator.cached_data_chunks) == 1
        assert simulator.cached_data_chunks[0].metadata["event_id"] == 3

    def test_no_spillover_leaves_the_simulator_alone(self):
        """A resumed run whose previous segment spilled nothing must not be handed an empty list.

        The distinction matters because ``None`` and ``TimeSeriesList()`` are both falsy: assigning
        unconditionally would work here and silently clobber chunks in any caller that had already
        populated them.
        """
        from gwmock.cli.simulate_utils import restore_batch_state

        simulator, batch = self._simulator_and_batch()
        simulator.cached_data_chunks = TimeSeriesList([_chunk(event_id=9)])

        restore_batch_state(simulator, batch, {"counter": 1}, None)

        assert len(simulator.cached_data_chunks) == 1
        assert simulator.cached_data_chunks[0].metadata["event_id"] == 9


class TestARetryDoesNotConsumeTheSpillover:
    """A retried batch must start from the same tail the first attempt did.

    `simulate` consumes `cached_data_chunks` and replaces it with the *new* tail, so restoring only
    `state` on retry leaves the second attempt working from consumed chunks. It would then write
    different data than the first attempt tried to -- and a retry that succeeds looks like a
    success, so nothing would say which of the two a frame came from.
    """

    def test_a_forced_failure_leaves_the_second_attempt_the_same_tail(self):
        """A real retry, not an inspection of the source that implements one.

        An earlier version of this test read `execute_plan`'s closure with `inspect.getsource`. That
        catches a revert to state-only and nothing else -- it cannot tell whether restoring the tail
        actually gives the second attempt the same input. This drives `retry_with_backoff` with the
        same capture-and-restore `execute_plan` builds, and a callable that fails once.

        The assertion is what each attempt *consumed*: with the restore, both attempts see the full
        tail; without it, the second sees what the first left behind.
        """
        import copy

        from gwmock.cli.simulate_utils import retry_with_backoff

        class _Simulator:
            def __init__(self):
                self.cached_data_chunks = TimeSeriesList([_chunk(event_id=4)])
                self.state = {"counter": 0}

        simulator = _Simulator()
        pre_batch_state = copy.deepcopy(simulator.state)
        pre_batch_spillover = copy.deepcopy(simulator.cached_data_chunks)

        consumed: list[int] = []

        def _attempt():
            # What `TimeSeriesMixin.simulate` does to this attribute: read it, then replace it with
            # the new tail. The second attempt therefore starts from whatever the first left.
            consumed.append(len(simulator.cached_data_chunks))
            simulator.cached_data_chunks = TimeSeriesList()
            if len(consumed) == 1:
                raise RuntimeError("forced write failure")
            return "ok"

        def _restore(_simulator=simulator, _state=pre_batch_state, _spillover=pre_batch_spillover):
            _simulator.state = copy.deepcopy(_state)
            _simulator.cached_data_chunks = copy.deepcopy(_spillover)

        result = retry_with_backoff(_attempt, max_retries=2, initial_delay=0.0, state_restore_func=_restore)

        assert result == "ok"
        assert consumed == [1, 1], (
            f"the two attempts consumed {consumed}; [1, 0] means the retry ran from chunks the "
            f"first attempt had already taken, so it would write different data"
        )


_RESUME_CONFIG = """
globals:
    simulator-arguments:
        sampling-frequency: 1024
        duration: 32
        total-duration: 96
        start-time: 1000000540
    working-directory: .
    output-directory: output
    metadata-directory: metadata

orchestration:
    population:
        backend: FilePopulationLoader
        source-type: bns
        arguments:
            path: pop.csv
    signal:
        waveform-model: IMRPhenomXPHM
        minimum-frequency: 30
        earth-rotation: true
        detectors:
            - ET-Triangle-Sardinia
        output:
            output_directory: signal
            file_name: 'E-{{ detectors }}_STRAIN_BNS-{{ start_time }}-{{ duration }}.gwf'
            arguments:
                channel: '{{ detectors }}:STRAIN'
"""

#: A 1.6+1.4 binary coalescing 6 s into the third 32 s segment. From 30 Hz its inspiral runs about
#: 48 s, so it spans all three -- which is the only reason this test can see anything.
_RESUME_POPULATION = (
    "detector_frame_mass_1,detector_frame_mass_2,coa_time,distance,"
    "declination,right_ascension,polarization_angle,inclination\n"
    "1.6,1.4,1000000610.0,100.0,0.3,1.1,0.2,0.5\n"
)


def _gwmock_executable() -> str:
    """Return the absolute path to the console script, or skip.

    Absolute rather than leaving PATH resolution to the subprocess: that is both a lint failure and
    a real ambiguity when more than one environment is active.
    """
    import shutil

    executable = shutil.which("gwmock")
    if executable is None:
        pytest.skip("the gwmock console script is not on PATH")
    return executable


def _run_to_completion(directory):
    import subprocess

    subprocess.run(  # noqa: S603 - absolute path, resolved by `_gwmock_executable`
        [_gwmock_executable(), "simulate", "config.yaml"],
        cwd=directory,
        capture_output=True,
        check=True,
        timeout=900,
    )


def _peaks(directory) -> list[float]:
    import glob

    from gwpy.io.gwf import iter_channel_names
    from gwpy.timeseries import TimeSeries as GWpyTimeSeries

    files = sorted(glob.glob(str(directory / "output" / "signal" / "*ET1*.gwf")))
    if not files:
        return []
    channel = next(iter(iter_channel_names(files[0])))
    return [float(np.max(np.abs(np.asarray(GWpyTimeSeries.read(f, channel).value)))) for f in files]


@pytest.mark.integration
def test_a_resumed_run_places_the_tail_end_to_end(tmp_path):
    """Interrupt a real run at a real checkpoint, resume it, and require the merger back.

    This is the only test here that guards the **save** side. Everything above hands the spillover
    to ``save_checkpoint`` itself, so gutting the one call site in ``execute_plan`` leaves them all
    green -- measured. It is also the only one that exercises the two halves together.

    Deterministic despite involving a signal: the interrupt waits for
    ``simulation.checkpoint.json`` to appear rather than for a wall-clock delay, so it lands after a
    batch has committed however slow the machine is. If the file never appears the test fails on the
    timeout rather than passing vacuously.

    Compared against an uninterrupted run in a separate directory rather than against a literal:
    the peaks depend on the waveform model and would otherwise need updating whenever ripple changes.
    On the unfixed code the third peak is exactly 0.0 against 7.280e-23.
    """
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    import json
    import signal as signal_module
    import subprocess
    import time

    executable = _gwmock_executable()

    reference = tmp_path / "reference"
    resumed = tmp_path / "resumed"
    for directory in (reference, resumed):
        directory.mkdir()
        (directory / "config.yaml").write_text(_RESUME_CONFIG)
        (directory / "pop.csv").write_text(_RESUME_POPULATION)

    _run_to_completion(reference)
    expected = _peaks(reference)
    assert len(expected) == 3, f"the reference run wrote {len(expected)} frames, not 3"
    assert expected[-1] > 0.0, "the reference run has no signal in the last frame, so this proves nothing"

    checkpoint = resumed / ".gwmock_checkpoints" / "simulation.checkpoint.json"
    process = subprocess.Popen(  # noqa: S603 - absolute path, resolved by `_gwmock_executable`
        [executable, "simulate", "config.yaml"], cwd=resumed, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        deadline = time.monotonic() + 600.0
        while not checkpoint.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert checkpoint.exists(), "no checkpoint was written, so there was nothing to resume from"
        # The one way this test could pass without testing anything: if the interrupt lands after
        # the *last* batch has checkpointed, the resume has nothing left to do and the frames match
        # trivially. Reading the checkpoint here rules that out rather than trusting the timing.
        completed = set(json.loads(checkpoint.read_text()).get("completed_batch_indices") or [])
        assert completed, "the checkpoint records no completed batch, so there is nothing to resume from"
        assert len(completed) < 3, (
            f"the interrupt landed with batches {sorted(completed)} complete out of 3; the resume "
            f"would have nothing to carry, so this run proves nothing"
        )
        process.send_signal(signal_module.SIGINT)
        process.wait(timeout=120)
    finally:
        if process.poll() is None:  # pragma: no cover - only on an unexpected hang
            process.kill()

    _run_to_completion(resumed)

    actual = _peaks(resumed)
    assert len(actual) == len(expected)
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        # atol=0.0: these are ~1e-23, and any default absolute tolerance would make two arbitrary
        # strain arrays compare equal, so the assertion could not fail.
        assert got == pytest.approx(want, rel=1e-9, abs=0.0), (
            f"frame {index} differs after a resume: {got:.3e} against {want:.3e} uninterrupted"
        )
