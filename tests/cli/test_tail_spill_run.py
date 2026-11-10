"""A run spills its signal tails beside the checkpoint, resumes from them, and removes them.

The mixin's half -- that a spilled tail places exactly what a tail in memory places -- is
``tests/mixin/test_tail_spill.py``. This is the run's half: ``execute_plan`` gives each simulator a
spill directory beside its checkpoint, the checkpoint records spilled tails by file, a resumed run
reads them back, a file is deleted once neither the checkpoint nor its backup needs it, and the
directory is gone when the run completes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from astropy.units import Quantity

from gwmock.cli.simulate_utils import execute_plan
from gwmock.cli.utils.checkpoint import CheckpointManager, spilled_tail_files
from gwmock.cli.utils.config import GlobalsConfig, SimulatorConfig, SimulatorOutputConfig
from gwmock.cli.utils.simulation_plan import SimulationBatch, SimulationPlan
from gwmock.data.time_series.spilled import SpilledTimeSeries
from gwmock.data.time_series.time_series import TimeSeries
from gwmock.data.time_series.time_series_list import TimeSeriesList
from gwmock.mixin.time_series import TimeSeriesMixin
from gwmock.simulator.base import Simulator

_RATE = 32.0
_DURATION = 4.0
_SEGMENTS = 7

#: The batch whose ``simulate`` raises, read at ``simulate`` time: ``execute_plan`` builds the
#: simulator from its configuration, so a test cannot reach the instance to arm it.
FAIL_AT_COUNTER: list[int] = []
#: The spilled-tail files on disk when each batch starts, keyed by counter.
FILES_AT_START: dict[int, set[str]] = {}


@pytest.fixture(autouse=True)
def _reset():
    FAIL_AT_COUNTER.clear()
    FILES_AT_START.clear()
    yield
    FAIL_AT_COUNTER.clear()
    FILES_AT_START.clear()


class LongTailSimulator(TimeSeriesMixin, Simulator):
    """Each segment generates one signal lasting 2.6 segments."""

    def _simulate(self, *args: Any, **kwargs: Any) -> TimeSeriesList:
        if self.tail_spill_directory is not None:
            root = Path(self.tail_spill_directory)
            FILES_AT_START[int(self.counter)] = {path.name for path in root.glob("*.npy")} if root.is_dir() else set()
        if int(self.counter) in FAIL_AT_COUNTER:
            raise RuntimeError(f"interrupted at batch {self.counter}")
        rng = np.random.default_rng(int(self.counter))
        chunk = TimeSeries(
            data=rng.standard_normal((1, int(2.6 * _DURATION * _RATE))),
            start_time=self.start_time + Quantity(1.0, unit="s"),
            sampling_frequency=self.sampling_frequency,
        )
        chunk.metadata.update({"injection_parameters": {"index": int(self.counter)}, "event_id": int(self.counter)})
        return TimeSeriesList([chunk])

    def update_state(self) -> None:
        super().update_state()
        self.start_time = self.start_time + self.duration

    def _save_data(self, data: Any, file_name: str | Path, **kwargs: Any) -> None:
        file_name = Path(file_name)
        file_name.parent.mkdir(parents=True, exist_ok=True)
        np.save(file_name, np.asarray(data[0].data))


def _plan(checkpoint_directory: Path) -> SimulationPlan:
    plan = SimulationPlan(checkpoint_directory=checkpoint_directory)
    for index in range(_SEGMENTS):
        plan.add_batch(
            SimulationBatch(
                simulator_name="long",
                simulator_config=SimulatorConfig(
                    class_="tests.cli.test_tail_spill_run.LongTailSimulator",
                    arguments={
                        "start_time": 1000,
                        "duration": _DURATION,
                        "total_duration": _SEGMENTS * _DURATION,
                        "sampling_frequency": _RATE,
                    },
                    output=SimulatorOutputConfig(file_name=f"segment-{index}.npy"),
                ),
                globals_config=GlobalsConfig(),
                batch_index=index,
            )
        )
    return plan


def _run(tmp_path: Path, name: str, expect_failure: bool = False) -> Path:
    root = tmp_path / name
    arguments = (_plan(root / "checkpoints"), root / "output", root / "metadata")
    if expect_failure:
        with pytest.raises(RuntimeError, match="interrupted"):
            execute_plan(*arguments, overwrite=True, max_retries=0)
    else:
        execute_plan(*arguments, overwrite=True, max_retries=0)
    return root


def _segments(root: Path) -> list[np.ndarray]:
    return [np.load(root / "output" / f"segment-{index}.npy") for index in range(_SEGMENTS)]


class TestARunSpillsItsTails:
    def test_the_checkpoint_records_tails_by_file(self, tmp_path):
        FAIL_AT_COUNTER.append(2)
        root = _run(tmp_path, "interrupted", expect_failure=True)

        text = (root / "checkpoints" / "simulation.checkpoint.json").read_text()
        tails = CheckpointManager(root / "checkpoints").load_checkpoint()["simulator_tails"]
        files = spilled_tail_files(tails)

        assert files, "the checkpoint holds no spilled tail"
        assert all(
            path.is_file() and path.parent.parent == (root / "checkpoints" / "tails").resolve() for path in files
        )
        assert '"__type__": "SpilledTimeSeries"' in text
        assert '"encoding": "base64"' not in text

    def test_a_resumed_run_writes_what_an_uninterrupted_one_writes(self, tmp_path):
        reference = _segments(_run(tmp_path, "reference"))

        FAIL_AT_COUNTER.append(2)
        root = _run(tmp_path, "resumed", expect_failure=True)
        FAIL_AT_COUNTER.clear()
        _run(tmp_path, "resumed")

        for index, (got, want) in enumerate(zip(_segments(root), reference, strict=True)):
            np.testing.assert_array_equal(got, want, err_msg=f"segment {index}")

    def test_a_tail_is_kept_exactly_while_the_checkpoint_or_its_backup_needs_it(self, tmp_path):
        """When batch k starts: the tails of segments k-3 to k-1, no more and no fewer.

        Each segment's signal lasts 2.6 segments, so it is consumed by the end of segment j+2: the
        checkpoint after batch k-1 references the signals of segments k-2 and k-1, and its backup --
        the checkpoint after batch k-2 -- those of k-3 and k-2.
        """
        _run(tmp_path, "run")

        assert [len(FILES_AT_START[k]) for k in range(_SEGMENTS)] == [min(k, 3) for k in range(_SEGMENTS)]
        # The last segment's tail lies past the end of the run and is never written.
        assert len(set().union(*FILES_AT_START.values())) == _SEGMENTS - 1

    def test_the_tails_are_gone_when_the_run_completes(self, tmp_path):
        root = _run(tmp_path, "run")

        assert not (root / "checkpoints" / "tails").exists()

    def test_a_fresh_run_removes_tails_an_earlier_attempt_left(self, tmp_path):
        stale = tmp_path / "run" / "checkpoints" / "tails" / "long" / "stale.npy"
        stale.parent.mkdir(parents=True)
        np.save(stale, np.zeros(4))

        _run(tmp_path, "run")

        assert "stale.npy" not in FILES_AT_START[0]
        assert not stale.exists()


def test_a_retried_batch_does_not_keep_the_failed_attempt_s_tail(tmp_path):
    """A tail written by an attempt that failed is unreferenced, so the next sweep removes it."""
    manager = CheckpointManager(tmp_path)
    kept = SpilledTimeSeries.spill(
        TimeSeries(
            data=np.ones((1, 8)), start_time=Quantity(0.0, unit="s"), sampling_frequency=Quantity(4.0, unit="Hz")
        ),
        manager.tail_directory / "long",
    )
    orphan = SpilledTimeSeries.spill(
        TimeSeries(
            data=np.ones((1, 8)), start_time=Quantity(0.0, unit="s"), sampling_frequency=Quantity(4.0, unit="Hz")
        ),
        manager.tail_directory / "long",
    )

    manager.remove_unreferenced_tails({kept.spill_path})

    assert kept.spill_path.is_file()
    assert not orphan.spill_path.exists()


@pytest.mark.requires_lalsuite
def test_each_event_is_placed_before_the_next_is_generated(tmp_path):
    """The per-event path streams: a segment never holds every new signal at once."""
    from tests.cli.test_events_ending_before_the_segment import (
        _install_events,
        _orchestrator,
        _start_time,
        _StubAdapter,
    )

    orchestrator = _orchestrator(tmp_path, execution="per-event")
    adapter = _StubAdapter(lead=0.0, tail=0.4)
    orchestrator.signal_adapter = adapter
    start = _start_time(orchestrator)
    _install_events(orchestrator, [start + 1.0, start + 5.0, start + 9.0])

    order: list[str] = []
    generate, place = adapter.simulate, orchestrator._place

    def _generate(parameters, **kwargs):
        order.append("generate")
        single = generate(parameters, **kwargs)
        # One channel per interferometer of the triangle, where the stub writes one.
        return TimeSeries(
            data=np.repeat(np.asarray(single[0].data)[None, :], orchestrator.num_of_channels, axis=0),
            start_time=single.start_time,
            sampling_frequency=single.sampling_frequency,
        )

    def _place(segment, chunk, preceding_gap):
        order.append("place")
        return place(segment, chunk, preceding_gap)

    adapter.simulate = _generate
    orchestrator._place = _place

    TimeSeriesMixin.simulate(orchestrator)

    assert order == ["generate", "place"] * 3
