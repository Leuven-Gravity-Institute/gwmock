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

"""The adapter must read the injector's glitch truth and stamp it with the batch's epoch.

gwmock owns where each batch sits in GPS time -- it is what the frame's name, its ``t0`` and its
metadata record carry -- so the epoch is handed *to* the injector rather than accumulated by it.
Getting that backwards is not a cosmetic error: a resumed run replays the chunks before the one it
is about to write, each replay advances an accumulating epoch, and the catalogue would then time
every glitch in the run against the wrong frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from gwmock.noise import NoiseAdapter

pytestmark = pytest.mark.unit

O3_EPOCH = 1256655618.0


class _CataloguingInjector:
    """An injector that records what it injected, as a released gwmock-noise does.

    Not a fake catalogue: the rows are derived from the samples this object actually produced, by
    finding the non-zero runs in a glitch-only chunk. What is under test here is gwmock's half --
    that the epoch reaches the injector before the chunk is drawn, that the rows for the chunk just
    drawn are the ones recorded, and that they reach the index and the lookup -- so the rows have to
    correspond to real samples, and they must not be a constant this test asserts against itself.
    """

    def __init__(self, chunks: list[dict[str, np.ndarray]], sampling_frequency: float) -> None:
        self.gps_start = 0.0
        # The four attributes the NoiseSimulator protocol requires, so this object can stand in
        # the stream where the real injector does.
        self.duration = 0.0
        self.sampling_frequency = sampling_frequency
        self.detectors: list[str] = []
        self.seed: int | None = None
        self._chunks = chunks
        self._position = 0
        self._segment_events: list[dict[str, Any]] = []
        # Per (model, detector) and cumulative over the stream, as the real injector's ordinals
        # are: an id has to be unique across a run, or the index would file two glitches under
        # one key and a lookup would answer with the wrong frame.
        self._ordinals: dict[str, int] = {}
        self.epochs_seen: list[float] = []

    def generate(self, duration, sampling_frequency, detectors, seed=None):
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = list(detectors)
        self.seed = seed
        chunk = self._chunks[self._position]
        self._position += 1
        self.epochs_seen.append(self.gps_start)
        self._segment_events = [
            {
                "event_id": f"{detector}-0-{self._next_ordinal(detector)}",
                "detector": detector,
                "model_index": 0,
                "kind": "blip",
                "glitch_class": None,
                "gps_start_time": self.gps_start + (start / self.sampling_frequency),
                "gps_peak_time": self.gps_start + (start / self.sampling_frequency),
                "duration_seconds": (stop - start) / self.sampling_frequency,
                "n_samples": stop - start,
                "segment_index": self._position - 1,
                "sample_index": start,
                "target_snr": None,
                "realized_snr": None,
                "amplitude": 1.0,
            }
            for detector in detectors
            for start, stop in _nonzero_runs(chunk[detector])
        ]
        self.gps_start += duration
        return chunk

    def _next_ordinal(self, detector: str) -> int:
        ordinal = self._ordinals.get(detector, 0)
        self._ordinals[detector] = ordinal + 1
        return ordinal

    def generate_stream(self, chunk_duration, sampling_frequency, detectors, seed=None):
        while True:
            yield self.generate(chunk_duration, sampling_frequency, detectors, seed)

    @property
    def segment_glitch_events(self) -> list[dict[str, Any]]:
        return list(self._segment_events)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "glitches": {
                "catalogue": {
                    "schema_version": "1.0.0",
                    "time_convention": "gps_start_time is the GPS time of the FIRST SAMPLE ...",
                    "columns": {"event_id": "...", "gps_start_time": "..."},
                    "events": [],
                }
            }
        }


class _InjectorWithoutCatalogue:
    """An injector from before the truth catalogue, i.e. an older installed gwmock-noise."""

    @property
    def metadata(self) -> dict[str, Any]:
        return {"glitches": {"models": [], "counts": []}}


def _nonzero_runs(strain: np.ndarray) -> list[tuple[int, int]]:
    occupied = strain != 0.0
    if not np.any(occupied):
        return []
    padded = np.concatenate(([False], occupied, [False]))
    edges = np.diff(padded.astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist(), strict=True))


def _chunk_with_glitch_at(n_samples: int, start: int, length: int) -> dict[str, np.ndarray]:
    strain = np.zeros(n_samples, dtype=float)
    strain[start : start + length] = 1e-21
    return {"H1": strain}


def test_adapter_reports_the_rows_of_the_chunk_just_generated() -> None:
    """Per batch, not cumulative: a writer needs the events of the frame it is about to write."""
    sampling_frequency = 16.0
    injector = _CataloguingInjector(
        [_chunk_with_glitch_at(64, 8, 4), _chunk_with_glitch_at(64, 32, 4)],
        sampling_frequency,
    )
    adapter = NoiseAdapter.from_backend()
    adapter._glitch_injector = injector

    adapter.set_segment_gps_start(O3_EPOCH)
    injector.generate(4.0, sampling_frequency, ["H1"])
    first = adapter.segment_glitch_events()

    adapter.set_segment_gps_start(O3_EPOCH + 4.0)
    injector.generate(4.0, sampling_frequency, ["H1"])
    second = adapter.segment_glitch_events()

    assert [row["gps_start_time"] for row in first] == [O3_EPOCH + 0.5]
    assert [row["gps_start_time"] for row in second] == [O3_EPOCH + 4.0 + 2.0]
    # The second read does not carry the first chunk's rows.
    assert len(second) == 1


def test_adapter_hands_the_batch_epoch_to_the_injector() -> None:
    """The epoch gwmock supplies is the one the rows are timed against."""
    injector = _CataloguingInjector([_chunk_with_glitch_at(64, 16, 4)], 16.0)
    adapter = NoiseAdapter.from_backend()
    adapter._glitch_injector = injector

    adapter.set_segment_gps_start(O3_EPOCH + 128.0)
    injector.generate(4.0, 16.0, ["H1"])

    assert injector.epochs_seen == [O3_EPOCH + 128.0]
    assert adapter.segment_glitch_events()[0]["gps_start_time"] == O3_EPOCH + 128.0 + 1.0


def test_adapter_returns_copies_so_a_consumer_cannot_edit_the_record() -> None:
    """The rows are a run's record of itself, handed out rather than shared."""
    injector = _CataloguingInjector([_chunk_with_glitch_at(64, 16, 4)], 16.0)
    adapter = NoiseAdapter.from_backend()
    adapter._glitch_injector = injector
    injector.generate(4.0, 16.0, ["H1"])

    rows = adapter.segment_glitch_events()
    rows[0]["gps_start_time"] = 0.0

    assert adapter.segment_glitch_events()[0]["gps_start_time"] != 0.0


def test_adapter_describes_the_catalogue_without_repeating_its_rows() -> None:
    """The column documentation travels with a batch; the run's whole catalogue does not."""
    injector = _CataloguingInjector([_chunk_with_glitch_at(64, 16, 4)], 16.0)
    adapter = NoiseAdapter.from_backend()
    adapter._glitch_injector = injector

    description = adapter.glitch_catalogue_description()

    assert description is not None
    assert description["schema_version"] == "1.0.0"
    assert "FIRST SAMPLE" in description["time_convention"]
    assert set(description["columns"]) == {"event_id", "gps_start_time"}
    # Not the rows: those are written per batch, and a per-batch copy of the whole run's
    # catalogue would grow quadratically across a long run.
    assert "events" not in description


def test_a_stream_without_glitches_reports_nothing_rather_than_failing() -> None:
    """Most runs have no glitches, and they must not pay for the catalogue at all."""
    adapter = NoiseAdapter.from_backend()

    assert adapter.segment_glitch_events() == []
    assert adapter.glitch_catalogue_description() is None
    adapter.set_segment_gps_start(O3_EPOCH)  # no injector to tell; must not raise


def test_an_older_gwmock_noise_degrades_visibly_rather_than_silently() -> None:
    """No rows *and* no description, which is how a consumer tells this from "injected none".

    gwmock floors its ``gwmock-noise`` dependency below the release that records glitches, so an
    installation can have glitches configured and an injector that cannot report them. Reporting
    an empty list with a description present would claim the run injected nothing.
    """
    adapter = NoiseAdapter.from_backend()
    adapter._glitch_injector = _InjectorWithoutCatalogue()

    assert adapter.segment_glitch_events() == []
    assert adapter.glitch_catalogue_description() is None
    adapter.set_segment_gps_start(O3_EPOCH)  # nothing to set; must not raise


def test_reconfiguring_a_stream_forgets_the_previous_injector(tmp_path: Path) -> None:
    """A reused adapter must not report the glitches of the stream it no longer runs."""
    adapter = NoiseAdapter.from_backend()
    adapter._glitch_injector = _CataloguingInjector([_chunk_with_glitch_at(64, 16, 4)], 16.0)

    adapter.open_stream(chunk_duration=4.0, sampling_frequency=16.0, detectors=["H1"], seed=3)

    assert adapter._glitch_injector is None
    assert adapter.segment_glitch_events() == []
    assert adapter.glitch_catalogue_description() is None


def test_a_glitch_stream_keeps_the_injector_for_the_catalogue(tmp_path: Path) -> None:
    """Opening a stream with glitches retains the injector the rows come from."""
    adapter = NoiseAdapter.from_backend()

    adapter.open_stream(
        chunk_duration=4.0,
        sampling_frequency=256.0,
        detectors=["H1"],
        seed=3,
        glitches=[
            {
                "kind": "blip",
                "rate": 1.0,
                "width": 0.01,
                "amplitude_distribution": {"distribution": "lognormal", "mean": 1.0, "std": 0.0},
            }
        ],
    )

    from gwmock_noise import InjectGlitches

    assert isinstance(adapter._glitch_injector, InjectGlitches)


def test_the_orchestrator_sets_the_epoch_only_after_realigning_the_stream(tmp_path: Path) -> None:
    """Order matters: realignment replays chunks, and each replay would consume an epoch.

    A resumed run opens a fresh stream and fast-forwards it to the current batch. Setting the epoch
    before that happens leaves it pointing at whatever the last replayed chunk advanced it to, so
    every glitch in the batch would be timed against a frame the run had already written.
    """
    from tests.cli.test_cli_orchestration import _fake_orchestration_config

    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    from gwmock.cli.adapter_orchestration import AdapterOrchestrator

    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)

    calls: list[str] = []
    real_ensure = orchestrator._ensure_noise_stream

    def _recording_ensure() -> None:
        calls.append("ensure")
        real_ensure()

    orchestrator._ensure_noise_stream = _recording_ensure  # type: ignore[method-assign]
    orchestrator.noise_adapter.set_segment_gps_start = lambda gps_start: calls.append(f"epoch:{gps_start}")  # type: ignore[method-assign]

    orchestrator._next_noise_chunk(gps_start=O3_EPOCH)

    assert calls == ["ensure", f"epoch:{O3_EPOCH}"]


def test_the_orchestrator_reports_this_batchs_glitches_in_its_metadata(tmp_path: Path) -> None:
    """The rows reach the metadata surface `_build_noise_section` reads, and are cleared with the batch."""
    from tests.cli.test_cli_orchestration import _fake_orchestration_config

    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    from gwmock.cli.adapter_orchestration import AdapterOrchestrator

    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)
    rows = [{"event_id": "H1-0-0", "gps_start_time": O3_EPOCH + 1.0}]
    orchestrator.noise_adapter.segment_glitch_events = lambda: list(rows)  # type: ignore[method-assign]

    orchestrator._next_noise_chunk(gps_start=O3_EPOCH)
    noise_metadata = orchestrator.metadata["orchestration"]["noise"]

    # Named as the record names it, so the sanitiser that withholds injection truth from a
    # released file reaches this copy too.
    assert noise_metadata["glitch_injections"] == rows
    # The configuration keeps its own name and is a different thing.
    assert "glitches" not in noise_metadata

    orchestrator.update_state()
    assert orchestrator.metadata["orchestration"]["noise"]["glitch_injections"] == []


def test_an_end_to_end_run_indexes_its_glitches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A full orchestration run writes the rows, the index, and answers the lookup.

    The injector is replaced with :class:`_CataloguingInjector`, which derives its rows from the
    samples it produced -- standing in for the released gwmock-noise, whose floor gwmock has not
    yet raised. Everything downstream of it is the real path: the orchestrator, the metadata
    record, the index writer and ``find_glitches``.
    """
    import gwmock.noise.adapter as adapter_module
    from gwmock.cli.simulate import _simulate_impl
    from gwmock.cli.utils.signal_lookup import find_glitches
    from tests.cli.test_cli_orchestration import _fake_orchestration_config

    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    # The default backend, because the glitch path is part of it: a run naming its own noise
    # backend gets no injector at all, and this test is about the injector's rows reaching disk.
    config.orchestration.noise.backend = None
    sampling_frequency = float(config.orchestration.noise.arguments["sampling_frequency"])
    duration = float(config.orchestration.noise.arguments["duration"])
    n_samples = round(duration * sampling_frequency)
    config.orchestration.noise.arguments["glitches"] = [
        {
            "kind": "blip",
            "rate": 1.0,
            "width": 0.01,
            "amplitude_distribution": {"distribution": "lognormal", "mean": 1.0, "std": 0.0},
        }
    ]

    # One glitch in each batch, at a different sample offset, so a wrong epoch or a reused
    # chunk shows up as a wrong time rather than passing by coincidence.
    chunks = [_chunk_with_glitch_at(n_samples, offset, 1) for offset in (1, 2, 3, 1, 2, 3)]
    injectors: list[_CataloguingInjector] = []

    def _fake_inject_glitches(simulator, glitch_models, *args, **kwargs):
        injector = _CataloguingInjector(chunks, sampling_frequency)
        injectors.append(injector)
        return injector

    monkeypatch.setattr(adapter_module, "InjectGlitches", _fake_inject_glitches)

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config.model_dump(by_alias=True, exclude_none=True), sort_keys=False))
    _simulate_impl(str(config_file), overwrite=True, metadata=True)

    metadata_directory = tmp_path / "metadata"
    index = yaml.safe_load((metadata_directory / "glitch_index.yaml").read_text())
    assert index, "test is vacuous: no glitch was indexed"

    # Every indexed glitch answers "which frame holds it", and its time falls inside the frame
    # the answer names.
    for event_id in index:
        matches = find_glitches(metadata_directory, event_id=event_id)
        assert len(matches) == 1, matches
        match = matches[0]
        assert match["frames"], match
        # One batch, because an id is unique across a run and a glitch is recorded against the
        # batch it starts in. Two would mean either a colliding id or a glitch filed twice.
        assert len(match["metadata"]) == 1, match
        batch_record = yaml.safe_load((metadata_directory / match["metadata"][0]).read_text())
        noise_output = next(output for output in batch_record["outputs"] if output["kind"] == "noise")
        assert noise_output["t0"] <= match["gps_start_time"] < noise_output["t0"] + noise_output["duration"]
