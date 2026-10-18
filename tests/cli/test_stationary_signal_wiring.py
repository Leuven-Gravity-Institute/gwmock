"""The stationary-signal branch of the orchestrator, exercised without a background model.

``test_sgwb_end_to_end.py`` drives the same branch through the real backend, but it is an
integration test: it is excluded from the default run, so nothing fast covered *what the
orchestrator hands the backend*. That is all wiring -- the canvas the background is accumulated
into, its epoch, its sample rate, its unit, and the seed for this segment -- and a fake backend
that records its arguments shows it exactly.

The canvas matters as much as the signal: it fixes the length and the time axis of everything the
stationary path produces, and a background of the wrong length or epoch is a segment of the wrong
duration or one placed at the wrong time, both of which a spectrum would not obviously reveal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from gwmock_data import TimeSeries as ChannelSeries
from gwmock_signal import DetectorStrainStack

from gwmock.cli.adapter_orchestration import AdapterOrchestrator
from gwmock.cli.utils.config import Config
from gwmock.simulator.seeds import derive_seed

pytestmark = pytest.mark.unit

FAKE_STATIONARY_BACKEND = "tests.cli.test_stationary_signal_wiring:FakeStationaryBackend"

_EPOCH = 1000000000.0
_FS = 8.0
_DURATION = 4.0
_DETECTORS = ["H1", "L1"]
_SEED = 7


class FakeStationaryBackend:
    """Records the background it is given and returns it plus a constant."""

    #: A stationary background needs no per-event parameters, but the backend protocol asks.
    required_params: ClassVar[frozenset[str]] = frozenset()

    #: Set by ``SignalAdapter.set_seed``, which only assigns it if the attribute already exists.
    seed: int | None = None

    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, **_kwargs: Any) -> None:
        self.seed = None

    def simulate(
        self,
        parameters: dict,
        detector_names: tuple[str, ...],
        background=None,
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        earth_rotation: bool = True,
        **_kwargs: Any,
    ) -> DetectorStrainStack:
        _ = minimum_frequency
        names = tuple(detector if isinstance(detector, str) else detector.name for detector in detector_names)
        if background is None:
            raise AssertionError("the stationary branch must pass a background to accumulate into")
        type(self).calls.append(
            {
                "parameters": dict(parameters),
                "seed": self.seed,
                "earth_rotation": earth_rotation,
                "detectors": list(names),
                "background": {
                    name: {
                        "samples": np.asarray(background[name].value, dtype=float),
                        "t0": float(background[name].t0),
                        "sample_rate": float(background[name].sample_rate),
                        "unit": str(background[name].unit),
                    }
                    for name in names
                },
            }
        )
        return DetectorStrainStack.from_mapping(
            names,
            {
                name: ChannelSeries(
                    np.asarray(background[name].value, dtype=float) + 1.0,
                    t0=float(background[name].t0),
                    sample_rate=sampling_frequency,
                )
                for name in names
            },
        )


@pytest.fixture(autouse=True)
def _clear_backend_calls():
    FakeStationaryBackend.calls.clear()
    yield
    FakeStationaryBackend.calls.clear()


def _config(tmp_path: Path, *, duration: float = _DURATION, fs: float = _FS) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": fs,
                "duration": duration,
                "total-duration": duration * 2,
                "start-time": _EPOCH,
                "seed": _SEED,
            },
            "working-directory": str(tmp_path),
            "output-directory": "output",
            "metadata-directory": "metadata",
        },
        "orchestration": {
            "signal": {
                "backend": FAKE_STATIONARY_BACKEND,
                "source-type": "sgwb",
                "detectors": list(_DETECTORS),
                "minimum-frequency": 4,
                "parameters": {"omega_ref": 1e-9, "spectral_index": 0.0},
                "output": {"output_directory": "signal", "file_name": "sgwb-{{ counter }}.hdf5"},
            },
        },
    }


def _orchestrator(tmp_path: Path, **kwargs: Any) -> AdapterOrchestrator:
    parsed = Config.model_validate(_config(tmp_path, **kwargs))
    return AdapterOrchestrator.from_config(
        parsed.orchestration, global_simulator_arguments=dict(parsed.globals.simulator_arguments)
    )


def _first_call(orchestrator: AdapterOrchestrator) -> dict[str, Any]:
    orchestrator._simulate()
    assert len(FakeStationaryBackend.calls) == 1, "the stationary branch should call the backend once per segment"
    return FakeStationaryBackend.calls[0]


class TestTheBranchIsTaken:
    def test_a_source_with_no_events_produces_one_chunk_for_the_segment(self, tmp_path):
        chunks = _orchestrator(tmp_path)._simulate()
        assert len(chunks) == 1

    def test_the_chunk_spans_the_segment(self, tmp_path):
        chunks = _orchestrator(tmp_path)._simulate()
        assert np.asarray(chunks[0][0], dtype=float).size == round(_DURATION * _FS)
        assert float(chunks[0].start_time.value) == _EPOCH

    def test_a_stationary_segment_records_no_injections(self, tmp_path):
        """There are no discrete events, so an injection list would be a claim about sources that
        do not exist."""
        orchestrator = _orchestrator(tmp_path)
        orchestrator._simulate()
        assert orchestrator._batch_injections == []


class TestTheBackgroundHandedToTheBackend:
    def test_it_is_as_long_as_the_segment(self, tmp_path):
        """Duration times sampling frequency: the two are multiplied, and any other combination of
        them gives a segment of the wrong length."""
        call = _first_call(_orchestrator(tmp_path, duration=4, fs=8))
        assert call["background"]["H1"]["samples"].size == 32

    def test_the_length_follows_the_sampling_frequency(self, tmp_path):
        call = _first_call(_orchestrator(tmp_path, duration=4, fs=16))
        assert call["background"]["H1"]["samples"].size == 64

    def test_the_length_follows_the_duration(self, tmp_path):
        call = _first_call(_orchestrator(tmp_path, duration=8, fs=8))
        assert call["background"]["H1"]["samples"].size == 64

    def test_a_fractional_sample_count_is_rounded(self, tmp_path):
        """A duration that is not a whole number of samples still has to produce an integer length
        rather than a truncated one."""
        call = _first_call(_orchestrator(tmp_path, duration=1.1, fs=9))
        assert call["background"]["H1"]["samples"].size == 10

    def test_it_starts_at_the_segment_epoch(self, tmp_path):
        """The epoch travels with the background: an unset one puts the whole segment at t=0."""
        call = _first_call(_orchestrator(tmp_path))
        assert call["background"]["H1"]["t0"] == _EPOCH

    def test_it_carries_the_sampling_frequency(self, tmp_path):
        call = _first_call(_orchestrator(tmp_path, fs=16))
        assert call["background"]["H1"]["sample_rate"] == 16.0

    def test_it_is_in_strain(self, tmp_path):
        """The unit is what makes the returned series comparable with the noise it is added to."""
        call = _first_call(_orchestrator(tmp_path))
        assert call["background"]["H1"]["unit"] == "strain"

    def test_it_starts_empty(self, tmp_path):
        """A blank canvas per segment: anything else would accumulate the previous segment again."""
        call = _first_call(_orchestrator(tmp_path))
        assert not np.any(call["background"]["H1"]["samples"])

    def test_there_is_one_series_per_detector_keyed_by_name(self, tmp_path):
        call = _first_call(_orchestrator(tmp_path))
        assert sorted(call["background"]) == sorted(_DETECTORS)

    def test_the_second_segment_gets_its_own_epoch(self, tmp_path):
        orchestrator = _orchestrator(tmp_path)
        orchestrator._simulate()
        orchestrator.update_state()
        orchestrator._simulate()
        assert [call["background"]["H1"]["t0"] for call in FakeStationaryBackend.calls] == [
            _EPOCH,
            _EPOCH + _DURATION,
        ]


class TestTheSeedForTheSegment:
    def test_the_backend_is_seeded_from_the_run_seed_and_the_segment(self, tmp_path):
        """Derived rather than reused, so two segments of one run are independent draws that a
        rerun still reproduces."""
        call = _first_call(_orchestrator(tmp_path))
        assert call["seed"] == derive_seed(_SEED, "signal", 0)

    def test_it_is_recorded_alongside_the_chunk(self, tmp_path):
        chunks = _orchestrator(tmp_path)._simulate()
        assert chunks[0].metadata["segment_seed"] == derive_seed(_SEED, "signal", 0)

    def test_each_segment_is_seeded_differently(self, tmp_path):
        orchestrator = _orchestrator(tmp_path)
        orchestrator._simulate()
        orchestrator.update_state()
        orchestrator._simulate()
        seeds = [call["seed"] for call in FakeStationaryBackend.calls]
        assert seeds == [derive_seed(_SEED, "signal", 0), derive_seed(_SEED, "signal", 1)]
        assert seeds[0] != seeds[1]


class TestTheParametersHandedToTheBackend:
    def test_the_configured_signal_parameters_are_passed(self, tmp_path):
        call = _first_call(_orchestrator(tmp_path))
        assert call["parameters"]["omega_ref"] == 1e-9
        assert call["parameters"]["spectral_index"] == 0.0

    def test_they_are_recorded_with_the_chunk(self, tmp_path):
        """The chunk's metadata is what provenance is written from, so it has to carry what was
        actually generated rather than an empty mapping."""
        chunks = _orchestrator(tmp_path)._simulate()
        assert chunks[0].metadata["signal_parameters"]["omega_ref"] == 1e-9

    def test_the_recorded_parameters_are_a_copy(self, tmp_path):
        """A live reference would let a later segment's edits rewrite this segment's record."""
        orchestrator = _orchestrator(tmp_path)
        chunks = orchestrator._simulate()
        recorded = chunks[0].metadata["signal_parameters"]
        recorded["omega_ref"] = 0.0
        assert orchestrator.signal_parameters["omega_ref"] == 1e-9
