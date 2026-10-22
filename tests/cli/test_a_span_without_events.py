"""A run whose span misses every event of its population must say so.

Shortening a run by changing only ``duration`` moves its end but not its start, and the span can
then miss every event: the run writes signal files holding only zeros and otherwise reports
success. The orchestrator warns, naming the span and the nearest event, so the cause is visible in
the log rather than discovered by reading the frames.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from gwmock.cli.utils.config import Config

_POPULATION_CSV = Path(__file__).resolve().parents[2] / "examples" / "signal" / "bbh_population.csv"
_START = 1577491218.0
_SEGMENT_DURATION = 64.0
_END = _START + _SEGMENT_DURATION

_EVENT: dict[str, Any] = {
    "detector_frame_mass_1": 30.0,
    "detector_frame_mass_2": 25.0,
    "distance": 400.0,
    "right_ascension": 1.0,
    "declination": 0.5,
    "polarization_angle": 0.2,
    "inclination": 0.3,
}


def _orchestrator(working_directory: Path):
    from gwmock.cli.adapter_orchestration import AdapterOrchestrator

    config = Config.model_validate(
        {
            "globals": {
                "simulator-arguments": {
                    "sampling-frequency": 1024.0,
                    "duration": _SEGMENT_DURATION,
                    "total-duration": _SEGMENT_DURATION,
                    "start-time": _START,
                    "seed": 1,
                },
                "working-directory": str(working_directory),
            },
            "orchestration": {
                "population": {
                    "backend": "FilePopulationLoader",
                    "source-type": "bbh",
                    "arguments": {"path": str(_POPULATION_CSV)},
                },
                "signal": {
                    "source-type": "bbh",
                    "waveform-model": "IMRPhenomD",
                    "minimum-frequency": 30,
                    "detectors": ["ET-Triangle-Sardinia"],
                },
            },
        }
    )
    return AdapterOrchestrator.from_config(
        config.orchestration, global_simulator_arguments=dict(config.globals.simulator_arguments)
    )


class _StubAdapter:
    """Answers the placement queries with fixed values and refuses to generate anything."""

    detector_names = ("E1",)

    def __init__(self, lead: float | None, tail: float | None) -> None:
        self._lead = lead
        self._tail = tail

    def pre_coalescence_duration(self, parameters, **_):
        del parameters
        return self._lead

    def post_coalescence_duration(self, parameters, **_):
        del parameters
        return self._tail

    def simulate(self, *args, **kwargs):
        raise AssertionError("an event outside the span was generated")


def _messages(caplog) -> str:
    return " ".join(
        record.getMessage() for record in caplog.records if "reaches the requested span" in record.getMessage()
    )


@pytest.mark.requires_lalsuite
def test_a_run_whose_span_misses_every_event_warns(tmp_path, caplog):
    """The shortened quick-start: the only event coalesces 18 s after the span ends."""
    orchestrator = _orchestrator(tmp_path)
    orchestrator.signal_adapter = _StubAdapter(lead=3.0, tail=0.4)
    orchestrator._population_events = ({**_EVENT, "coa_time": _END + 18.0},)

    with caplog.at_level(logging.WARNING, logger="gwmock"):
        chunks = orchestrator._simulate()

    assert len(chunks) == 0
    message = _messages(caplog)
    assert "1577491218" in message, "the warning must name the span"
    assert "1577491300" in message, "the warning must name the nearest event"
    assert "start-time" in message


@pytest.mark.requires_lalsuite
def test_the_warning_is_given_once(tmp_path, caplog):
    orchestrator = _orchestrator(tmp_path)
    orchestrator.signal_adapter = _StubAdapter(lead=3.0, tail=0.4)
    orchestrator._population_events = ({**_EVENT, "coa_time": _END + 18.0},)

    with caplog.at_level(logging.WARNING, logger="gwmock"):
        orchestrator._warn_if_no_event_reaches_the_run()
        orchestrator._warn_if_no_event_reaches_the_run()

    assert len([r for r in caplog.records if "reaches the requested span" in r.getMessage()]) == 1


@pytest.mark.requires_lalsuite
@pytest.mark.parametrize(
    ("coa_time", "lead", "tail"),
    [
        (_START + 10.0, 3.0, 0.4),  # coalesces inside the span
        (_END + 2.0, 3.0, 0.4),  # coalesces after it, but its inspiral starts inside
        (_START - 0.2, 3.0, 0.4),  # coalesced before it, but its tail reaches in
    ],
)
def test_an_event_whose_waveform_reaches_the_span_is_not_reported(tmp_path, caplog, coa_time, lead, tail):
    orchestrator = _orchestrator(tmp_path)
    orchestrator.signal_adapter = _StubAdapter(lead=lead, tail=tail)
    orchestrator._population_events = ({**_EVENT, "coa_time": coa_time},)

    with caplog.at_level(logging.WARNING, logger="gwmock"):
        orchestrator._warn_if_no_event_reaches_the_run()

    assert _messages(caplog) == ""


@pytest.mark.requires_lalsuite
def test_the_nearest_event_is_named_from_either_side(tmp_path, caplog):
    orchestrator = _orchestrator(tmp_path)
    orchestrator.signal_adapter = _StubAdapter(lead=None, tail=None)
    orchestrator._population_events = (
        {**_EVENT, "coa_time": _START - 500.0},
        {**_EVENT, "coa_time": _START - 5.0},
        {**_EVENT, "coa_time": _END + 50.0},
    )

    with caplog.at_level(logging.WARNING, logger="gwmock"):
        orchestrator._warn_if_no_event_reaches_the_run()

    message = _messages(caplog)
    assert "3 event(s)" in message
    assert str(_START - 5.0) in message
