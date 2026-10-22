"""A refusal decided by the configuration and inputs alone is raised at once, not retried.

``retry_with_backoff`` re-runs a failed batch on the same configuration and inputs, so a refusal
of a waveform setting, a population column or a placement fails identically on every attempt --
after a backoff, and with the same warning each time.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gwmock.cli.simulate_utils import retry_with_backoff
from gwmock.signal.adapter import SignalAdapter


class _RejectingBackend:
    """Rejects the keys in ``REJECTED`` the way gwmock-signal's waveform backends do."""

    REJECTED = frozenset({"f_ref", "spin1_z"})

    def __init__(self) -> None:
        self.waveform_model = "IMRPhenomD"
        self.required_params = frozenset({"coa_time"})

    def simulate(self, params, detector_names, background=None, **_):
        rejected = sorted(set(params) & self.REJECTED)
        raise ValueError(f"Unsupported LAL waveform parameters: {', '.join(rejected)}")

    def pre_coalescence_duration(self, params, **_):
        raise ValueError(f"coa_time {params['coa_time']} is outside the simulator's span")


def _adapter() -> SignalAdapter:
    return SignalAdapter(
        source_type="bbh", backend=_RejectingBackend(), network=SimpleNamespace(detector_names=("H1",))
    )


def _attempts(operation) -> int:
    attempts = 0

    def counted():
        nonlocal attempts
        attempts += 1
        operation()

    with pytest.raises(ValueError, match=r"waveform|span|column"):
        retry_with_backoff(counted, max_retries=3, initial_delay=0.0)
    return attempts


def test_a_refused_waveform_argument_is_attempted_once():
    adapter = _adapter()

    assert (
        _attempts(
            lambda: adapter.simulate_stack(
                {"mass_1": 30.0, "coa_time": 0.0},
                sampling_frequency=4.0,
                minimum_frequency=20.0,
                waveform_arguments={"f_ref": 2.0},
            )
        )
        == 1
    )


def test_a_refused_population_column_is_attempted_once():
    adapter = _adapter()

    assert (
        _attempts(
            lambda: adapter.simulate_stack(
                {"mass_1": 30.0, "spin1_z": 0.3, "coa_time": 0.0}, sampling_frequency=4.0, minimum_frequency=20.0
            )
        )
        == 1
    )


@pytest.mark.requires_lalsuite
def test_a_refused_placement_query_is_attempted_once(tmp_path):
    from tests.cli.test_inspiral_segment_placement import _COMPLETE_EVENT, _end_time, _orchestrator

    orchestrator = _orchestrator(tmp_path)
    orchestrator.signal_adapter = SignalAdapter.from_backend(
        source_type="bbh", backend=_RejectingBackend(), detectors=["E1"]
    )
    event = {**_COMPLETE_EVENT, "coa_time": _end_time(orchestrator) - 0.5}

    assert _attempts(lambda: orchestrator._event_starts_before_segment_end(event)) == 1


def test_a_transient_failure_is_still_retried():
    def flaky():
        raise OSError("disk briefly unavailable")

    assert _attempts_of(flaky) == 4


def _attempts_of(operation) -> int:
    attempts = 0

    def counted():
        nonlocal attempts
        attempts += 1
        operation()

    with pytest.raises(OSError, match="disk briefly unavailable"):
        retry_with_backoff(counted, max_retries=3, initial_delay=0.0)
    return attempts
