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


def _attempts_any(operation) -> int:
    """Count attempts of an operation expected to fail with a refusal (a ``ValueError``)."""
    attempts = 0

    def counted():
        nonlocal attempts
        attempts += 1
        operation()

    with pytest.raises(ValueError, match=r"."):
        retry_with_backoff(counted, max_retries=3, initial_delay=0.0)
    return attempts


class _OptionlessBackend(_RejectingBackend):
    """An older simulator that cannot carry waveform options at all."""

    REJECTED = frozenset({"waveform_arguments"})


def test_waveform_options_the_simulator_cannot_carry_are_attempted_once():
    adapter = SignalAdapter(
        source_type="bbh", backend=_OptionlessBackend(), network=SimpleNamespace(detector_names=("H1",))
    )

    assert (
        _attempts_any(
            lambda: adapter.simulate_stack(
                {"mass_1": 30.0, "coa_time": 0.0},
                sampling_frequency=4.0,
                minimum_frequency=20.0,
                waveform_options={"ModeArray": [[2, 2]]},
            )
        )
        == 1
    )


class _AliasRejectingBackend(_RejectingBackend):
    def simulate(self, params, detector_names, background=None, **_):
        raise ValueError("Do not mix aliases for 'detector_frame_mass_1': detector_frame_mass_1, mass1")


def test_a_simulator_validation_error_is_attempted_once():
    """Generation is a pure function of its inputs: the simulator's own refusal recurs on retry."""
    adapter = SignalAdapter(
        source_type="bbh", backend=_AliasRejectingBackend(), network=SimpleNamespace(detector_names=("H1",))
    )

    assert (
        _attempts_any(
            lambda: adapter.simulate_stack(
                {"mass1": 30.0, "detector_frame_mass_1": 31.0, "coa_time": 0.0},
                sampling_frequency=4.0,
                minimum_frequency=20.0,
            )
        )
        == 1
    )


@pytest.mark.requires_lalsuite
def test_a_fixed_argument_the_batched_path_cannot_apply_is_attempted_once(tmp_path):
    from tests.cli.test_device_execution import _orchestrator

    orchestrator = _orchestrator(
        tmp_path, "batched", waveform_backend=None, **{"waveform-arguments": {"not_a_parameter": 1.0}}
    )

    assert _attempts_any(orchestrator._simulate) == 1


@pytest.mark.requires_lalsuite
def test_an_alias_clash_in_the_catalogue_is_attempted_once(tmp_path):
    from tests.cli.test_device_execution import _COMPLETE_EVENT, _orchestrator

    orchestrator = _orchestrator(tmp_path, "batched", waveform_backend=None)
    end_time = float(getattr(orchestrator.end_time, "value", orchestrator.end_time))
    orchestrator._population_events = ({**_COMPLETE_EVENT, "mass1": 31.0, "coa_time": end_time - 1.0},)

    assert _attempts_any(orchestrator._simulate) == 1


def _noise_orchestrator(tmp_path, **noise_arguments):
    from gwmock.cli.adapter_orchestration import AdapterOrchestrator
    from gwmock.cli.utils.config import Config

    config = Config.model_validate(
        {
            "globals": {
                "simulator-arguments": {"sampling-frequency": 64, "duration": 4, "total-duration": 4, "seed": 1},
                "working-directory": str(tmp_path),
            },
            "orchestration": {
                "noise": {
                    "arguments": {"detectors": ["H1"], **noise_arguments},
                    "output": {"output_directory": "noise", "file_name": "n-{{ detectors }}.hdf5"},
                }
            },
        }
    )
    return AdapterOrchestrator.from_config(
        config.orchestration, global_simulator_arguments=dict(config.globals.simulator_arguments)
    )


def test_a_noise_configuration_the_stream_refuses_is_attempted_once(tmp_path):
    orchestrator = _noise_orchestrator(tmp_path, spectral_lines=[])

    assert _attempts_any(orchestrator._ensure_noise_stream) == 1


def test_an_unsupported_noise_output_argument_is_attempted_once(tmp_path):
    orchestrator = _noise_orchestrator(tmp_path)
    orchestrator._active_noise_output_directory = tmp_path / "noise"
    orchestrator._active_overwrite = True
    orchestrator._active_noise_output_arguments = {"channel_prefix": "MOCK"}

    assert _attempts_any(orchestrator._run_noise_batch) == 1
