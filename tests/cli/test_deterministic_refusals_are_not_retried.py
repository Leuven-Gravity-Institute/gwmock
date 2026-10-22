"""A refusal decided by the configuration and inputs alone is raised at once, not retried.

``retry_with_backoff`` re-runs a failed batch on the same configuration and inputs, so a refusal
of a waveform setting, a population column or a placement fails identically on every attempt --
after a backoff, and with the same warning each time.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

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


def test_a_download_failure_while_opening_the_noise_stream_is_retried(tmp_path, monkeypatch):
    """A connection error is not the configuration's: the batch is retried, and a later attempt works.

    The stream opening resolves remote glitch population files through ``download_file``; a
    failure there used to come back as a ``ValueError``, which the stream-opening guard then
    marked final.
    """
    import requests

    from gwmock.utils.download import download_file

    calls = {"get": 0}

    class _Response:
        headers: ClassVar[dict] = {}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            del chunk_size
            yield b"population"

    def flaky_get(*_args, **_kwargs):
        calls["get"] += 1
        if calls["get"] == 1:
            raise requests.ConnectionError("connection reset")
        return _Response()

    monkeypatch.setattr("gwmock.utils.download.requests.get", flaky_get)
    orchestrator = _noise_orchestrator(tmp_path)

    def open_stream(**_kwargs):
        download_file("https://example.com/population.csv", outdir=tmp_path / "cache")
        return iter(())

    monkeypatch.setattr(orchestrator.noise_adapter, "open_stream", open_stream)

    attempts = 0

    def counted():
        nonlocal attempts
        attempts += 1
        orchestrator._noise_stream = None
        orchestrator._ensure_noise_stream()

    retry_with_backoff(counted, max_retries=3, initial_delay=0.0)

    assert attempts == 2
    assert (tmp_path / "cache" / "population.csv").read_bytes() == b"population"


def test_an_unknown_template_variable_is_attempted_once(tmp_path):
    """``set_batch_context`` expands the output templates; a variable the run lacks never appears."""
    orchestrator = _noise_orchestrator(tmp_path)
    orchestrator.orchestration_config.noise.output.arguments = {"channel": "{{ not_a_variable }}:STRAIN"}
    batch = SimpleNamespace(simulator_config=orchestrator.orchestration_config)

    assert (
        _attempts_any(lambda: orchestrator.set_batch_context(batch=batch, output_directory=tmp_path, overwrite=True))
        == 1
    )


def test_an_unknown_template_variable_is_still_an_attribute_error(tmp_path):
    from gwmock.cli.utils.template import expand_template_variables

    with pytest.raises(AttributeError, match="Template variable 'not_a_variable' not found"):
        expand_template_variables("{{ not_a_variable }}", _noise_orchestrator(tmp_path))


def _stream_open_attempts(tmp_path, monkeypatch, open_stream) -> int:
    """Count the attempts a batch makes when opening the noise stream runs *open_stream*."""
    orchestrator = _noise_orchestrator(tmp_path)
    monkeypatch.setattr(orchestrator.noise_adapter, "open_stream", open_stream)
    attempts = 0

    def counted():
        nonlocal attempts
        attempts += 1
        orchestrator._noise_stream = None
        orchestrator._ensure_noise_stream()

    with pytest.raises(Exception, match=r"."):
        retry_with_backoff(counted, max_retries=3, initial_delay=0.0)
    return attempts


def _http_status(monkeypatch, status: int) -> None:
    import requests

    response = requests.Response()
    response.status_code = status
    monkeypatch.setattr(
        "gwmock.utils.download.requests.get",
        lambda *_a, **_k: (_ for _ in ()).throw(requests.HTTPError(str(status), response=response)),
    )


@pytest.mark.parametrize("status", [401, 404])
def test_a_permanent_http_failure_while_opening_the_noise_stream_is_attempted_once(tmp_path, monkeypatch, status):
    from gwmock.utils.download import download_file

    _http_status(monkeypatch, status)

    def open_stream(**_kwargs):
        download_file("https://example.com/population.csv", outdir=tmp_path / "cache")

    assert _stream_open_attempts(tmp_path, monkeypatch, open_stream) == 1


def test_a_server_error_while_opening_the_noise_stream_is_retried(tmp_path, monkeypatch):
    from gwmock.utils.download import download_file

    _http_status(monkeypatch, 503)

    def open_stream(**_kwargs):
        download_file("https://example.com/population.csv", outdir=tmp_path / "cache")

    assert _stream_open_attempts(tmp_path, monkeypatch, open_stream) == 4


def test_a_value_error_raised_from_none_while_handling_an_os_error_is_final(tmp_path, monkeypatch):
    """``from None`` says the earlier failure is not the reason; the context is not followed."""

    def open_stream(**_kwargs):
        try:
            raise FileNotFoundError("no cached copy")
        except OSError:
            raise ValueError("psd_file names no known PSD") from None

    assert _stream_open_attempts(tmp_path, monkeypatch, open_stream) == 1


def test_a_refusal_wrapping_an_os_error_is_final_however_deep(tmp_path, monkeypatch):
    """Something along the chain already decided the failure is final; an I/O error beneath it does not undo that."""
    from gwmock.utils.errors import DeterministicRefusalError

    def open_stream(**_kwargs):
        try:
            try:
                raise ConnectionError("reset")
            except OSError as io_error:
                raise DeterministicRefusalError("the glitch model refuses this file") from io_error
        except DeterministicRefusalError as refusal:
            raise ValueError("could not configure the glitch models") from refusal

    assert _stream_open_attempts(tmp_path, monkeypatch, open_stream) == 1


def test_an_implicit_wrap_of_a_network_failure_stays_retryable(tmp_path, monkeypatch):
    """A library raising ValueError while handling a connection error, without chaining it."""

    def open_stream(**_kwargs):
        try:
            raise ConnectionError("reset")
        except OSError:
            raise ValueError("Failed to fetch data")  # noqa: B904 -- the implicit wrap under test

    assert _stream_open_attempts(tmp_path, monkeypatch, open_stream) == 4


class TestTheWholeChainIsRead:
    """A final failure anywhere in the chain wins over transient-looking wrappers above it."""

    @staticmethod
    def _requests_404():
        import requests

        response = requests.Response()
        response.status_code = 404
        return requests.HTTPError("404", response=response)

    def test_an_os_error_wrapping_a_404_is_not_transient(self) -> None:
        from gwmock.utils.errors import wraps_transient_failure

        try:
            try:
                raise self._requests_404()
            except OSError as http_error:
                raise OSError("could not fetch the population") from http_error
        except OSError as wrapper:
            error = ValueError("glitch model")
            error.__cause__ = wrapper
        assert not wraps_transient_failure(error)

    def test_a_urllib_404_is_permanent(self) -> None:
        import urllib.error

        from gwmock.utils.errors import is_permanent_http_failure, wraps_transient_failure

        not_found = urllib.error.HTTPError("https://example.com/x", 404, "Not Found", None, None)
        assert is_permanent_http_failure(not_found)
        error = ValueError("fetch failed")
        error.__cause__ = not_found
        assert not wraps_transient_failure(error)
        unavailable = urllib.error.HTTPError("https://example.com/x", 503, "Unavailable", None, None)
        error.__cause__ = unavailable
        assert wraps_transient_failure(error)

    def test_a_cycle_ends_the_walk(self) -> None:
        from gwmock.utils.errors import wraps_transient_failure

        first, second = ValueError("a"), ConnectionError("b")
        first.__cause__ = second
        second.__cause__ = first
        assert wraps_transient_failure(first)
        third = ValueError("c")
        third.__cause__ = third
        assert not wraps_transient_failure(third)

    def test_a_404_through_a_wrapper_while_opening_the_noise_stream_is_attempted_once(self, tmp_path, monkeypatch):
        def open_stream(**_kwargs):
            try:
                raise self._requests_404()
            except OSError as http_error:
                raise OSError("could not fetch the glitch population") from http_error

        def wrapped(**kwargs):
            try:
                open_stream(**kwargs)
            except OSError as wrapper:
                raise ValueError("glitch model could not be configured") from wrapper

        assert _stream_open_attempts(tmp_path, monkeypatch, wrapped) == 1


class TestAContextLinkMeansRaisedWhileHandling:
    """An implicit context is an earlier failure, not the cause of the one raised while handling it."""

    @staticmethod
    def _requests_404():
        import requests

        response = requests.Response()
        response.status_code = 404
        return requests.HTTPError("404", response=response)

    def _mixed_chain(self) -> ValueError:
        """ValueError <- context ConnectionError <- context 404."""
        try:
            try:
                try:
                    raise self._requests_404()
                except OSError:
                    raise ConnectionError("reset while falling back")  # noqa: B904 -- the implicit link under test
            except OSError:
                raise ValueError("glitch model could not be configured")  # noqa: B904
        except ValueError as error:
            return error

    def test_a_connection_failure_raised_while_handling_a_404_is_transient(self) -> None:
        from gwmock.utils.errors import wraps_transient_failure

        assert wraps_transient_failure(self._mixed_chain())

    def test_the_mixed_chain_is_retried_while_opening_the_noise_stream(self, tmp_path, monkeypatch):
        chain = self._mixed_chain()

        def open_stream(**_kwargs):
            raise chain

        assert _stream_open_attempts(tmp_path, monkeypatch, open_stream) == 4

    def test_causes_from_an_os_error_down_to_a_404_are_final(self) -> None:
        from gwmock.utils.errors import wraps_transient_failure

        wrapper = OSError("could not fetch")
        wrapper.__cause__ = self._requests_404()
        error = ValueError("glitch model")
        error.__cause__ = wrapper
        assert not wraps_transient_failure(error)

    def test_a_value_error_raised_while_handling_a_404_is_final(self) -> None:
        from gwmock.utils.errors import wraps_transient_failure

        caught: ValueError | None = None
        try:
            try:
                raise self._requests_404()
            except OSError:
                raise ValueError("no catalogue at that URL")  # noqa: B904 -- the implicit link under test
        except ValueError as error:
            caught = error
        assert caught is not None
        assert not wraps_transient_failure(caught)
