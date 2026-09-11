"""Selecting the projection implementation from a configuration file.

gwmock-signal has projected with two implementations of the same algorithm for a while -- a
NumPy one on the host and a JAX one -- but a gwmock configuration could not reach the second,
so every run took the first. That is not a micro-optimisation: projection is 97% of the
per-event generation cost at the ET production configuration (1024 s, 8192 Hz, five detectors),
and the device path is 2.7x faster on the same CPU.

``signal.projection-backend`` is that switch. These tests cover what a configuration can say and
what happens to it, in two halves: the constraints that are refused while the configuration is
parsed, and the value reaching the simulator that will use it. The numerical claim -- that the
two implementations return the same strain -- belongs to gwmock-signal and is pinned there,
where both are reachable without a released dependency.
"""

from __future__ import annotations

import builtins
import inspect
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from gwmock.cli.utils.config import Config, SignalConfig
from gwmock.signal.projection_backend import (
    MAX_PROJECTION_SPAN_SECONDS,
    require_backend_accepts_projection_backend,
    require_projection_backend_runnable,
)

_POPULATION_CSV = Path(__file__).resolve().parents[2] / "examples" / "signal" / "bbh_population.csv"


def _real_simulator_takes_it() -> bool:
    """Whether the installed gwmock-signal's compact-binary simulator accepts the setting."""
    from gwmock_signal import CBCSimulator

    return "projection_backend" in inspect.signature(CBCSimulator).parameters


_REAL_SIMULATOR_TAKES_IT = _real_simulator_takes_it()


@pytest.fixture(autouse=True)
def _restore_x64():
    """Restore ``jax_enable_x64`` around every test in this module.

    Validating ``projection-backend: jax`` turns 64-bit mode on process-wide, which is right for
    a run -- the projection refuses to work without it -- and wrong to leave behind in a test
    session, where tests run in random order and the flag changes the dtype of unrelated JAX
    work.
    """
    try:
        import jax
    except ImportError:  # pragma: no cover - JAX is a hard dependency of gwmock-pop
        yield
        return
    previous = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", previous)


def _config_dict(working_directory: Path, *, duration: int = 8, **signal_overrides: Any) -> dict[str, Any]:
    """Return a minimal single-segment BBH config, with *signal_overrides* merged in."""
    signal: dict[str, Any] = {
        "source-type": "bbh",
        "waveform-model": "IMRPhenomD",
        "minimum-frequency": 25,
        "detectors": ["ET-Triangle-Sardinia"],
    }
    signal.update(signal_overrides)
    return {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": 2048,
                "duration": duration,
                "total-duration": duration,
                "start-time": 1577491296,
                "seed": 7,
            },
            "working-directory": str(working_directory),
        },
        "orchestration": {
            "population": {
                "backend": "FilePopulationLoader",
                "source-type": "bbh",
                "n-samples": 1,
                "arguments": {"path": str(_POPULATION_CSV)},
            },
            "signal": signal,
        },
    }


def _load(working_directory: Path, **kwargs: Any) -> Config:
    """Validate a config dict the way the CLI does, through YAML."""
    raw = yaml.safe_dump(_config_dict(working_directory, **kwargs))
    return Config.model_validate(yaml.safe_load(raw))


class _AcceptingSimulator:
    """A signal backend that takes the setting, standing in for a current gwmock-signal."""

    def __init__(self, *, projection_backend: str = "numpy", **_: Any) -> None:
        self.projection_backend = projection_backend

    @property
    def required_params(self) -> frozenset[str]:
        return frozenset()

    def simulate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - never called here
        raise NotImplementedError


class _RefusingSimulator:
    """A signal backend that does not, standing in for one too old to know about it."""

    def __init__(self) -> None:
        pass

    @property
    def required_params(self) -> frozenset[str]:
        return frozenset()

    def simulate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - never called here
        raise NotImplementedError


class _SwallowingSimulator:
    """A signal backend whose constructor takes anything and keeps nothing.

    The shape that made the check necessary: it accepts ``projection_backend`` without error and
    projects with its own default, so nothing between the configuration and the output says the
    request was dropped.
    """

    def __init__(self, **_: Any) -> None:
        pass

    @property
    def required_params(self) -> frozenset[str]:
        return frozenset()

    def simulate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - never called here
        raise NotImplementedError


class TestWhatTheConfigurationAccepts:
    """The name itself, and the default that keeps every existing run unchanged."""

    def test_it_is_absent_unless_asked_for(self):
        """`None` is not `"numpy"`: it leaves each backend on whatever it chooses for itself.

        The distinction matters because the continuous-wave simulator defaults to the *device*
        path -- projection is 99% of one of its segments -- so a gwmock that forwarded "numpy"
        whenever the key was omitted would quietly take that away.
        """
        assert SignalConfig(detectors=["H1"]).projection_backend is None

    def test_an_unknown_name_is_refused(self):
        """Named once and used much later, so an unchecked typo would surface far from here."""
        with pytest.raises(ValidationError, match="projection-backend"):
            SignalConfig(detectors=["H1"], **{"projection-backend": "cuda"})

    def test_the_host_path_is_accepted_explicitly(self):
        """Writing the default down must not be an error, and must not require JAX."""
        assert SignalConfig(detectors=["H1"], **{"projection-backend": "numpy"}).projection_backend == "numpy"


class TestWhatIsRefusedWhileParsing:
    """Constraints gwmock-signal enforces anyway. What is new is that they are found now."""

    def test_the_device_path_needs_earth_rotation(self):
        """The constant-pattern branch has no device implementation, so the pair cannot be served."""
        with pytest.raises(ValidationError, match="earth-rotation"):
            SignalConfig(
                detectors=["H1"],
                **{"projection-backend": "jax", "earth-rotation": False},
            )

    def test_a_segment_longer_than_the_validated_span_is_refused(self, tmp_path):
        """The device path extrapolates sidereal time from one anchor and is validated to a day."""
        with pytest.raises(ValidationError, match="at most 86400 s"):
            _load(
                tmp_path,
                duration=int(MAX_PROJECTION_SPAN_SECONDS) + 1,
                **{"projection-backend": "jax"},
            )

    def test_a_long_segment_is_fine_on_the_host_path(self, tmp_path):
        """The limit belongs to the device path alone, so it must not leak onto the default."""
        config = _load(tmp_path, duration=int(MAX_PROJECTION_SPAN_SECONDS) + 1)
        assert config.orchestration.signal is not None

    def test_a_configuration_without_a_duration_is_left_alone(self, tmp_path):
        """Nothing to measure the span against, so this check has nothing to say about it.

        The run takes its own default duration elsewhere; refusing here, or guessing that
        default, would make the projection setting the thing that reports an unrelated omission.
        """
        raw = _config_dict(tmp_path, **{"projection-backend": "jax"})
        del raw["globals"]["simulator-arguments"]["duration"]

        config = Config.model_validate(raw)

        assert config.orchestration.signal is not None

    def test_an_unparsable_duration_is_not_this_check_s_error_to_report(self, tmp_path):
        """A duration that is not a number fails where the simulation is set up, not here.

        Raising here would answer a malformed duration with a message about a projection backend
        the author may not even have connected to it, and would hide the real mistake behind it.
        """
        raw = _config_dict(tmp_path, **{"projection-backend": "jax"})
        raw["globals"]["simulator-arguments"]["duration"] = "not-a-number"

        config = Config.model_validate(raw)

        assert config.orchestration.signal is not None

    def test_an_ordinary_segment_on_the_device_path_is_accepted(self, tmp_path):
        """The whole point is that this configuration loads; a refusal here refuses the feature."""
        config = _load(tmp_path, duration=1024, **{"projection-backend": "jax"})
        assert config.orchestration.signal is not None
        assert config.orchestration.signal.projection_backend == "jax"


class TestWhatTheEnvironmentMustProvide:
    """JAX, in 64-bit mode. Both are environment problems, so both say what to change."""

    def test_the_host_path_checks_nothing(self):
        """It has no optional dependency, so it must not import one."""
        require_projection_backend_runnable("numpy")

    def test_a_missing_jax_names_the_extra(self, monkeypatch):
        """`gwmock[jax]`, and specifically not ripple, which a reader would otherwise assume."""
        real_import = builtins.__import__

        def _refuse_jax(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "jax":
                raise ImportError("No module named 'jax'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _refuse_jax)
        with pytest.raises(ValueError, match=r"gwmock\[jax\]"):
            require_projection_backend_runnable("jax")

    def test_it_enables_sixty_four_bit_mode_rather_than_only_reporting_it(self):
        """Enabled here because it must be set before any array exists, and parsing is earliest.

        Without this the feature would be unreachable in the configuration that most needs it: a
        LAL-generated run never imports ripple, which is the only thing that has ever turned x64
        on by accident, so `projection-backend: jax` would always refuse.
        """
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", False)

        require_projection_backend_runnable("jax")

        assert jax.config.jax_enable_x64 is True

    def test_a_forced_off_sixty_four_bit_mode_is_refused(self, monkeypatch):
        """Verified rather than assumed: in 32-bit mode the projection is silently wrong.

        Simulated by neutering ``update`` with the flag off, which is what an environment that
        overrode the setting would look like from inside this function.
        """
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", False)
        monkeypatch.setattr(jax.config, "update", lambda *args, **kwargs: None)

        with pytest.raises(ValueError, match="64-bit"):
            require_projection_backend_runnable("jax")


class TestWhatReachesTheSimulator:
    """A setting that is validated and then not forwarded is the failure mode this guards."""

    @staticmethod
    def _adapter(monkeypatch, backend_class: type[Any], **signal_overrides: Any):
        from gwmock.cli import adapter_orchestration
        from gwmock.cli.adapter_orchestration import AdapterOrchestrator

        monkeypatch.setattr(adapter_orchestration, "resolve_backend_class", lambda kind, name: backend_class)
        signal_config = SignalConfig(detectors=["H1"], **signal_overrides)
        network, _ = AdapterOrchestrator._resolve_detector_network(["H1"])
        adapter, _ = AdapterOrchestrator._instantiate_signal_adapter(
            signal_config,
            source_type="bbh",
            detector_network=network,
            duration=8.0,
        )
        return adapter

    def test_the_configured_backend_reaches_the_simulator(self, monkeypatch):
        """The whole path, from the YAML key to the object that will project with it."""
        adapter = self._adapter(monkeypatch, _AcceptingSimulator, **{"projection-backend": "jax"})

        assert adapter._backend.projection_backend == "jax"

    def test_nothing_is_forwarded_when_the_key_is_absent(self, monkeypatch):
        """Absent means "leave it alone", so the simulator must see its own default."""
        adapter = self._adapter(monkeypatch, _AcceptingSimulator)

        assert adapter._backend.projection_backend == "numpy"

    def test_a_simulator_that_cannot_take_it_is_refused(self, monkeypatch):
        """Rather than a TypeError naming neither the setting nor what to do about it."""
        with pytest.raises(ValueError, match="projection-backend"):
            self._adapter(monkeypatch, _RefusingSimulator, **{"projection-backend": "jax"})

    def test_a_simulator_that_would_swallow_it_is_refused(self, monkeypatch):
        """The guard is wired into the path a configuration actually takes, not merely present.

        This is the failure the whole check exists for: `_SwallowingSimulator` accepts
        `projection_backend="jax"` and drops it, so without the refusal the run would finish, the
        metadata would record `jax`, and the strain would be what the host path produced.
        """
        with pytest.raises(ValueError, match="silently discard"):
            self._adapter(monkeypatch, _SwallowingSimulator, **{"projection-backend": "jax"})

    @pytest.mark.skipif(
        not _REAL_SIMULATOR_TAKES_IT,
        reason="the installed gwmock-signal predates the projection_backend argument",
    )
    def test_it_reaches_the_real_compact_binary_simulator(self, monkeypatch):
        """The stubs above pin the wiring; this pins that gwmock-signal's own class matches it.

        Skipped rather than dropped while the dependency floor is behind the release that adds
        the argument: until then `test_a_simulator_that_cannot_take_it_is_refused` is what the
        real class exercises, and this turns itself on at the bump.
        """
        from gwmock_signal import CBCSimulator

        adapter = self._adapter(
            monkeypatch,
            CBCSimulator,
            **{"waveform-model": "IMRPhenomD", "projection-backend": "jax"},
        )

        assert adapter._backend.projection_backend == "jax"


class TestTheBatchedPathRefusesIt:
    """`execution: batched` projects on device unconditionally, so the key would be ignored."""

    def test_it_is_named_with_a_reason(self):
        from gwmock.signal.execution_support import require_execution_supports_configuration

        signal_config = SignalConfig(
            detectors=["H1"],
            **{"projection-backend": "jax", "execution": "batched"},
        )
        with pytest.raises(ValueError, match="projection-backend"):
            require_execution_supports_configuration(signal_config, "batched")


def test_the_span_limit_is_read_from_gwmock_signal():
    """A limit restated here would drift from the one the projection actually enforces."""
    from gwmock_signal.projection import network

    assert (
        getattr(
            network,
            "MAX_LINEAR_SIDEREAL_SPAN_SECONDS",
            getattr(network, "_MAX_LINEAR_SIDEREAL_SPAN_SECONDS", None),
        )
        == MAX_PROJECTION_SPAN_SECONDS
    )


class TestOnlyAProvablyConsumedSettingIsAllowed:
    """Accepting the keyword is not evidence of using it, and only evidence is accepted.

    The dangerous shape is a constructor ending in ``**kwargs``: it takes
    ``projection_backend="jax"`` without complaint, discards it, projects on the host, and
    leaves a run whose configuration and metadata both record the implementation that never
    ran. There is no later point at which that surfaces, so it is refused here.
    """

    def test_arbitrary_keywords_are_not_evidence(self):
        """The regression: this constructor used to be waved through and silently swallow it."""

        class _Flexible:
            def __init__(self, **_: Any) -> None: ...

        with pytest.raises(ValueError, match="silently discard"):
            require_backend_accepts_projection_backend(_Flexible, "jax")

    def test_arbitrary_keywords_alongside_a_named_parameter_are_fine(self):
        """Naming it is the promise; what else the constructor accepts is its own business."""

        class _FlexibleButExplicit:
            def __init__(self, *, projection_backend: str = "numpy", **_: Any) -> None: ...

        require_backend_accepts_projection_backend(_FlexibleButExplicit, "jax")

    def test_a_positional_only_parameter_of_that_name_is_not_enough(self):
        """The setting is forwarded by keyword, so a positional-only parameter cannot receive it.

        Its name would be a promise about a parameter this code can never reach, and passing the
        keyword anyway would land in ``**kwargs`` -- back to the silent case.
        """

        class _PositionalOnly:
            def __init__(self, projection_backend: str = "numpy", /, **_: Any) -> None: ...

        with pytest.raises(ValueError, match="silently discard"):
            require_backend_accepts_projection_backend(_PositionalOnly, "jax")

    def test_an_unreadable_signature_is_refused(self, monkeypatch):
        """No signature is no evidence either, and the failure it leads to is the silent one."""

        class _Opaque:
            def __init__(self, **_: Any) -> None: ...

        def _no_signature(_: Any) -> Any:
            raise ValueError("no signature found")

        monkeypatch.setattr(inspect, "signature", _no_signature)
        with pytest.raises(ValueError, match="cannot be read"):
            require_backend_accepts_projection_backend(_Opaque, "jax")

    def test_a_constructor_with_no_such_parameter_names_the_likely_cause(self):
        """The common case in practice: a gwmock-signal older than the argument."""

        class _Old:
            def __init__(self, waveform_model: str | None = None) -> None: ...

        with pytest.raises(ValueError, match="upgrade it"):
            require_backend_accepts_projection_backend(_Old, "jax")
