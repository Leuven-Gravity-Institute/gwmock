"""Choosing the projection effects in the configuration, and the effects reaching gwmock-signal.

``orchestration.signal.effects`` names, one key each, the physical effects gwmock-signal's projection
includes. Every one is on unless the configuration turns it off. These tests pin four things: how the
block is read (defaults, each key, a misspelling, the deprecated top-level ``earth-rotation``), that
the resolved effects reach gwmock-signal on the per-event and the batched path, that the run's
metadata records all of them, and that the run's identity changes with each one.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from gwmock_signal.projection import ProjectionEffects
from pydantic import ValidationError

from gwmock.cli.utils.checkpoint import run_fingerprint
from gwmock.cli.utils.config import Config, ProjectionEffectsConfig, SignalConfig

#: The configuration key of each effect, and the ``ProjectionEffects`` field it sets.
_KEYS = {
    "earth-rotation": "earth_rotation",
    "finite-arm-length": "finite_arm_length",
    "source-precession": "source_precession",
    "roemer-delay": "roemer_delay",
    "einstein-delay": "einstein_delay",
    "shapiro-delay": "shapiro_delay",
}

_POPULATION_CSV = Path(__file__).resolve().parents[2] / "examples" / "signal" / "bbh_population.csv"
_START = 1577491296.0


def _signal(**settings: Any) -> SignalConfig:
    return SignalConfig.model_validate({"detectors": ["H1"], **settings})


class TestReadingTheBlock:
    """How ``effects`` is parsed, before anything runs."""

    def test_left_out_every_effect_is_on(self):
        """gwmock-signal's default, so an omitted block means the complete projection."""
        assert _signal().effects.to_projection_effects() == ProjectionEffects()

    def test_the_keys_are_exactly_the_fields_of_projection_effects(self):
        """A field added to gwmock-signal without a key here could not be chosen from a configuration."""
        from dataclasses import fields

        assert set(_KEYS.values()) == {field.name for field in fields(ProjectionEffects)}
        assert {info.alias for info in ProjectionEffectsConfig.model_fields.values()} == set(_KEYS)

    @pytest.mark.parametrize(("key", "field"), list(_KEYS.items()))
    def test_each_key_turns_off_its_own_effect_only(self, key, field):
        effects = _signal(effects={key: False}).effects.to_projection_effects()

        assert effects == ProjectionEffects(**{field: False})

    def test_all_off_is_the_long_wavelength_convention(self):
        """What LAL, Bilby and PyCBC's default projection use."""
        effects = _signal(effects=dict.fromkeys(_KEYS, False)).effects.to_projection_effects()

        assert effects == ProjectionEffects.long_wavelength()

    def test_a_misspelled_key_is_refused(self):
        """Ignored, it would leave the effect on while the configuration reads as if it were off."""
        with pytest.raises(ValidationError, match="earth-rotations"):
            _signal(effects={"earth-rotations": False})

    def test_giving_the_block_raises_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _signal(effects={"earth-rotation": False})

    @pytest.mark.parametrize("key", ["earth-rotation", "earth_rotation"])
    @pytest.mark.parametrize("value", [True, False])
    def test_the_deprecated_key_sets_earth_rotation_and_warns(self, key, value):
        """The top-level key still means what it did; the other effects keep their defaults."""
        with pytest.warns(FutureWarning, match=rf"orchestration\.signal\.{key} is deprecated"):
            signal = _signal(**{key: value})

        assert signal.effects.to_projection_effects() == ProjectionEffects(earth_rotation=value)

    @pytest.mark.parametrize("effects", [{}, {"earth-rotation": True}, {"shapiro-delay": False}])
    def test_the_deprecated_key_and_the_block_together_are_refused(self, effects):
        """Merging them would need a rule for which wins, and the file would say one thing twice."""
        with pytest.raises(ValidationError, match="'earth-rotation' and 'effects' are both given"):
            _signal(**{"earth-rotation": False, "effects": effects})

    def test_the_whole_configuration_dumps_the_block_by_its_keys(self):
        """Metadata stores the configuration dumped by alias, and must load again as the same effects."""
        dumped = _signal(effects={"einstein-delay": False}).model_dump(by_alias=True)

        assert dumped["effects"] == {**dict.fromkeys(_KEYS, True), "einstein-delay": False}
        assert "earth-rotation" not in dumped
        assert _signal(effects=dumped["effects"]).effects == _signal(effects={"einstein-delay": False}).effects


def _config(working_directory: Path, execution: str, effects: dict[str, bool] | None) -> dict[str, Any]:
    signal: dict[str, Any] = {
        "source-type": "bbh",
        "waveform-model": "IMRPhenomD",
        "waveform-backend": "ripple",
        "execution": execution,
        "minimum-frequency": 30,
        "detectors": ["ET-Triangle-Sardinia"],
        "output": {"output_directory": "signal", "file_name": "sig-{{ detectors }}.hdf5"},
    }
    if effects is not None:
        signal["effects"] = effects
    return {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": 1024.0,
                "duration": 16,
                "total-duration": 16,
                "start-time": _START,
                "seed": 20261105,
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
            "signal": signal,
        },
    }


def _orchestrator(working_directory: Path, execution: str, effects: dict[str, bool] | None):
    from gwmock.cli.adapter_orchestration import AdapterOrchestrator

    working_directory.mkdir(parents=True, exist_ok=True)
    config = Config.model_validate(_config(working_directory, execution, effects))
    return AdapterOrchestrator.from_config(
        config.orchestration, global_simulator_arguments=dict(config.globals.simulator_arguments)
    )


#: Configured blocks and the effects each must arrive as. A mixture rather than only all-on, so a
#: path that dropped the block and fell back to gwmock-signal's default would fail.
_THREADED = [
    (None, ProjectionEffects()),
    (
        {"finite-arm-length": False, "shapiro-delay": False},
        ProjectionEffects(finite_arm_length=False, shapiro_delay=False),
    ),
    (
        {**dict.fromkeys(_KEYS, False), "earth-rotation": True},
        ProjectionEffects.long_wavelength(earth_rotation=True),
    ),
]


class TestTheEffectsReachGwmockSignal:
    """The resolved effects are what gwmock-signal is asked for, as ``effects`` and never ``earth_rotation``."""

    @pytest.mark.parametrize(("configured", "expected"), _THREADED)
    def test_on_the_per_event_path(self, tmp_path, monkeypatch, configured, expected):
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        orchestrator = _orchestrator(tmp_path, "per-event", configured)
        backend = orchestrator.signal_adapter._backend
        calls: list[dict[str, Any]] = []
        real = backend.simulate

        def spy(*args: Any, **kwargs: Any):
            calls.append(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(backend, "simulate", spy)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error", message=r".*earth_rotation=\.\.\.\) is deprecated", category=DeprecationWarning
            )
            orchestrator._simulate()

        assert calls, "the spy never saw a call, so this test proves nothing"
        assert [call["effects"] for call in calls] == [expected] * len(calls)
        assert not [call for call in calls if "earth_rotation" in call]

    @pytest.mark.parametrize(("configured", "expected"), _THREADED)
    def test_on_the_batched_path(self, tmp_path, monkeypatch, configured, expected):
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        import gwmock_signal

        orchestrator = _orchestrator(tmp_path, "batched", configured)
        calls: list[dict[str, Any]] = []
        real = gwmock_signal.simulate_cbc_batch

        def spy(*args: Any, **kwargs: Any):
            calls.append(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(gwmock_signal, "simulate_cbc_batch", spy)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error", message=r".*earth_rotation=\.\.\.\) is deprecated", category=DeprecationWarning
            )
            orchestrator._simulate()

        assert calls, "the spy never saw a call, so this test proves nothing"
        assert [call["effects"] for call in calls] == [expected] * len(calls)
        assert not [call for call in calls if "earth_rotation" in call]

    @pytest.mark.parametrize(("configured", "expected"), _THREADED)
    def test_the_metadata_records_every_effect(self, tmp_path, configured, expected):
        """All six, defaults included: the defaults are gwmock-signal's and have changed before."""
        from dataclasses import asdict

        orchestrator = _orchestrator(tmp_path, "per-event", configured)

        recorded = orchestrator.metadata["orchestration"]["signal"]

        assert recorded["effects"] == asdict(expected)
        assert "earth_rotation" not in recorded


def _plan(*signals: SignalConfig | None) -> SimpleNamespace:
    """A plan whose batches carry these signal blocks; ``None`` for a batch without one."""
    return SimpleNamespace(
        batches=[
            SimpleNamespace(simulator_config=SimpleNamespace(signal=signal, population=None)) for signal in signals
        ]
    )


def _identity(*signals: SignalConfig | None) -> str:
    from gwmock.cli.simulate_utils import _resolved_projection_effects

    plan = _plan(*signals)
    return run_fingerprint(
        ["c" * 64] * len(plan.batches),
        Path("output"),
        Path("metadata"),
        resolved_settings=_resolved_projection_effects(plan),
    )


class TestTheRunIdentity:
    """Two runs whose effects differ never share a checkpoint, even from the same configuration bytes.

    The config hash already separates two files that set different effects. What it cannot see is an
    effect a file leaves out, whose value is gwmock-signal's default: the same bytes under a different
    default are a different run. So the identity carries every effect as resolved.
    """

    @pytest.mark.parametrize("key", list(_KEYS))
    def test_each_effect_changes_the_identity(self, key):
        assert _identity(_signal(effects={key: False})) != _identity(_signal())

    def test_the_same_effects_give_the_same_identity(self):
        """Written out or left to the defaults, the resolved effects are the same, and so is this part."""
        assert _identity(_signal(effects=dict.fromkeys(_KEYS, True))) == _identity(_signal())

    def test_every_batch_counts(self):
        """A plan can mix configurations; checking only the first would let the rest through."""
        assert _identity(_signal(), _signal(effects={"roemer-delay": False})) != _identity(_signal(), _signal())

    def test_a_run_without_a_signal_keeps_the_identity_it_had(self):
        """Nothing to resolve, so nothing is added, and a noise-only checkpoint is still accepted."""
        assert _identity(None) == run_fingerprint(["c" * 64], Path("output"), Path("metadata"))

    def test_a_checkpoint_from_before_the_effects_were_recorded_is_refused(self):
        """Its identity was the config hash alone, and the default projection has changed since."""
        assert _identity(_signal()) != run_fingerprint(["c" * 64], Path("output"), Path("metadata"))
