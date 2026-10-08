"""Selecting the waveform library from a configuration file.

``signal.backend`` was already taken -- it names the *simulator* (``bbh``, ``bns``, ``sgwb``),
resolved by ``resolve_simulator_backend``. So there was no way to ask for a different waveform
library, and PyCBC and ripple were supported by gwmock-signal but unreachable from YAML.

``signal.waveform-backend`` fills that gap. It selects a *library*, not a compute device:
``ripple`` generates ripple waveforms through the same per-event path, because the batched
device entry point still has no caller in the orchestration layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from gwmock.cli.utils.config import Config

_POPULATION_CSV = Path(__file__).resolve().parents[2] / "examples" / "signal" / "bbh_population.csv"


def _config_dict(working_directory: Path, **signal_overrides: Any) -> dict[str, Any]:
    """Return a minimal single-segment BBH config, with *signal_overrides* merged in.

    The start time places the population's only event (GPS 1577491300) inside the segment;
    without that the run completes and writes files full of zeros, which is a real trap --
    see ``test_the_reference_run_actually_contains_signal``.
    """
    signal: dict[str, Any] = {
        "source-type": "bbh",
        "waveform-model": "IMRPhenomD",
        "minimum-frequency": 25,
        "detectors": ["ET-Triangle-Sardinia"],
        "output": {
            "output_directory": "signal",
            "file_name": "sig-{{ detectors }}.gwf",
            "arguments": {"channel": "{{ detectors }}:STRAIN"},
        },
    }
    signal.update(signal_overrides)
    return {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": 2048,
                "duration": 8,
                "total-duration": 8,
                "start-time": 1577491296,
                "seed": 7,
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


def _orchestrator(working_directory: Path, **signal_overrides: Any):
    """Build an orchestrator from a config dict, as the CLI does."""
    from gwmock.cli.adapter_orchestration import AdapterOrchestrator

    config_path = working_directory / "config.yaml"
    raw = yaml.safe_dump(_config_dict(working_directory, **signal_overrides))
    config_path.write_text(raw, encoding="utf-8")
    config = Config.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    return AdapterOrchestrator.from_config(
        config.orchestration,
        global_simulator_arguments=dict(config.globals.simulator_arguments),
    )


def _active_waveform_backend(orchestrator) -> Any:
    """Return the waveform backend instance the simulator will actually generate with.

    Reaches through ``WaveformFactory``'s private attribute deliberately: the wiring from a
    YAML string to the object that generates strain is exactly what these tests exist to
    check, and no public accessor exposes it.
    """
    factory = orchestrator.signal_adapter._backend._waveform_factory
    return factory._backend


class TestSelection:
    """Which library a config selects."""

    def test_omitting_the_key_keeps_the_default(self, tmp_path):
        """Existing configs must be unaffected; the default stays LAL."""
        backend = _active_waveform_backend(_orchestrator(tmp_path))

        assert type(backend).__name__ == "LALSimulationBackend"

    def test_an_alias_selects_that_library(self, tmp_path):
        """``ripple`` must reach the factory as a ``RippleBackend`` instance, not as a string."""
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        backend = _active_waveform_backend(_orchestrator(tmp_path, **{"waveform-backend": "ripple"}))

        assert type(backend).__name__ == "RippleBackend"

    def test_backend_arguments_are_applied(self, tmp_path):
        """``taper_fraction`` is ripple-specific and had no route from config before this."""
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        orchestrator = _orchestrator(
            tmp_path,
            **{"waveform-backend": "ripple", "waveform-backend-arguments": {"taper_fraction": 0.02}},
        )

        assert _active_waveform_backend(orchestrator).taper_fraction == pytest.approx(0.02)

    def test_arguments_without_a_backend_name_are_still_applied(self, tmp_path):
        """Tuning the default library is legitimate, and must not be silently discarded.

        Gating on the backend name dropped ``waveform-backend-arguments`` from any config that
        named no library -- while the run metadata still recorded them, so provenance claimed
        settings that never reached the waveform. ``f_ref`` on LAL is the ordinary case.
        """
        orchestrator = _orchestrator(tmp_path, **{"waveform-backend-arguments": {"f_ref": 30.0}})
        backend = _active_waveform_backend(orchestrator)

        assert type(backend).__name__ == "LALSimulationBackend", "the library must not change"
        # `_f_ref`: LALSimulationBackend stores it privately and exposes no accessor, unlike
        # ripple's public `taper_fraction`. The alternative is asserting nothing at all here.
        assert backend._f_ref == pytest.approx(30.0), "the argument was dropped"

    def test_the_default_backend_matches_what_the_factory_would_have_built(self, tmp_path):
        """The fallback library is stated in gwmock and again inside ``WaveformFactory``.

        Two copies of one decision, so they are checked against each other rather than trusted:
        if gwmock-signal changed its internal default, supplying arguments alone would silently
        switch which library generates the waveforms.
        """
        from gwmock_signal.waveform.factory import WaveformFactory

        factory_default = type(WaveformFactory()._backend)
        explicit = type(_active_waveform_backend(_orchestrator(tmp_path, **{"waveform-backend": "lal"})))

        assert explicit is factory_default, (
            f"gwmock falls back to 'lal' ({explicit.__name__}) but WaveformFactory defaults to "
            f"{factory_default.__name__}; supplying arguments alone would change the library."
        )

    def test_an_empty_backend_name_is_rejected_rather_than_ignored(self, tmp_path):
        """An empty string is a mistake, and must not read as "use the default".

        This is why the check is ``is not None`` rather than a truth test.
        """
        with pytest.raises(ValueError, match="non-empty string"):
            _orchestrator(tmp_path, **{"waveform-backend": ""})

    def test_an_unknown_name_is_reported_against_the_setting(self, tmp_path):
        """Resolution happens up front, so the error names the library rather than a type.

        Passed through to ``WaveformFactory`` instead, a string arrives as
        ``AttributeError: 'str' object has no attribute 'available_approximants'``.
        """
        with pytest.raises(ValueError, match="Unknown waveform backend"):
            _orchestrator(tmp_path, **{"waveform-backend": "nosuchlibrary"})


class TestGeneratedData:
    """That the selection reaches the strain, not just the object graph."""

    @staticmethod
    def _generate(working_directory: Path, **signal_overrides: Any):
        """Run one segment through the real CLI and return the first channel's samples."""
        import numpy as np
        from gwpy.timeseries import TimeSeries

        from gwmock.cli.simulate import _simulate_impl

        working_directory.mkdir(parents=True, exist_ok=True)
        config_path = working_directory / "config.yaml"
        config_path.write_text(yaml.safe_dump(_config_dict(working_directory, **signal_overrides)), encoding="utf-8")
        _simulate_impl(str(config_path))
        written = working_directory / "output" / "signal" / "sig-ET1_SARD.gwf"
        assert written.is_file(), f"no signal file was written to {written}"
        return np.asarray(TimeSeries.read(written, channel="ET1_SARD:STRAIN").value)

    def test_the_reference_run_actually_contains_signal(self, tmp_path):
        """An all-zero output is the trap here: the run still reports success.

        The population's only event sits at GPS 1577491300, so a segment placed anywhere else
        completes, writes correctly-shaped files, and contains nothing. Asserting non-zero
        content is what separates "the pipeline ran" from "the pipeline produced data".
        """
        import numpy as np

        samples = self._generate(tmp_path)

        assert np.count_nonzero(samples) > 0, "the run wrote a file containing no signal"
        assert np.all(np.isfinite(samples))

    def test_ripple_and_lal_produce_different_strain(self, tmp_path):
        """The selection has to change the data, otherwise the key is decorative.

        Both are asked for the same approximant, so they should agree closely and differ in
        detail. The merger must land in the same sample -- that is what distinguishes "a
        different implementation ran" from "one of them is wrong".

        "The same sample" is a statement about the time offset between the two signals, so it is
        checked as the lag that maximises their cross-correlation. The sample of each signal's own
        peak is not a substitute: near merger adjacent cycles have almost equal amplitude, and a
        change to the detector's antenna weights can make a neighbouring cycle the tallest in one
        signal and not the other while the two stay aligned.

        ``atol=0`` because strain is ~1e-22: the default absolute tolerance would call any two
        strain arrays equal and this assertion could not fail.
        """
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        import numpy as np

        lal = self._generate(tmp_path / "lal", **{"waveform-backend": "lal"})
        ripple = self._generate(tmp_path / "ripple", **{"waveform-backend": "ripple"})

        assert not np.allclose(lal, ripple, rtol=1e-6, atol=0.0), "selecting ripple changed nothing"
        lag = int(np.argmax(np.correlate(lal, ripple, mode="full"))) - (len(ripple) - 1)
        assert lag == 0, f"the two libraries put the merger {lag} samples apart, so one of them is wrong"
        assert np.max(np.abs(ripple)) == pytest.approx(np.max(np.abs(lal)), rel=0.05), (
            "the same approximant from two libraries should agree to a few percent"
        )


class TestProvenance:
    """The selected library has to be recorded, because it changes the data."""

    def test_the_selected_backend_is_recorded(self, tmp_path):
        """Two runs whose strain differs must not carry identical provenance."""
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        orchestrator = _orchestrator(
            tmp_path,
            **{"waveform-backend": "ripple", "waveform-backend-arguments": {"taper_fraction": 0.02}},
        )
        signal_metadata = orchestrator.metadata["orchestration"]["signal"]

        assert signal_metadata["waveform_backend"] == "ripple"
        assert signal_metadata["waveform_backend_arguments"] == {"taper_fraction": 0.02}

    def test_the_default_is_recorded_as_unset(self, tmp_path):
        """``None`` distinguishes "not requested" from a library that was explicitly named."""
        signal_metadata = _orchestrator(tmp_path).metadata["orchestration"]["signal"]

        assert signal_metadata["waveform_backend"] is None
        assert signal_metadata["waveform_backend_arguments"] == {}
