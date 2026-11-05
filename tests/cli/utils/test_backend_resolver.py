"""Tests for backend discovery and validation."""

from __future__ import annotations

import builtins
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gwmock_noise import BaseNoiseSimulator, NoiseConfig, SimulationResult
from gwmock_signal import DetectorStrainStack, GWSimulator

from gwmock.cli.utils.backend_resolver import instantiate_backend, resolve_backend_class, validate_backend


class ModulePopulationBackend:
    """Protocol-conformant backend used for import-path tests."""

    parameter_names = ("mass_1",)
    source_type = "bbh"

    def simulate(self, n_samples: int, **_kwargs):
        return {"mass_1": np.ones(n_samples)}


class LegacyPopulationBackend(ModulePopulationBackend):
    """Protocol-conformant backend used for legacy dotted-path tests."""


class InvalidPopulationBackend:
    """Backend missing required protocol members."""

    parameter_names = ("mass_1",)


class DuckSignalBackend:
    """Duck-typed signal backend with the public ``simulate`` surface."""

    required_params = frozenset({"coa_time"})

    def simulate(
        self,
        params: Mapping[str, object],
        detector_names,
        background=None,
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        effects: Any = None,
        interpolate_if_offset: bool = True,
    ) -> DetectorStrainStack:
        _ = background, minimum_frequency, effects, interpolate_if_offset
        return DetectorStrainStack.from_mapping(
            detector_names,
            {detector: np.zeros(int(sampling_frequency)) + float(params["coa_time"]) for detector in detector_names},
        )


class ProtocolNoiseBackend:
    """Runtime-checkable noise backend used for validation tests."""

    def __init__(self) -> None:
        self.duration = 4.0
        self.sampling_frequency = 8.0
        self.detectors = ["H1"]
        self.seed = None

    def generate(self, duration: float, sampling_frequency: float, detectors: list[str], seed: int | None = None):
        _ = duration, sampling_frequency, seed
        return {detector: np.zeros(4) for detector in detectors}

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ):
        _ = chunk_duration, sampling_frequency, seed
        yield {detector: np.zeros(4) for detector in detectors}

    @property
    def metadata(self) -> dict[str, object]:
        return {"kind": "protocol"}


class RunOnlyNoiseBackend(BaseNoiseSimulator):
    """BaseNoiseSimulator compatibility should remain valid for orchestration."""

    def run(self, config: NoiseConfig) -> SimulationResult:
        return SimulationResult(output_paths={}, config=config)


def test_resolve_population_builtin_alias():
    """Built-in aliases should resolve before other lookup modes."""
    backend_class = resolve_backend_class("population", "file")

    assert backend_class.__name__ == "FilePopulationLoader"


def test_resolve_backend_entry_point_from_installed_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Entry-point discovery should work for an installed third-party package."""
    alias = "tiny_alias_backend_resolver_test"
    package_dir = tmp_path / "plugin-src"
    site_packages = tmp_path / "site-packages"
    (package_dir / "tiny_backend_pkg").mkdir(parents=True)
    (package_dir / "tiny_backend_pkg" / "__init__.py").write_text("")
    (package_dir / "tiny_backend_pkg" / "population.py").write_text(
        textwrap.dedent(
            """
            import numpy as np

            class EntryPointPopulationBackend:
                parameter_names = ("mass_1",)
                source_type = "bbh"

                def simulate(self, n_samples: int, **_kwargs):
                    return {"mass_1": np.full(n_samples, 7.0)}
            """
        )
    )
    (package_dir / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [build-system]
            requires = ["setuptools>=61"]
            build-backend = "setuptools.build_meta"

            [project]
            name = "tiny-backend-pkg"
            version = "0.0.1"

            [project.entry-points."gwmock.population"]
            {alias} = "tiny_backend_pkg.population:EntryPointPopulationBackend"
            """
        ).format(alias=alias)
    )

    uv_path = shutil.which("uv")
    if uv_path is None:  # pragma: no cover - repository tests run with uv available
        raise AssertionError("uv executable is required for entry-point installation tests.")
    subprocess.run(  # noqa: S603
        [
            uv_path,
            "pip",
            "install",
            "--python",
            sys.executable,
            "--quiet",
            "--no-deps",
            "--target",
            str(site_packages),
            str(package_dir),
        ],
        check=True,
    )
    monkeypatch.syspath_prepend(str(site_packages))

    backend = instantiate_backend("population", alias)

    assert backend.__class__.__name__ == "EntryPointPopulationBackend"
    assert backend.simulate(1)["mass_1"][0] == pytest.approx(7.0)


def test_resolve_module_class_literal():
    """Explicit ``module:Class`` paths should resolve directly."""
    backend = instantiate_backend(
        "population",
        "tests.cli.utils.test_backend_resolver:ModulePopulationBackend",
    )

    assert backend.__class__ is ModulePopulationBackend


def test_resolve_legacy_dotted_path_warns_once(monkeypatch):
    """Legacy dotted import paths should warn once and continue to work."""
    from gwmock.cli.utils import backend_resolver

    monkeypatch.setattr(backend_resolver, "_LEGACY_PATH_WARNINGS", set())
    with pytest.warns(DeprecationWarning, match="use 'tests.cli.utils.test_backend_resolver:LegacyPopulationBackend'"):
        first = resolve_backend_class("population", "tests.cli.utils.test_backend_resolver.LegacyPopulationBackend")
    second = resolve_backend_class("population", "tests.cli.utils.test_backend_resolver.LegacyPopulationBackend")

    assert first is LegacyPopulationBackend
    assert second is LegacyPopulationBackend


def test_invalid_population_backend_names_missing_member():
    """Validation failures should be reported at the GWPopSimulator boundary."""
    with pytest.raises(TypeError, match="does not satisfy GWPopSimulator"):
        instantiate_backend("population", "tests.cli.utils.test_backend_resolver:InvalidPopulationBackend")


def test_validate_signal_backend_accepts_duck_typed_simulator():
    """Signal backends may match by public surface without subclassing ``GWSimulator``."""
    backend = DuckSignalBackend()

    validate_backend("signal", "duck", DuckSignalBackend, backend)


def test_validate_signal_backend_accepts_gwsimulator_subclass():
    """Subclass-based signal backends remain valid."""

    class ConcreteSignalBackend(GWSimulator):
        @property
        def required_params(self) -> frozenset[str]:
            return frozenset({"coa_time"})

        def simulate(
            self,
            params,
            detector_names,
            background=None,
            *,
            sampling_frequency,
            minimum_frequency,
            effects=None,
            interpolate_if_offset=True,
        ):
            _ = params, background, sampling_frequency, minimum_frequency, effects, interpolate_if_offset
            return DetectorStrainStack.from_mapping(
                detector_names, {detector: np.zeros(4) for detector in detector_names}
            )

    validate_backend("signal", "concrete", ConcreteSignalBackend, ConcreteSignalBackend())


def test_validate_noise_backend_accepts_protocol_instance():
    """Noise backends may match the runtime-checkable public protocol."""
    validate_backend("noise", "protocol", ProtocolNoiseBackend, ProtocolNoiseBackend())


def test_validate_noise_backend_accepts_run_boundary_class():
    """Run-boundary adapters remain compatible during the orchestration transition."""
    validate_backend("noise", "run-only", RunOnlyNoiseBackend, RunOnlyNoiseBackend())


class NotAWaveformBackend:
    """Resolvable class that does not implement the waveform contract."""


class StubWaveformBackend:
    """Minimal waveform backend, so argument forwarding is testable without the [jax] extra.

    Duck-typed rather than a ``WaveformBackend`` subclass, which exercises the validator's
    match-by-public-surface path -- the same one a third-party entry-point backend relies on.
    """

    def __init__(self, *, taper_fraction: float = 0.0) -> None:
        self.taper_fraction = taper_fraction

    def available_approximants(self) -> tuple[str, ...]:
        return ("StubApproximant",)

    def generate_td_waveform(self, *args, **kwargs):
        raise NotImplementedError("the stub is never asked to generate")


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("lal", "LALSimulationBackend"),
        ("lalsimulation", "LALSimulationBackend"),
        ("LALSimulationBackend", "LALSimulationBackend"),
        ("pycbc", "PyCBCBackend"),
        ("ripple", "RippleBackend"),
        ("gwsignal", "GWSignalBackend"),
    ],
)
def test_resolve_waveform_builtin_aliases(alias: str, expected: str):
    """Each waveform library is selectable by a short alias and by its class name."""
    assert resolve_backend_class("waveform", alias).__name__ == expected


def test_resolving_a_waveform_backend_does_not_require_the_others():
    """Aliases map to import paths, not imported classes, so one absent library is not fatal.

    ``lal`` must resolve in an installation with no ripplegw. Asserted by blocking the
    ripplegw import outright rather than by uninstalling it.
    """
    real_import = builtins.__import__

    def blocked(name: str, *args, **kwargs):
        if name.split(".", maxsplit=1)[0] == "ripplegw":
            raise ImportError("simulated: ripplegw is not installed")
        return real_import(name, *args, **kwargs)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(builtins, "__import__", blocked)
    try:
        assert resolve_backend_class("waveform", "lal").__name__ == "LALSimulationBackend"
    finally:
        monkey.undo()


def test_waveform_backend_arguments_reach_the_constructor():
    """Constructor arguments must be forwarded to the backend being built.

    Checked against a stub rather than ``RippleBackend``, so this runs in an installation
    without the optional stack. What is under test is the forwarding, not any one library --
    and instantiating ripple needs ripplegw even though *resolving* it does not.
    """
    backend = instantiate_backend(
        "waveform",
        "tests.cli.utils.test_backend_resolver:StubWaveformBackend",
        init_kwargs={"taper_fraction": 0.02},
    )

    assert backend.taper_fraction == pytest.approx(0.02)


def test_ripple_accepts_its_taper_fraction():
    """The same forwarding against the real class, which is how it is set from a config file.

    Separate from the stub test because this one needs ripplegw: ``taper_fraction`` is
    ripple-specific, so a stub cannot show that ripple actually accepts it.
    """
    pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
    backend = instantiate_backend("waveform", "ripple", init_kwargs={"taper_fraction": 0.02})

    assert backend.taper_fraction == pytest.approx(0.02)


def test_unknown_waveform_backend_lists_the_available_aliases():
    """The error has to name the alternatives; the set is not guessable."""
    with pytest.raises(ValueError, match="Unknown waveform backend") as raised:
        resolve_backend_class("waveform", "nosuchlibrary")

    message = str(raised.value)
    assert "ripple" in message
    assert "gwmock.waveform" in message, "the entry-point group is part of the contract"


def test_validate_waveform_backend_accepts_a_duck_typed_backend():
    """A third-party backend must not have to subclass gwmock-signal's ABC to be usable.

    The plugin story is that a `gwmock.waveform` entry point works on the same terms as the
    other backend kinds, whose validators also match by public surface. `WaveformFactory` only
    calls these two methods, so requiring the base class would be a stricter contract than the
    code actually needs.
    """
    validate_backend("waveform", "stub", StubWaveformBackend, StubWaveformBackend())


def test_validate_waveform_backend_rejects_a_non_backend():
    """A resolvable-but-wrong class must fail here, not inside WaveformFactory.

    Handed to ``WaveformFactory`` instead, it surfaces as ``AttributeError: 'str' object has
    no attribute 'available_approximants'`` -- a message about a type, naming neither the
    setting at fault nor what it should have been.
    """
    with pytest.raises(TypeError, match="does not satisfy WaveformBackend"):
        instantiate_backend("waveform", "tests.cli.utils.test_backend_resolver:NotAWaveformBackend")
