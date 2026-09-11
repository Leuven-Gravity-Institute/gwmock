"""Adapter-backed orchestration for the primary gwmock CLI path."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from gwmock_noise import SimulationResult
from gwmock_pop import GWPopSimulator
from gwmock_signal import DetectorStrainStack, Network
from gwpy.timeseries import TimeSeries as GWpyTimeSeries

from gwmock.cli.utils.backend_resolver import instantiate_backend, resolve_backend_class, validate_backend
from gwmock.cli.utils.config import OrchestrationConfig
from gwmock.cli.utils.config_resolution import resolve_max_samples
from gwmock.cli.utils.template import expand_template_variables
from gwmock.data.time_series.time_series import TimeSeries
from gwmock.data.time_series.time_series_list import TimeSeriesList
from gwmock.mixin.time_series import TimeSeriesMixin
from gwmock.noise import NoiseAdapter
from gwmock.population import PopulationAdapter
from gwmock.signal import SignalAdapter
from gwmock.signal.projection_backend import require_backend_accepts_projection_backend
from gwmock.simulator.base import Simulator
from gwmock.simulator.seeds import derive_seed
from gwmock.simulator.state import StateAttribute
from gwmock.strain_schema import declare_strain_schema

#: Waveform library used when a config gives ``waveform-backend-arguments`` but names no
#: backend. It must stay the same library ``WaveformFactory`` defaults to internally, or
#: supplying arguments would quietly change which library generates the waveforms.
_DEFAULT_WAVEFORM_BACKEND = "lal"

#: Sentinel recorded when a batched config needed ripple for its placement queries and could not
#: have it. Distinct from ``None`` -- which means nothing was substituted and the default genuinely
#: applies -- because the two must behave differently: one answers, the other must refuse.
_RIPPLE_UNAVAILABLE = "__ripple_unavailable__"

logger = logging.getLogger("gwmock")


@dataclass(slots=True)
class AdapterOrchestrationResult:
    """Artifacts produced by one adapter-backed orchestration batch."""

    signal_segment: TimeSeries | None
    noise_result: SimulationResult | None


def _normalize_keys(values: dict[str, Any]) -> dict[str, Any]:
    """Convert YAML-style hyphenated keys into Python identifiers."""
    return {key.replace("-", "_"): value for key, value in values.items()}


class _NoiseTemplateProxy:
    """Forward attribute access to an orchestrator, substituting noise detectors.

    ``expand_template_variables`` resolves ``{{ detectors }}`` via ``getattr``.
    This proxy redirects that lookup to the noise detector list instead of the
    signal detector list stored on the real orchestrator.
    """

    __slots__ = ("_wrapped", "detectors")

    def __init__(self, wrapped: Any, noise_detectors: list[str]) -> None:
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "detectors", noise_detectors)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_wrapped"), name)


class AdapterOrchestrator(TimeSeriesMixin, Simulator):
    """Compose population, signal, and noise adapters inside gwmock."""

    population_index = StateAttribute(0)
    noise_stream_committed_count = StateAttribute(0)

    def __init__(  # noqa: PLR0913
        self,
        *,
        population_events: list[dict[str, Any]],
        population_metadata: dict[str, Any],
        source_type: str,
        source_detector_specs: list[str],
        detector_network: Network,
        detector_resolution: dict[str, Any],
        waveform_model: str | None,
        waveform_arguments: dict[str, Any],
        waveform_options: dict[str, Any],
        signal_parameters: dict[str, Any],
        detectors: list[str],
        duration: float,
        sampling_frequency: float,
        start_time: float,
        max_samples: int,
        minimum_frequency: float,
        earth_rotation: bool,
        noise_arguments: dict[str, Any],
        orchestration_config: OrchestrationConfig,
        population_seed: int | None = None,
        signal_adapter: SignalAdapter | None = None,
        noise_adapter: NoiseAdapter | None = None,
        substituted_waveform_backend: str | None = None,
    ) -> None:
        self._population_events = tuple(population_events)
        self._population_metadata = dict(population_metadata)
        self._source_type = source_type
        self._configuration_checked = False
        self._source_detector_specs = tuple(source_detector_specs)
        self._signal_network = detector_network
        self._detector_resolution = detector_resolution
        self._population_seed = population_seed
        self.waveform_model = waveform_model
        self.waveform_arguments = waveform_arguments
        self.waveform_options = waveform_options
        self.signal_parameters = signal_parameters
        self.minimum_frequency = minimum_frequency
        self.earth_rotation = earth_rotation
        self.orchestration_config = orchestration_config
        if signal_adapter is not None:
            self.signal_adapter = signal_adapter
        elif orchestration_config.signal is not None:
            self.signal_adapter = SignalAdapter.from_source_type(
                source_type=source_type,
                waveform_model=waveform_model,
                backend_arguments={"duration": duration} if source_type == "sgwb" else None,
                network=detector_network,
            )
        else:
            self.signal_adapter = None
        self.detectors = (
            list(self.signal_adapter.detector_names) if self.signal_adapter is not None else list(detectors)
        )
        self.noise_arguments = noise_arguments
        if noise_adapter is not None:
            self.noise_adapter = noise_adapter
        elif orchestration_config.noise is not None:
            self.noise_adapter = NoiseAdapter.from_backend()
        else:
            self.noise_adapter = None
        self._active_signal_output_directory = Path("signal")
        self._active_noise_output_directory = Path("noise")
        self._active_noise_output_arguments: dict[str, Any] = {}
        self._active_noise_file_name: str | list[str] | None = None
        self._active_overwrite = False
        self._noise_stream: Iterator[dict[str, Any]] | None = None
        self._noise_stream_position = 0
        # The glitch truth rows for the chunk this batch writes, read off the injector when
        # the chunk is generated. Per-batch and derived, like `_batch_injections`, so it is
        # not simulator state: a resumed run regenerates the chunk and reads them again.
        self._batch_glitches: list[dict[str, Any]] = []
        # Source parameters of the signals injected into the current batch, in
        # injection order: [{"event_id": <population index>, "parameters": {...}}].
        # An event is attributed to the batch whose segment its *waveform starts* in,
        # which for a compact binary is at or before the segment its coa_time falls in.
        # Recorded into per-batch metadata so a frame's sources are self-describing.
        self._batch_injections: list[dict[str, Any]] = []
        self._pending_noise_chunk: dict[str, Any] | None = None
        # Warned at most once per distinct reason, not once per event: a catalogue whose parameters
        # the query cannot read fails for every row alike, and the message would otherwise arrive
        # thousands of times. Keyed by the failure text rather than a flag, because a single
        # malformed row fails only its own query and its cause differs from a whole-catalogue one --
        # collapsing both into one warning would name the wrong cause for everything after it.
        #: Library the run was switched to when the config named none -- ripple for a batched
        #: config, because that is what `execution: batched` generates with. `None` when nothing
        #: was substituted. Provenance reads this so it names the library that actually ran.
        self._substituted_waveform_backend = substituted_waveform_backend
        self._pre_coalescence_query_failures: set[str] = set()
        self._post_coalescence_query_failures: set[str] = set()
        #: How far the last batched walk traversed, set by `_events_for_this_segment` and consumed
        #: by `_commit_consumed_events`. `None` means no walk has happened since the last commit.
        self._traversed_this_batch: int | None = None
        # Consumption order, as positions into `_population_events`. Established lazily by
        # `_placement_order`, because it costs one query per event and the continuous-wave and
        # stochastic paths never consume the catalogue this way.
        self._placement_order_cache: tuple[int, ...] | None = None

        super().__init__(
            max_samples=max_samples,
            start_time=start_time,
            duration=duration,
            sampling_frequency=sampling_frequency,
            num_of_channels=len(self.detectors),
        )

    @classmethod
    def from_config(
        cls,
        orchestration_config: OrchestrationConfig,
        global_simulator_arguments: dict[str, Any] | None = None,
    ) -> AdapterOrchestrator:
        """Instantiate the composite adapter-backed orchestration path."""
        global_args = _normalize_keys(global_simulator_arguments or {})
        max_samples = resolve_max_samples(simulator_args={}, global_args=global_args)
        duration = float(global_args.get("duration", 4.0))
        sampling_frequency = float(global_args.get("sampling_frequency", 4096.0))
        start_time = float(global_args.get("start_time", 0.0))

        has_population = orchestration_config.population is not None
        has_signal = orchestration_config.signal is not None

        noise_arguments, detector_network, detector_resolution, resolved_detectors = (
            cls._resolve_detectors_and_noise_args(orchestration_config, global_args)
        )
        top_level_seed = int(noise_arguments["seed"]) if noise_arguments.get("seed") is not None else None
        population_seed = derive_seed(top_level_seed, "population") if top_level_seed is not None else None

        population_events: list[dict[str, Any]] = []
        population_metadata: dict[str, Any] = {}
        source_type = ""
        source_detector_specs: list[str] = []
        waveform_model: str | None = None
        waveform_arguments: dict[str, Any] = {}
        waveform_options: dict[str, Any] = {}
        signal_parameters: dict[str, Any] = {}
        minimum_frequency = 5.0
        earth_rotation = True
        signal_adapter: SignalAdapter | None = None
        noise_adapter: NoiseAdapter | None = None

        if has_population:
            population_events, population_metadata, source_type = cls._build_population_context(
                orchestration_config.population,  # type: ignore[arg-type]
                population_seed,
            )
        elif has_signal:
            source_type = str(orchestration_config.signal.source_type)  # type: ignore[union-attr]

        substituted_waveform_backend: str | None = None
        if has_signal:
            signal_adapter, substituted_waveform_backend = cls._instantiate_signal_adapter(
                orchestration_config.signal,
                source_type=source_type,
                detector_network=detector_network,
                duration=duration,
            )
            source_detector_specs = list(orchestration_config.signal.detectors)  # type: ignore[union-attr]
            waveform_model = orchestration_config.signal.waveform_model  # type: ignore[union-attr]
            waveform_arguments = orchestration_config.signal.waveform_arguments  # type: ignore[union-attr]
            waveform_options = orchestration_config.signal.waveform_options  # type: ignore[union-attr]
            signal_parameters = orchestration_config.signal.parameters  # type: ignore[union-attr]
            minimum_frequency = orchestration_config.signal.minimum_frequency  # type: ignore[union-attr]
            earth_rotation = orchestration_config.signal.earth_rotation  # type: ignore[union-attr]
            if orchestration_config.noise is None:
                noise_arguments["detectors"] = resolved_detectors

        if orchestration_config.noise is not None:
            noise_adapter = cls._instantiate_noise_adapter(orchestration_config.noise)

        return cls(
            population_events=population_events,
            population_metadata=population_metadata,
            source_type=source_type,
            source_detector_specs=source_detector_specs,
            detector_network=detector_network,
            detector_resolution=detector_resolution,
            waveform_model=waveform_model,
            waveform_arguments=waveform_arguments,
            waveform_options=waveform_options,
            signal_parameters=signal_parameters,
            detectors=resolved_detectors,
            duration=duration,
            sampling_frequency=sampling_frequency,
            start_time=start_time,
            max_samples=max_samples,
            minimum_frequency=minimum_frequency,
            earth_rotation=earth_rotation,
            noise_arguments=noise_arguments,
            orchestration_config=orchestration_config,
            population_seed=population_seed,
            signal_adapter=signal_adapter,
            substituted_waveform_backend=substituted_waveform_backend,
            noise_adapter=noise_adapter,
        )

    @classmethod
    def _resolve_detectors_and_noise_args(
        cls,
        orchestration_config: OrchestrationConfig,
        global_args: dict[str, Any],
    ) -> tuple[dict[str, Any], Network | None, dict[str, Any], list[str]]:
        """Resolve noise arguments and detector lists for all active adapters."""
        has_noise = orchestration_config.noise is not None
        has_signal = orchestration_config.signal is not None

        noise_arguments: dict[str, Any] = (
            _normalize_keys(orchestration_config.noise.arguments)  # type: ignore[union-attr]
            if has_noise
            else {}
        )
        if global_args.get("seed") is not None:
            noise_arguments.setdefault("seed", int(global_args["seed"]))

        detector_network: Network | None = None
        detector_resolution: dict[str, Any] = {}
        resolved_detectors: list[str] = []

        if has_signal:
            detector_network, detector_resolution = cls._resolve_detector_network(
                orchestration_config.signal.detectors  # type: ignore[union-attr]
            )
            resolved_detectors = cls._network_detector_names(detector_network)

        if has_noise:
            if "detectors" in noise_arguments:
                noise_network, _ = cls._resolve_detector_network(noise_arguments["detectors"])
                noise_arguments["detectors"] = cls._network_detector_names(noise_network)
                if not has_signal:
                    resolved_detectors = noise_arguments["detectors"]
            elif has_signal:
                noise_arguments["detectors"] = resolved_detectors
            else:
                raise ValueError(
                    "noise orchestration without a signal section must specify detectors under noise.arguments.detectors"
                )

        return noise_arguments, detector_network, detector_resolution, resolved_detectors

    @classmethod
    def _build_population_context(
        cls,
        population_config: Any,
        population_seed: int | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
        """Materialise population events and return (events, metadata, source_type)."""
        population_backend = cls._instantiate_population_backend(population_config)
        population_adapter = PopulationAdapter.from_backend(
            population_backend,
            n_samples=population_config.n_samples,
            **({"seed": population_seed} if population_seed is not None else {}),
        )
        population_events = list(population_adapter.iter_event_parameters())
        sort_key = population_config.sort_by
        if sort_key and population_events:
            missing = [sort_key not in event for event in population_events]
            explicit = "sort_by" in getattr(population_config, "model_fields_set", set())
            source_type = (population_config.source_type or "").lower()
            if all(missing) and not explicit and source_type == "cw":
                # The default key is `coa_time`, which only compact-binary catalogues carry. A
                # continuous-wave catalogue has no coalescence time and no ordering that means
                # anything -- every source contributes to every segment -- so the file's own order
                # stands.
                #
                # Gated on the source type, not merely on the key being absent everywhere. Without
                # that gate a compact-binary catalogue that lost its `coa_time` column, to a header
                # typo say, would stop raising and start running unsorted: the per-event loop never
                # breaks when `coa_time` is None, so every event would land in the first segment
                # and the output would look entirely reasonable with every coalescence time wrong.
                sort_key = None
            elif any(missing):
                raise ValueError(f"Population event ordering key '{sort_key}' is missing from one or more events.")
        if sort_key and population_events:
            population_events.sort(key=lambda event: event[sort_key])
        return population_events, population_adapter.metadata, population_adapter.source_type

    @staticmethod
    def _instantiate_population_backend(population_config) -> GWPopSimulator:
        backend_arguments = dict(population_config.arguments)
        if population_config.source_type is not None:
            backend_arguments.setdefault("source_type", population_config.source_type)

        return instantiate_backend(
            "population",
            population_config.backend,
            init_kwargs=backend_arguments,
        )

    @staticmethod
    def _instantiate_signal_adapter(
        signal_config,
        *,
        source_type: str,
        detector_network: Network,
        duration: float,
    ) -> tuple[SignalAdapter, str | None]:
        """Return the adapter and the backend name substituted for it, if any."""
        substituted: str | None = None
        backend_name = signal_config.backend or source_type
        backend_class = resolve_backend_class("signal", backend_name)
        backend_arguments = _normalize_keys(dict(signal_config.arguments))
        if source_type == "sgwb":
            backend_arguments.setdefault("duration", duration)

        # `waveform-backend` names a library; the simulator wants an instance. Resolved here so an
        # unknown name is reported against the setting, rather than reaching WaveformFactory as a
        # string and failing as an AttributeError about `str`.
        #
        # Arguments alone are enough to require this. Gating on the name would silently discard
        # `waveform-backend-arguments` from a config that only tunes the default library --
        # `f_ref` on LAL, say -- while the run metadata still recorded them as applied. So when
        # arguments are given without a name, the default is constructed explicitly; that is the
        # same class WaveformFactory would have built for itself, so nothing else changes.
        #
        # `is not None` rather than a truth test, so an empty configured name reaches the
        # resolver and is rejected there instead of being treated as absent.
        waveform_backend_name = getattr(signal_config, "waveform_backend", None)
        waveform_backend_arguments = _normalize_keys(dict(signal_config.waveform_backend_arguments))

        # `execution: batched` always generates with ripple -- `_batched_waveform_backend` refuses
        # any other library rather than substituting one. So when a batched config names no backend,
        # the adapter must be built on ripple too, or the *placement queries* answer for a LAL buffer
        # while generation produces a ripple one.
        #
        # That mismatch deletes signal, and not hypothetically. The two libraries round buffer
        # lengths onto different grids, so which tail is longer flips with mass and cutoff: measured
        # across 24 mass/cutoff combinations, 9 have ripple's tail longer than LAL's -- 1.4+1.35 at
        # 5 Hz answers 819.20 s against a generated 934.17 s, so an event up to 115 s of whose
        # content lands inside the segment is reported finished and skipped. Low cutoffs and low
        # masses are exactly the ET regime. Aligning the two here removes the whole class rather
        # than relying on which way the rounding happens to fall.
        batched = str(getattr(signal_config, "execution", "per-event")) == "batched"
        if waveform_backend_name is None and batched:
            # Not eager: a reviewer showed that instantiating ripple here breaks every caller that
            # only wants to *construct* an orchestrator without the [jax] extra -- config
            # validation, stubbed placement, the tests that pass waveform_backend=None -- and worse,
            # it masks a real configuration error behind "rippleGW is not installed".
            #
            # Falling back is safe precisely where it applies: without ripple the batched path
            # cannot generate at all, so no placement decision it might get wrong is ever acted on,
            # and generation raises the actionable install error instead of this one.
            try:
                backend_arguments["waveform_backend"] = instantiate_backend(
                    "waveform", "ripple", init_kwargs=waveform_backend_arguments
                )
                waveform_backend_name = "ripple"
                substituted = "ripple"
            except ImportError:
                # Construction still succeeds -- one reviewer showed that raising here breaks config
                # validation, stubbed placement and every caller that only builds an orchestrator.
                # But it must not proceed silently either: another reviewer then showed that a
                # LAL-backed adapter answers placement, skips an event ripple would still be
                # producing, and advances `population_index` -- and because the batched path returns
                # early on an empty batch, generation never runs and the install error never
                # surfaces. The event is consumed, never generated, and nothing is reported.
                #
                # So the failure is deferred to the first placement query instead of being dropped:
                # constructing is harmless, asking a question whose answer would be wrong is not.
                substituted = _RIPPLE_UNAVAILABLE
                logger.debug("ripple is unavailable; placement queries will refuse rather than answer.")

        if waveform_backend_name is not None or waveform_backend_arguments:
            backend_arguments["waveform_backend"] = instantiate_backend(
                "waveform",
                waveform_backend_name if waveform_backend_name is not None else _DEFAULT_WAVEFORM_BACKEND,
                init_kwargs=waveform_backend_arguments,
            )
        # `projection-backend` reaches gwmock-signal as a constructor argument, and only when the
        # user set it. Passing the default explicitly would be a behaviour change rather than a
        # no-op: each simulator picks its own -- the host path for compact binaries, the device
        # path for continuous waves, where projection is 99% of a segment -- and forwarding
        # "numpy" everywhere would silently take the second one off it.
        projection_backend = getattr(signal_config, "projection_backend", None)
        if projection_backend is not None:
            require_backend_accepts_projection_backend(backend_class, projection_backend)
            backend_arguments["projection_backend"] = projection_backend

        backend_instance = SignalAdapter.instantiate_backend(
            backend_class,
            waveform_model=signal_config.waveform_model,
            backend_arguments=backend_arguments,
        )
        validate_backend("signal", backend_name, backend_class, backend_instance)
        adapter = SignalAdapter.from_backend(
            source_type=source_type,
            backend=backend_instance,
            network=detector_network,
        )
        return adapter, substituted

    @staticmethod
    def _instantiate_noise_adapter(noise_config) -> NoiseAdapter:
        if noise_config.backend is None:
            return NoiseAdapter.from_backend()
        backend_instance = instantiate_backend(
            "noise",
            noise_config.backend,
            init_kwargs=dict(noise_config.arguments),
        )
        return NoiseAdapter.from_backend(backend_instance)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return orchestration metadata for reproducibility."""
        signal_segment_seed = self._signal_segment_seed()
        return {
            **super().metadata,
            "orchestration": {
                "source_type": self._source_type,
                "population_events_total": len(self._population_events),
                # "Remaining" counts events still to be consumed, which only means something for a
                # source consumed once. Every continuous-wave source is present in every segment,
                # so nothing is ever consumed and the count would read as the full catalogue
                # forever -- true but misleading. Reported as null there instead.
                "population_events_remaining": (
                    None if self._source_type == "cw" else len(self._population_events) - int(self.population_index)
                ),
                "population": {
                    "metadata": self._population_metadata,
                    "seed": self._population_seed,
                },
                "signal": {
                    "waveform_model": self.waveform_model,
                    # Which library generated the polarizations. Recorded because it changes the
                    # data: the same approximant from LAL and from ripple agree closely but not
                    # exactly, so without this two runs with materially different strain would
                    # carry identical provenance. Read from the config rather than stored
                    # separately, to keep one source of truth. ``None`` means none was requested
                    # *and* none was substituted, so gwmock-signal's own default (LAL) applied.
                    #
                    # A batched config naming no backend is such a substitution: the adapter is
                    # built on ripple, because that is what `execution: batched` generates with.
                    # Reading the raw key here would record ``None`` for a run that used ripple, so
                    # provenance and replay would name a different library than the one that
                    # produced the data -- and the note above is exactly why that matters.
                    "waveform_backend": self._effective_waveform_backend(),
                    "waveform_backend_arguments": self._configured_waveform_backend_arguments(),
                    "waveform_arguments": self.waveform_arguments,
                    "waveform_options": self.waveform_options,
                    "parameters": self.signal_parameters,
                    "minimum_frequency": self.minimum_frequency,
                    "earth_rotation": self.earth_rotation,
                    "detector_specs": list(self._source_detector_specs),
                    "detectors": self.detectors,
                    "network_resolution": self._detector_resolution,
                    "segment_seed": signal_segment_seed,
                    "injections": self._segment_injections(),
                },
                "noise": {
                    "arguments": self.noise_arguments,
                    "stream_seed": self._noise_stream_seed(),
                    "state_model": "gwmock consumes one shared gwmock_noise.open_stream() iterator across batches.",
                    # The per-event glitch truth for this batch, and what its columns mean.
                    # `glitches` under `arguments` above is the configuration -- which models at
                    # which rates -- and this is what those models actually injected.
                    #
                    # Spelled `glitch_injections` here and in `noise.glitch_injections`, the same
                    # name at both depths, because that is the name a released file's sanitiser
                    # strips: this blob is copied verbatim into the record, so a second spelling
                    # would be a second copy of the truth that the strip did not reach -- which is
                    # exactly how the signal parameters once leaked.
                    "glitch_injections": list(self._batch_glitches),
                    "glitch_catalogue": (
                        None if self.noise_adapter is None else self.noise_adapter.glitch_catalogue_description()
                    ),
                },
                "segment_seeds": self.segment_seeds(),
            },
        }

    def _configured_waveform_backend(self) -> str | None:
        """Return the requested waveform-backend name, or ``None`` if the default applied."""
        signal_config = self.orchestration_config.signal
        return None if signal_config is None else getattr(signal_config, "waveform_backend", None)

    def _effective_waveform_backend(self) -> str | None:
        """Return the waveform library this run actually used, not the one the config named.

        The two differ in exactly one case, and it is a case provenance must not misreport: a
        batched config naming no backend is built on ripple, because ``execution: batched``
        generates with ripple and the placement queries have to describe the buffer generation will
        produce. Recording the raw key would file such a run under the default library.

        Returns:
            The backend name, or ``None`` when none was requested and none was substituted.
        """
        configured = self._configured_waveform_backend()
        if configured is not None:
            return configured
        if self._substituted_waveform_backend == _RIPPLE_UNAVAILABLE:
            # Nothing was substituted, and nothing generated either -- such a run cannot get past
            # its first placement query. Reporting the sentinel as a library name would be worse
            # than reporting the truth, which is that the default applied.
            return None
        # Recorded by the code that made the substitution rather than recovered from the adapter.
        # The first version of this walked `_backend._waveform_factory` looking for a RippleBackend,
        # guessed the attribute name wrong, and silently returned None for every run -- reporting
        # the library correctly only by accident. Two private attributes across a package boundary
        # is what this codebase warns about elsewhere, and this is why.
        return self._substituted_waveform_backend

    def _configured_waveform_backend_arguments(self) -> dict[str, Any]:
        """Return the waveform-backend constructor arguments, empty when none were given."""
        signal_config = self.orchestration_config.signal
        if signal_config is None:
            return {}
        return dict(getattr(signal_config, "waveform_backend_arguments", {}) or {})

    def resolved_config(self) -> dict[str, Any]:
        """Return runtime-resolved config overrides, shaped like OrchestrationConfig.

        Aggregates each sub-adapter's ``resolved_config()`` (only noise today)
        into an ``OrchestrationConfig``-shaped fragment — e.g.
        ``{"noise": {"arguments": {"glitches": [...]}}}`` — that a caller deep-
        merges over the input config to obtain a fully-resolved, replayable
        config. Returns an empty mapping when nothing needed resolving, so
        callers can skip writing a resolved layer for a purely parametric run.
        """
        resolved: dict[str, Any] = {}
        noise_resolved = self.noise_adapter.resolved_config() if self.noise_adapter is not None else {}
        if noise_resolved:
            resolved["noise"] = {"arguments": noise_resolved}
        return resolved

    def set_batch_context(self, *, batch: Any, output_directory: Path, overwrite: bool) -> None:
        """Resolve per-batch output directories and runtime arguments."""
        if batch.simulator_config.signal is not None:
            signal_output = batch.simulator_config.signal.output
            self._active_signal_output_directory = self._resolve_output_directory(
                output_directory,
                signal_output.output_directory,
                fallback_subdir="signal",
            )
        if batch.simulator_config.noise is not None:
            noise_output = batch.simulator_config.noise.output
            self._active_noise_output_directory = self._resolve_output_directory(
                output_directory,
                noise_output.output_directory,
                fallback_subdir="noise",
            )
            noise_detectors = list(self.noise_arguments.get("detectors", []))
            proxy = _NoiseTemplateProxy(self, noise_detectors)
            self._active_noise_output_arguments = expand_template_variables(noise_output.arguments or {}, proxy)
            self._active_noise_file_name = noise_output.file_name
        self._active_overwrite = overwrite

    def simulate(self) -> AdapterOrchestrationResult:
        """Generate signal and/or noise for the current batch depending on active adapters."""
        signal_segment = TimeSeriesMixin.simulate(self) if self.signal_adapter is not None else None
        noise_result = self._run_noise_batch() if self.noise_adapter is not None else None
        return AdapterOrchestrationResult(signal_segment=signal_segment, noise_result=noise_result)

    def _simulate(self) -> TimeSeriesList:
        """Generate signal chunks for the current segment from population events."""
        # Ahead of the general check, which would otherwise answer a continuous-wave request for
        # the batched path by complaining that `signal.arguments` would be ignored. True, but not
        # the reason: the batched entry point generates coalescences from a catalogue of events,
        # and a continuous wave is not one. The specific message is the useful one.
        if self._source_type == "cw" and self._execution_mode() == "batched":
            raise ValueError(
                "execution: batched is not available for continuous waves. The batched entry point "
                "generates compact-binary events from a catalogue of coalescences; a continuous "
                "wave has no coalescence and is present in every segment. Use execution: per-event, "
                "which is the default."
            )
        self._require_configuration_supported()

        if not self._population_events and self._source_type == "sgwb":
            return self._simulate_stationary_signal_segment()

        # Continuous waves fit neither branch above. They are stationary -- on for the whole run,
        # with no coalescence time and no spill-over into later segments -- but unlike a stochastic
        # background they have discrete sources, a catalogue of pulsars that must all be summed
        # into every segment. So the events are *not* consumed as the per-event loop consumes them:
        # `population_index` never advances, because every pulsar contributes to every segment.
        if self._source_type == "cw":
            return self._simulate_continuous_wave_segment()

        if self._execution_mode() == "batched":
            return self._simulate_batched_segment()

        chunks = TimeSeriesList()
        self._batch_injections = []
        order = self._placement_order()
        while self.population_index < len(order):
            # `population_index` counts events consumed, and `order` turns that into the catalogue
            # position -- which is also the event id, so provenance keeps naming events by their
            # place in the loaded catalogue rather than by the order generation happened to take.
            event_id = int(order[int(self.population_index)])
            parameters = dict(self._population_events[event_id])
            if self._event_ended_before_segment_start(parameters):
                # Consumed without being generated, the same as in `_events_for_this_segment`, and
                # stepped over rather than breaking so the events behind it are still reached.
                self.population_index = cast(int, self.population_index) + 1
                continue
            if not self._event_starts_before_segment_end(parameters):
                break
            strain = self.signal_adapter.simulate(
                parameters,
                sampling_frequency=float(self.sampling_frequency.value),
                minimum_frequency=self.minimum_frequency,
                waveform_arguments=self.waveform_arguments,
                waveform_options=self.waveform_options,
                earth_rotation=self.earth_rotation,
            )
            # `event_id` alongside the parameters, and both survive onto a tail, so a chunk that
            # crosses a segment boundary can still be attributed to its event in the next segment.
            strain.metadata.update({"injection_parameters": dict(parameters), "event_id": event_id})
            self._batch_injections.append({"event_id": event_id, "parameters": dict(parameters)})
            chunks.append(strain)
            self.population_index = cast(int, self.population_index) + 1
            if strain.start_time >= self.end_time:
                break
        return chunks

    def _execution_mode(self) -> str:
        """Return how this segment's events should be generated."""
        signal_config = self.orchestration_config.signal
        return "per-event" if signal_config is None else str(getattr(signal_config, "execution", "per-event"))

    def _require_configuration_supported(self) -> None:
        """Check the signal settings against what the chosen execution path actually reads.

        Runs before any events are consumed: ``population_index`` is checkpointed state, so failing
        after advancing it would make a resumed run skip the events this call rejected. Runs once
        per orchestrator rather than once per segment, so a warning is not repeated for every
        segment of a long run.
        """
        if self._configuration_checked or self.orchestration_config.signal is None:
            return

        from gwmock.signal.execution_support import require_execution_supports_configuration  # noqa: PLC0415

        require_execution_supports_configuration(
            self.orchestration_config.signal, self._execution_mode(), self._source_type
        )
        # Only after it passes. `retry_with_backoff` re-runs a failed batch on this same instance
        # and restores `state`, which this flag is not part of -- so setting it first would let the
        # retry sail past a configuration the first attempt refused.
        self._configuration_checked = True

    def _placement_order(self) -> tuple[int, ...]:
        """Return catalogue positions ordered by when each event's waveform starts.

        The loops consume the catalogue in order and stop at the first event that does not belong to
        the current segment. That is only sound if "belongs" is a prefix property, and under the
        waveform-start rule it is not for a ``coa_time``-sorted catalogue: the lead varies with the
        source, roughly 3 s for a heavy binary black hole against ~100 s for a binary neutron star at
        a low cutoff. So a long-lead event can sit *after* a short-lead one whose waveform start has
        already crossed the boundary::

            segment [100, 116):
              coa=117 lead=10 -> start 107   belongs
              coa=118 lead= 1 -> start 117   stops the walk
              coa=119 lead=20 -> start  99   belongs, but is never reached

        The third event would then be generated a segment late and cropped -- the very loss this rule
        exists to prevent, reintroduced by catalogue ordering alone. Sorting by the placement key
        makes the prefix property hold again, which keeps ``population_index`` meaning "every event
        before this one is consumed" -- the invariant resume depends on.

        Stable, so events sharing a start keep their catalogue order, and events with no ``coa_time``
        sort first because they belong to every segment. The key is ``coa_time - lead``, with the lead
        taken as zero when unknown, matching the fallback in
        :meth:`_event_starts_before_segment_end`.

        Computed once. It costs one query per event, which is arithmetic over the conditioning
        settings rather than a generation -- orders of magnitude below the cost of generating the same
        event, which the run is about to pay anyway.

        Returns:
            Catalogue positions in consumption order.
        """
        if self._placement_order_cache is None:
            self._placement_order_cache = tuple(sorted(range(len(self._population_events)), key=self._placement_key))
        return self._placement_order_cache

    def _placement_key(self, position: int) -> float:
        """Return the time the event at *position* starts, for ordering consumption.

        Args:
            position: Index into the catalogue as loaded.

        Returns:
            The waveform's start time, or ``-inf`` for an event with no coalescence time, which
            belongs to every segment and so must never delay one.
        """
        parameters = self._population_events[position]
        coa_time = parameters.get("coa_time")
        if coa_time is None:
            return float("-inf")
        lead = self._pre_coalescence_duration(parameters)
        return float(coa_time) if lead is None else float(coa_time) - float(lead)

    def _event_starts_before_segment_end(self, parameters: Mapping[str, Any]) -> bool:
        """Return whether this event's *waveform* begins before the current segment ends.

        The segment boundary rule, in one place because two loops apply it -- the per-event loop and
        :meth:`_events_for_this_segment` -- and a run that switches between execution modes mid-way
        must not skip or repeat an event because the two disagreed.

        The rule is the waveform's start, not ``coa_time``. A compact binary's buffer begins seconds
        before coalescence, so an event whose ``coa_time`` lands just past a boundary has content
        belonging to the segment before it. Claiming that event by ``coa_time`` puts its buffer start
        outside the claiming segment, where injection crops it and it is gone: the earlier segments
        are already written. Claiming it by its start instead places the whole waveform, and the
        forward overflow already carried between segments delivers the rest.

        ``None`` from the query means *unknown*, and unknown falls back to the ``coa_time`` rule --
        the previous behaviour, whose loss is reported by
        :meth:`~gwmock.data.time_series.time_series.TimeSeries._report_content_before_segment`.
        Treating unknown as zero would be the same fallback while looking like an answer.

        The query is arithmetic over the conditioning settings rather than a generation, so it is
        cheap enough to ask per candidate event; it is not cached, and the cost is one query per
        event plus one per segment for the event that ends it.

        Args:
            parameters: The candidate event's source parameters.

        Returns:
            Whether the event belongs to this segment. Events without a ``coa_time`` always do,
            which is how non-coalescing sources reach the loop unchanged.
        """
        coa_time = parameters.get("coa_time")
        if coa_time is None:
            return True
        end_time_value = float(getattr(self.end_time, "value", self.end_time))
        lead = self._pre_coalescence_duration(parameters)
        start_time_value = float(coa_time) if lead is None else float(coa_time) - float(lead)
        return start_time_value < end_time_value

    @property
    def signal_adapter(self):
        """The adapter answering placement queries and generating signals."""
        return self._signal_adapter

    @signal_adapter.setter
    def signal_adapter(self, adapter) -> None:
        """Replace the adapter, and stop claiming a substitution that described the old one.

        ``_substituted_waveform_backend`` records which library `from_config` built the adapter on.
        Replacing the adapter -- which tests do routinely, and which library callers may do --
        invalidates that: a reviewer showed the recorded value survived the swap, so provenance
        reported ripple while a LAL-backed stand-in answered every query. Clearing it makes the
        record describe nothing rather than describe the wrong thing, and a caller who knows better
        can set it explicitly.
        """
        self._signal_adapter = adapter
        if getattr(self, "_substituted_waveform_backend", None) is not None:
            self._substituted_waveform_backend = None

    def _placement_backend_can_answer(self) -> bool:
        """Whether the tail query describes the buffer this run will actually generate.

        A batched config with no ``waveform-backend`` needs ripple, because that is what it
        generates with. When ripple is missing the adapter is left on the default library, whose
        tail is shorter than ripple's at some masses and cutoffs -- so *acting* on that answer can
        skip an event ripple would still be producing.

        Only the skip is dangerous, so only the skip is withheld: the tail reports *unknown*, which
        by this module's standing rule claims the event rather than dropping it. The batch is then
        non-empty, generation runs, and ripple's own install error surfaces where it belongs.
        Refusing to answer at all was tried and was too broad -- it made every batched unit test and
        every construct-and-inspect caller raise, in an environment where nothing was going to
        generate anyway.

        Returns:
            Whether the answering backend is the one that will generate.
        """
        return self._substituted_waveform_backend != _RIPPLE_UNAVAILABLE

    def _event_ended_before_segment_start(self, parameters: Mapping[str, Any]) -> bool:
        """Return whether this event's waveform has finished before the current segment begins.

        The lower bound, and the reason it exists: :meth:`_event_starts_before_segment_end` is
        one-sided. In a sequential run that is invisible, because the earlier segments have already
        consumed the earlier events. In a run whose ``start-time`` is later than its population's
        first event -- resuming a campaign mid-catalogue, or slicing a long population into per-day
        jobs -- nothing has consumed them, every one satisfies the upper bound, and the first
        segment claims the whole back catalogue. Measured on the shipped ET config: 126 events
        batched for a segment containing 13, refused by the preflight at an estimated 12.5 GiB.

        **This must not be folded into the upper-bound predicate.** Both loops ``break`` on the
        first event that does not belong, and ``population_index`` means "every event before this
        one is consumed". An early event answering "does not belong" would stop the walk at
        position 0, consume nothing, and repeat that on every following segment -- the run would
        stall permanently, which is worse than the over-batching being fixed. A finished event is
        *stepped over and consumed*, which is what the callers do with this answer.

        ``None`` means *unknown*, and unknown claims the event. The tail is a fixed fraction of the
        buffer rather than a physical ringdown -- 0.4 s for a 4 s binary-black-hole buffer, 25.6 s
        for a 256 s binary-neutron-star one -- so an event coalescing seconds before the segment
        may still be producing signal inside it. Reading an unknown tail as zero would conclude
        every such event finished at ``coa_time`` and discard it, silently and for a whole backend
        at a time; claiming it merely generates a waveform injection then crops.

        Args:
            parameters: The candidate event's source parameters.

        Returns:
            Whether the event's content is wholly behind this segment. Events without a
            ``coa_time``, and events whose tail is unknown, are never reported as finished.
        """
        coa_time = parameters.get("coa_time")
        if coa_time is None:
            return False
        tail = self._post_coalescence_duration(parameters)
        if tail is None:
            return False
        start_time_value = float(getattr(self.start_time, "value", self.start_time))
        # Strict, so an event ending exactly on the boundary is claimed: its final sample lands on
        # the segment's first sample. The upper bound is strict at its own end for the same reason.
        return float(coa_time) + float(tail) < start_time_value

    def _post_coalescence_duration(self, parameters: Mapping[str, Any]) -> float | None:
        """Return the event's waveform tail in seconds, or ``None`` when it cannot be established.

        The complement of :meth:`_pre_coalescence_duration`, with the same swallow-and-report
        handling and for the same reason: this query decides *placement*, and letting it fail a run
        would turn an incomplete catalogue column into a crash in an unrelated part of the loop.

        The fallback differs from the pre side's in what it costs. There, an unknown lead falls
        back to the ``coa_time`` rule and the loss is reported per signal. Here, an unknown tail
        means the event is claimed and generated -- work that may be discarded, which is exactly
        today's behaviour and the thing being improved. Failing safe here is failing *slow*, not
        failing wrong.

        Args:
            parameters: The candidate event's source parameters.

        Returns:
            Seconds after ``coa_time`` at which the waveform ends, or ``None`` for unknown.
        """
        if self.signal_adapter is None:
            return None
        if not self._placement_backend_can_answer():
            # Unknown, not zero: the caller claims the event and generation raises the real error.
            reason = "ripple is not installed, so the tail would describe a different library"
            if reason not in self._post_coalescence_query_failures:
                self._post_coalescence_query_failures.add(reason)
                logger.warning(
                    "Cannot establish how long after coalescence a waveform ends (%s). Every event "
                    "is claimed rather than skipped, which is the conservative direction; "
                    "generation will fail with the install instruction.",
                    reason,
                )
            return None
        # Called directly, exactly as the pre side calls its own query, and deliberately *not*
        # behind a `getattr` guard. A guard here returns `None` for an adapter without the method
        # and says nothing, which is precisely how this change shipped as a silent no-op: the
        # orchestrator's own `SignalAdapter` had no `post_coalescence_duration`, every tail came
        # back unknown, and the whole lower bound was dead in production while the stub-driven
        # tests passed. Letting the `AttributeError` fall into the handler below produces the same
        # conservative `None` *and* warns once, which is the codebase's stated rule: the fallback
        # is the previous behaviour, and its cost is reported rather than silent.
        try:
            return self.signal_adapter.post_coalescence_duration(
                parameters,
                sampling_frequency=float(self.sampling_frequency.value),
                minimum_frequency=self.minimum_frequency,
                waveform_arguments=self.waveform_arguments,
                waveform_options=self.waveform_options,
            )
        except Exception as exc:
            # Deliberately broad, deduplicated by reason rather than by a flag -- see
            # `_pre_coalescence_duration` for why a flag mislabels a second, different failure.
            reason = f"{type(exc).__name__}: {exc}"
            if reason not in self._post_coalescence_query_failures:
                self._post_coalescence_query_failures.add(reason)
                logger.warning(
                    "Cannot establish how long after coalescence a waveform ends (%s). Affected "
                    "events are claimed by every segment their coalescence precedes, so a run "
                    "starting later than its population's first event may batch events whose "
                    "signal it then crops.",
                    reason,
                )
            return None

    def _pre_coalescence_duration(self, parameters: Mapping[str, Any]) -> float | None:
        """Return the event's waveform lead in seconds, or ``None`` when it cannot be established.

        Args:
            parameters: The candidate event's source parameters.

        Returns:
            Seconds before ``coa_time`` at which the waveform starts, or ``None`` for unknown.
        """
        if self.signal_adapter is None:
            return None
        try:
            return self.signal_adapter.pre_coalescence_duration(
                parameters,
                sampling_frequency=float(self.sampling_frequency.value),
                minimum_frequency=self.minimum_frequency,
                waveform_arguments=self.waveform_arguments,
                waveform_options=self.waveform_options,
            )
        except Exception as exc:
            # Deliberately broad, and it does not hide the failure it swallows. This query decides
            # *placement*; the same parameters are handed to generation immediately afterwards, which
            # raises whatever this raised -- with the context of the event it was generating rather
            # than of a boundary test. Letting placement be the thing that fails a run would turn an
            # incomplete catalogue column into a crash in an unrelated part of the loop.
            #
            # The fallback is the previous behaviour, and its cost is reported rather than silent:
            # any inspiral cropped by it is named by `_report_content_before_segment`.
            #
            # Deduplicated by reason rather than by a single flag. A whole catalogue the query cannot
            # read fails identically for every row, and one message is right for it; but a single
            # malformed row fails only its own query, and a flag would let every later row with a
            # *different* problem fall back silently under a warning naming the first one.
            reason = f"{type(exc).__name__}: {exc}"
            if reason not in self._pre_coalescence_query_failures:
                self._pre_coalescence_query_failures.add(reason)
                logger.warning(
                    "Cannot establish how long before coalescence a waveform starts (%s). Affected "
                    "events are claimed by coa_time alone, which crops the start of any waveform "
                    "beginning before its segment; the amount lost is reported per signal.",
                    reason,
                )
            return None

    def _events_for_this_segment(self) -> tuple[list[int], list[dict[str, Any]]]:
        """Return the population events to generate for the current segment.

        Stops on :meth:`_event_starts_before_segment_end`, the same boundary the per-event loop
        applies. That matters for resume: ``population_index`` is checkpointed state, so a run
        switched between modes must not skip or repeat events.

        Events the walk *steps over* -- those :meth:`_event_ended_before_segment_start` reports as
        finished -- are consumed without being returned, so the count of events generated is not
        the distance the index must advance. The walk records that distance in
        ``_traversed_this_batch`` for :meth:`_commit_consumed_events`, which still does the
        advancing once generation has succeeded. The single exception is a batch in which
        *everything* was skipped: it advances here, because the caller returns early on an empty
        batch and never reaches the commit.

        One difference, stated rather than glossed: the per-event loop also breaks when a *generated*
        strain starts at or after ``end_time``, a condition that cannot be evaluated before
        generating. With the boundary now taken from the waveform's predicted start, reaching it
        requires the prediction to disagree with the buffer generation actually produces -- the two
        come from the same backend helpers, so the bundled waveforms do not reach it. This helper
        remains the weaker of the two rules where they could differ.

        Returns:
            The population indices and their parameter mappings, in consumption order -- which is
            :meth:`_placement_order`, so the indices need not be ascending.
        """
        event_ids: list[int] = []
        events: list[dict[str, Any]] = []

        # Read without advancing. `population_index` is checkpointed, and everything after this --
        # transposing to a struct-of-arrays, canonicalising, generating -- can still fail. Advancing
        # here would let a caller observe a consumed index after an exception, so the commit happens
        # in `_commit_consumed_events` once generation has succeeded. The CLI's retry wrapper
        # restores pre-batch state, so this is about direct library callers rather than the runner.
        order = self._placement_order()
        position = int(self.population_index)
        skipped = 0
        while position < len(order):
            event_id = int(order[position])
            parameters = dict(self._population_events[event_id])
            if self._event_ended_before_segment_start(parameters):
                skipped += 1
                position += 1
                continue
            if not self._event_starts_before_segment_end(parameters):
                break
            event_ids.append(event_id)
            events.append(parameters)
            position += 1

        if skipped:
            logger.info(
                "Skipped %d event(s) whose waveform ends before this segment starts; they belong "
                "to no segment this run writes.",
                skipped,
            )

        # How far the walk got, which is what consumption has to advance by -- not the number of
        # events generated. Skipped and claimed events interleave: consumption is ordered by
        # waveform *start* while skipping is decided by waveform *end*, so a long event starting
        # early and overlapping the segment can be followed by a short one that starts later and
        # already finished. Advancing by the skip count alone would leave the index pointing past a
        # claimed event that had not been generated yet, and a direct caller retrying after a
        # generation failure would never generate it -- silently dropping a real event, which is
        # worse than the wasted work this change removes.
        self._traversed_this_batch = position - int(self.population_index)

        # The all-skipped batch is the one case that must advance here. Nothing can fail for events
        # that are never generated, and `_simulate_batched_segment` returns early on an empty
        # batch, so the commit never runs -- the next segment would reconsider the same events, and
        # every segment after it too.
        if not event_ids and skipped:
            self.population_index = cast(int, self.population_index) + skipped
            self._traversed_this_batch = 0
        return event_ids, events

    def _commit_consumed_events(self, event_ids: list[int]) -> None:
        """Advance the checkpointed index past events whose generation succeeded.

        Advances by *count*, not to ``max(event_ids) + 1``: the ids are catalogue positions and
        consumption follows :meth:`_placement_order`, so they need not be contiguous or ascending.

        The count is how far the walk *traversed*, not how many events it generated. Those differ
        once events can be skipped: a skipped event sitting between two claimed ones is consumed
        without being generated, and advancing by the generated count alone would leave the index
        short, so the next segment would re-walk -- and re-generate -- events already written.
        ``_traversed_this_batch`` is set by :meth:`_events_for_this_segment` and falls back to the
        generated count for any caller that commits without having walked.

        Args:
            event_ids: Catalogue positions of the events this batch generated.
        """
        if event_ids:
            traversed = self._traversed_this_batch
            advance = len(event_ids) if traversed is None else traversed
            self.population_index = cast(int, self.population_index) + advance
        self._traversed_this_batch = None

    def _segment_injections(self) -> list[dict[str, Any]]:
        """Return one record per signal present in the segment just built.

        Not the same list as ``_batch_injections``, and the difference is the whole point of
        `gwmock/per-event-provenance-on-segments`: that list holds what this batch *generated*, while
        a segment also contains the continuing part of any signal generated earlier. A 1.6+1.4
        binary from 30 Hz runs about 48 s, so across 32 s segments it appears in three frames and
        only one of them generated it -- and the frame holding the merger, the loudest part, was not
        that one.

        Ordered generated-first so the common case reads as it did before, then the carried ones.
        Deduplicated on ``event_id``, because a multi-detector batch emits one chunk per detector
        carrying the same record.

        Returns:
            Injection records for this segment, generated and carried alike.
        """
        from gwmock.mixin.time_series import _merge_injection_records  # noqa: PLC0415

        return _merge_injection_records(self._batch_injections, getattr(self, "carried_injections", []))

    def _batched_waveform_backend(self) -> Any:
        """Return the ripple backend the batched path should generate with.

        The batched entry point is ripple-only, and it takes a backend *instance*. Without passing
        one, ``waveform-backend-arguments`` -- ripple's ``taper_fraction``, ``f_ref``,
        ``ringdown_fraction`` -- are silently discarded and the run uses ripple's defaults while the
        configuration says otherwise. That is the same silent-drop this codebase already fixed once
        for the per-event path.

        A configuration asking for a library the batched path cannot provide is refused rather than
        quietly served with ripple, because the output of the wrong library looks entirely normal.

        Returns:
            A configured ``RippleBackend``, or ``None`` to let gwmock-signal build its default.

        Raises:
            ValueError: If the configuration selects a library other than ripple. Settings the path
                cannot apply at all are refused earlier, by
                :func:`~gwmock.signal.execution_support.require_execution_supports_configuration`.
        """
        signal_config = self.orchestration_config.signal
        if signal_config is None:
            return None

        requested = getattr(signal_config, "waveform_backend", None)
        if requested is not None and resolve_backend_class("waveform", requested).__name__ != "RippleBackend":
            raise ValueError(
                f"execution: batched always generates with ripple, but waveform-backend is "
                f"{requested!r}. Substituting ripple would produce a different waveform than the "
                f"configuration asks for. Use execution: per-event, or waveform-backend: ripple."
            )

        from gwmock.signal.device_chunks import BATCHED_BACKEND_ARGUMENTS  # noqa: PLC0415

        arguments = _normalize_keys(dict(getattr(signal_config, "waveform_backend_arguments", {}) or {}))

        # `f_ref` is written under `waveform-arguments` because that is where the per-event path
        # takes it, but gwmock-signal treats it as a backend option -- its own reserved-argument
        # table says "f_ref is configured on the backend, not through waveform_arguments". Left in
        # the parameter mapping it would reach `simulate_cbc_batch` and do nothing, so the same
        # config key would set the reference frequency in one execution mode and be inert in the
        # other. Routed here instead, so the two modes agree.
        for name in sorted(BATCHED_BACKEND_ARGUMENTS & set(self.waveform_arguments)):
            from_arguments = self.waveform_arguments[name]
            if name in arguments and arguments[name] != from_arguments:
                raise ValueError(
                    f"{name} is set to {from_arguments!r} in waveform-arguments and "
                    f"{arguments[name]!r} in waveform-backend-arguments. One value has to win and "
                    f"picking silently would make the waveform depend on which of two keys a reader "
                    f"happened to look at. Set it in one place."
                )
            arguments[name] = from_arguments

        if not arguments:
            return None
        return instantiate_backend("waveform", "ripple", init_kwargs=arguments)

    def _simulate_batched_segment(self) -> TimeSeriesList:
        """Generate this segment's events together, through gwmock-signal's batched path.

        The device produces one buffer per event, aligned to this segment's sample lattice, and
        gwmock's own assembler places them. Chunks stay per-event so ``inject_from_list`` keeps
        handling spill-over into later segments and provenance stays per-injection -- the batched
        path is a different way to *generate*, not a different way to assemble.

        Returns:
            One chunk per event, ready for injection, or an empty list if the segment has no events.
        """
        from gwmock_signal import SamplingGrid, simulate_cbc_batch  # noqa: PLC0415

        from gwmock.signal.device_chunks import (  # noqa: PLC0415
            BATCHED_BACKEND_ARGUMENTS,
            batched_strain_to_chunks,
            canonicalise_parameters,
            require_batched_parameters_supported,
        )

        # The per-key waveform-arguments check is separate from the whole-config one run by
        # `_require_configuration_supported`: that field *is* honoured, but only for the canonical
        # parameters the batched entry point reads.
        waveform_backend = self._batched_waveform_backend()
        require_batched_parameters_supported(canonicalise_parameters(dict(self.waveform_arguments)))

        event_ids, events = self._events_for_this_segment()
        self._batch_injections = []
        if not events:
            return TimeSeriesList()
        # Backend options are excluded: they went to the constructor in `_batched_waveform_backend`,
        # and leaving them here too would hand `simulate_cbc_batch` a key it does not read.
        fixed_arguments = {
            key: value for key, value in self.waveform_arguments.items() if key not in BATCHED_BACKEND_ARGUMENTS
        }
        parameters = canonicalise_parameters({**fixed_arguments, **SignalAdapter.events_to_struct_of_arrays(events)})
        sampling_frequency = float(self.sampling_frequency.value)

        # The grid is this segment's own lattice, so every buffer starts on a sample of it and
        # injection is an integer-offset add rather than a resample.
        grid = SamplingGrid(float(getattr(self.start_time, "value", self.start_time)), sampling_frequency)

        batch = simulate_cbc_batch(
            self.signal_adapter.device_approximant(),
            list(self._signal_network.detector_names),
            sampling_frequency=sampling_frequency,
            minimum_frequency=self.minimum_frequency,
            parameters=parameters,
            backend=waveform_backend,
            earth_rotation=self.earth_rotation,
            output_grid=grid,
        )

        chunks = batched_strain_to_chunks(batch, expected_detector_names=tuple(self.detectors))

        # Generation succeeded, so these events are now genuinely consumed.
        self._commit_consumed_events(event_ids)
        # Provenance records what the *catalogue* said, not the canonicalised and merged mapping
        # handed to the device, so the two execution modes describe an injection the same way. The
        # per-event path records `dict(parameters)` straight from the population.
        self._batch_injections = [
            {"event_id": int(event_id), "parameters": dict(event)}
            for event_id, event in zip(event_ids, events, strict=True)
        ]
        for chunk, record in zip(chunks, self._batch_injections, strict=True):
            chunk.metadata.update({"injection_parameters": dict(record["parameters"]), "event_id": record["event_id"]})
        return chunks

    def _simulate_continuous_wave_segment(self) -> TimeSeriesList:
        """Sum every pulsar in the catalogue into one chunk spanning this segment.

        Every source contributes to every segment, so this walks the whole catalogue each time and
        leaves ``population_index`` alone -- advancing it would drop pulsars from later segments.
        Each call adds to the running total by passing the previous result back as the background,
        which is how the backend composes sources.

        Returns:
            A single chunk covering the segment, or an empty list if there are no pulsars.
        """
        if self.signal_adapter is None or not self._population_events:
            return TimeSeriesList()

        n_samples = round(float(self.duration.value) * float(self.sampling_frequency.value))
        epoch = float(self.start_time.value)
        sampling_frequency = float(self.sampling_frequency.value)
        total = {
            detector: GWpyTimeSeries(
                np.zeros(n_samples, dtype=float),
                t0=epoch,
                sample_rate=sampling_frequency,
                unit="strain",
            )
            for detector in self.signal_adapter.detector_names
        }

        strain = None
        self._batch_injections = []
        for source_id, source in enumerate(self._population_events):
            parameters = {**self.waveform_arguments, **dict(source)}
            strain = self.signal_adapter.simulate(
                parameters,
                sampling_frequency=sampling_frequency,
                minimum_frequency=self.minimum_frequency,
                waveform_options=self.waveform_options,
                background=total,
                earth_rotation=self.earth_rotation,
            )
            # Feed the running total back in, so the next pulsar adds to it rather than to zeros.
            total = {
                detector: GWpyTimeSeries(
                    np.asarray(strain[index], dtype=float),
                    t0=epoch,
                    sample_rate=sampling_frequency,
                    unit="strain",
                )
                for index, detector in enumerate(self.signal_adapter.detector_names)
            }
            # Recorded for every segment, not once: a continuous wave is present in all of them, so
            # attributing it to one frame the way a transient is attributed would be wrong.
            #
            # This used to be undone downstream: `update_signal_index` assigned rather than
            # merged, so the id fast path kept only the last frame written for each pulsar and
            # `gwmock find-signal --id N` named one frame where a continuous wave is in all of
            # them. The index now accumulates per batch, so appending here is what it looks like.
            self._batch_injections.append({"event_id": source_id, "parameters": dict(source)})

        if strain is None:  # pragma: no cover - guarded by the empty-catalogue check above
            return TimeSeriesList()
        strain.metadata.update({"continuous_wave_sources": len(self._population_events)})
        return TimeSeriesList([strain])

    def _simulate_stationary_signal_segment(self) -> TimeSeriesList:
        """Generate one stationary signal chunk spanning the active segment."""
        if self.signal_adapter is None:
            return TimeSeriesList()
        # A stationary (e.g. SGWB) segment has no discrete source events.
        self._batch_injections = []
        segment_seed = self._signal_segment_seed()
        self.signal_adapter.set_seed(segment_seed)
        n_samples = round(float(self.duration.value) * float(self.sampling_frequency.value))
        background = {
            detector: GWpyTimeSeries(
                np.zeros(n_samples, dtype=float),
                t0=float(self.start_time.value),
                sample_rate=float(self.sampling_frequency.value),
                unit="strain",
            )
            for detector in self.signal_adapter.detector_names
        }
        parameters = {**self.waveform_arguments, **self.signal_parameters}
        strain = self.signal_adapter.simulate(
            parameters,
            sampling_frequency=float(self.sampling_frequency.value),
            minimum_frequency=self.minimum_frequency,
            waveform_options=self.waveform_options,
            background=background,
            earth_rotation=self.earth_rotation,
        )
        strain.metadata.update({"signal_parameters": dict(parameters), "segment_seed": segment_seed})
        return TimeSeriesList([strain])

    def update_state(self) -> None:
        """Advance to the next segment."""
        self.noise_stream_committed_count = max(
            int(self.noise_stream_committed_count), int(self._noise_stream_position)
        )
        self.counter = cast(int, self.counter) + 1
        self.start_time += self.duration
        self._pending_noise_chunk = None
        # Cleared with the chunk they describe, so a batch that writes no noise cannot record the
        # previous batch's glitches as its own.
        self._batch_glitches = []

    def signal_output_directory(self) -> Path:
        """Return the active signal output directory."""
        return self._active_signal_output_directory

    def signal_output_arguments(self) -> dict[str, Any]:
        """Return the active signal output keyword arguments."""
        if self.orchestration_config.signal is None:
            return {}
        return dict(self.orchestration_config.signal.output.arguments or {})

    def _save_data(
        self,
        data: TimeSeries,
        file_name: str | Path | np.ndarray[Any, np.dtype[np.object_]],
        **kwargs,
    ) -> None:
        """Persist orchestration signal output through ``DetectorStrainStack.write``."""
        if not isinstance(data, TimeSeries):
            raise TypeError(f"AdapterOrchestrator can only save TimeSeries signal data, got {type(data)}.")

        if isinstance(file_name, list):
            file_name = np.asarray(file_name, dtype=object)

        channel_names = self._resolve_signal_channels(data=data, channel_spec=kwargs.pop("channel", None))
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise ValueError(
                f"Signal orchestration output arguments only support channel. Unsupported keys: {unsupported}."
            )

        if isinstance(file_name, (str, Path)):
            output_path = Path(file_name)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._build_signal_stack(data=data, channel_names=channel_names).write(
                output_path,
                format=self._infer_signal_output_format(output_path),
            )
            declare_strain_schema(output_path)
            return

        if len(file_name.shape) != 1 or file_name.shape[0] != data.num_of_channels:
            raise ValueError(
                "Resolved signal output paths must be a single path or a one-dimensional array "
                "matching the number of detector channels."
            )

        for index in range(data.num_of_channels):
            detector_name = self.detectors[index]
            output_path = Path(file_name[index])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._build_signal_stack(
                data=data,
                detector_names=[detector_name],
                channel_names=channel_names[index : index + 1],
                indices=[index],
            ).write(
                output_path,
                format=self._infer_signal_output_format(output_path),
            )
            declare_strain_schema(output_path)

    def _run_noise_batch(self) -> SimulationResult:
        # Source the file_name from the active per-batch context so that reproduction from
        # metadata (where each batch carries its own already-expanded literal file_name) writes
        # to the correct per-batch path instead of reusing the first batch's filenames. Falls
        # back to the instantiation-time config when no batch context has been set.
        file_name = (
            self._active_noise_file_name
            if self._active_noise_file_name is not None
            else self.orchestration_config.noise.output.file_name
        )
        output_format = self._infer_noise_output_format(file_name)
        output_paths = self._expand_noise_output_paths(file_name)
        output_arguments = dict(self._active_noise_output_arguments)
        if "channel_prefix" in output_arguments:
            raise ValueError(
                "Noise output argument 'channel_prefix' is no longer supported. "
                "Rename it to 'channel' (e.g. channel: MOCK_NOISE)."
            )
        channel_raw = output_arguments.pop("channel", "MOCK_NOISE")
        gps_start = float(output_arguments.pop("gps_start", float(self.start_time.value)))
        if output_arguments:
            unsupported = ", ".join(sorted(output_arguments))
            raise ValueError(
                "Noise orchestration output arguments only support channel and gps_start. "
                f"Unsupported keys: {unsupported}."
            )

        noise_detectors = list(self.noise_arguments["detectors"])
        if isinstance(channel_raw, list):
            if len(channel_raw) != len(noise_detectors):
                raise ValueError(
                    f"Noise channel list expanded to {len(channel_raw)} entries "
                    f"but there are {len(noise_detectors)} noise detectors."
                )
            channels_dict: dict[str, str] | None = dict(
                zip(noise_detectors, [str(c) for c in channel_raw], strict=True)
            )
            first = str(channel_raw[0])
            channel_str = first.split(":", 1)[1] if ":" in first else first
        else:
            channel_str = str(channel_raw)
            channels_dict = None

        if not self._active_overwrite:
            existing = [path for path in dict.fromkeys(output_paths) if path.exists()]
            if existing:
                raise FileExistsError(
                    f"Noise adapter output(s) already exist: {', '.join(str(path) for path in existing)}. "
                    "Use overwrite=True to overwrite them."
                )

        # The batch's epoch reaches the injector before the chunk is drawn, so each glitch is
        # recorded against the GPS time of the frame it lands in rather than a time the injector
        # accumulated for itself. It has to be after `_ensure_noise_stream` has realigned the
        # stream -- a resumed run replays the chunks before this one, and each replay advances the
        # injector's epoch -- which is why `_next_noise_chunk` takes it rather than this function
        # setting it here.
        chunk = self._next_noise_chunk(gps_start=gps_start)
        output_paths_by_detector: dict[str, Path] = {}

        for i, detector in enumerate(noise_detectors):
            output_path = output_paths[i] if len(output_paths) > 1 else output_paths[0]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_format in {"gwf", "hdf5"}:
                channel_id = channels_dict[detector] if channels_dict is not None else f"{detector}:{channel_str}"
                gwpy_ts = GWpyTimeSeries(
                    np.asarray(chunk[detector], dtype=float),
                    t0=gps_start,
                    sample_rate=float(self.sampling_frequency.value),
                )
                DetectorStrainStack.from_mapping([channel_id], {channel_id: gwpy_ts}).write(
                    output_path, format=output_format
                )
            else:
                np.save(output_path, chunk[detector])
            # Every strain artifact a run writes declares the contract it meets, whichever of the three
            # writers above produced it; the call is a no-op for the formats with no attribute space.
            declare_strain_schema(output_path)
            output_paths_by_detector[detector] = output_path

        self.noise_stream_committed_count = max(
            int(self.noise_stream_committed_count), int(self._noise_stream_position)
        )
        config = self.noise_adapter.build_config(
            detectors=noise_detectors,
            duration=float(self.duration.value),
            sampling_frequency=float(self.sampling_frequency.value),
            output_directory=self._active_noise_output_directory,
            output_prefix="",
            output_format=output_format,
            gps_start=gps_start,
            channel=channel_str,
            channels=channels_dict,
            seed=self._noise_stream_seed(),
            psd_file=self.noise_arguments.get("psd_file"),
            psd_schedule=self.noise_arguments.get("psd_schedule"),
            psd_files=self.noise_arguments.get("psd_files"),
            csd_files=self.noise_arguments.get("csd_files"),
            low_frequency_cutoff=self.noise_arguments.get("low_frequency_cutoff", 2.0),
            high_frequency_cutoff=self.noise_arguments.get("high_frequency_cutoff"),
            spectral_lines=self.noise_arguments.get("spectral_lines"),
            glitches=self.noise_arguments.get("glitches"),
        )
        return SimulationResult(output_paths=output_paths_by_detector, config=config)

    def segment_seeds(self) -> list[int]:
        """Return the deterministic per-segment seeds derived locally by gwmock."""
        return [seed for seed in (self._signal_segment_seed(),) if seed is not None]

    def _root_seed(self) -> int | None:
        base_seed = self.noise_arguments.get("seed")
        if base_seed is None:
            return None
        return int(base_seed)

    def _signal_segment_seed(self) -> int | None:
        root_seed = self._root_seed()
        if root_seed is None:
            return None
        return derive_seed(root_seed, "signal", int(self.counter))

    def _noise_stream_seed(self) -> int | None:
        root_seed = self._root_seed()
        if root_seed is None:
            return None
        return derive_seed(root_seed, "noise", "stream")

    @staticmethod
    def _infer_signal_output_format(path: Path) -> Literal["gwf", "hdf5", "npy", "txt"]:
        suffix = path.suffix.lower().lstrip(".")
        if suffix == "h5":
            suffix = "hdf5"
        if suffix not in {"gwf", "hdf5", "npy", "txt"}:
            raise ValueError("Signal output files must end with .gwf, .hdf5, .h5, .npy, or .txt.")
        return cast(Literal["gwf", "hdf5", "npy", "txt"], suffix)

    def _infer_noise_output_format(self, file_name_template: str | list[str]) -> Literal["npy", "gwf", "hdf5"]:
        """Return the noise output format named by the file_name template's extension.

        `.h5` folds into `hdf5` here as it does on the signal side, so the two halves of one run answer
        the same question the same way: a template is not a different format for being spelled with the
        shorter extension.

        Args:
            file_name_template: The configured noise file_name, one template or one per detector.

        Returns:
            The format the extension names.

        Raises:
            ValueError: If the template list is empty, names an unsupported extension, or -- given one
                template per detector -- does not name a single format for all of them.
        """
        names = list(file_name_template) if isinstance(file_name_template, list) else [file_name_template]
        if not names:
            raise ValueError("Noise file_name list must contain at least one entry.")
        suffixes = {"hdf5" if (s := Path(name).suffix.lower().lstrip(".")) == "h5" else s for name in names}
        if not suffixes <= {"npy", "gwf", "hdf5"}:
            raise ValueError("Noise output templates must end with .npy, .gwf, .hdf5, or .h5.")
        if len(suffixes) != 1:
            raise ValueError("All noise file_name entries must use the same format (.npy, .gwf, or .hdf5/.h5).")
        return cast(Literal["npy", "gwf", "hdf5"], suffixes.pop())

    def _expand_noise_output_paths(self, file_name_template: str | list[str]) -> list[Path]:
        """Expand the noise file_name template to one output path per detector."""
        noise_detectors = list(self.noise_arguments["detectors"])
        if isinstance(file_name_template, list):
            if len(file_name_template) != len(noise_detectors):
                raise ValueError(
                    f"Noise file_name list has {len(file_name_template)} entries "
                    f"but there are {len(noise_detectors)} noise detectors."
                )
            paths = [self._active_noise_output_directory / str(p) for p in file_name_template]
        else:
            proxy = _NoiseTemplateProxy(self, noise_detectors)
            expanded = expand_template_variables(file_name_template, proxy)
            if isinstance(expanded, list):
                if len(expanded) != len(noise_detectors):
                    raise ValueError(
                        f"Noise file_name template expanded to {len(expanded)} paths "
                        f"but there are {len(noise_detectors)} noise detectors."
                    )
                paths = [self._active_noise_output_directory / str(p) for p in expanded]
            else:
                paths = [self._active_noise_output_directory / str(expanded)] * len(noise_detectors)
        if len(set(paths)) < len(paths):
            raise ValueError(
                "Noise file_name template produces identical paths for multiple detectors. "
                "Include {{ detectors }} in the template to generate per-detector paths."
            )
        return paths

    @staticmethod
    def _resolve_output_directory(
        base_output_directory: Path, configured_directory: str | None, fallback_subdir: str
    ) -> Path:
        if configured_directory is None:
            return base_output_directory / fallback_subdir
        configured_path = Path(configured_directory)
        return configured_path if configured_path.is_absolute() else base_output_directory / configured_path

    def _next_noise_chunk(self, gps_start: float | None = None) -> dict[str, Any]:
        """Return the chunk for the current batch, reusing it across retries.

        Args:
            gps_start: GPS time of the chunk's first sample, handed to the glitch injector so
                its truth catalogue is timed against the frame this batch writes. Passed here
                rather than set by the caller because the stream may first be realigned to this
                batch by replaying earlier chunks, each of which advances the injector's own
                epoch. Omitted by callers that only need the samples.

        Returns:
            The chunk, one array per detector.
        """
        if self._pending_noise_chunk is not None:
            # Already drawn for this batch, so its rows are already recorded and its epoch was
            # applied when it was drawn.
            return self._pending_noise_chunk

        self._ensure_noise_stream()
        if self._noise_stream is None:
            raise RuntimeError("Noise stream was not initialized.")
        if gps_start is not None and self.noise_adapter is not None:
            self.noise_adapter.set_segment_gps_start(gps_start)
        try:
            self._pending_noise_chunk = next(self._noise_stream)
        except StopIteration as error:
            raise ValueError("Noise stream ended before all orchestration batches were generated.") from error
        self._noise_stream_position += 1
        self._batch_glitches = [] if self.noise_adapter is None else self.noise_adapter.segment_glitch_events()
        return self._pending_noise_chunk

    def _ensure_noise_stream(self) -> None:
        """Open or realign the shared upstream noise stream to the current batch index."""
        target_position = max(int(self.counter), int(self.noise_stream_committed_count))
        if self._noise_stream is not None and self._noise_stream_position == target_position:
            return

        if int(self.counter) > 0 and self._root_seed() is None:
            raise ValueError(
                "Cannot resume an unseeded noise stream from a non-zero batch index; "
                "the upstream stream is non-deterministic without a seed."
            )
        self._pending_noise_chunk = None

        self._noise_stream = self.noise_adapter.open_stream(
            chunk_duration=float(self.duration.value),
            sampling_frequency=float(self.sampling_frequency.value),
            detectors=list(self.noise_arguments["detectors"]),
            seed=self._noise_stream_seed(),
            psd_file=self.noise_arguments.get("psd_file"),
            psd_schedule=self.noise_arguments.get("psd_schedule"),
            psd_files=self.noise_arguments.get("psd_files"),
            csd_files=self.noise_arguments.get("csd_files"),
            low_frequency_cutoff=self.noise_arguments.get("low_frequency_cutoff", 2.0),
            high_frequency_cutoff=self.noise_arguments.get("high_frequency_cutoff"),
            spectral_lines=self.noise_arguments.get("spectral_lines"),
            glitches=self.noise_arguments.get("glitches"),
        )
        self._noise_stream_position = 0
        for _ in range(target_position):
            try:
                next(self._noise_stream)
            except StopIteration as error:
                raise ValueError(
                    "Noise stream ended before the saved orchestration state could be restored."
                ) from error
            self._noise_stream_position += 1

    @classmethod
    def _resolve_detector_network(cls, detector_specs: Sequence[str]) -> tuple[Network, dict[str, Any]]:
        resolved_detectors: list[str | Any] = []
        resolution_steps: list[dict[str, Any]] = []
        for detector_spec in detector_specs:
            resolved, resolution_step = SignalAdapter.resolve_detector_spec(str(detector_spec))
            resolved_detectors.extend(resolved)
            resolution_steps.append(resolution_step)

        network = Network.from_detectors(tuple(resolved_detectors))
        return network, {
            "inputs": [str(detector_spec) for detector_spec in detector_specs],
            "detector_names": cls._network_detector_names(network),
            "steps": resolution_steps,
        }

    @staticmethod
    def _network_detector_names(network: Network) -> list[str]:
        return [detector if isinstance(detector, str) else detector.name for detector in network.detector_names]

    def _resolve_signal_channels(self, *, data: TimeSeries, channel_spec: Any) -> list[str | None]:
        if channel_spec is None:
            return [None] * data.num_of_channels

        channel_value = expand_template_variables(channel_spec, self)
        if isinstance(channel_value, str):
            return [channel_value] * data.num_of_channels

        channel_names = [str(channel) for channel in list(channel_value)]
        if len(channel_names) != data.num_of_channels:
            raise ValueError("Length of channel list must match number of channels in data.")
        return channel_names

    def _build_signal_stack(
        self,
        *,
        data: TimeSeries,
        channel_names: list[str | None],
        detector_names: list[str] | None = None,
        indices: list[int] | None = None,
    ) -> DetectorStrainStack:
        active_detector_names = self.detectors if detector_names is None else detector_names
        active_indices = list(range(data.num_of_channels)) if indices is None else indices
        if len(active_detector_names) != len(active_indices) or len(channel_names) != len(active_indices):
            raise ValueError("Signal detector, channel, and data selections must have matching lengths.")

        mapping = {}
        effective_names = []
        for detector_name, channel_name, index in zip(
            active_detector_names, channel_names, active_indices, strict=True
        ):
            series = data[index].copy()
            effective_name = channel_name if channel_name is not None else detector_name
            if effective_name in mapping:
                effective_name = f"{effective_name}__{detector_name}"
            effective_names.append(effective_name)
            mapping[effective_name] = series
        return DetectorStrainStack.from_mapping(effective_names, mapping)
