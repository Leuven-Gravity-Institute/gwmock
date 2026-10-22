"""Adapter from gwmock orchestration to public ``gwmock_signal`` APIs."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from gwmock_signal import CustomDetector, DetectorStrainStack, Network, resolve_simulator_backend

from gwmock.data.time_series.time_series import TimeSeries
from gwmock.signal.population_columns import require_population_columns_read, require_zero_only_columns_zero
from gwmock.utils.errors import DeterministicRefusalError

logger = logging.getLogger("gwmock")

#: gwmock-signal raises ``ValueError("Unsupported <backend> waveform parameters: a, b")``
#: when a backend rejects parameters it does not recognise. The backend token varies
#: (``LAL``, ``ripple``, ``PyCBC``), so match any of them and capture the comma-separated
#: parameter list that follows.
_UNSUPPORTED_PARAMS_RE = re.compile(r"^Unsupported (.+?) waveform parameters:\s*(.*)$")

#: Canonical parameter names the on-device batched path requires, with no aliases accepted.
#:
#: The per-event path forwards whatever it is given and lets the backend resolve aliases
#: (``mass1`` for ``detector_frame_mass_1``, and so on). The batched path does not: it reads these
#: names directly from a struct-of-arrays, so a missing or aliased key would surface as a
#: ``KeyError`` from inside a jitted kernel rather than as a statement about the input.
_DEVICE_REQUIRED_PARAMETERS = (
    "coa_time",
    "declination",
    "detector_frame_mass_1",
    "detector_frame_mass_2",
    "polarization_angle",
    "right_ascension",
)

_DEFAULT_WAVEFORM_MODEL = "IMRPhenomXPHM"
_LEGACY_SINGLE_DETECTOR_ALIASES = {
    "E1_triangle_sardinia": ("ET-Triangle-Sardinia", "ET1_SARD"),
    "E2_triangle_sardinia": ("ET-Triangle-Sardinia", "ET2_SARD"),
    "E3_triangle_sardinia": ("ET-Triangle-Sardinia", "ET3_SARD"),
    "E1_triangle_emr": ("ET-Triangle-EMR", "ET1_EMR"),
    "E2_triangle_emr": ("ET-Triangle-EMR", "ET2_EMR"),
    "E3_triangle_emr": ("ET-Triangle-EMR", "ET3_EMR"),
    "E1_2L_aligned_sardinia": ("ET-2L-Aligned", "ET1_2L_ALIGNED_SARD"),
    "E2_2L_aligned_emr": ("ET-2L-Aligned", "ET2_2L_ALIGNED_EMR"),
    "E1_2L_misaligned_sardinia": ("ET-2L-Misaligned", "ET1_2L_MISALIGNED_SARD"),
    "E2_2L_misaligned_emr": ("ET-2L-Misaligned", "ET2_2L_MISALIGNED_EMR"),
}
DetectorSpec = str | CustomDetector


#: Marks a registry name generated for a *callable* waveform model rather than a real approximant.
#: The device path tests for it to reject callables with an explanation, so the prefix has one
#: definition rather than being spelled out at both the producing and the consuming end.
_CALLABLE_WAVEFORM_PREFIX = "__gwmock_custom__"


def _callable_waveform_registry_key(func: Callable[..., Any]) -> str:
    """Return a unique registry name for *func* on a ``WaveformFactory`` instance."""
    qual = getattr(func, "__qualname__", type(func).__name__)
    mod = getattr(func, "__module__", "")
    return f"{_CALLABLE_WAVEFORM_PREFIX}{mod}:{qual}__{id(func):#x}"


def _register_callable_waveform(backend: Any, registry_key: str, factory: Callable[..., Any]) -> None:
    """Register *factory* under *registry_key* through the public backend API."""
    backend.register_waveform_model(registry_key, factory)


def _resolve_detector_path(detector_spec: str) -> Path | None:
    """Resolve a detector spec to an on-disk network file when one exists."""
    detector_path = Path(detector_spec)
    if detector_path.is_file():
        return detector_path

    return None


def _network_detector_names(detector_specs: Sequence[DetectorSpec]) -> list[str]:
    """Normalize detector specs to their output detector-name strings."""
    return [detector if isinstance(detector, str) else detector.name for detector in detector_specs]


@lru_cache(maxsize=1)
def _single_detector_catalog() -> dict[str, CustomDetector]:
    """Build a catalog of public single-detector aliases backed by public presets."""
    catalog: dict[str, CustomDetector] = {}
    for preset_name in Network.list_names():
        try:
            detector_specs = Network.from_preset(preset_name).detector_names
        except ValueError:
            continue
        for detector in detector_specs:
            if isinstance(detector, str):
                continue
            catalog.setdefault(detector.name, detector)
            catalog.setdefault(detector.name.lower(), detector)

    for alias, (preset_name, detector_name) in _LEGACY_SINGLE_DETECTOR_ALIASES.items():
        detector = next(
            spec
            for spec in Network.from_preset(preset_name).detector_names
            if not isinstance(spec, str) and spec.name == detector_name
        )
        catalog[alias] = detector

    return catalog


def _resolve_single_detector_alias(detector_spec: str) -> tuple[CustomDetector, ...] | None:
    """Resolve one public or legacy single-detector alias via public preset geometry."""
    detector = _single_detector_catalog().get(detector_spec) or _single_detector_catalog().get(detector_spec.lower())
    if detector is None:
        return None
    return (detector,)


def _resolve_detector_spec(detector_spec: str) -> tuple[tuple[DetectorSpec, ...], dict[str, Any]]:
    """Resolve one detector spec and return both the detector specs and metadata."""
    detector_alias = str(detector_spec)
    detector_path = _resolve_detector_path(detector_alias)
    if detector_path is not None:
        resolved = tuple(Network.from_file(detector_path).detector_names)
        return resolved, {
            "input": detector_alias,
            "resolver": "file",
            "source": str(detector_path),
            "detector_names": _network_detector_names(resolved),
        }

    try:
        resolved = tuple(Network.from_preset(detector_alias).detector_names)
        return resolved, {
            "input": detector_alias,
            "resolver": "preset",
            "detector_names": _network_detector_names(resolved),
        }
    except ValueError:
        pass

    try:
        resolved = tuple(Network.from_name(detector_alias).detector_names)
        return resolved, {
            "input": detector_alias,
            "resolver": "name",
            "detector_names": _network_detector_names(resolved),
        }
    except ValueError:
        pass

    resolved_single = _resolve_single_detector_alias(detector_alias)
    if resolved_single is not None:
        return resolved_single, {
            "input": detector_alias,
            "resolver": "preset-detector",
            "detector_names": _network_detector_names(resolved_single),
        }

    return (detector_alias,), {
        "input": detector_alias,
        "resolver": "detector",
        "detector_names": [detector_alias],
    }


def _refuse_configured_rejections(
    rejected: Collection[str],
    waveform_arguments: Mapping[str, Any] | None,
    *,
    cause: BaseException | None = None,
) -> None:
    """Raise if the waveform backend rejects a key the configuration set under ``waveform-arguments``.

    Args:
        rejected: Parameter names the backend refused.
        waveform_arguments: The configured fixed waveform arguments.
        cause: The backend's own error, chained so its message stays visible.

    Raises:
        ValueError: If any rejected name is a configured waveform argument.
    """
    configured = sorted(set(rejected) & set(waveform_arguments or {}))
    if not configured:
        return
    message = (
        f"The waveform backend does not accept {', '.join(configured)}, given under "
        "signal.waveform-arguments. The setting would not be applied, so the run would generate a "
        "different waveform than the one configured. Remove it, or -- if it configures the backend "
        "itself, as f_ref does for LAL -- move it to signal.waveform-backend-arguments."
    )
    if cause is not None:
        raise DeterministicRefusalError(f"{message} Backend error: {cause}") from cause
    raise DeterministicRefusalError(message)


class SignalAdapter:
    """Bridge gwmock population/orchestration state to public gwmock-signal APIs."""

    def __init__(
        self,
        *,
        source_type: str,
        backend: Any,
        network: Network,
    ) -> None:
        """Store the resolved backend and detector network.

        Args:
            source_type: The source type to use for the backend.
            backend: The backend to use.
            network: The network to use.
        """
        self._source_type = source_type
        self._backend = backend
        self._network = network
        self._detector_names = tuple(
            detector if isinstance(detector, str) else detector.name for detector in self._network.detector_names
        )
        self._unsupported_params: set[str] = set()
        #: Population columns the configuration declares are not waveform inputs, left out quietly.
        self._ignored_population_columns: frozenset[str] = frozenset()

    def ignore_population_columns(self, names: Collection[str]) -> None:
        """Declare population columns that are not waveform inputs, to be left out without refusal.

        Args:
            names: Column names, as ``signal.ignored-population-columns`` lists them.
        """
        self._ignored_population_columns = frozenset(names)

    @classmethod
    def from_source_type(
        cls,
        *,
        source_type: str,
        waveform_model: str | Callable[..., Any] | None,
        backend_arguments: Mapping[str, Any] | None = None,
        detectors: Sequence[str] | None = None,
        network: Network | None = None,
    ) -> SignalAdapter:
        """Resolve the public gwmock-signal backend for one source type.

        Args:
            source_type: The source type to use for the backend.
            waveform_model: The waveform model to use.
            backend_arguments: Constructor arguments for the backend.
            detectors: The detectors to use.
            network: The network to use.

        Returns:
            A SignalAdapter instance.

        Raises:
            ValueError: If detectors is not a non-empty sequence.
        """
        backend_class = resolve_simulator_backend(source_type)
        backend = cls.instantiate_backend(
            backend_class,
            waveform_model=waveform_model,
            backend_arguments=backend_arguments,
        )
        return cls(
            source_type=source_type,
            backend=backend,
            network=cls._require_network(detectors=detectors, network=network),
        )

    @classmethod
    def from_backend(
        cls,
        *,
        source_type: str,
        backend: Any,
        detectors: Sequence[str] | None = None,
        network: Network | None = None,
    ) -> SignalAdapter:
        """Build an adapter from an already-instantiated backend.

        Args:
            source_type: The source type to use for the backend.
            backend: The backend to use.
            detectors: The detectors to use.
            network: The network to use.

        Returns:
            A SignalAdapter instance.
        """
        return cls(
            source_type=source_type,
            backend=backend,
            network=cls._require_network(detectors=detectors, network=network),
        )

    @staticmethod
    def instantiate_backend(
        backend_class: type[Any],
        *,
        waveform_model: str | Callable[..., Any] | None,
        backend_arguments: Mapping[str, Any] | None = None,
    ) -> Any:
        """Instantiate a signal backend class while preserving callable waveform support.

        Args:
            backend_class: The backend class to instantiate.
            waveform_model: The waveform model to use.
            backend_arguments: Constructor arguments for the backend.

        Returns:
            An instantiated backend.
        """
        init_kwargs = dict(backend_arguments or {})
        if waveform_model is None:
            try:
                return backend_class(waveform_model=_DEFAULT_WAVEFORM_MODEL, **init_kwargs)
            except TypeError:
                return backend_class(**init_kwargs)

        if callable(waveform_model):
            registry_key = _callable_waveform_registry_key(waveform_model)
            backend = backend_class(waveform_model=registry_key, **init_kwargs)
            _register_callable_waveform(backend, registry_key, waveform_model)
            return backend

        return backend_class(waveform_model=waveform_model, **init_kwargs)

    @classmethod
    def _require_network(cls, *, detectors: Sequence[str] | None, network: Network | None) -> Network:
        if network is not None:
            return network
        if not detectors:
            raise ValueError("detectors must be a non-empty sequence.")
        return cls.resolve_detector_network(detectors)

    @staticmethod
    def resolve_detector_path(detector_spec: str) -> Path | None:
        """Resolve a detector spec to an on-disk network file when one exists.

        Args:
            detector_spec: The detector spec to resolve.

        Returns:
            The resolved detector path or None if not found.
        """
        return _resolve_detector_path(detector_spec)

    @staticmethod
    def resolve_detector_spec(detector_spec: str) -> tuple[tuple[DetectorSpec, ...], dict[str, Any]]:
        """Resolve one detector spec into public detector objects or built-in detector names.

        Args:
            detector_spec: The detector spec to resolve.

        Returns:
            A tuple of the resolved detector specs and metadata.
        """
        return _resolve_detector_spec(detector_spec)

    @staticmethod
    def resolve_detector_network(detector_specs: Sequence[str]) -> Network:
        """Resolve detector specs into one public ``Network`` instance.

        Args:
            detector_specs: The detector specs to resolve.

        Returns:
            A resolved detector network.
        """
        resolved_detectors: list[DetectorSpec] = []
        for detector_spec in detector_specs:
            resolved, _ = _resolve_detector_spec(str(detector_spec))
            resolved_detectors.extend(resolved)

        return Network.from_detectors(tuple(resolved_detectors))

    @property
    def source_type(self) -> str:
        """Return the source-type routing key used for backend resolution.

        Returns:
            The source-type routing key used for backend resolution.
        """
        return self._source_type

    @property
    def detector_names(self) -> tuple[str, ...]:
        """Return the ordered detector names used for output stacking.

        Returns:
            The ordered detector names used for output stacking.
        """
        return self._detector_names

    @property
    def network(self) -> Network:
        """Return the resolved public detector network.

        Returns:
            The resolved public detector network.
        """
        return self._network

    @staticmethod
    def events_to_struct_of_arrays(
        events: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Transpose a sequence of per-event parameter mappings into arrays.

        The orchestration layer holds one mapping per event, while the batched device path takes one
        array per parameter. Every event must therefore carry the same keys.

        A ragged catalogue is rejected rather than reduced to the shared keys. Dropping a key that
        only some events carry does not avoid fabricating physics, it fabricates it wholesale: if one
        event omits ``spin_1z``, intersecting removes the column entirely and the backend default is
        then applied to *every* event, silently. A loader that emits ragged events is malformed, and
        saying so is the only outcome that does not quietly alter the simulation.

        Args:
            events: Per-event parameter mappings, as the population loader yields them.

        Returns:
            One entry per key, each a list of that key's value across events, in order.

        Raises:
            ValueError: If *events* is empty, or if the events do not all carry the same keys.
        """
        if not events:
            raise ValueError("events must be non-empty to build a batch.")
        expected = set(events[0])
        for index, event in enumerate(events[1:], start=1):
            if set(event) != expected:
                absent = sorted(expected - set(event))
                extra = sorted(set(event) - expected)
                raise ValueError(
                    f"every event must carry the same parameters, but event {index} differs from the "
                    f"first: missing {absent or 'nothing'}, unexpected {extra or 'nothing'}."
                )
        return {key: [event[key] for event in events] for key in sorted(expected)}

    def simulate_segments(
        self,
        parameters: Mapping[str, Any],
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        segment_duration: float,
        start_time: float,
        end_time: float,
        waveform_arguments: Mapping[str, Any] | None = None,
        earth_rotation: bool = True,
        n_chirp_mass_bins: int = 1,
        chunk_size: int | None = None,
    ) -> list[DetectorStrainStack]:
        """Generate a whole catalogue on device and return it as fixed-duration segments.

        The device counterpart of :meth:`simulate_stack`. That method takes one event and returns one
        stack; this takes the catalogue as a struct-of-arrays and returns the assembled segments, so
        the per-event Python loop disappears and the waveforms are generated under one ``vmap``.

        Kept as a separate entry point rather than folded into :meth:`simulate_stack`, because the
        per-event path must keep working unchanged for the LAL and PyCBC backends, which have no
        batched form.

        .. warning::

           **This path always generates with Ripple**, whatever backend the adapter was configured
           with, because Ripple is the only JAX implementation and therefore the only batchable one.
           The approximant carries over, but its implementation does not: an adapter configured for
           LAL that runs here gets Ripple's version of the same approximant, which agrees closely but
           not bit-for-bit. To keep that from being silent, an approximant Ripple does not implement
           is rejected rather than substituted -- see :meth:`_device_approximant`.

        Args:
            parameters: Canonical catalogue parameters as a struct-of-arrays — one array per
                parameter, aligned across parameters. See :meth:`events_to_struct_of_arrays`.
            sampling_frequency: Sample rate in Hz.
            minimum_frequency: Low-frequency cutoff in Hz. Note that with a tapered cutoff this is
                where the waveform reaches full amplitude, not where its content begins.
            segment_duration: Duration of each output segment in seconds.
            start_time: GPS start of the first segment.
            end_time: GPS end of the span to tile.
            waveform_arguments: Fixed parameters merged into the catalogue; per-event values win.
            earth_rotation: Whether to include earth rotation, as on :meth:`simulate_stack`.
            n_chirp_mass_bins: Generate heavier events on shorter grids. Bounds buffer length at the
                cost of exact reproducibility against a single grid.
            chunk_size: Events per batched call. ``None`` lets gwmock-signal size it from device
                memory, which is the safer default.

        Returns:
            One :class:`~gwmock_signal.DetectorStrainStack` per segment, in time order.

        Raises:
            ValueError: If a required canonical parameter is missing, or the arrays disagree in
                length.
            RuntimeError: If the installed gwmock-signal does not export the device path.
        """
        try:
            # Imported here, not at module scope: an older gwmock-signal without this export would
            # otherwise break `import gwmock.signal.adapter` outright, taking the per-event path down
            # with it. Locally, only the device path fails, and it says why.
            from gwmock_signal import simulate_cbc_catalogue  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on the installed version
            raise RuntimeError(
                "The installed gwmock-signal does not export simulate_cbc_catalogue, which the "
                "device path needs. Upgrade gwmock-signal, and install it with the [jax] extra."
            ) from exc

        merged = {**(waveform_arguments or {}), **dict(parameters)}
        missing = [name for name in _DEVICE_REQUIRED_PARAMETERS if name not in merged]
        if missing:
            raise ValueError(
                f"The device path needs these canonical parameters, which are absent: "
                f"{', '.join(missing)}. Unlike the per-event path it does not accept aliases, so "
                f"e.g. 'mass1' must be given as 'detector_frame_mass_1'."
            )
        # The required parameters are per-event by definition, so each must be a column. A scalar is
        # rejected rather than broadcast: it means every event shares one value, and for `coa_time`
        # that is every signal landing at the same instant. Left unchecked it surfaces from inside
        # batching as "too many indices for array: array is 0-dimensional".
        #
        # Only the required ones. A scalar elsewhere is the documented way to fix a parameter across
        # the catalogue -- `f_ref=20.0` via waveform_arguments -- so scalars stay legal in general.
        scalars = [name for name in _DEVICE_REQUIRED_PARAMETERS if np.ndim(merged[name]) == 0]
        if scalars:
            raise ValueError(
                f"These parameters vary per event and must be given as arrays, but were scalars: "
                f"{', '.join(scalars)}. To hold one fixed across the catalogue, repeat it per event."
            )

        # Sized by np.ndim rather than isinstance, so NumPy and JAX columns are checked too. Testing
        # for list/tuple alone let an ndarray of the wrong length through to fail inside batching,
        # where the message no longer names the parameter.
        lengths = {name: len(values) for name, values in merged.items() if np.ndim(values) > 0}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"catalogue parameters disagree in length: {lengths}")

        return simulate_cbc_catalogue(
            self._device_approximant(),
            list(self._network.detector_names),
            sampling_frequency=sampling_frequency,
            minimum_frequency=minimum_frequency,
            parameters=merged,
            segment_duration=segment_duration,
            start_time=start_time,
            end_time=end_time,
            earth_rotation=earth_rotation,
            n_chirp_mass_bins=n_chirp_mass_bins,
            chunk_size=chunk_size,
        )

    def device_approximant(self) -> str:
        """Return the approximant the batched path will generate, or explain why there is not one.

        Public because the orchestrator needs it to drive the batched path from another module;
        the logic lives in :meth:`_device_approximant`.
        """
        return self._device_approximant()

    def _device_approximant(self) -> str:
        """Return the approximant to generate on device, or explain why there is not one.

        The batched entry point takes an approximant *name*, while the per-event path goes through
        the backend's own waveform registry, so the name has to be read back off the backend. That is
        the public ``waveform_model`` property, which is the contract every simulator in
        gwmock-signal exposes -- deliberately not a search across candidate attributes, since a
        backend that grew a similarly-named attribute for another purpose would then be picked up
        silently.

        Two things cannot cross to the device path, and both are rejected here rather than left to
        fail further in:

        * A **callable** waveform model, which the per-event path supports by registering it under a
          generated key. Ripple cannot execute arbitrary Python, and the key would otherwise be
          handed over as if it were an approximant name.
        * An approximant **Ripple does not implement**. Substituting the nearest one would change the
          waveform without saying so, which is the failure this whole method exists to prevent.

        Returns:
            The approximant name, known to be one Ripple implements.

        Raises:
            RuntimeError: If the backend exposes no usable approximant name.
            ValueError: If the model is a callable, or Ripple has no such approximant.
        """
        name = getattr(self._backend, "waveform_model", None)
        if not isinstance(name, str) or not name:
            raise RuntimeError(
                f"The signal backend {type(self._backend).__name__} exposes no 'waveform_model' "
                f"string, which the device path needs to select an approximant."
            )
        if name.startswith(_CALLABLE_WAVEFORM_PREFIX):
            raise ValueError(
                "This adapter was built with a callable waveform model, which the device path "
                "cannot run: Ripple generates from its own compiled approximants, not from Python "
                "callbacks. Use a string approximant, or the per-event path."
            )

        # Imported only once the cheap checks pass, so both of the above stay reachable -- and
        # therefore testable, and covered in CI -- in an installation without the [jax] extra.
        from gwmock_signal.waveform.backends.ripple import RippleBackend  # noqa: PLC0415

        available = RippleBackend().available_approximants()
        if name not in available:
            raise ValueError(
                f"Ripple does not implement the approximant '{name}', and the device path always "
                f"generates with Ripple. It is rejected rather than substituted, because silently "
                f"generating a different waveform than the one configured would corrupt the "
                f"simulation. Available: {', '.join(sorted(available))}."
            )
        return name

    def _backend_parameters(
        self,
        parameters: Mapping[str, Any],
        waveform_arguments: Mapping[str, Any] | None,
        waveform_options: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Build the mapping a backend call takes, from per-event and fixed settings.

        Shared by generation and by :meth:`pre_coalescence_duration`, because the buffer length the
        query reports depends on settings that arrive this way -- ripple's ``ringdown_fraction`` and
        ``segment_duration`` among them. Preparing the two mappings separately would let the query
        describe a buffer other than the one generation goes on to produce, and the discrepancy
        would appear as dropped signal rather than as an error.

        Args:
            parameters: Per-event source parameters.
            waveform_arguments: Fixed waveform parameters merged flat; per-event values win.
            waveform_options: Extra options passed through as the backend's ``waveform_arguments``.

        Returns:
            The merged mapping, less any parameters this backend has already rejected.

        Raises:
            ValueError: If *waveform_options* and a ``waveform_arguments`` parameter are both given.
        """
        backend_parameters = {**(waveform_arguments or {}), **dict(parameters)}
        if waveform_options:
            if "waveform_arguments" in backend_parameters:
                raise ValueError(
                    "Specify extra waveform options either via waveform_options or as a "
                    "waveform_arguments parameter, not both"
                )
            backend_parameters["waveform_arguments"] = dict(waveform_options)
        if self._unsupported_params:
            _refuse_configured_rejections(self._unsupported_params, waveform_arguments)
            # The columns were classified when the backend first rejected them; a zero-only column
            # is still checked per event, because only a zero value may be left out.
            require_zero_only_columns_zero(
                self._unsupported_params,
                ignored=self._ignored_population_columns,
                reader="the waveform backend",
                events=(parameters,),
            )
            backend_parameters = {k: v for k, v in backend_parameters.items() if k not in self._unsupported_params}
        return backend_parameters

    def _without_unsupported(
        self,
        exc: ValueError,
        backend_parameters: Mapping[str, Any],
        waveform_arguments: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Return *backend_parameters* less what *exc* says the backend rejects, or ``None``.

        ``None`` means *exc* is not an unsupported-parameter complaint and the caller should
        re-raise it. Records the rejected names on the adapter, so the next call drops them up front
        rather than paying a failed backend call per event.

        Only population columns that are known not to be waveform inputs -- ``redshift``,
        source-frame masses, labels; see :mod:`gwmock.signal.population_columns` -- or that the
        configuration declares ignored are dropped, quietly; they are still recorded in the
        injection parameters. Any other rejected column stops the run: a misspelt spin column would
        otherwise vanish. A key the user wrote under ``waveform-arguments`` stops it too: it is a
        setting the run was asked to apply.

        Args:
            exc: The ``ValueError`` a backend call raised.
            backend_parameters: The mapping that call was given.
            waveform_arguments: The configured fixed waveform arguments merged into that mapping.

        Returns:
            The filtered mapping, or ``None`` when *exc* is some other failure.

        Raises:
            ValueError: If the rejected parameter is ``waveform_arguments`` itself, which no amount
                of filtering fixes -- the installed gwmock-signal cannot carry the requested
                waveform options at all, and dropping them would run the wrong waveform silently;
                or if the backend rejects a key given in *waveform_arguments*, or a population column
                that is neither known to be population-only nor declared ignored.
        """
        match = _UNSUPPORTED_PARAMS_RE.match(str(exc))
        if match is None:
            return None
        extras = {token.strip() for token in match.group(2).split(",") if token.strip()}
        if "waveform_arguments" in extras:
            raise ValueError(
                "The installed gwmock-signal does not support the waveform_arguments "
                "parameter required by waveform_options; upgrade gwmock-signal"
            ) from exc
        _refuse_configured_rejections(extras, waveform_arguments, cause=exc)
        new = extras - self._unsupported_params
        if new:
            try:
                require_population_columns_read(
                    new,
                    ignored=self._ignored_population_columns,
                    reader=f"the {match.group(1)} waveform backend",
                    events=(backend_parameters,),
                )
            except ValueError as refusal:
                raise refusal from exc
            self._unsupported_params |= new
        return {k: v for k, v in backend_parameters.items() if k not in self._unsupported_params}

    def pre_coalescence_duration(
        self,
        parameters: Mapping[str, Any],
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        waveform_arguments: Mapping[str, Any] | None = None,
        waveform_options: Mapping[str, Any] | None = None,
    ) -> float | None:
        """Return how long before ``coa_time`` this event's waveform starts, in seconds.

        Asked before generating, so a segment can claim the event whose waveform *begins* inside it
        rather than the one its ``coa_time`` falls in. A compact binary's buffer starts well before
        coalescence, and a segment chosen from ``coa_time`` alone crops that lead away with nowhere
        to put it -- the earlier segments are already written.

        Args:
            parameters: Per-event source parameters, as :meth:`simulate` takes them.
            sampling_frequency: Sample rate in Hz.
            minimum_frequency: Low-frequency cutoff in Hz.
            waveform_arguments: Fixed waveform parameters, as :meth:`simulate` takes them.
            waveform_options: Extra waveform options, as :meth:`simulate` takes them.

        Returns:
            Seconds between the first sample and coalescence, positive; or ``None`` when the answer
            is *unknown* -- a backend that cannot say (PyCBC), a source type with no coalescence, or
            an installed gwmock-signal predating the query. Never read ``None`` as zero: zero is a
            claim that the waveform starts at coalescence, and acting on it drops the whole inspiral.
        """
        query = getattr(self._backend, "pre_coalescence_duration", None)
        if query is None:
            return None
        backend_parameters = self._backend_parameters(parameters, waveform_arguments, waveform_options)
        try:
            return query(
                backend_parameters,
                sampling_frequency=sampling_frequency,
                minimum_frequency=minimum_frequency,
            )
        except ValueError as exc:
            # The same unsupported-parameter dance generation does, for the same reason: a
            # population column the backend does not know must not be the thing that decides
            # whether an inspiral is placed. Anything else propagates.
            filtered = self._without_unsupported(exc, backend_parameters, waveform_arguments)
            if filtered is None:
                raise
            return query(
                filtered,
                sampling_frequency=sampling_frequency,
                minimum_frequency=minimum_frequency,
            )

    def post_coalescence_duration(
        self,
        parameters: Mapping[str, Any],
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        waveform_arguments: Mapping[str, Any] | None = None,
        waveform_options: Mapping[str, Any] | None = None,
    ) -> float | None:
        """Return how long after ``coa_time`` this event's waveform ends, in seconds.

        The complement of :meth:`pre_coalescence_duration`, and the reason it exists: knowing only
        where a buffer *starts* tells a caller that an event begins before a segment, never that it
        has finished before one. Without the tail, a run whose ``start-time`` is later than its
        population's first event cannot tell that the earlier events are behind it, and claims
        every one of them into its first segment.

        Args:
            parameters: Per-event source parameters, as :meth:`simulate` takes them.
            sampling_frequency: Sample rate in Hz.
            minimum_frequency: Low-frequency cutoff in Hz.
            waveform_arguments: Fixed waveform parameters, as :meth:`simulate` takes them.
            waveform_options: Extra waveform options, as :meth:`simulate` takes them.

        Returns:
            Seconds between coalescence and one sample past the buffer's end, positive; or ``None``
            when the answer is *unknown* -- a backend that cannot say (PyCBC), a source type with no
            coalescence, or an installed gwmock-signal predating the query. Never read ``None`` as
            zero: zero claims the waveform stops at coalescence, and acting on it discards an event
            whose tail still reaches into the segment. The tail is a fraction of the buffer rather
            than a physical ringdown, so it is 0.4 s for a 4 s binary-black-hole buffer and 25.6 s
            for a 256 s binary-neutron-star one.
        """
        query = getattr(self._backend, "post_coalescence_duration", None)
        if query is None:
            return None
        backend_parameters = self._backend_parameters(parameters, waveform_arguments, waveform_options)
        try:
            return query(
                backend_parameters,
                sampling_frequency=sampling_frequency,
                minimum_frequency=minimum_frequency,
            )
        except ValueError as exc:
            # As on the pre side: a population column the backend does not know must not be what
            # decides whether an event is placed. Anything else propagates.
            filtered = self._without_unsupported(exc, backend_parameters, waveform_arguments)
            if filtered is None:
                raise
            return query(
                filtered,
                sampling_frequency=sampling_frequency,
                minimum_frequency=minimum_frequency,
            )

    def simulate_stack(
        self,
        parameters: Mapping[str, Any],
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        waveform_arguments: Mapping[str, Any] | None = None,
        waveform_options: Mapping[str, Any] | None = None,
        background: Mapping[str, Any] | None = None,
        earth_rotation: bool = True,
    ) -> DetectorStrainStack:
        """Generate one detector strain stack via the public ``gwmock_signal`` API.

        Args:
            parameters: The parameters to use for the simulation.
            sampling_frequency: The sampling frequency to use for the simulation.
            minimum_frequency: The minimum frequency to use for the simulation.
            waveform_arguments: Fixed waveform parameters merged (flat) into the
                per-event parameters; per-event values win.
            waveform_options: Extra waveform options (e.g. LAL dictionary
                entries) forwarded to the backend as its ``waveform_arguments``
                parameter, without flattening.
            background: Optional background mapping forwarded to the signal backend.
            earth_rotation: Whether to include earth rotation in the simulation.

        Returns:
            A DetectorStrainStack instance.
        """
        backend_parameters = self._backend_parameters(parameters, waveform_arguments, waveform_options)
        try:
            return self._backend.simulate(
                backend_parameters,
                self._network.detector_names,
                background=background,
                sampling_frequency=sampling_frequency,
                minimum_frequency=minimum_frequency,
                earth_rotation=earth_rotation,
            )
        except ValueError as exc:
            filtered = self._without_unsupported(exc, backend_parameters, waveform_arguments)
            if filtered is None:
                raise
            return self._backend.simulate(
                filtered,
                self._network.detector_names,
                background=background,
                sampling_frequency=sampling_frequency,
                minimum_frequency=minimum_frequency,
                earth_rotation=earth_rotation,
            )

    def simulate(
        self,
        parameters: Mapping[str, Any],
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        waveform_arguments: Mapping[str, Any] | None = None,
        waveform_options: Mapping[str, Any] | None = None,
        background: Mapping[str, Any] | None = None,
        earth_rotation: bool = True,
    ) -> TimeSeries:
        """Generate one signal chunk via the public gwmock-signal ``simulate`` API.

        Args:
            parameters: The parameters to use for the simulation.
            sampling_frequency: The sampling frequency to use for the simulation.
            minimum_frequency: The minimum frequency to use for the simulation.
            waveform_arguments: Fixed waveform parameters merged (flat) into the
                per-event parameters; per-event values win.
            waveform_options: Extra waveform options (e.g. LAL dictionary
                entries) forwarded to the backend as its ``waveform_arguments``
                parameter, without flattening.
            background: Optional background mapping forwarded to the signal backend.
            earth_rotation: Whether to include earth rotation in the simulation.

        Returns:
            A TimeSeries instance.
        """
        strain_stack = self.simulate_stack(
            parameters,
            sampling_frequency=sampling_frequency,
            minimum_frequency=minimum_frequency,
            waveform_arguments=waveform_arguments,
            waveform_options=waveform_options,
            background=background,
            earth_rotation=earth_rotation,
        )
        return TimeSeries(
            data=strain_stack.data,
            start_time=strain_stack.t0,
            sampling_frequency=strain_stack.sample_rate,
        )

    def set_seed(self, seed: int | None) -> None:
        """Set a backend seed attribute when the backend exposes one."""
        if hasattr(self._backend, "seed"):
            self._backend.seed = seed
