"""Tests for the gwmock-side gwmock_signal adapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from gwmock_data import TimeSeries as ChannelSeries
from gwmock_signal import DetectorStrainStack

from gwmock.signal.adapter import SignalAdapter


class FakeBackend:
    """Minimal backend exposing the public gwmock_signal ``simulate`` interface."""

    def __init__(self, *, waveform_model: str) -> None:
        self.waveform_model = waveform_model
        self.required_params = frozenset({"coa_time"})
        self.simulate_calls: list[dict[str, object]] = []

    def register_waveform_model(self, name: str, factory) -> None:
        self.waveform_model = name
        self._registered = (name, factory)

    def simulate(
        self,
        params: dict,
        detector_names,
        background=None,
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        earth_rotation: bool = True,
        interpolate_if_offset: bool = True,
    ) -> DetectorStrainStack:
        """Record the call and return simple per-detector strains."""
        names = tuple(detector if isinstance(detector, str) else detector.name for detector in detector_names)
        self.simulate_calls.append(
            {
                "params": params,
                "detector_names": tuple(detector_names),
                "background": background,
                "sampling_frequency": sampling_frequency,
                "minimum_frequency": minimum_frequency,
                "earth_rotation": earth_rotation,
                "interpolate_if_offset": interpolate_if_offset,
            }
        )
        return DetectorStrainStack.from_mapping(
            names,
            {
                name: ChannelSeries(
                    [10.0 + index, 20.0 + index, 30.0 + index, 40.0 + index], t0=8.0, sample_rate=sampling_frequency
                )
                for index, name in enumerate(names)
            },
        )


class NoRegistrationBackend:
    """Backend without waveform registration support."""

    def __init__(self, *, waveform_model: str) -> None:
        self.waveform_model = waveform_model


class TestSignalAdapter:
    """Test the gwmock-side signal adapter."""

    def test_from_source_type_merges_fixed_waveform_arguments(self):
        """The adapter should merge waveform args and call the public backend ``simulate``."""
        reference_frequency = 50.0
        detector_frame_mass_1 = 30.0
        expected = {
            "H1": [10.0, 20.0, 30.0, 40.0],
            "L1": [11.0, 21.0, 31.0, 41.0],
        }
        parameters = {
            "detector_frame_mass_1": detector_frame_mass_1,
            "detector_frame_mass_2": 25.0,
            "coa_time": 12.0,
            "distance": 800.0,
            "inclination": 0.3,
            "right_ascension": 1.1,
            "declination": -0.8,
            "polarization_angle": 0.2,
        }

        with (
            patch("gwmock.signal.adapter.resolve_simulator_backend", return_value=FakeBackend),
        ):
            adapter = SignalAdapter.from_source_type(
                source_type="bbh",
                waveform_model="IMRPhenomD",
                detectors=["H1", "L1"],
            )
            strain = adapter.simulate(
                parameters,
                sampling_frequency=4.0,
                minimum_frequency=5.0,
                waveform_arguments={"reference_frequency": reference_frequency},
            )

        backend = adapter._backend
        assert adapter.detector_names == ("H1", "L1")
        assert backend.simulate_calls[0]["params"]["reference_frequency"] == reference_frequency
        assert backend.simulate_calls[0]["params"]["detector_frame_mass_1"] == detector_frame_mass_1
        assert backend.simulate_calls[0]["detector_names"] == ("H1", "L1")
        assert backend.simulate_calls[0]["background"] is None
        assert backend.simulate_calls[0]["sampling_frequency"] == 4.0
        assert backend.simulate_calls[0]["minimum_frequency"] == 5.0
        assert backend.simulate_calls[0]["earth_rotation"] is True
        assert strain.shape == (2, 4)
        np.testing.assert_allclose(strain[0].value, expected["H1"])

    def test_from_source_type_loads_detector_config_files_via_network_from_file(self, tmp_path):
        """Detector config paths should be loaded through ``Network.from_file``."""
        config_path = tmp_path / "custom.interferometer"
        config_path.write_text("# dummy\n")
        sentinel_detector = SimpleNamespace(name="custom")

        with (
            patch("gwmock.signal.adapter.resolve_simulator_backend", return_value=FakeBackend),
            patch(
                "gwmock.signal.adapter.Network.from_file",
                return_value=SimpleNamespace(detector_names=(sentinel_detector,)),
            ) as mock_from_file,
            patch(
                "gwmock.signal.adapter.Network.from_detectors",
                return_value=SimpleNamespace(detector_names=(sentinel_detector,)),
            ) as mock_from_detectors,
        ):
            adapter = SignalAdapter.from_source_type(
                source_type="bbh",
                waveform_model="IMRPhenomD",
                detectors=[str(config_path)],
            )

        assert adapter.detector_names == ("custom",)
        mock_from_file.assert_called_once_with(config_path)
        mock_from_detectors.assert_called_once_with((sentinel_detector,))

    def test_from_source_type_resolves_named_networks_via_public_network_api(self):
        """Named detector presets should be resolved through ``Network.from_preset``."""
        sentinel_detector = SimpleNamespace(name="ET1_SARD")

        with (
            patch("gwmock.signal.adapter.resolve_simulator_backend", return_value=FakeBackend),
            patch(
                "gwmock.signal.adapter.Network.from_preset",
                return_value=SimpleNamespace(detector_names=(sentinel_detector,)),
            ) as mock_from_preset,
            patch(
                "gwmock.signal.adapter.Network.from_detectors",
                return_value=SimpleNamespace(detector_names=(sentinel_detector,)),
            ) as mock_from_detectors,
        ):
            adapter = SignalAdapter.from_source_type(
                source_type="bbh",
                waveform_model="IMRPhenomD",
                detectors=["ET-Triangle-Sardinia"],
            )

        assert adapter.detector_names == ("ET1_SARD",)
        mock_from_preset.assert_called_once_with("ET-Triangle-Sardinia")
        mock_from_detectors.assert_called_once_with((sentinel_detector,))

    def test_from_source_type_resolves_single_public_detector_aliases_via_public_presets(self):
        """Single ET-detector aliases should resolve through the public preset catalog."""
        with patch("gwmock.signal.adapter.resolve_simulator_backend", return_value=FakeBackend):
            adapter = SignalAdapter.from_source_type(
                source_type="bbh",
                waveform_model="IMRPhenomD",
                detectors=["ET1_SARD"],
            )

        assert adapter.detector_names == ("ET1_SARD",)

    def test_from_source_type_callable_requires_public_waveform_registration(self):
        """Backends without ``register_waveform_model`` cannot host a callable ``waveform_model``."""

        def _dummy(**_kwargs):
            return {"plus": None, "cross": None}

        with (
            patch("gwmock.signal.adapter.resolve_simulator_backend", return_value=NoRegistrationBackend),
            pytest.raises(AttributeError, match="register_waveform_model"),
        ):
            SignalAdapter.from_source_type(
                source_type="bbh",
                waveform_model=_dummy,
                detectors=["H1"],
            )

    @pytest.mark.requires_lalsuite
    def test_from_source_type_callable_waveform_with_real_backend(self):
        """A callable should be registered and invoked on the public CBC backend."""
        from gwmock_data import TimeSeries

        n = 32
        merged_calls: list[dict] = []

        def custom_waveform(*, waveform_model, tc, sampling_frequency, minimum_frequency, **params):
            merged_calls.append(
                {"waveform_model": waveform_model, "tc": tc, "minimum_frequency": minimum_frequency, **params}
            )
            t0 = float(tc) - (n / 2) / sampling_frequency
            hp = TimeSeries(np.linspace(0, 1, n), t0=t0, sample_rate=sampling_frequency)
            hc = TimeSeries(np.linspace(1, 0, n), t0=t0, sample_rate=sampling_frequency)
            return {"plus": hp, "cross": hc}

        parameters = {
            "detector_frame_mass_1": 30.0,
            "detector_frame_mass_2": 25.0,
            "coa_time": 12.0,
            "distance": 800.0,
            "inclination": 0.3,
            "right_ascension": 1.1,
            "declination": -0.8,
            "polarization_angle": 0.2,
        }

        adapter = SignalAdapter.from_source_type(
            source_type="bbh",
            waveform_model=custom_waveform,
            detectors=["H1", "L1"],
        )
        strain = adapter.simulate(
            parameters,
            sampling_frequency=4.0,
            minimum_frequency=5.0,
            waveform_arguments={"reference_frequency": 50.0},
        )

        assert strain.shape == (2, n)
        backend = adapter._backend
        assert isinstance(backend.waveform_model, str)
        assert backend.waveform_model.startswith("__gwmock_custom__")
        assert merged_calls[0]["reference_frequency"] == 50.0
        assert merged_calls[0]["detector_frame_mass_1"] == 30.0


class FailingBackend:
    """Backend that raises 'Unsupported <backend> waveform parameters' on the first call."""

    def __init__(self, *, waveform_model: str, backend_label: str = "LAL") -> None:
        self.waveform_model = waveform_model
        self.backend_label = backend_label
        self.required_params = frozenset({"coa_time"})
        self.call_count = 0
        self.received_params: list[dict] = []

    def simulate(
        self,
        params,
        detector_names,
        background=None,
        *,
        sampling_frequency,
        minimum_frequency,
        earth_rotation=True,
        interpolate_if_offset=True,
    ):
        self.call_count += 1
        self.received_params.append(dict(params))
        if "redshift" in params or "lambda_1" in params:
            raise ValueError(f"Unsupported {self.backend_label} waveform parameters: lambda_1, redshift")
        names = tuple(d if isinstance(d, str) else d.name for d in detector_names)
        return DetectorStrainStack.from_mapping(
            names,
            {name: ChannelSeries([1.0, 2.0], t0=0.0, sample_rate=sampling_frequency) for name in names},
        )


def _make_network(detector_names=("H1",)):
    """Return a minimal Network-like stub."""
    return SimpleNamespace(detector_names=detector_names)


def _make_adapter_with_backend(backend):
    """Build a SignalAdapter using an already-instantiated backend stub."""
    network = _make_network()
    return SignalAdapter(source_type="bns", backend=backend, network=network)


class TestSimulateStackUnsupportedParams:
    """Verify catch-parse-retry for 'Unsupported LAL waveform parameters' errors."""

    @pytest.mark.parametrize("backend_label", ["LAL", "ripple", "PyCBC"])
    def test_unsupported_params_trigger_warning_and_retry(self, backend_label):
        """On first failure the adapter logs a WARNING and retries without bad keys.

        The parse works for every backend's error prefix, not only LAL.
        """
        backend = FailingBackend(waveform_model="IMRPhenomXPHM", backend_label=backend_label)
        adapter = _make_adapter_with_backend(backend)

        params = {"mass_1": 1.4, "redshift": 0.05, "lambda_1": 300.0, "coa_time": 0.0}

        with patch("gwmock.signal.adapter.logger") as mock_logger:
            adapter.simulate_stack(params, sampling_frequency=4.0, minimum_frequency=20.0)

        # Warning was emitted listing the ignored keys
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0] % mock_logger.warning.call_args[0][1:]
        assert "lambda_1" in warning_msg
        assert "redshift" in warning_msg
        assert "injection_parameters" in warning_msg
        # Two backend calls: initial (failing) + retry (filtered)
        assert backend.call_count == 2
        assert "redshift" not in backend.received_params[1]
        assert "lambda_1" not in backend.received_params[1]
        assert "mass_1" in backend.received_params[1]

    def test_unsupported_params_cached_on_second_call(self):
        """Subsequent calls pre-filter and do NOT trigger a second backend failure."""
        backend = FailingBackend(waveform_model="IMRPhenomXPHM")
        adapter = _make_adapter_with_backend(backend)

        params = {"mass_1": 1.4, "redshift": 0.05, "lambda_1": 300.0, "coa_time": 0.0}

        with patch("gwmock.signal.adapter.logger") as mock_logger:
            adapter.simulate_stack(params, sampling_frequency=4.0, minimum_frequency=20.0)
            first_warning_count = mock_logger.warning.call_count
            adapter.simulate_stack(params, sampling_frequency=4.0, minimum_frequency=20.0)
            second_warning_count = mock_logger.warning.call_count

        # No additional warning on the second call
        assert second_warning_count == first_warning_count
        # Second call: only one backend call (pre-filtered, no failure)
        assert backend.call_count == 3  # 2 from first simulate_stack + 1 from second
        assert "redshift" not in backend.received_params[2]
        assert "lambda_1" not in backend.received_params[2]

    def test_genuine_value_error_is_reraised(self):
        """A ValueError not matching the prefix must propagate unchanged."""
        backend = MagicMock()
        backend.simulate.side_effect = ValueError("mass_1 must be positive")
        network = _make_network()
        adapter = SignalAdapter(source_type="bbh", backend=backend, network=network)

        with pytest.raises(ValueError, match="mass_1 must be positive"):
            adapter.simulate_stack({"mass_1": -1.0}, sampling_frequency=4.0, minimum_frequency=20.0)


class RecordingBackend:
    """Backend stub that records the parameters it receives."""

    def __init__(self, *, waveform_model: str) -> None:
        self.waveform_model = waveform_model
        self.required_params = frozenset({"coa_time"})
        self.received_params: list[dict] = []

    def simulate(
        self,
        params,
        detector_names,
        background=None,
        *,
        sampling_frequency,
        minimum_frequency,
        earth_rotation=True,
        interpolate_if_offset=True,
    ):
        self.received_params.append(dict(params))
        names = tuple(d if isinstance(d, str) else d.name for d in detector_names)
        return DetectorStrainStack.from_mapping(
            names,
            {name: ChannelSeries([1.0, 2.0], t0=0.0, sample_rate=sampling_frequency) for name in names},
        )


class WaveformArgumentsRejectingBackend(RecordingBackend):
    """Backend stub predating gwmock-signal's waveform_arguments parameter."""

    def __init__(self, *, waveform_model: str, backend_label: str = "LAL") -> None:
        super().__init__(waveform_model=waveform_model)
        self.backend_label = backend_label

    def simulate(self, params, detector_names, background=None, **kwargs):
        if "waveform_arguments" in params:
            raise ValueError(f"Unsupported {self.backend_label} waveform parameters: waveform_arguments")
        return super().simulate(params, detector_names, background=background, **kwargs)


class TestWaveformOptions:
    def test_options_forwarded_as_waveform_arguments_parameter(self):
        """waveform_options reach the backend nested, not flattened."""
        backend = RecordingBackend(waveform_model="IMRPhenomXHM")
        adapter = _make_adapter_with_backend(backend)
        options = {"ModeArray": [(2, 2), (2, -2)]}
        adapter.simulate_stack(
            {"mass_1": 30.0, "coa_time": 0.0},
            sampling_frequency=4.0,
            minimum_frequency=20.0,
            waveform_options=options,
        )
        received = backend.received_params[0]
        assert received["waveform_arguments"] == options
        assert "ModeArray" not in received

    def test_options_and_parameter_conflict_raises(self):
        """Specifying options both ways is rejected."""
        adapter = _make_adapter_with_backend(RecordingBackend(waveform_model="IMRPhenomXHM"))
        with pytest.raises(ValueError, match="not both"):
            adapter.simulate_stack(
                {"mass_1": 30.0, "coa_time": 0.0, "waveform_arguments": {"ModeArray": [(2, 2)]}},
                sampling_frequency=4.0,
                minimum_frequency=20.0,
                waveform_options={"ModeArray": [(3, 3)]},
            )

    @pytest.mark.parametrize("backend_label", ["LAL", "ripple", "PyCBC"])
    def test_old_backend_raises_instead_of_dropping_options(self, backend_label):
        """A gwmock-signal without waveform_arguments support must not silently drop options.

        The upgrade error is raised regardless of which backend rejected it.
        """
        adapter = _make_adapter_with_backend(
            WaveformArgumentsRejectingBackend(waveform_model="IMRPhenomXHM", backend_label=backend_label)
        )
        with pytest.raises(ValueError, match="upgrade gwmock-signal"):
            adapter.simulate_stack(
                {"mass_1": 30.0, "coa_time": 0.0},
                sampling_frequency=4.0,
                minimum_frequency=20.0,
                waveform_options={"ModeArray": [(2, 2)]},
            )

    def test_empty_options_change_nothing(self):
        """No waveform_arguments key is injected when options are empty."""
        backend = RecordingBackend(waveform_model="IMRPhenomXHM")
        adapter = _make_adapter_with_backend(backend)
        adapter.simulate_stack(
            {"mass_1": 30.0, "coa_time": 0.0},
            sampling_frequency=4.0,
            minimum_frequency=20.0,
            waveform_options={},
        )
        assert "waveform_arguments" not in backend.received_params[0]


class TailQueryingBackend:
    """Backend answering the tail query, rejecting unsupported parameters on the first call.

    Deliberately separate from :class:`FailingBackend`: that one fails inside ``simulate``, and the
    point here is the *query*, which runs before any generation and has its own retry.
    """

    def __init__(self, *, waveform_model: str = "IMRPhenomD", backend_label: str = "LAL") -> None:
        self.waveform_model = waveform_model
        self.backend_label = backend_label
        self.required_params = frozenset({"coa_time"})
        self.query_calls: list[dict] = []

    def post_coalescence_duration(self, params, *, sampling_frequency, minimum_frequency):
        del sampling_frequency, minimum_frequency
        self.query_calls.append(dict(params))
        if "redshift" in params:
            raise ValueError(f"Unsupported {self.backend_label} waveform parameters: redshift")
        return 0.4


class TestPostCoalescenceDurationThroughTheAdapter:
    """The tail query's own parameter handling, which generation's tests do not cover.

    Both cases were named by a reviewer as residual risk after the round-2 fixes: the query is the
    thing that decides *placement*, so if it and generation disagree about what reaches the
    backend, the number describes a waveform the run will not produce.
    """

    def test_a_population_column_the_backend_rejects_does_not_lose_the_answer(self):
        """Catch-parse-retry on the query, mirroring what generation already does.

        Without it a catalogue carrying an extra column -- ``redshift`` is the usual one -- makes
        every tail unknown, and the placement rule silently degrades to the behaviour it replaced.
        """
        backend = TailQueryingBackend()
        adapter = _make_adapter_with_backend(backend)

        tail = adapter.post_coalescence_duration(
            {"mass_1": 30.0, "coa_time": 0.0, "redshift": 0.1},
            sampling_frequency=1024.0,
            minimum_frequency=20.0,
        )

        assert tail == 0.4
        assert len(backend.query_calls) == 2, "the query was not retried without the rejected column"
        assert "redshift" not in backend.query_calls[1]

    def test_waveform_arguments_and_options_reach_the_query(self):
        """The query must see the same merged mapping generation sees.

        A query answering from defaults while generation runs with a pinned ``segment_duration`` or
        a non-default ``ringdown_fraction`` would size a different buffer than the one produced,
        and the placement decision would be made about a waveform that does not exist.
        """
        backend = TailQueryingBackend()
        adapter = _make_adapter_with_backend(backend)

        adapter.post_coalescence_duration(
            {"mass_1": 30.0, "coa_time": 0.0},
            sampling_frequency=1024.0,
            minimum_frequency=20.0,
            waveform_arguments={"reference_frequency": 50.0},
            waveform_options={"ModeArray": [(2, 2)]},
        )

        received = backend.query_calls[0]
        assert received["reference_frequency"] == 50.0
        assert "waveform_arguments" in received
