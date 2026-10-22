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

        # The projected strain is the waveform plus, where gwmock-signal pads the projection, the
        # largest detector light-travel delay at each end -- not the waveform's own length.
        from gwmock_signal.projection.resampling import edge_padding

        padding = (strain.shape[1] - n) // 2
        assert strain.shape[0] == 2
        assert padding in {0, edge_padding(4.0)}, f"{strain.shape[1]} samples for an {n}-sample waveform"
        assert strain.shape[1] == n + 2 * padding
        backend = adapter._backend
        assert isinstance(backend.waveform_model, str)
        assert backend.waveform_model.startswith("__gwmock_custom__")
        assert merged_calls[0]["reference_frequency"] == 50.0
        assert merged_calls[0]["detector_frame_mass_1"] == 30.0


class FailingBackend:
    """Backend that raises 'Unsupported <backend> waveform parameters' for the keys in ``REJECTED``."""

    #: What the backend does not read: two population-only columns, a waveform setting, a
    #: misspelt spin and an orbital parameter it cannot apply.
    REJECTED = frozenset({"redshift", "source_frame_mass_1", "lambda_1", "spin1_z", "eccentricity"})

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
        rejected = sorted(set(params) & self.REJECTED)
        if rejected:
            raise ValueError(f"Unsupported {self.backend_label} waveform parameters: {', '.join(rejected)}")
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
    def test_known_population_only_columns_are_dropped_quietly(self, backend_label):
        """``redshift`` and source-frame masses are not waveform inputs: dropped, retried, no warning.

        The parse works for every backend's error prefix, not only LAL.
        """
        backend = FailingBackend(waveform_model="IMRPhenomXPHM", backend_label=backend_label)
        adapter = _make_adapter_with_backend(backend)

        params = {"detector_frame_mass_1": 1.43, "redshift": 0.1, "source_frame_mass_1": 1.3, "coa_time": 0.0}

        with patch("gwmock.signal.adapter.logger") as mock_logger:
            adapter.simulate_stack(params, sampling_frequency=4.0, minimum_frequency=20.0)

        mock_logger.warning.assert_not_called()
        assert backend.call_count == 2
        assert "redshift" not in backend.received_params[1]
        assert "source_frame_mass_1" not in backend.received_params[1]
        assert "detector_frame_mass_1" in backend.received_params[1]

    def test_known_columns_are_cached_after_the_first_call(self):
        """Subsequent calls pre-filter and do NOT trigger a second backend failure."""
        backend = FailingBackend(waveform_model="IMRPhenomXPHM")
        adapter = _make_adapter_with_backend(backend)

        params = {"mass_1": 1.4, "redshift": 0.05, "coa_time": 0.0}
        adapter.simulate_stack(params, sampling_frequency=4.0, minimum_frequency=20.0)
        adapter.simulate_stack(params, sampling_frequency=4.0, minimum_frequency=20.0)

        assert backend.call_count == 3  # 2 from first simulate_stack + 1 from second
        assert "redshift" not in backend.received_params[2]

    @pytest.mark.parametrize("backend_label", ["LAL", "ripple", "PyCBC"])
    def test_a_misspelt_spin_column_stops_the_run(self, backend_label):
        """``spin1_z`` is not ``spin_1z``: dropping it would generate a non-spinning binary silently."""
        backend = FailingBackend(waveform_model="IMRPhenomXPHM", backend_label=backend_label)
        adapter = _make_adapter_with_backend(backend)

        with pytest.raises(ValueError, match="spin1_z") as excinfo:
            adapter.simulate_stack(
                {"mass_1": 1.4, "spin1_z": 0.3, "redshift": 0.05, "coa_time": 0.0},
                sampling_frequency=4.0,
                minimum_frequency=20.0,
            )
        message = str(excinfo.value)
        assert f"the {backend_label} waveform backend" in message
        assert "signal.ignored-population-columns" in message
        assert "redshift" not in message.split("Check the spelling", maxsplit=1)[0], (
            "a known column was named as refused"
        )
        assert backend.call_count == 1, "the call was retried without the misspelt column"

    def test_every_refused_column_is_named(self):
        backend = FailingBackend(waveform_model="IMRPhenomXPHM")
        adapter = _make_adapter_with_backend(backend)

        with pytest.raises(ValueError, match=r"lambda_1, spin1_z"):
            adapter.simulate_stack(
                {"mass_1": 1.4, "spin1_z": 0.3, "lambda_1": 300.0, "coa_time": 0.0},
                sampling_frequency=4.0,
                minimum_frequency=20.0,
            )

    def test_a_declared_column_is_dropped(self):
        backend = FailingBackend(waveform_model="IMRPhenomXPHM")
        adapter = _make_adapter_with_backend(backend)
        adapter.ignore_population_columns(["spin1_z"])

        adapter.simulate_stack(
            {"mass_1": 1.4, "spin1_z": 0.3, "coa_time": 0.0}, sampling_frequency=4.0, minimum_frequency=20.0
        )

        assert "spin1_z" not in backend.received_params[-1]

    def test_a_zero_eccentricity_is_dropped_and_a_nonzero_one_stops_the_run(self):
        """A quasi-circular waveform is exact at zero eccentricity and wrong at any other value."""
        backend = FailingBackend(waveform_model="IMRPhenomXPHM")
        adapter = _make_adapter_with_backend(backend)

        adapter.simulate_stack(
            {"mass_1": 1.4, "eccentricity": 0.0, "coa_time": 0.0}, sampling_frequency=4.0, minimum_frequency=20.0
        )
        with pytest.raises(ValueError, match=r"eccentricity is 0\.1"):
            adapter.simulate_stack(
                {"mass_1": 1.4, "eccentricity": 0.1, "coa_time": 0.0}, sampling_frequency=4.0, minimum_frequency=20.0
            )

    def test_spins_given_only_as_magnitude_and_tilt_stop_the_run(self):
        """Dropping them would generate a non-spinning binary from a spinning catalogue."""
        backend = FailingBackend(waveform_model="IMRPhenomXPHM")
        backend.REJECTED = frozenset({"spin_1", "tilt_1"})
        adapter = _make_adapter_with_backend(backend)

        with pytest.raises(ValueError, match="spin_1, tilt_1 are given without all of spin_1x"):
            adapter.simulate_stack(
                {"mass_1": 1.4, "spin_1": 0.5, "tilt_1": 0.6, "coa_time": 0.0},
                sampling_frequency=4.0,
                minimum_frequency=20.0,
            )

    def test_a_per_event_f_ref_must_be_the_one_the_backend_uses(self):
        """Without a configured f_ref the backends reference the spins to the call's cutoff."""
        backend = FailingBackend(waveform_model="IMRPhenomXPHM")
        backend.REJECTED = frozenset({"f_ref"})
        adapter = _make_adapter_with_backend(backend)
        event = {"mass_1": 1.4, "f_ref": 20.0, "coa_time": 0.0}

        adapter.simulate_stack(event, sampling_frequency=4.0, minimum_frequency=20.0)
        with pytest.raises(ValueError, match=r"f_ref is 20\.0 Hz.*5\.0 Hz"):
            adapter.simulate_stack(event, sampling_frequency=4.0, minimum_frequency=5.0)

        adapter.configure_reference_frequency(20.0)
        adapter.simulate_stack(event, sampling_frequency=4.0, minimum_frequency=5.0)

    def test_a_rejected_configured_waveform_argument_fails_the_run(self):
        """A key the user set under ``waveform-arguments`` is refused, not dropped.

        Dropping it would generate a waveform other than the one configured -- an ``f_ref`` that
        never reaches the backend runs at the backend's default -- while the run reports success.
        """
        backend = FailingBackend(waveform_model="IMRPhenomXPHM")
        adapter = _make_adapter_with_backend(backend)

        with pytest.raises(ValueError, match=r"lambda_1.*waveform-arguments") as excinfo:
            adapter.simulate_stack(
                {"mass_1": 1.4, "coa_time": 0.0},
                sampling_frequency=4.0,
                minimum_frequency=20.0,
                waveform_arguments={"lambda_1": 300.0},
            )
        assert "Unsupported LAL waveform parameters" in str(excinfo.value)
        assert backend.call_count == 1, "the call was retried without the configured key"

    def test_a_rejected_configured_waveform_argument_fails_every_entry_point(self):
        """The placement queries refuse it too, so placement cannot run on a silently changed setting."""
        backend = TailQueryingBackend()
        adapter = _make_adapter_with_backend(backend)

        with pytest.raises(ValueError, match="waveform-arguments"):
            adapter.post_coalescence_duration(
                {"mass_1": 30.0, "coa_time": 0.0},
                sampling_frequency=1024.0,
                minimum_frequency=20.0,
                waveform_arguments={"redshift": 0.1},
            )

    def test_a_configured_key_already_rejected_as_a_column_is_not_dropped_silently(self):
        """The cache of rejected population columns must not swallow a configured key of the same name."""
        backend = FailingBackend(waveform_model="IMRPhenomXPHM")
        adapter = _make_adapter_with_backend(backend)
        adapter.simulate_stack(
            {"mass_1": 1.4, "redshift": 0.05, "coa_time": 0.0},
            sampling_frequency=4.0,
            minimum_frequency=20.0,
        )

        with pytest.raises(ValueError, match="redshift"):
            adapter.simulate_stack(
                {"mass_1": 1.4, "coa_time": 0.0},
                sampling_frequency=4.0,
                minimum_frequency=20.0,
                waveform_arguments={"redshift": 0.05},
            )

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
