"""Tests for the batched (on-device) entry point on the gwmock-side adapter.

gwmock's orchestration generates one event at a time: ``cli/adapter_orchestration.py`` loops over
population events calling ``simulate``, producing a strain per event. That is the only shape the
LAL and PyCBC backends offer, and it is why gwmock could not run a GPU simulation at all — the
batched path in gwmock-signal had no caller here.

``simulate_segments`` is the batched counterpart: catalogue in as a struct-of-arrays, assembled
segments out. It is deliberately a *second* entry point rather than a change to ``simulate_stack``,
because the per-event path has to keep working unchanged for backends with no batched form.

The tests that actually generate waveforms need the ``[jax]`` extra and are skipped without it; the
input-handling tests do not, and run everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwmock.signal.adapter import SignalAdapter

_FS = 2048.0
_START = 1400000000.0
_SEGMENT = 64.0
_SPAN = 256.0


def _events(n: int = 4) -> list[dict[str, float]]:
    """Return *n* BNS events with every canonical parameter the device path requires."""
    rng = np.random.default_rng(0)
    return [
        {
            "detector_frame_mass_1": 1.4,
            "detector_frame_mass_2": 1.35,
            "luminosity_distance": float(distance),
            "inclination": float(inclination),
            "coa_phase": float(phase),
            "right_ascension": float(ascension),
            "declination": float(declination),
            "polarization_angle": float(polarization),
            "coa_time": float(coalescence),
            "spin_1z": 0.0,
            "spin_2z": 0.0,
        }
        for distance, inclination, phase, ascension, declination, polarization, coalescence in zip(
            rng.uniform(200.0, 2000.0, n),
            rng.uniform(0.0, np.pi, n),
            rng.uniform(0.0, 2 * np.pi, n),
            rng.uniform(0.0, 2 * np.pi, n),
            np.arcsin(rng.uniform(-1.0, 1.0, n)),
            rng.uniform(0.0, np.pi, n),
            _START + np.sort(rng.uniform(0.0, _SPAN, n)),
            strict=True,
        )
    ]


def _adapter() -> SignalAdapter:
    """An adapter on the bundled ET triangle preset, which resolves to CustomDetector instances."""
    return SignalAdapter.from_source_type(
        source_type="bns", waveform_model="IMRPhenomD", detectors=["ET-Triangle-Sardinia"]
    )


class TestEventsToStructOfArrays:
    """Transposing per-event mappings into the columns the batched path takes."""

    def test_transposes_shared_keys_in_event_order(self):
        """Each column holds one key's value across events, in the order given."""
        events = [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]
        assert SignalAdapter.events_to_struct_of_arrays(events) == {"a": [1.0, 3.0], "b": [2.0, 4.0]}

    def test_rejects_events_whose_keys_differ(self):
        """A ragged catalogue is an error, not something to reduce to the shared keys.

        Intersecting would be the more forgiving choice and the wrong one: dropping a key that only
        some events carry does not avoid fabricating physics, it fabricates it for the whole batch,
        because the backend default then applies to every event with nothing said.
        """
        events = [{"a": 1.0, "spin_1z": 0.4}, {"a": 2.0}]
        with pytest.raises(ValueError, match="same parameters") as raised:
            SignalAdapter.events_to_struct_of_arrays(events)
        assert "spin_1z" in str(raised.value), "the message must name the parameter that went missing"

    def test_names_the_offending_event_and_both_directions(self):
        """The message points at which event differs, and how, so a loader bug is locatable."""
        events = [{"a": 1.0}, {"a": 2.0}, {"a": 3.0, "surprise": 0.0}]
        with pytest.raises(ValueError, match="event 2") as raised:
            SignalAdapter.events_to_struct_of_arrays(events)
        assert "surprise" in str(raised.value)

    def test_rejects_an_empty_catalogue(self):
        """No events means no batch, and saying so beats returning an empty mapping."""
        with pytest.raises(ValueError, match="non-empty"):
            SignalAdapter.events_to_struct_of_arrays([])

    def test_rejects_events_with_no_key_in_common(self):
        """Wholly disjoint events are the extreme case of ragged, and fail the same way."""
        with pytest.raises(ValueError, match="same parameters"):
            SignalAdapter.events_to_struct_of_arrays([{"a": 1.0}, {"b": 2.0}])


class TestSimulateSegmentsValidation:
    """Input checks that run before any device work, so they need no extra installed."""

    @pytest.mark.requires_lalsuite
    def test_missing_canonical_parameters_are_named(self):
        """The message must list what is absent and say aliases are not accepted.

        The batched path reads canonical names straight from the struct-of-arrays, so an alias such
        as ``mass1`` would otherwise surface as a ``KeyError`` from inside a jitted kernel rather
        than as a statement about the input.
        """
        parameters = {"mass1": [1.4], "mass2": [1.35]}
        with pytest.raises(ValueError, match="detector_frame_mass_1") as raised:
            _adapter().simulate_segments(
                parameters,
                sampling_frequency=_FS,
                minimum_frequency=30.0,
                segment_duration=_SEGMENT,
                start_time=_START,
                end_time=_START + _SPAN,
            )
        message = str(raised.value)
        assert "coa_time" in message
        assert "alias" in message.lower()

    @pytest.mark.requires_lalsuite
    def test_ragged_columns_are_rejected(self):
        """Columns of different lengths cannot describe one catalogue."""
        events = _events(3)
        parameters = SignalAdapter.events_to_struct_of_arrays(events)
        parameters["coa_time"] = parameters["coa_time"][:2]
        with pytest.raises(ValueError, match="disagree in length"):
            _adapter().simulate_segments(
                parameters,
                sampling_frequency=_FS,
                minimum_frequency=30.0,
                segment_duration=_SEGMENT,
                start_time=_START,
                end_time=_START + _SPAN,
            )

    @pytest.mark.requires_lalsuite
    def test_a_scalar_required_parameter_is_rejected(self):
        """A per-event parameter given once is not a column, whatever it would broadcast to.

        Verified to be a real failure rather than a hypothetical: passing ``coa_time`` as a scalar
        reaches batching and raises ``too many indices for array: array is 0-dimensional``, which
        names neither the parameter nor the mistake.
        """
        parameters = SignalAdapter.events_to_struct_of_arrays(_events(2))
        parameters["coa_time"] = _START
        with pytest.raises(ValueError, match="must be given as arrays") as raised:
            _adapter().simulate_segments(
                parameters,
                sampling_frequency=_FS,
                minimum_frequency=30.0,
                segment_duration=_SEGMENT,
                start_time=_START,
                end_time=_START + _SPAN,
            )
        assert "coa_time" in str(raised.value)

    @pytest.mark.requires_lalsuite
    def test_a_scalar_stays_legal_for_a_parameter_that_is_not_per_event(self):
        """Only the required per-event names are forced to be arrays.

        Fixing a value across the catalogue with a scalar -- ``f_ref=20.0`` -- is the documented use
        of ``waveform_arguments``, so the new check must not sweep it up.

        The assertion is the *absence* of the scalar rejection rather than a specific exception,
        because what happens after validation differs by installation: with the extra the
        approximant is resolved and rejected as unknown, without it the Ripple import fails first.
        Both mean validation was cleared, which is the whole claim.
        """
        parameters = SignalAdapter.events_to_struct_of_arrays(_events(2))
        adapter = SignalAdapter.from_source_type(
            source_type="bns", waveform_model="NotARippleApproximant", detectors=["ET-Triangle-Sardinia"]
        )
        with pytest.raises(Exception) as raised:  # noqa: PT011 - see the docstring
            adapter.simulate_segments(
                parameters,
                sampling_frequency=_FS,
                minimum_frequency=30.0,
                segment_duration=_SEGMENT,
                start_time=_START,
                end_time=_START + _SPAN,
                waveform_arguments={"f_ref": 20.0},
            )
        assert "must be given as arrays" not in str(raised.value), "a scalar f_ref must not be rejected"

    @pytest.mark.requires_lalsuite
    def test_ragged_numpy_columns_are_rejected(self):
        """Length validation must cover arrays, not only lists and tuples.

        An ``ndarray`` of the wrong length used to pass this check and fail deep inside batching,
        where the error no longer names the parameter at fault.
        """
        parameters = SignalAdapter.events_to_struct_of_arrays(_events(3))
        parameters = {key: np.asarray(value) for key, value in parameters.items()}
        parameters["coa_time"] = parameters["coa_time"][:2]
        with pytest.raises(ValueError, match="disagree in length"):
            _adapter().simulate_segments(
                parameters,
                sampling_frequency=_FS,
                minimum_frequency=30.0,
                segment_duration=_SEGMENT,
                start_time=_START,
                end_time=_START + _SPAN,
            )


class TestForwardingToTheBatchedCall:
    """What the adapter actually hands to ``simulate_cbc_catalogue``.

    These stub out the generation but still resolve the approximant, which instantiates
    ``RippleBackend``, so they need the ``[jax]`` extra.
    """

    @pytest.fixture(autouse=True)
    def _require_device_stack(self):
        pytest.importorskip("jax", reason="the [jax] extra is not installed")
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")

    def test_a_catalogue_value_overrides_the_same_key_in_waveform_arguments(self, monkeypatch):
        """Fixed arguments fill gaps; per-event values win where both supply a key.

        Asserted on what actually reaches ``simulate_cbc_catalogue`` rather than through a failure
        message. A message-based check would pass for any implementation that merges the two
        mappings in *either* order, so it could not detect the precedence being backwards -- which is
        the only thing this test is for.
        """
        captured = {}

        def _capture(*args, **kwargs):
            captured.update(kwargs)
            return []

        monkeypatch.setattr("gwmock_signal.simulate_cbc_catalogue", _capture)

        adapter = _adapter()
        parameters = SignalAdapter.events_to_struct_of_arrays(_events(2))
        adapter.simulate_segments(
            parameters,
            sampling_frequency=_FS,
            minimum_frequency=30.0,
            segment_duration=_SEGMENT,
            start_time=_START,
            end_time=_START + _SPAN,
            waveform_arguments={"detector_frame_mass_1": [99.0, 99.0], "f_ref": 20.0},
        )
        merged = captured["parameters"]
        assert merged["detector_frame_mass_1"] == [1.4, 1.4], "the catalogue must win over a fixed value"
        assert merged["f_ref"] == 20.0, "a fixed key absent from the catalogue must still be passed"

    def test_earth_rotation_reaches_the_batched_call(self, monkeypatch):
        """The flag must be forwarded, not silently left at the gwmock-signal default.

        Without this the setting would be honoured on the per-event path and ignored here, so the
        two paths would disagree for anyone who turns it off.
        """
        captured = {}
        monkeypatch.setattr("gwmock_signal.simulate_cbc_catalogue", lambda *a, **k: captured.update(k) or [])

        _adapter().simulate_segments(
            SignalAdapter.events_to_struct_of_arrays(_events(2)),
            sampling_frequency=_FS,
            minimum_frequency=30.0,
            segment_duration=_SEGMENT,
            start_time=_START,
            end_time=_START + _SPAN,
            earth_rotation=False,
        )
        assert captured["earth_rotation"] is False


class TestDeviceApproximantWithoutTheExtra:
    """Model rejections that do not need Ripple, and so must stay reachable without the extra.

    ``_device_approximant`` runs these before importing ``RippleBackend`` precisely so they hold in
    a base installation. Placing them here rather than behind ``importorskip`` is the assertion.
    """

    @pytest.mark.requires_lalsuite
    def test_a_callable_waveform_model_is_rejected_with_its_own_message(self):
        """A callable is registered under a generated key, which is not an approximant name.

        Left unchecked that key reaches Ripple as if it were one, and the failure talks about an
        unsupported approximant named ``__gwmock_custom__...`` rather than about the callable.
        """

        def custom(*args, **kwargs):  # pragma: no cover - never called
            raise AssertionError("the device path must not reach the callable")

        adapter = SignalAdapter.from_source_type(
            source_type="bns", waveform_model=custom, detectors=["ET-Triangle-Sardinia"]
        )
        with pytest.raises(ValueError, match="callable waveform model"):
            adapter.simulate_segments(
                SignalAdapter.events_to_struct_of_arrays(_events(2)),
                sampling_frequency=_FS,
                minimum_frequency=30.0,
                segment_duration=_SEGMENT,
                start_time=_START,
                end_time=_START + _SPAN,
            )


class TestDeviceApproximant:
    """Which configured models can and cannot cross to the device path."""

    @pytest.fixture(autouse=True)
    def _require_device_stack(self):
        pytest.importorskip("jax", reason="the [jax] extra is not installed")
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")

    def test_an_approximant_ripple_lacks_is_rejected_not_substituted(self):
        """Generating a different waveform than the one configured would corrupt the simulation."""
        adapter = SignalAdapter.from_source_type(
            source_type="bns", waveform_model="SEOBNRv4", detectors=["ET-Triangle-Sardinia"]
        )
        with pytest.raises(ValueError, match="does not implement the approximant 'SEOBNRv4'") as raised:
            adapter.simulate_segments(
                SignalAdapter.events_to_struct_of_arrays(_events(2)),
                sampling_frequency=_FS,
                minimum_frequency=30.0,
                segment_duration=_SEGMENT,
                start_time=_START,
                end_time=_START + _SPAN,
            )
        assert "IMRPhenomD" in str(raised.value), "the message must list what is available instead"


class TestSimulateSegmentsOnDevice:
    """End-to-end through the real batched path. Requires the ``[jax]`` extra."""

    @pytest.fixture(autouse=True)
    def _require_device_stack(self):
        pytest.importorskip("jax", reason="the [jax] extra is not installed")
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")

    def test_returns_one_stack_per_segment(self):
        """The span is tiled into fixed-duration segments, each a full detector stack."""
        adapter = _adapter()
        segments = adapter.simulate_segments(
            SignalAdapter.events_to_struct_of_arrays(_events()),
            sampling_frequency=_FS,
            minimum_frequency=30.0,
            segment_duration=_SEGMENT,
            start_time=_START,
            end_time=_START + _SPAN,
        )
        assert len(segments) == int(_SPAN // _SEGMENT)
        assert segments[0].detector_names == ("ET1_SARD", "ET2_SARD", "ET3_SARD")
        assert segments[0].data.shape == (3, round(_SEGMENT * _FS))

    def test_the_segments_contain_signal(self):
        """A catalogue that overlaps the span must leave non-zero strain in it.

        Asserted on finite, non-zero content rather than on exact values: this test exists to show
        the device path is reached and produces a waveform, not to re-verify gwmock-signal's physics,
        which its own suite covers.
        """
        adapter = _adapter()
        segments = adapter.simulate_segments(
            SignalAdapter.events_to_struct_of_arrays(_events()),
            sampling_frequency=_FS,
            minimum_frequency=30.0,
            segment_duration=_SEGMENT,
            start_time=_START,
            end_time=_START + _SPAN,
        )
        channel = segments[0].detector_names[0]
        occupied = sum(int(np.count_nonzero(segment.to_dict()[channel].value)) for segment in segments)
        peak = max(float(np.max(np.abs(segment.to_dict()[channel].value))) for segment in segments)
        assert occupied > 0, "no segment contains signal, so the device path produced nothing"
        assert np.isfinite(peak)
        assert peak > 0.0

    def test_every_detector_in_the_network_differs(self):
        """Three ET channels must carry three different projections, not one repeated.

        atol=0 because strain is ~1e-23: the default absolute tolerance would call any two strain
        arrays equal and the assertion could not fail.
        """
        adapter = _adapter()
        segments = adapter.simulate_segments(
            SignalAdapter.events_to_struct_of_arrays(_events()),
            sampling_frequency=_FS,
            minimum_frequency=30.0,
            segment_duration=_SEGMENT,
            start_time=_START,
            end_time=_START + _SPAN,
        )
        with_signal = max(segments, key=lambda segment: np.count_nonzero(segment.data))
        first, second, third = with_signal.data
        assert not np.allclose(first, second, rtol=1e-6, atol=0.0)
        assert not np.allclose(second, third, rtol=1e-6, atol=0.0)
