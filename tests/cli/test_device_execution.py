"""Selecting batched execution from a configuration file.

``execution: batched`` generates a segment's events together through gwmock-signal's batched entry
point instead of looping over them. That is what makes GPU execution reachable from a config, though
whether it *runs* on a GPU depends on the JAX device present, not on this key.

The chunks the batched path produces stay per-event, so gwmock's own assembler still handles
spill-over into later segments and provenance stays per-injection. The tests below check that the two
modes consume the catalogue identically and produce the same physics, because a difference in either
would make the choice of mode change the dataset rather than only how it was computed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from gwmock.cli.utils.config import Config, SignalConfig
from gwmock.signal.device_chunks import canonicalise_parameters

_POPULATION_CSV = Path(__file__).resolve().parents[2] / "examples" / "signal" / "bbh_population.csv"
_START = 1577491296.0
_SAMPLING_FREQUENCY = 1024.0
#: A row the signal simulator can size, so the placement queries answer rather than refuse.
_COMPLETE_EVENT: dict[str, Any] = {
    "detector_frame_mass_1": 30.0,
    "detector_frame_mass_2": 25.0,
    "distance": 400.0,
    "right_ascension": 1.0,
    "declination": 0.5,
    "polarization_angle": 0.2,
    "inclination": 0.3,
}


def _config(working_directory: Path, execution: str, waveform_backend: str | None = "ripple") -> dict[str, Any]:
    """Return a one-segment BBH config in the given execution mode.

    ``waveform_backend`` is omitted entirely when ``None``. That matters for the tests that must run
    without the ``[jax]`` extra: naming ripple makes *constructing* the orchestrator instantiate
    ``RippleBackend``, which needs ripplegw long before any generation happens.
    """
    return {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": _SAMPLING_FREQUENCY,
                "duration": 16,
                "total-duration": 16,
                "start-time": _START,
                "seed": 20260731,
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
            "signal": {
                "source-type": "bbh",
                "waveform-model": "IMRPhenomD",
                # Naming ripple for both modes makes the comparison isolate batching rather than
                # also changing the waveform library.
                **({"waveform-backend": waveform_backend} if waveform_backend is not None else {}),
                "execution": execution,
                "minimum-frequency": 30,
                "detectors": ["ET-Triangle-Sardinia"],
                "output": {
                    "output_directory": "signal",
                    "file_name": "sig-{{ detectors }}.gwf",
                    "arguments": {"channel": "{{ detectors }}:STRAIN"},
                },
            },
        },
    }


def _orchestrator(
    working_directory: Path,
    execution: str,
    waveform_backend: str | None = "ripple",
    **signal_overrides: Any,
):
    from gwmock.cli.adapter_orchestration import AdapterOrchestrator

    working_directory.mkdir(parents=True, exist_ok=True)
    raw = _config(working_directory, execution, waveform_backend=waveform_backend)
    raw["orchestration"]["signal"].update(signal_overrides)
    config = Config.model_validate(raw)
    return AdapterOrchestrator.from_config(
        config.orchestration,
        global_simulator_arguments=dict(config.globals.simulator_arguments),
    )


class TestConfigKey:
    """The key itself, before anything runs."""

    def test_the_default_is_per_event(self):
        """Existing configs must be unaffected."""
        assert SignalConfig(detectors=["H1"]).execution == "per-event"

    def test_batched_is_accepted(self):
        assert SignalConfig.model_validate({"detectors": ["H1"], "execution": "batched"}).execution == "batched"

    def test_an_unknown_mode_is_rejected(self):
        """A typo must not leave the run silently on the default path.

        The output of a per-event run looks entirely normal, so an author who mistyped the mode
        would have no way to notice they had not switched it.
        """
        with pytest.raises(ValueError, match="'execution' must be one of"):
            SignalConfig.model_validate({"detectors": ["H1"], "execution": "batchd"})


class TestCanonicalParameters:
    """Aliases, so switching mode does not change whether a config runs at all."""

    def test_a_known_alias_is_renamed(self):
        """The bundled BBH catalogue uses ``distance``; the batched path needs the canonical name.

        The per-event backends resolve this internally, so without the rename the same config runs
        one way and fails the other -- which is how this was found.
        """
        assert canonicalise_parameters({"distance": [400.0]}) == {"luminosity_distance": [400.0]}

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [("mass1", "detector_frame_mass_1"), ("spin1z", "spin_1z"), ("tidal_2", "lambda_2")],
    )
    def test_the_alias_table_matches_gwmock_signals(self, alias: str, canonical: str):
        """Taken from gwmock-signal's LAL backend rather than invented."""
        assert canonicalise_parameters({alias: [1.0]}) == {canonical: [1.0]}

    def test_an_unknown_key_is_left_alone(self):
        assert canonicalise_parameters({"coa_time": [1.0]}) == {"coa_time": [1.0]}

    def test_a_conflicting_alias_and_canonical_pair_is_refused(self):
        """Choosing one silently would pick which physics to simulate."""
        with pytest.raises(ValueError, match="both 'luminosity_distance' and its alias"):
            canonicalise_parameters({"distance": [400.0], "luminosity_distance": [500.0]})

    def test_an_agreeing_pair_is_accepted(self):
        """Redundant but consistent input is not an error."""
        assert canonicalise_parameters({"distance": [400.0], "luminosity_distance": [400.0]}) == {
            "luminosity_distance": [400.0]
        }


class TestWaveformConfigurationIsHonoured:
    """Batched mode must not quietly ignore the waveform settings the config asks for.

    The batched entry point takes a backend *instance*, so omitting it discards
    ``waveform-backend-arguments`` and silently uses ripple's defaults -- the same silent drop this
    codebase already fixed once for the per-event path. Output from the wrong settings looks
    entirely normal, which is what makes it worth testing rather than assuming.
    """

    def test_backend_arguments_reach_the_device(self, tmp_path):
        """``taper_fraction`` must change the generated waveform.

        It lowers the effective start frequency, which lengthens the inspiral, so the buffer's
        length changes -- an unmistakable signal that the argument was applied rather than dropped.
        """
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        default = _orchestrator(tmp_path / "default", "batched")._simulate()
        tapered = _orchestrator(
            tmp_path / "tapered", "batched", **{"waveform-backend-arguments": {"taper_fraction": 0.25}}
        )._simulate()

        assert np.asarray(default[0]).shape != np.asarray(tapered[0]).shape, (
            "taper_fraction did not change the waveform, so waveform-backend-arguments are being "
            "discarded and the run silently uses ripple's defaults"
        )

    @pytest.mark.requires_lalsuite
    def test_a_non_ripple_library_is_refused(self, tmp_path):
        """The batched path is ripple-only; substituting it would ignore the configuration.

        Deliberately not gated on ``ripplegw``: the guard runs before anything touches the device,
        so gating it would hide this branch from the CI job that has no extras -- which is the job
        measuring coverage.
        """
        with pytest.raises(ValueError, match="always generates with ripple"):
            _orchestrator(tmp_path, "batched", waveform_backend="lal")._simulate()

    @pytest.mark.requires_lalsuite
    def test_waveform_options_are_refused(self, tmp_path):
        """There is no equivalent parameter on the batched entry point, so silence would lose them.

        Refused by the general consumption check rather than a rule of its own -- see
        ``tests/signal/test_execution_support.py``. Kept here because this exercises it through the
        orchestrator, confirming the check is actually reached on the real path.
        """
        with pytest.raises(ValueError, match="would ignore settings"):
            _orchestrator(
                tmp_path, "batched", waveform_backend=None, **{"waveform-options": {"ModeArray": [[2, 2]]}}
            )._simulate()


class TestTheCheckIsReachedOnThePerEventPath:
    """The per-event warnings are worth nothing if nothing calls them.

    Every other test of this check calls the helper directly, which cannot tell a wired-up check
    from a dead one. These go through ``_simulate`` on the default path instead. The population is
    emptied first so the per-event loop returns without generating anything -- the check runs at the
    top of ``_simulate``, before any of that.
    """

    @staticmethod
    def _simulate_nothing(tmp_path: Path, **signal_overrides: Any) -> None:
        orchestrator = _orchestrator(tmp_path, "per-event", waveform_backend=None, **signal_overrides)
        orchestrator._population_events = []
        orchestrator._simulate()

    @pytest.mark.requires_lalsuite
    def test_an_unrecognised_key_warns_through_the_orchestrator(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="gwmock"):
            self._simulate_nothing(tmp_path, **{"not-a-real-setting": "x"})

        assert "not a recognised setting" in caplog.text

    @pytest.mark.requires_lalsuite
    def test_an_ordinary_configuration_warns_about_nothing(self, tmp_path, caplog):
        """Guards the opposite failure: a check that fires on every run stops being read."""
        with caplog.at_level(logging.WARNING, logger="gwmock"):
            self._simulate_nothing(tmp_path)

        assert caplog.text == ""

    @pytest.mark.requires_lalsuite
    def test_a_refused_configuration_is_refused_again_on_retry(self, tmp_path):
        """The once-per-orchestrator guard must not turn into a once-per-orchestrator bypass.

        ``retry_with_backoff`` re-runs a failed batch on the same instance and restores
        ``simulator.state``, which this flag is not part of. Marking the check done before it passes
        would let the second attempt run the very configuration the first one rejected -- the silent
        drop again, reached by a different route.
        """
        orchestrator = _orchestrator(
            tmp_path, "batched", waveform_backend=None, **{"waveform-options": {"ModeArray": [[2, 2]]}}
        )

        with pytest.raises(ValueError, match="would ignore settings"):
            orchestrator._simulate()

        # The retry. Without the fix this returns normally, generating from a rejected config.
        with pytest.raises(ValueError, match="would ignore settings"):
            orchestrator._simulate()

    @pytest.mark.requires_lalsuite
    def test_the_check_runs_once_rather_than_once_per_segment(self, tmp_path, caplog):
        """A warning repeated for every segment of a long run is noise, not a warning."""
        orchestrator = _orchestrator(tmp_path, "per-event", waveform_backend=None, **{"not-a-real-setting": "x"})
        orchestrator._population_events = []
        with caplog.at_level(logging.WARNING, logger="gwmock"):
            orchestrator._simulate()
            orchestrator._simulate()

        assert caplog.text.count("not a recognised setting") == 1


class TestSegmentEventSelection:
    """Which events a segment claims, tested without generating anything.

    ``_events_for_this_segment`` is pure catalogue walking, so it needs no device -- and testing it
    here rather than behind an ``importorskip`` is what keeps it covered in the CI job that installs
    no extras.
    """

    @pytest.mark.requires_lalsuite
    def test_events_beyond_the_segment_are_left_for_the_next_one(self, tmp_path):
        """The boundary rule is the waveform's start against ``end_time``, and it is checkpointed state."""
        orchestrator = _orchestrator(tmp_path, "batched", waveform_backend=None)
        end_time = float(getattr(orchestrator.end_time, "value", orchestrator.end_time))
        lead = orchestrator._pre_coalescence_duration({**_COMPLETE_EVENT, "coa_time": end_time})
        assert lead is not None
        assert lead > 0
        orchestrator._population_events = (
            {**_COMPLETE_EVENT, "coa_time": end_time - 1.0},
            {**_COMPLETE_EVENT, "coa_time": end_time + lead},
            {**_COMPLETE_EVENT, "coa_time": end_time + lead + 5.0},
        )

        event_ids, events = orchestrator._events_for_this_segment()

        assert event_ids == [0], "only the event before the segment end belongs to this segment"
        assert len(events) == 1
        assert int(orchestrator.population_index) == 0, (
            "reading the segment's events must not advance the checkpointed index: generation can "
            "still fail afterwards, and a caller would then see events consumed that were never "
            "produced"
        )

        orchestrator._commit_consumed_events(event_ids)

        assert int(orchestrator.population_index) == 1, (
            "committing must advance past exactly the events generated, so the next segment resumes "
            "at the boundary rather than skipping or repeating"
        )

    @pytest.mark.requires_lalsuite
    def test_an_empty_segment_consumes_nothing(self, tmp_path):
        """A segment with no events must not advance the index or invent provenance."""
        orchestrator = _orchestrator(tmp_path, "batched", waveform_backend=None)
        end_time = float(getattr(orchestrator.end_time, "value", orchestrator.end_time))
        orchestrator._population_events = ({**_COMPLETE_EVENT, "coa_time": end_time + 100.0},)

        chunks = orchestrator._simulate()

        assert len(chunks) == 0
        assert int(orchestrator.population_index) == 0
        assert orchestrator._batch_injections == []

    @pytest.mark.requires_lalsuite
    def test_the_execution_mode_comes_from_the_config(self, tmp_path):
        assert _orchestrator(tmp_path / "a", "batched", waveform_backend=None)._execution_mode() == "batched"
        assert _orchestrator(tmp_path / "b", "per-event", waveform_backend=None)._execution_mode() == "per-event"


class TestCatalogueConsumption:
    """Both modes must walk the catalogue the same way, or a resumed run skips or repeats events."""

    def test_both_modes_consume_the_same_events(self, tmp_path):
        """``population_index`` is checkpointed state, so the two paths must agree on it.

        A run switched between modes partway through would otherwise lose or duplicate events at the
        boundary, and nothing downstream would notice.
        """
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        per_event = _orchestrator(tmp_path / "per", "per-event")
        batched = _orchestrator(tmp_path / "bat", "batched")

        per_event._simulate()
        batched._simulate()

        assert int(batched.population_index) == int(per_event.population_index)

    def test_the_batched_path_records_provenance_per_event(self, tmp_path):
        """Segment-shaped output must not thin provenance down to one record per segment."""
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        batched = _orchestrator(tmp_path, "batched")

        chunks = batched._simulate()

        assert len(batched._batch_injections) == len(chunks)
        assert all("event_id" in record and "parameters" in record for record in batched._batch_injections)
        assert all("injection_parameters" in chunk.metadata for chunk in chunks)

    def test_both_modes_record_the_same_provenance(self, tmp_path):
        """An injection must be described identically whichever mode produced it.

        The batched path canonicalises names and merges fixed waveform arguments before handing
        parameters to the device. Recording *that* mapping would make the same event appear with
        ``luminosity_distance`` in one mode and ``distance`` in the other, so provenance is taken
        from the catalogue instead.
        """
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        per_event = _orchestrator(tmp_path / "per", "per-event")
        batched = _orchestrator(tmp_path / "bat", "batched")

        per_event._simulate()
        batched._simulate()

        assert batched._batch_injections == per_event._batch_injections


@pytest.mark.e2e
class TestBothModesAgree:
    """The batched path must produce the same physics, not merely run.

    Marked ``e2e`` because it drives the CLI twice and ripple compiles on first use. This is the
    assertion the whole device path rests on: if batching changed the data, the mode would be a
    choice about the dataset rather than about how it was computed.
    """

    @staticmethod
    def _generate(working_directory: Path, execution: str) -> np.ndarray:
        from gwmock.cli.simulate import _simulate_impl

        working_directory.mkdir(parents=True, exist_ok=True)
        config_path = working_directory / "config.yaml"
        config_path.write_text(yaml.safe_dump(_config(working_directory, execution)), encoding="utf-8")
        _simulate_impl(str(config_path))

        from gwpy.timeseries import TimeSeries

        written = working_directory / "output" / "signal" / "sig-ET1_SARD.gwf"
        assert written.is_file(), f"no output from the {execution} run"
        return np.asarray(TimeSeries.read(written, channel="ET1_SARD:STRAIN").value)

    @pytest.mark.requires_gwpy
    def test_the_two_modes_agree_to_floating_point(self, tmp_path):
        """Same events, same library, different execution: the strain must match.

        Not bit-identical, and it should not be expected to be -- a batched ``vmap`` accumulates in a
        different order than a loop. Measured at 1.03e-12 of the peak on this configuration, which is
        floating-point reassociation rather than a difference in physics.

        Compared against the **peak**, not element-wise. ``np.allclose`` with ``atol=0`` applies a
        relative tolerance per sample, which fails wherever the strain is near zero -- an absolute
        difference of 1e-34 against a sample of 1e-30 is a relative difference of 1e-4, and most of a
        segment is near-zero. Scaling to the peak asks the question that matters: is the waveform the
        same, to a fraction of its own amplitude. The default ``atol`` is not usable either, since at
        ~1e-22 it would call any two strain arrays equal.
        """
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        per_event = self._generate(tmp_path / "per", "per-event")
        batched = self._generate(tmp_path / "bat", "batched")

        assert per_event.shape == batched.shape
        assert np.count_nonzero(per_event) == np.count_nonzero(batched), (
            "the two modes placed the signal over a different number of samples"
        )
        assert int(np.argmax(np.abs(per_event))) == int(np.argmax(np.abs(batched))), (
            "the peak moved between modes, so the signal is at a different time"
        )
        peak = float(np.max(np.abs(per_event)))
        assert peak > 0.0, "the per-event run produced no signal, so there is nothing to compare"
        largest_difference = float(np.max(np.abs(per_event - batched)))

        assert largest_difference <= 1e-9 * peak, (
            f"the batched path produced materially different strain: largest difference "
            f"{largest_difference:.3e} is {largest_difference / peak:.3e} of the peak. That makes "
            f"the execution mode a choice about the dataset rather than about how it was computed."
        )


class TestReferenceFrequencyReachesTheBackend:
    """`f_ref` is written under waveform-arguments but is a backend option.

    gwmock-signal's own reserved-argument table says so: "f_ref is configured on the backend, not
    through waveform_arguments". Left in the parameter mapping it reaches `simulate_cbc_batch` and is
    read by nobody, so the same config key would set the reference frequency on the per-event path
    and do nothing on the batched one -- two execution modes silently disagreeing about the waveform.
    """

    def test_it_is_forwarded_to_the_ripple_backend(self, tmp_path):
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        orchestrator = _orchestrator(tmp_path, "batched", **{"waveform-arguments": {"f_ref": 25.0}})

        backend = orchestrator._batched_waveform_backend()

        assert backend is not None, "f_ref alone must still produce a configured backend"
        # `_f_ref` because the backend keeps it private; there is no public reader, and the
        # alternative -- trusting that construction succeeded -- would pass without forwarding.
        assert backend._f_ref == 25.0

    def test_it_does_not_also_go_into_the_parameter_mapping(self, tmp_path):
        """Passing it both ways would hand the batch entry point a key it does not read."""
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        captured = {}
        orchestrator = _orchestrator(tmp_path, "batched", **{"waveform-arguments": {"f_ref": 25.0}})

        import gwmock_signal

        real = gwmock_signal.simulate_cbc_batch

        def spy(*args, **kwargs):
            captured.update(kwargs.get("parameters", {}))
            return real(*args, **kwargs)

        gwmock_signal.simulate_cbc_batch = spy
        try:
            orchestrator._simulate()
        finally:
            gwmock_signal.simulate_cbc_batch = real

        assert captured, "the spy never saw a call, so this test proves nothing"
        assert "f_ref" not in captured

    @pytest.mark.requires_lalsuite
    def test_disagreeing_with_waveform_backend_arguments_is_refused(self, tmp_path):
        """Two keys setting one physical quantity: pick silently and the waveform depends on which
        one a reader happened to look at.

        Deliberately not gated on ripplegw -- the conflict is caught before the backend is built, so
        gating it would hide the branch from the CI job that installs no extras.
        """
        orchestrator = _orchestrator(
            tmp_path,
            "batched",
            waveform_backend=None,
            **{"waveform-arguments": {"f_ref": 25.0}, "waveform-backend-arguments": {"f_ref": 30.0}},
        )

        with pytest.raises(ValueError, match="f_ref is set to"):
            orchestrator._batched_waveform_backend()

    def test_agreeing_with_waveform_backend_arguments_is_accepted(self, tmp_path):
        """Refusing a redundant but consistent pair would be gratuitous."""
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        orchestrator = _orchestrator(
            tmp_path,
            "batched",
            **{"waveform-arguments": {"f_ref": 25.0}, "waveform-backend-arguments": {"f_ref": 25.0}},
        )

        assert orchestrator._batched_waveform_backend()._f_ref == 25.0


#: A complete BBH parameter set, so both the placement query and generation can answer.
_SKIP_EVENT: dict[str, Any] = {
    "detector_frame_mass_1": 30.0,
    "detector_frame_mass_2": 25.0,
    "distance": 400.0,
    "right_ascension": 1.0,
    "declination": 0.5,
    "polarization_angle": 0.2,
    "inclination": 0.3,
}


def _projection_padding_in_the_query(raw: float, answered: float) -> float:
    """Return how much of *answered* is projection padding, checking it is all or nothing.

    gwmock-signal pads the projected strain at both ends by ``edge_padding(fs)`` samples -- the
    largest detector light-travel delay -- so the signal a delay moves past the waveform buffer is
    kept, and its duration queries cover that padded output. Releases before the padding answer with
    the waveform buffer alone. Either way the query must describe the strain that is *written*, so
    the difference from the raw waveform buffer is exactly the padding or exactly zero; anything
    else is a query describing some other buffer.

    Args:
        raw: The generating backend's own answer for the waveform buffer.
        answered: The placement query's answer.

    Returns:
        The padding, in seconds, the answer includes.
    """
    from gwmock_signal.projection.resampling import edge_padding

    padding = edge_padding(_SAMPLING_FREQUENCY) / _SAMPLING_FREQUENCY
    assert padding > 0
    included = answered - raw
    tolerance = 0.5 / _SAMPLING_FREQUENCY
    assert included == pytest.approx(0.0, abs=tolerance) or included == pytest.approx(padding, abs=tolerance), (
        f"the query answers {answered} s against a {raw} s waveform buffer: {included} s more, which is "
        f"neither nothing nor the {padding} s projection padding"
    )
    return padding if included > tolerance else 0.0


def _ripple_simulator_answer(side: str, parameters: dict[str, Any], minimum_frequency: float) -> float | None:
    """Return gwmock-signal's own answer for a ripple-backed compact-binary simulator.

    The simulator's queries state what its projected output covers, padding included where it
    pads, so this is the contract the placement query has to match -- without restating any
    number gwmock-signal owns.
    """
    from gwmock_signal import CBCSimulator
    from gwmock_signal.waveform.backends.ripple import RippleBackend

    simulator = CBCSimulator("IMRPhenomD", waveform_backend=RippleBackend())
    query = simulator.pre_coalescence_duration if side == "pre" else simulator.post_coalescence_duration
    return query(parameters, _SAMPLING_FREQUENCY, minimum_frequency)


def _install(orchestrator, coa_times: list[float]) -> None:
    """Replace the catalogue and drop the cached placement order."""
    orchestrator._population_events = [{**_SKIP_EVENT, "coa_time": t} for t in coa_times]
    orchestrator._placement_order_cache = None
    orchestrator.population_index = 0


class TestTheBatchedPathActuallySkips:
    """The lower bound on the path a GPU run takes, with real ripple numbers.

    ``TestCatalogueConsumption`` already asserts the two modes agree on ``population_index`` -- but
    every event in the bundled catalogue coalesces inside the segment, so it agrees with **zero
    skips taken**, which is the trivial case. The skip decision itself has been exercised with stubs
    and with the default LAL adapter; never end to end through ripple, which is what
    ``execution: batched`` generates with.
    """

    def test_a_finished_event_is_skipped_and_consumed(self, tmp_path):
        """Real query, real generation, no substitution."""
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        orchestrator = _orchestrator(tmp_path, "batched")
        start = float(getattr(orchestrator.start_time, "value", orchestrator.start_time))
        _install(orchestrator, [start - 900.0, start - 600.0, start + 2.0, start + 6.0])

        event_ids, events = orchestrator._events_for_this_segment()

        assert len(events) == 2, f"batched {len(events)} events for a segment containing 2"
        assert sorted(event_ids) == [2, 3]

    def test_both_modes_agree_when_skips_are_actually_taken(self, tmp_path):
        """The agreement the existing test cannot show, because it never skips anything.

        ``population_index`` is checkpointed, so a run switched between modes must not lose or
        repeat an event -- and the interesting case is precisely the one where the two paths each
        have to step over the same finished events.
        """
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        start = _START
        catalogue = [start - 900.0, start - 600.0, start + 2.0, start + 6.0]

        per_event = _orchestrator(tmp_path / "per", "per-event")
        batched = _orchestrator(tmp_path / "bat", "batched")
        _install(per_event, catalogue)
        _install(batched, catalogue)

        per_event._simulate()
        batched._simulate()

        assert int(batched.population_index) == int(per_event.population_index)
        # The index alone cannot see this. A walk that stopped skipping would *generate* the
        # finished events instead of stepping over them and still arrive at 4, so agreement on the
        # index is satisfied by both the fixed and the broken path -- a mutation proved exactly that.
        # What separates them is how many events each one actually produced.
        assert len(batched._batch_injections) == 2, (
            f"the batched path generated {len(batched._batch_injections)} events; the finished ones "
            "were not skipped, so this test is not covering what it exists for"
        )
        assert len(per_event._batch_injections) == 2, (
            f"the per-event path generated {len(per_event._batch_injections)} events; it agreed on "
            "the index while doing different work"
        )
        assert int(batched.population_index) == 4


class TestTheQueryAnswersForTheBackendThatGenerates:
    """A batched run must not ask one library where another library's buffer ends.

    ``execution: batched`` always generates with ripple. The adapter answering the placement query
    used to default to LAL, so the tail deciding whether an event was finished described a different
    library's buffer -- and the two round buffer lengths onto different grids, so which is longer
    flips with mass and cutoff.

    That was not a hypothetical. Measured across 24 mass/cutoff combinations before the fix, **9 had
    ripple's tail longer than LAL's**: 1.4+1.35 at 5 Hz answered 819.20 s against a generated
    934.17 s, so an event with up to 115 s of content inside the segment was reported finished and
    skipped. Low masses and low cutoffs are the Einstein Telescope regime. Two reviewers found this
    independently, from a test that had pinned a single safe point (30+25 at 30 Hz) as though the
    asymmetry were an invariant.
    """

    @pytest.mark.parametrize(
        ("mass_1", "mass_2", "minimum_frequency"),
        [
            (30.0, 25.0, 30.0),  # was safe before the fix
            (1.4, 1.35, 30.0),  # reversed: 6.4004 answered against 8.4375 generated
            (1.4, 1.35, 5.0),  # reversed worst: 819.20 against 934.17
            (4.5, 3.0, 30.0),  # reversed: 1.5996 against 1.8750
            (3.0, 3.0, 5.0),  # reversed: 204.80 against 256.00
            (30.0, 25.0, 20.0),  # reversed narrowly: 0.4004 against 0.4502
        ],
    )
    def test_the_answering_tail_is_the_generated_one(self, tmp_path, mass_1, mass_2, minimum_frequency):
        """Equality, not an inequality: the query and generation must describe one buffer.

        The previous version of this test asserted ``answered >= generated`` -- an inequality that
        holds at some parameters and not others, so it enshrined a coin-flip of rounding as a safety
        property. With the adapter built on the generating backend there is nothing to compare
        conservatively: the two are the same number.
        """
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        from gwmock_signal.waveform.backends.ripple import RippleBackend

        orchestrator = _orchestrator(
            tmp_path / f"{mass_1}_{mass_2}_{minimum_frequency}",
            "batched",
            waveform_backend=None,
            **{"minimum-frequency": minimum_frequency},
        )
        parameters = {
            **_SKIP_EVENT,
            "detector_frame_mass_1": mass_1,
            "detector_frame_mass_2": mass_2,
            "coa_time": _START + 4.0,
        }

        answered = orchestrator._post_coalescence_duration(parameters)
        generated = RippleBackend().post_coalescence_duration(
            "IMRPhenomD",
            _SAMPLING_FREQUENCY,
            minimum_frequency,
            mass1=mass_1,
            mass2=mass_2,
            distance=_SKIP_EVENT["distance"],
            inclination=_SKIP_EVENT["inclination"],
        )

        assert answered is not None
        assert generated is not None
        # The written strain is ripple's buffer plus the projection padding, when the installed
        # gwmock-signal pads; ripple's own query sizes only the waveform buffer.
        generated += _projection_padding_in_the_query(generated, answered)
        assert answered == pytest.approx(generated, abs=0.5 / _SAMPLING_FREQUENCY), (
            f"the placement query answers {answered} s while ripple generates {generated} s; an "
            f"event whose content reaches {generated - answered} s further than the query admits "
            "can be skipped while it is still contributing to the segment"
        )
        assert answered == pytest.approx(
            _ripple_simulator_answer("post", parameters, minimum_frequency), abs=0.5 / _SAMPLING_FREQUENCY
        ), "the placement query disagrees with gwmock-signal's ripple-backed simulator"


class TestTheLeadIsAlignedToo:
    """The lead moved as well, and that is what actually relocates boundary events.

    A reviewer found this after the tail fix: the adapter answers both queries, so aligning it to
    ripple changed ``pre_coalescence_duration`` too -- and the lead drives ``_placement_key`` and the
    upper bound. LAL and ripple disagree there as well (30+25 at 30 Hz: 3.60 s against 2.81 s with
    gwmock-signal 0.21.1, 6.71 s against 2.81 s with 0.21.2, whose LAL buffer starts below the
    cutoff), so an event at ``end + 3.0`` was claimed by this segment before the fix and belongs to
    the next one after it.

    That is the same defect as the tail, on the other side: the query described a buffer a different
    library would produce. It is a real behaviour change, it moves boundary events for configs where
    the tail direction was already safe, and the tail sweep does not see it.
    """

    @pytest.mark.parametrize(
        ("mass_1", "mass_2", "minimum_frequency"),
        [(30.0, 25.0, 30.0), (1.4, 1.35, 5.0), (4.5, 3.0, 30.0)],
    )
    def test_the_answering_lead_is_the_generated_one(self, tmp_path, mass_1, mass_2, minimum_frequency):
        """Equality against the generating backend, exactly as for the tail."""
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        from gwmock_signal.waveform.backends.ripple import RippleBackend

        orchestrator = _orchestrator(
            tmp_path / f"lead_{mass_1}_{minimum_frequency}",
            "batched",
            waveform_backend=None,
            **{"minimum-frequency": minimum_frequency},
        )
        parameters = {
            **_SKIP_EVENT,
            "detector_frame_mass_1": mass_1,
            "detector_frame_mass_2": mass_2,
            "coa_time": _START + 4.0,
        }

        answered = orchestrator._pre_coalescence_duration(parameters)
        generated = RippleBackend().pre_coalescence_duration(
            "IMRPhenomD",
            _SAMPLING_FREQUENCY,
            minimum_frequency,
            mass1=mass_1,
            mass2=mass_2,
            distance=_SKIP_EVENT["distance"],
            inclination=_SKIP_EVENT["inclination"],
        )

        assert answered is not None
        assert generated is not None
        generated += _projection_padding_in_the_query(generated, answered)
        assert answered == pytest.approx(generated, abs=0.5 / _SAMPLING_FREQUENCY), (
            f"the placement query answers a {answered} s lead while ripple generates {generated} s; "
            "the segment an event is claimed by is decided from a different library's buffer"
        )
        assert answered == pytest.approx(
            _ripple_simulator_answer("pre", parameters, minimum_frequency), abs=0.5 / _SAMPLING_FREQUENCY
        ), "the placement query disagrees with gwmock-signal's ripple-backed simulator"


class TestTheRippleSubstitutionIsHonest:
    """Substituting ripple for a batched config must not break construction or lie in provenance.

    Both cases are reviewer findings against the first version of the fix, which instantiated ripple
    eagerly in ``from_config`` and left the metadata reading the raw config key.
    """

    def test_metadata_names_the_library_that_actually_ran(self, tmp_path):
        """Provenance must say ripple, because ripple is what answered and generated.

        The metadata comment says ``None`` means gwmock-signal's default (LAL) applied. Reading the
        raw config key would file a ripple run under LAL -- and that field exists precisely because
        the two libraries produce different strain, so a wrong value is worse than a missing one.
        """
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        orchestrator = _orchestrator(tmp_path, "batched", waveform_backend=None)

        assert orchestrator._effective_waveform_backend() == "ripple"

    @pytest.mark.requires_lalsuite
    def test_a_per_event_config_still_reports_no_substitution(self, tmp_path):
        """The substitution is batched-only, so per-event provenance must be unchanged."""
        orchestrator = _orchestrator(tmp_path, "per-event", waveform_backend=None)

        assert orchestrator._effective_waveform_backend() is None

    def test_an_explicit_backend_is_reported_as_configured(self, tmp_path):
        """A named backend is not a substitution and must be reported as the config wrote it."""
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        orchestrator = _orchestrator(tmp_path, "batched", waveform_backend="ripple")

        assert orchestrator._effective_waveform_backend() == "ripple"

    @pytest.mark.requires_lalsuite
    def test_construction_survives_without_the_jax_extra(self, tmp_path, monkeypatch):
        """A batched orchestrator must still *construct* when ripple is not installed.

        The first version of this fix instantiated ripple eagerly, so config validation, stubbed
        placement and every test passing ``waveform_backend=None`` began failing with "rippleGW is
        not installed" -- and a genuine configuration error would surface as that message instead of
        its own. Falling back is safe here and only here: without ripple the batched path cannot
        generate at all, so no placement answer it might get wrong is ever acted on.
        """
        from gwmock.cli import adapter_orchestration as module

        real = module.instantiate_backend

        def _no_ripple(kind, name, *args, **kwargs):
            if kind == "waveform" and name == "ripple":
                raise ImportError("ripple (rippleGW) is not installed. Run: pip install 'gwmock-signal[jax]'")
            return real(kind, name, *args, **kwargs)

        monkeypatch.setattr(module, "instantiate_backend", _no_ripple)

        orchestrator = _orchestrator(tmp_path, "batched", waveform_backend=None)

        assert orchestrator.signal_adapter is not None
        # And it does not then claim a library it failed to load.
        assert orchestrator._effective_waveform_backend() is None

    def test_replacing_the_adapter_stops_the_substitution_claim(self, tmp_path):
        """The record describes the adapter `from_config` built, so a swap must invalidate it.

        A reviewer found the value surviving the replacement: provenance kept reporting ripple while
        a LAL-backed stand-in answered every placement query. Tests substitute adapters routinely, so
        this is the common path, not an exotic one. Clearing on assignment makes the record describe
        nothing rather than describe the wrong thing.
        """
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        orchestrator = _orchestrator(tmp_path, "batched", waveform_backend=None)
        assert orchestrator._effective_waveform_backend() == "ripple"

        class _StandIn:
            detector_names = ("E1",)

        orchestrator.signal_adapter = _StandIn()

        assert orchestrator._effective_waveform_backend() is None, (
            "provenance still claims the library the replaced adapter was built on"
        )

    @pytest.mark.requires_lalsuite
    def test_nothing_is_skipped_on_an_answer_the_run_cannot_stand_behind(self, tmp_path, monkeypatch):
        """The reproducer a reviewer built against the first version of this fallback.

        Constructing is harmless; *answering* is not. With ripple missing, the LAL-backed adapter
        reported a 1.4+1.35 event at 5 Hz as finished on its 819.20 s tail, when ripple would have
        produced 934.17 s and still been contributing to the segment. `_events_for_this_segment`
        skipped it and advanced `population_index`; `_simulate_batched_segment` then returned early
        on the empty batch, so generation never ran and the install error never surfaced. The event
        was consumed, never generated, and nothing was reported -- exactly the outcome my "no ripple
        means no generation, so nothing acts on a wrong answer" reasoning ruled out.
        """
        from gwmock.cli import adapter_orchestration as module

        real = module.instantiate_backend

        def _no_ripple(kind, name, *args, **kwargs):
            if kind == "waveform" and name == "ripple":
                raise ImportError("ripple (rippleGW) is not installed.")
            return real(kind, name, *args, **kwargs)

        monkeypatch.setattr(module, "instantiate_backend", _no_ripple)
        orchestrator = _orchestrator(tmp_path, "batched", waveform_backend=None, **{"minimum-frequency": 5.0})
        start = float(getattr(orchestrator.start_time, "value", orchestrator.start_time))
        _install(orchestrator, [start - 850.0])
        orchestrator._population_events = [
            {**_SKIP_EVENT, "detector_frame_mass_1": 1.4, "detector_frame_mass_2": 1.35, "coa_time": start - 850.0}
        ]
        orchestrator._placement_order_cache = None
        orchestrator.population_index = 0

        # Nothing may be skipped on an answer this run cannot stand behind, so the tail reports
        # unknown and the event is claimed. The batch is then non-empty, and on a machine without
        # ripple generation raises the install error -- loudly, where it belongs, instead of an
        # empty batch quietly completing and checkpointing the events as done.
        #
        # The generation half is deliberately not asserted here: this test forces the *construction*
        # import to fail, so on a machine that has ripple -- which is where the suite runs --
        # generation succeeds and consumes the event legitimately. Asserting a raise would pass only
        # where ripple is absent and mislead everywhere else.
        event = orchestrator._population_events[0]

        assert orchestrator._post_coalescence_duration(event) is None, (
            "the tail was answered by a library that is not the one this run generates with"
        )
        assert not orchestrator._event_ended_before_segment_start(event), (
            "the event was skipped on a tail the run could not stand behind"
        )


class TestThePopulationColumnsTheBatchedPathIgnores:
    """The batched entry point reads a fixed set and ignores every other column without complaint.

    So it is checked against the same rule as the per-event path, once, before anything is
    generated or consumed.
    """

    @staticmethod
    def _with_columns(orchestrator, **extra):
        end_time = float(getattr(orchestrator.end_time, "value", orchestrator.end_time))
        orchestrator._population_events = ({**_COMPLETE_EVENT, "coa_time": end_time - 1.0, **extra},)
        return orchestrator

    @pytest.mark.requires_lalsuite
    def test_a_misspelt_spin_column_stops_the_run_before_anything_is_consumed(self, tmp_path):
        orchestrator = self._with_columns(_orchestrator(tmp_path, "batched", waveform_backend=None), spin1_z=0.3)

        with pytest.raises(ValueError, match=r"spin1_z.*batched path.*signal\.ignored-population-columns"):
            orchestrator._simulate()
        assert int(orchestrator.population_index) == 0

    @pytest.mark.requires_lalsuite
    def test_known_population_only_columns_pass(self, tmp_path):
        orchestrator = self._with_columns(
            _orchestrator(tmp_path, "batched", waveform_backend=None),
            redshift=0.1,
            source_frame_mass_1=_COMPLETE_EVENT["detector_frame_mass_1"] / 1.1,
            eccentricity=0.0,
        )

        orchestrator._require_batched_population_columns()

    @pytest.mark.requires_lalsuite
    def test_a_declared_column_passes(self, tmp_path):
        orchestrator = self._with_columns(
            _orchestrator(tmp_path, "batched", waveform_backend=None, **{"ignored-population-columns": ["spin1_z"]}),
            spin1_z=0.3,
        )

        orchestrator._require_batched_population_columns()

    @pytest.mark.requires_lalsuite
    def test_a_nonzero_eccentricity_stops_the_run(self, tmp_path):
        orchestrator = self._with_columns(_orchestrator(tmp_path, "batched", waveform_backend=None), eccentricity=0.2)

        with pytest.raises(ValueError, match=r"eccentricity is 0\.2"):
            orchestrator._require_batched_population_columns()

    @pytest.mark.requires_lalsuite
    def test_spins_fixed_under_waveform_arguments_count_for_theta_jn(self, tmp_path):
        """The device reads the population merged with the fixed arguments; the check reads the same."""
        orchestrator = _orchestrator(
            tmp_path, "batched", waveform_backend=None, **{"waveform-arguments": {"spin_1z": 0.0, "spin_2z": 0.0}}
        )
        end_time = float(getattr(orchestrator.end_time, "value", orchestrator.end_time))
        orchestrator._population_events = (
            {**_COMPLETE_EVENT, "coa_time": end_time - 1.0, "theta_jn": _COMPLETE_EVENT["inclination"]},
        )

        orchestrator._require_batched_population_columns()
