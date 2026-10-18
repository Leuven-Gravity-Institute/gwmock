"""The continuous-wave branch of the orchestrator, exercised without a waveform library.

``test_continuous_wave_orchestration.py`` covers the same branch against the real backend, which is
where phase coherence and the numerical composition are established. That module needs ``ripplegw``
and the LALPulsar ephemeris tables, so it runs in the `test-jax` job rather than the default one.

This module runs everywhere, which is the point of separating them: the wiring is covered without
the optional extra, and only claims that need real strain depend on it.

What is checked here is the wiring: which route a `cw` configuration takes, what it refuses, what
it records, and that the catalogue is summed rather than overwritten. None of that involves a
waveform, so a fake signal backend serves. The fake returns ``background + a per-source constant``,
which makes the expected total exact and any composition error a mismatch rather than a tolerance
question.

Deliberately not covered here: anything about the *signal*. A fake backend cannot show that the
phase joins up across a segment boundary, and a test here that appeared to would be lying.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from gwmock_data import TimeSeries as ChannelSeries
from gwmock_signal import DetectorStrainStack

from gwmock.cli.adapter_orchestration import AdapterOrchestrator
from gwmock.cli.utils.config import Config

FAKE_CW_SIGNAL_BACKEND = "tests.cli.test_continuous_wave_wiring:FakeContinuousWaveBackend"

_EPOCH = 1577491218.0
_FS = 4.0
_DETECTORS = ["H1", "L1"]

#: Two pulsars whose amplitudes differ, so a sum and an overwrite cannot agree by accident.
_PULSARS = (
    "right_ascension,declination,frequency,initial_phase,amplitude_plus,amplitude_cross\n"
    "1.1,0.3,20.0,0.4,3.0,0.0\n"
    "2.7,-0.6,15.0,1.9,5.0,0.0\n"
)


class FakeContinuousWaveBackend:
    """Adds ``amplitude_plus`` to the background, so the expected total is exactly the sum."""

    required_params: ClassVar[frozenset[str]] = frozenset({"frequency", "amplitude_plus"})

    #: Every ``simulate`` call, so a test can assert what the orchestrator passed down.
    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, reference_time_ssb: float | None = None, **_kwargs: Any) -> None:
        self.reference_time_ssb = reference_time_ssb

    def simulate(
        self,
        parameters: dict,
        detector_names: tuple[str, ...],
        background=None,
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        earth_rotation: bool = True,
        interpolate_if_offset: bool = True,
    ) -> DetectorStrainStack:
        _ = minimum_frequency, interpolate_if_offset
        names = tuple(detector if isinstance(detector, str) else detector.name for detector in detector_names)
        type(self).calls.append(
            {
                "parameters": dict(parameters),
                "reference_time_ssb": self.reference_time_ssb,
                "earth_rotation": earth_rotation,
                "background_peak": None if background is None else float(np.max(np.abs(background[names[0]].value))),
            }
        )
        if background is None:
            raise AssertionError("the continuous-wave branch must always pass a background to accumulate into")
        return DetectorStrainStack.from_mapping(
            names,
            {
                detector: ChannelSeries(
                    np.asarray(background[detector].value, dtype=float) + float(parameters["amplitude_plus"]),
                    t0=float(background[detector].t0),
                    sample_rate=sampling_frequency,
                )
                for detector in names
            },
        )


@pytest.fixture(autouse=True)
def _clear_backend_calls():
    FakeContinuousWaveBackend.calls.clear()
    yield
    FakeContinuousWaveBackend.calls.clear()


def _config(tmp_path: Path, *, duration: float, total: float, n_pulsars: int = 2) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    catalogue = tmp_path / "pulsars.csv"
    rows = _PULSARS.splitlines()
    catalogue.write_text("\n".join([rows[0], *rows[1 : 1 + n_pulsars]]) + "\n")
    return {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": _FS,
                "duration": duration,
                "total-duration": total,
                "start-time": _EPOCH,
                "seed": 7,
            },
            "working-directory": str(tmp_path),
            "output-directory": "output",
            "metadata-directory": "metadata",
        },
        "orchestration": {
            "population": {
                "backend": "FilePopulationLoader",
                "source-type": "cw",
                "n-samples": n_pulsars,
                "arguments": {"path": str(catalogue)},
            },
            "signal": {
                "backend": FAKE_CW_SIGNAL_BACKEND,
                "source-type": "cw",
                "detectors": list(_DETECTORS),
                "minimum-frequency": 5,
                "earth-rotation": True,
                "arguments": {"reference_time_ssb": _EPOCH},
                "output": {
                    "output_directory": "signal",
                    "file_name": "cw.gwf",
                    "arguments": {"channel": "{{ detectors }}:STRAIN"},
                },
            },
        },
    }


def _orchestrator(tmp_path: Path, **kwargs: Any) -> AdapterOrchestrator:
    parsed = Config.model_validate(_config(tmp_path, **kwargs))
    return AdapterOrchestrator.from_config(
        parsed.orchestration, global_simulator_arguments=dict(parsed.globals.simulator_arguments)
    )


def _segment(orchestrator: AdapterOrchestrator) -> np.ndarray:
    chunks = orchestrator._simulate()
    assert len(chunks) == 1, f"expected one chunk spanning the segment, got {len(chunks)}"
    return np.asarray(chunks[0][0], dtype=float)


class TestTheBranchIsTaken:
    """A `cw` configuration must reach the continuous-wave route and not one that merely works."""

    def test_a_cw_configuration_produces_one_chunk_spanning_the_segment(self, tmp_path):
        """The per-event route emits a chunk per event; this route emits one for the whole span."""
        orchestrator = _orchestrator(tmp_path, duration=8, total=8)

        chunks = orchestrator._simulate()

        assert len(chunks) == 1
        samples = np.asarray(chunks[0][0], dtype=float)
        assert samples.size == round(8 * _FS)
        assert float(chunks[0].start_time.value) == _EPOCH

    def test_the_chunk_records_how_many_sources_went_into_it(self, tmp_path):
        """A summed chunk hides its source count unless it is written down."""
        orchestrator = _orchestrator(tmp_path, duration=8, total=8, n_pulsars=2)

        chunks = orchestrator._simulate()

        assert chunks[0].metadata["continuous_wave_sources"] == 2

    def test_an_empty_catalogue_yields_no_chunks(self, tmp_path):
        """No sources means nothing to write, rather than a chunk of zeros claiming to be signal."""
        orchestrator = _orchestrator(tmp_path, duration=8, total=8, n_pulsars=1)
        orchestrator._population_events = []

        assert len(orchestrator._simulate()) == 0


class TestTheCatalogueIsSummed:
    """Each pulsar adds to the running total, which is what feeding the background back achieves."""

    def test_the_total_is_the_sum_of_every_source(self, tmp_path):
        """Pins the composition rather than merely that a second source changed something.

        The fake adds its ``amplitude_plus`` to whatever background it is handed, so two pulsars at
        3.0 and 5.0 must give exactly 8.0. An implementation that overwrote would give 5.0, one
        that dropped the feedback 5.0 as well, and one that double-counted 11.0 or more -- all
        distinguishable, none within a tolerance of the right answer.
        """
        samples = _segment(_orchestrator(tmp_path, duration=8, total=8, n_pulsars=2))

        assert np.array_equal(samples, np.full(round(8 * _FS), 8.0))

    def test_each_source_sees_the_running_total_as_its_background(self, tmp_path):
        """The second call must receive the first source's output, not zeros.

        Expressed against the order the loader actually produced rather than the order in the CSV.
        ``FilePopulationLoader`` hands the rows back reversed, and an assertion written to the file
        would encode that as though it were the contract under test.
        """
        orchestrator = _orchestrator(tmp_path, duration=8, total=8, n_pulsars=2)
        first_amplitude = float(orchestrator._population_events[0]["amplitude_plus"])

        _segment(orchestrator)

        backgrounds = [call["background_peak"] for call in FakeContinuousWaveBackend.calls]
        assert backgrounds == [0.0, first_amplitude], f"expected the total to accumulate, got backgrounds {backgrounds}"

    def test_every_detector_in_the_network_is_summed(self, tmp_path):
        """A sum that only held for the first detector would pass a single-detector check."""
        orchestrator = _orchestrator(tmp_path, duration=8, total=8, n_pulsars=2)

        chunks = orchestrator._simulate()

        assert len(orchestrator.signal_adapter.detector_names) == len(_DETECTORS)
        for index in range(len(_DETECTORS)):
            assert np.array_equal(np.asarray(chunks[0][index], dtype=float), np.full(round(8 * _FS), 8.0))


class TestTheCatalogueReachesEverySegment:
    """Nothing is consumed: every pulsar is present in every segment."""

    def test_the_population_index_never_advances(self, tmp_path):
        """Advancing it would silently shorten later segments while they still looked valid."""
        orchestrator = _orchestrator(tmp_path, duration=8, total=16)

        _segment(orchestrator)
        assert int(orchestrator.population_index) == 0

        orchestrator.update_state()
        _segment(orchestrator)
        assert int(orchestrator.population_index) == 0

    def test_the_second_segment_still_contains_every_source(self, tmp_path):
        """The index staying put is the mechanism; this is the consequence that matters."""
        orchestrator = _orchestrator(tmp_path, duration=8, total=16, n_pulsars=2)

        _segment(orchestrator)
        orchestrator.update_state()

        assert np.array_equal(_segment(orchestrator), np.full(round(8 * _FS), 8.0))

    def test_every_source_is_recorded_for_every_segment(self, tmp_path):
        """Provenance lists all pulsars each time, because all of them are present each time."""
        orchestrator = _orchestrator(tmp_path, duration=8, total=16, n_pulsars=2)

        _segment(orchestrator)
        assert [record["event_id"] for record in orchestrator._batch_injections] == [0, 1]

        orchestrator.update_state()
        _segment(orchestrator)
        assert [record["event_id"] for record in orchestrator._batch_injections] == [0, 1]


class TestWhatIsReported:
    """Metadata must not describe a continuous wave in terms that only fit a transient."""

    def test_remaining_events_are_reported_as_unknown(self, tmp_path):
        """Nothing is ever consumed, so a count of what is left would read as the whole catalogue
        in the final frame as much as the first -- true, and misleading."""
        orchestrator = _orchestrator(tmp_path, duration=8, total=16, n_pulsars=2)

        orchestration_metadata = orchestrator.metadata["orchestration"]

        assert orchestration_metadata["population_events_total"] == 2
        assert orchestration_metadata["population_events_remaining"] is None

    def test_a_transient_still_counts_down(self, tmp_path):
        """The null is for continuous waves only; removing that gate must break something."""
        config = _config(tmp_path, duration=8, total=16, n_pulsars=2)
        orchestrator = AdapterOrchestrator.from_config(
            Config.model_validate(config).orchestration,
            global_simulator_arguments=dict(Config.model_validate(config).globals.simulator_arguments),
        )
        orchestrator._source_type = "bbh"

        assert orchestrator.metadata["orchestration"]["population_events_remaining"] == 2


class TestUnsupportedCombinations:
    """What the branch refuses, and why refusing beats quietly substituting."""

    def test_the_batched_execution_mode_is_refused(self, tmp_path):
        """Otherwise a run asking for `batched` gets the per-source loop, with nothing saying so."""
        config = _config(tmp_path, duration=8, total=8)
        config["orchestration"]["signal"]["execution"] = "batched"
        parsed = Config.model_validate(config)
        orchestrator = AdapterOrchestrator.from_config(
            parsed.orchestration, global_simulator_arguments=dict(parsed.globals.simulator_arguments)
        )

        with pytest.raises(ValueError, match="execution: batched is not available for continuous waves"):
            orchestrator._simulate()


class TestTheOrderingExemption:
    """Skipping the `coa_time` sort is for continuous waves only, and that gate is load-bearing."""

    def test_a_pulsar_catalogue_without_coa_time_is_accepted(self, tmp_path):
        """The default ordering key means nothing here, so its absence must not be an error."""
        orchestrator = _orchestrator(tmp_path, duration=8, total=8, n_pulsars=2)

        assert len(orchestrator._population_events) == 2

    def test_the_loaded_order_reaches_the_backend_untouched(self, tmp_path):
        """No sort applied means the events reach the backend in the order they were loaded.

        Checked against the loaded order, not the CSV: ``FilePopulationLoader`` reverses the rows,
        so a literal ``[3.0, 5.0]`` here would be asserting that quirk and would break if the
        loader stopped doing it, while a reordering bug in the orchestrator went unnoticed.
        """
        orchestrator = _orchestrator(tmp_path, duration=8, total=8, n_pulsars=2)
        loaded = [float(event["amplitude_plus"]) for event in orchestrator._population_events]

        _segment(orchestrator)

        seen = [float(call["parameters"]["amplitude_plus"]) for call in FakeContinuousWaveBackend.calls]
        assert seen == loaded
        # Guards the guard: if both were sorted the comparison above would hold vacuously.
        assert loaded != sorted(loaded), "the fixture no longer distinguishes loaded order from sorted order"

    def test_a_compact_binary_catalogue_without_coa_time_still_raises(self, tmp_path):
        """Without the source-type gate this regresses into silent, plausible corruption.

        The per-event loop breaks on ``coa_time >= end_time``; when the key is absent that test is
        never true, so every event lands in the first segment. A `bbh` catalogue that lost the
        column -- a header typo, a dropped field -- would produce a full-looking run with every
        coalescence time wrong. It used to raise, and must keep raising.
        """
        catalogue = tmp_path / "no_coa_time.csv"
        catalogue.write_text("detector_frame_mass_1,detector_frame_mass_2,luminosity_distance\n30.0,25.0,400.0\n")
        config = _config(tmp_path, duration=8, total=8, n_pulsars=1)
        config["orchestration"]["population"].update(
            {"source-type": "bbh", "n-samples": 1, "arguments": {"path": str(catalogue)}}
        )
        config["orchestration"]["signal"]["source-type"] = "bbh"
        parsed = Config.model_validate(config)

        with pytest.raises(ValueError, match="ordering key 'coa_time' is missing"):
            AdapterOrchestrator.from_config(
                parsed.orchestration, global_simulator_arguments=dict(parsed.globals.simulator_arguments)
            )

    def test_an_explicit_ordering_key_that_is_missing_still_raises(self, tmp_path):
        """The exemption is for the *default* key. Asking for one that is absent is a mistake."""
        config = _config(tmp_path, duration=8, total=8, n_pulsars=2)
        config["orchestration"]["population"]["sort-by"] = "coa_time"
        parsed = Config.model_validate(config)

        with pytest.raises(ValueError, match="ordering key 'coa_time' is missing"):
            AdapterOrchestrator.from_config(
                parsed.orchestration, global_simulator_arguments=dict(parsed.globals.simulator_arguments)
            )


class TestWhatTheBackendIsGiven:
    """The two constructor-level guarantees the design depends on."""

    def test_the_reference_epoch_is_the_same_for_every_segment(self, tmp_path):
        """A per-segment epoch resets the phase at each boundary while every frame looks normal.

        The backend cannot detect that itself -- it is handed one value and trusts it -- so the
        check belongs here, on what the orchestrator passes down across segments.
        """
        orchestrator = _orchestrator(tmp_path, duration=8, total=16, n_pulsars=2)

        _segment(orchestrator)
        orchestrator.update_state()
        _segment(orchestrator)

        epochs = {call["reference_time_ssb"] for call in FakeContinuousWaveBackend.calls}
        assert epochs == {_EPOCH}, f"the reference epoch varied across the run: {sorted(epochs)}"

    def test_earth_rotation_reaches_the_backend(self, tmp_path):
        """The simulator refuses a constant antenna pattern, which needs the flag to arrive."""
        _segment(_orchestrator(tmp_path, duration=8, total=8, n_pulsars=1))

        assert [call["earth_rotation"] for call in FakeContinuousWaveBackend.calls] == [True]
