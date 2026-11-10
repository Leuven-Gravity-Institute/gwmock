"""``orchestration.duty-cycle``: the state vector, the segments and the observing flags a run records.

The duty-cycle model and its anchors are gwmock-noise's. What is pinned here is what a run does with
it: the configuration validates at load time, one realization covers the whole run, every frame's
state vector is that realization at the frame's samples, the segment files hold it, every injected
signal and glitch is flagged by it, and ``gwmock merge --copy-channel`` carries the state vector into
the merged file unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from gwmock_noise.duty_cycle import MAINTENANCE_BIT, OBSERVING_BIT, DutyCycleModel, MaintenanceSchedule

from gwmock.cli.duty_cycle import SEGMENTS_FILE, STATE_VECTOR_DTYPE, DutyCycle
from gwmock.cli.utils.config import DutyCycleConfig, OrchestrationConfig

try:
    from gwpy.timeseries import TimeSeries
except ModuleNotFoundError as exc:  # Python 3.15: GWpy does not install yet; its tests are skipped
    if exc.name != "gwpy":
        raise
    TimeSeries = None

needs_gwpy = pytest.mark.skipif(TimeSeries is None, reason="GWpy is not installed")

pytestmark = pytest.mark.unit

START = 1_000_000_000.0
TRIANGLE = ("ET1_EMR", "ET2_EMR", "ET3_EMR")


def _duty_cycle(**kwargs) -> DutyCycle:
    """A duty cycle over three days, with maintenance windows inside it."""
    schedule = MaintenanceSchedule(strategy="coincident", period=8192.0, duration=1024.0, offset=0.0)
    defaults = {
        "model": DutyCycleModel(maintenance=schedule),
        "detectors": TRIANGLE,
        "start": START,
        "span": 3 * 86400.0,
        "seed": 3,
        "sampling_frequency": 16.0,
    }
    return DutyCycle(**{**defaults, **kwargs})


# ---------------------------------------------------------------------------------------
# The configuration
# ---------------------------------------------------------------------------------------


def test_an_unset_field_takes_gwmock_noises_default() -> None:
    """The defaults are defined once, in gwmock-noise."""
    assert DutyCycleConfig().to_model() == DutyCycleModel()
    model = DutyCycleConfig.model_validate({"mean-uptime": 3600.0, "maintenance": {"strategy": "none"}}).to_model()
    assert model.mean_uptime == 3600.0
    assert model.mean_downtime == DutyCycleModel().mean_downtime
    assert model.maintenance.strategy == "none"


@pytest.mark.parametrize(
    "fields",
    [
        {"mean-uptime": -1.0},
        {"correlated-downtime-fraction": 1.0},
        {"maintenance": {"strategy": "staggered"}},
        {"maintenance": {"period": 100.0, "duration": 200.0}},
        {"sampling-frequency": 0.0},
        {"unknown": 1},
        {"maintenance": {"unknown": 1}},
    ],
)
def test_an_invalid_duty_cycle_is_refused_at_load_time(fields: dict) -> None:
    """gwmock-noise's own checks, and unknown keys, fail when the configuration is read."""
    with pytest.raises(ValueError, match="validation error"):
        DutyCycleConfig.model_validate(fields)


def test_a_duty_cycle_needs_interferometers() -> None:
    """Without noise or a signal there is no network to draw it for."""
    with pytest.raises(ValueError, match="needs the interferometers"):
        OrchestrationConfig.model_validate({"population": {"backend": "x"}, "duty-cycle": {}})


# ---------------------------------------------------------------------------------------
# The realization and its products
# ---------------------------------------------------------------------------------------


def test_the_state_vector_is_the_realization_at_the_frames_samples() -> None:
    """Both bits, at every sample, as uint32."""
    duty_cycle = _duty_cycle()
    t0, duration = START + 4096.0, 8192.0
    state = duty_cycle.state_vector("ET1_EMR", t0, duration)
    assert state.dtype == STATE_VECTOR_DTYPE
    times = t0 + np.arange(state.size) / 16.0
    np.testing.assert_array_equal((state & OBSERVING_BIT) > 0, duty_cycle.realization.is_observing("ET1_EMR", times))
    assert np.any(state & MAINTENANCE_BIT)
    assert not np.any((state & OBSERVING_BIT) & (state & MAINTENANCE_BIT))


def test_a_frame_must_hold_a_whole_number_of_samples() -> None:
    """At 16 Hz a frame of 1.03 s would end mid-sample."""
    with pytest.raises(ValueError, match="whole number"):
        _duty_cycle().state_vector("ET1_EMR", START, 1.03)


def test_the_realization_is_one_draw_for_the_whole_span() -> None:
    """The same seed gives the same realization, so a resumed run sees the same duty cycle."""
    first, second = _duty_cycle(), _duty_cycle()
    for detector in TRIANGLE:
        np.testing.assert_array_equal(first.realization.observing[detector], second.realization.observing[detector])
    other = _duty_cycle(seed=4)
    assert any(not np.array_equal(first.realization.observing[d], other.realization.observing[d]) for d in TRIANGLE)


def test_the_segment_files_hold_the_realization(tmp_path: Path) -> None:
    """One JSON file with every interferometer, and one text list of observing intervals each."""
    duty_cycle = _duty_cycle()
    paths = duty_cycle.write_segment_files(tmp_path)
    assert [path.name for path in paths] == [SEGMENTS_FILE] + [f"{d}-observing-segments.txt" for d in TRIANGLE]
    record = json.loads(paths[0].read_text(encoding="utf-8"))
    assert record["schema"] == "gwmock-duty-cycle-segments"
    assert record["span"] == [START, START + 3 * 86400.0]
    for detector, text_path in zip(TRIANGLE, paths[1:], strict=True):
        observing = duty_cycle.realization.observing[detector]
        np.testing.assert_array_equal(record["detectors"][detector]["observing"], observing)
        np.testing.assert_array_equal(
            record["detectors"][detector]["maintenance"], duty_cycle.realization.maintenance[detector]
        )
        rows = np.loadtxt(text_path, ndmin=2)
        np.testing.assert_array_equal(rows, observing)


@needs_gwpy
@pytest.mark.parametrize("output_format", ["gwf", "hdf5"])
def test_a_written_state_vector_reads_back_unchanged(tmp_path: Path, output_format: str) -> None:
    """Values, dtype, rate and start time survive the file."""
    duty_cycle = _duty_cycle()
    paths = {detector: tmp_path / f"{detector}.{output_format}" for detector in TRIANGLE}
    channels = {detector: f"{detector}:STATE_VECTOR" for detector in TRIANGLE}
    duty_cycle.write_frame(paths, channels, START + 2048.0, 2048.0, output_format)
    for detector in TRIANGLE:
        series = TimeSeries.read(str(paths[detector]), channels[detector])
        assert series.dtype == STATE_VECTOR_DTYPE
        assert float(series.sample_rate.value) == 16.0
        assert float(series.t0.value) == START + 2048.0
        np.testing.assert_array_equal(series.value, duty_cycle.state_vector(detector, START + 2048.0, 2048.0))


# ---------------------------------------------------------------------------------------
# A run
# ---------------------------------------------------------------------------------------


def _config(tmp_path: Path, *, noise_seed: int | None = 7, duty_cycle: dict | None = None) -> Path:
    arguments: dict = {
        "psd_file": "ET2026a_10km_psd",
        "low_frequency_cutoff": 5.0,
        "detectors": ["ET-Triangle-EMR"],
        "glitches": [
            {
                "kind": "blip",
                "rate": 0.01,
                "width": 0.01,
                "psd_file": "ET2026a_10km_psd",
                "snr": 20.0,
                "amplitude_distribution": {"distribution": "lognormal", "mean": 1.0, "std": 0.0},
            }
        ],
    }
    if noise_seed is not None:
        arguments["seed"] = noise_seed
    config = {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": 512,
                "duration": 4096,
                "total-duration": 12288,
                "start-time": int(START),
            },
            "working-directory": str(tmp_path),
            "output-directory": "output",
            "metadata-directory": "metadata",
        },
        "orchestration": {
            "noise": {
                "arguments": arguments,
                "output": {
                    "output_directory": "noise",
                    "file_name": "E-{{ detectors }}_STRAIN-{{ start_time }}-{{ duration }}.hdf5",
                    "arguments": {"channel": "STRAIN"},
                },
            },
            "duty-cycle": duty_cycle
            if duty_cycle is not None
            else {"maintenance": {"strategy": "coincident", "period": 8192, "duration": 1024, "offset": 0}},
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _simulate(config: Path) -> None:
    from gwmock.cli.simulate import simulate_command

    cwd = Path.cwd()
    os.chdir(config.parent)
    try:
        simulate_command([str(config)])
    finally:
        os.chdir(cwd)


def test_a_run_writes_state_vectors_segments_and_flags(tmp_path: Path) -> None:
    """Every frame's state vector, the segments once, and a flag on every glitch, all from one draw."""
    _simulate(_config(tmp_path))
    from gwmock_data.io import read_hdf5

    duty = tmp_path / "output" / "duty-cycle"
    record = json.loads((duty / SEGMENTS_FILE).read_text(encoding="utf-8"))
    assert record["span"] == [START, START + 12288.0]
    observing = {detector: np.asarray(block["observing"]) for detector, block in record["detectors"].items()}
    assert sorted(observing) == sorted(TRIANGLE)

    def observed(detector: str, times: np.ndarray) -> np.ndarray:
        segments = observing[detector].reshape(-1, 2)
        return np.array([bool(((segments[:, 0] <= t) & (t < segments[:, 1])).any()) for t in times])

    records = sorted((tmp_path / "metadata").glob("*.metadata.json"))
    assert len(records) == 3
    flagged = 0
    for index, path in enumerate(records):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        kinds = [output["kind"] for output in metadata["outputs"]]
        assert kinds.count("duty-cycle") == 3
        assert kinds.count("duty-cycle-segments") == (4 if index == 0 else 0)
        t0 = START + 4096.0 * index
        for detector in TRIANGLE:
            state_file = duty / f"duty-cycle-{detector}-{index}.hdf5"
            series = read_hdf5(state_file)[f"{detector}:STATE_VECTOR"]
            assert series.t0 == t0
            times = t0 + np.arange(series.data.size) / 16.0
            np.testing.assert_array_equal((np.asarray(series.data) & OBSERVING_BIT) > 0, observed(detector, times))
        for glitch in metadata["noise"]["glitch_injections"]:
            assert glitch["observing"] == observed(glitch["detector"], [glitch["gps_peak_time"]])[0]
            flagged += 1
    assert flagged > 0


def test_a_run_without_a_seed_is_refused(tmp_path: Path) -> None:
    """An unseeded duty cycle would change when the run is resumed."""
    import typer

    from gwmock.utils.errors import DeterministicRefusalError

    with pytest.raises(typer.Exit) as excinfo:
        _simulate(_config(tmp_path, noise_seed=None))
    assert isinstance(excinfo.value.__cause__, DeterministicRefusalError)
    assert "needs a seed" in str(excinfo.value.__cause__)
    assert not (tmp_path / "output" / "duty-cycle").exists()


def test_an_injected_signal_is_flagged_at_its_merger_time(tmp_path: Path) -> None:
    """Rows with a merger time get one flag per interferometer; rows without one are left as they are."""
    from gwmock.cli.adapter_orchestration import AdapterOrchestrator
    from gwmock.cli.utils.config import load_config

    config = load_config(_config(tmp_path))
    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)
    duty_cycle = orchestrator.duty_cycle
    assert duty_cycle is not None
    segments = duty_cycle.realization.observing["ET1_EMR"]
    inside, outside = segments[0, 0] + 1.0, segments[0, 1] + 1e-6
    rows = [
        {"event_id": 0, "parameters": {"coa_time": inside}},
        {"event_id": 1, "parameters": {"coa_time": outside}},
        {"event_id": 2, "parameters": {"frequency": 100.0}},
    ]
    flagged = orchestrator._flag_observing_injections(rows)
    assert flagged[0]["observing"]["ET1_EMR"] is True
    assert flagged[1]["observing"]["ET1_EMR"] is duty_cycle.observing("ET1_EMR", outside)
    assert set(flagged[0]["observing"]) == set(TRIANGLE)
    assert "observing" not in flagged[2]


def _noise_files(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "output" / "noise").glob("*")) if (tmp_path / "output" / "noise").exists() else []


@pytest.mark.parametrize(
    ("duty_cycle", "message"),
    [
        # Three interferometers maintained one after another need 3 x 1024 s within 2048 s.
        ({"maintenance": {"strategy": "rotating", "period": 2048, "duration": 1024}}, "rotating schedule"),
        # 4096 s at 0.3 Hz is 1228.8 samples.
        ({"sampling-frequency": 0.3}, "whole number"),
    ],
)
def test_a_duty_cycle_the_network_or_frame_cannot_take_is_refused_before_anything_is_written(
    tmp_path: Path, duty_cycle: dict, message: str
) -> None:
    """Refused once, with its own message, and not retried into a complaint about the noise files."""
    import typer

    from gwmock.utils.errors import DeterministicRefusalError

    with pytest.raises(typer.Exit) as excinfo:
        _simulate(_config(tmp_path, duty_cycle=duty_cycle))
    assert isinstance(excinfo.value.__cause__, DeterministicRefusalError)
    assert message in str(excinfo.value.__cause__)
    assert _noise_files(tmp_path) == []


def test_a_second_run_into_the_same_directory_is_refused_without_overwrite(tmp_path: Path) -> None:
    """The state vectors follow the rule the noise files follow, and are checked first."""
    import typer

    from gwmock.cli.simulate import simulate_command

    config = _config(tmp_path)
    _simulate(config)
    for path in _noise_files(tmp_path):
        path.unlink()
    with pytest.raises(typer.Exit) as excinfo:
        _simulate(config)
    assert isinstance(excinfo.value.__cause__, FileExistsError)
    assert "Duty-cycle output(s) already exist" in str(excinfo.value.__cause__)
    assert _noise_files(tmp_path) == []

    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        simulate_command([str(config)], overwrite=True)
    finally:
        os.chdir(cwd)


def test_a_file_name_that_does_not_vary_per_frame_is_refused(tmp_path: Path) -> None:
    """Without it the second frame would replace the first frame's state vector."""
    import typer

    duty_cycle = {"output": {"file_name": "state-{{ detectors }}.hdf5", "output_directory": "duty-cycle"}}
    with pytest.raises(typer.Exit) as excinfo:
        _simulate(_config(tmp_path, duty_cycle=duty_cycle))
    assert isinstance(excinfo.value.__cause__, FileExistsError)


def test_a_segment_file_from_another_run_is_refused_without_overwrite(tmp_path: Path) -> None:
    """An existing segment file is kept only if it holds this run's realization."""
    import typer

    duty = tmp_path / "output" / "duty-cycle"
    duty.mkdir(parents=True)
    (duty / SEGMENTS_FILE).write_text('{"seed": 1}\n', encoding="utf-8")
    with pytest.raises(typer.Exit) as excinfo:
        _simulate(_config(tmp_path))
    assert SEGMENTS_FILE in str(excinfo.value.__cause__)
    assert (duty / SEGMENTS_FILE).read_text(encoding="utf-8") == '{"seed": 1}\n'


def test_the_channel_is_expanded_like_the_noise_channel(tmp_path: Path) -> None:
    """``{{ detectors }}:STATE_VECTOR`` names each interferometer's channel once."""
    from gwmock_data.io import read_hdf5

    duty_cycle = {
        "output": {
            "file_name": "state-{{ detectors }}-{{ counter }}.hdf5",
            "output_directory": "duty-cycle",
            "arguments": {"channel": "{{ detectors }}:STATE_VECTOR"},
        }
    }
    _simulate(_config(tmp_path, duty_cycle=duty_cycle))
    for detector in TRIANGLE:
        assert list(read_hdf5(tmp_path / "output" / "duty-cycle" / f"state-{detector}-0.hdf5")) == [
            f"{detector}:STATE_VECTOR"
        ]


def test_an_unknown_output_argument_is_refused(tmp_path: Path) -> None:
    import typer

    duty_cycle = {
        "output": {
            "file_name": "state-{{ detectors }}-{{ counter }}.hdf5",
            "arguments": {"channel": "STATE_VECTOR", "gps_start": 123.0},
        }
    }
    with pytest.raises(typer.Exit) as excinfo:
        _simulate(_config(tmp_path, duty_cycle=duty_cycle))
    assert "Unsupported keys: gps_start" in str(excinfo.value.__cause__)
    assert _noise_files(tmp_path) == []


def test_a_signal_only_run_draws_the_duty_cycle_for_the_signals_interferometers(tmp_path: Path, monkeypatch) -> None:
    """With no noise, the signal's interferometers; a background has no merger time, so no flags."""
    from gwmock.cli.simulate import _simulate_impl
    from gwmock.cli.utils.config import DutyCycleConfig
    from tests.cli.test_cli_orchestration import _fake_sgwb_config, _write_signal_file

    config = _fake_sgwb_config(tmp_path)
    config.orchestration.duty_cycle = DutyCycleConfig.model_validate({"seed": 5})
    config_file = tmp_path / "sgwb.yaml"
    config_file.write_text(yaml.safe_dump(config.model_dump(by_alias=True, exclude_none=True), sort_keys=False))
    monkeypatch.setattr("gwmock.cli.adapter_orchestration.DetectorStrainStack.write", _write_signal_file)

    _simulate_impl(str(config_file), overwrite=True, metadata=True)

    record = json.loads((tmp_path / "output" / "duty-cycle" / SEGMENTS_FILE).read_text(encoding="utf-8"))
    assert sorted(record["detectors"]) == ["H1", "L1"]
    metadata = yaml.safe_load((tmp_path / "metadata" / "orchestration-0.metadata.json").read_text())
    assert metadata["simulator_metadata"]["orchestration"]["duty_cycle"]["detectors"] == ["H1", "L1"]
    assert all(
        "observing" not in row for row in metadata["simulator_metadata"]["orchestration"]["signal"]["injections"]
    )
    assert [o["kind"] for o in metadata["outputs"]].count("duty-cycle") == 2
