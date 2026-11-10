"""``orchestration.duty-cycle``: the state vector, the segments and the observing flags a run records.

The duty-cycle model and its anchors are gwmock-noise's. What is pinned here is what a run does with
it: the configuration validates at load time, one realization covers the whole run, every frame's
state vector is that realization at the frame's samples, the segment files hold it, every injected
signal and glitch is flagged by it, and ``gwmock merge --copy-channel`` carries the state vector into
the merged file unchanged.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from gwmock_noise.duty_cycle import MAINTENANCE_BIT, OBSERVING_MASK, DutyCycleModel, MaintenanceSchedule

from gwmock.cli.duty_cycle import SEGMENTS_FILE, SEGMENTS_XML_FILE, STATE_VECTOR_DTYPE, DutyCycle
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
#: A maintenance window every 8192 s, so a three-frame run has some.
_MAINTENANCE = {"strategy": "coincident", "period": 8192, "duration": 1024, "offset": 0}
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
    """The bits at every sample, as uint32; a LIGO-style mask of bits 0 and 1 selects observing time."""
    duty_cycle = _duty_cycle()
    t0, duration = START + 4096.0, 8192.0
    state = duty_cycle.state_vector("ET1_EMR", t0, duration)
    assert state.dtype == STATE_VECTOR_DTYPE
    times = t0 + np.arange(state.size) / 16.0
    science = _whole_seconds(duty_cycle.realization.observing["ET1_EMR"])
    np.testing.assert_array_equal((state & 0b11) == 0b11, _observed(science, times))
    assert set(np.unique(state).tolist()) <= {0, OBSERVING_MASK, MAINTENANCE_BIT}
    assert (state == MAINTENANCE_BIT).any()
    assert (state == OBSERVING_MASK).any()


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
    """One JSON file with every interferometer, the XML file, and one text list of observing intervals each."""
    duty_cycle = _duty_cycle()
    paths = duty_cycle.write_segment_files(tmp_path)
    assert [path.name for path in paths] == [SEGMENTS_FILE, SEGMENTS_XML_FILE] + [
        f"{d}-observing-segments.txt" for d in TRIANGLE
    ]
    record = json.loads(paths[0].read_text(encoding="utf-8"))
    assert record["schema"] == "gwmock-duty-cycle-segments"
    assert record["span"] == [START, START + 3 * 86400.0]
    for detector, text_path in zip(TRIANGLE, paths[2:], strict=True):
        observing = duty_cycle.realization.observing[detector]
        science = _whole_seconds(observing)
        np.testing.assert_array_equal(record["detectors"][detector]["observing"], observing)
        np.testing.assert_array_equal(np.reshape(record["detectors"][detector]["science"], (-1, 2)), science)
        np.testing.assert_array_equal(
            record["detectors"][detector]["maintenance"], duty_cycle.realization.maintenance[detector]
        )
        assert text_path.read_text().splitlines() == [f"{int(a)} {int(b)}" for a, b in science]
        np.testing.assert_array_equal(np.loadtxt(text_path, ndmin=2).reshape(-1, 2), science)


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
            "duty-cycle": duty_cycle if duty_cycle is not None else {"maintenance": _MAINTENANCE},
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
    observing = {detector: _whole_seconds(block["observing"]) for detector, block in record["detectors"].items()}
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
        assert kinds.count("duty-cycle-segments") == (5 if index == 0 else 0)
        t0 = START + 4096.0 * index
        for detector in TRIANGLE:
            state_file = duty / f"duty-cycle-{detector}-{index}.hdf5"
            series = read_hdf5(state_file)[f"{detector}:STATE_VECTOR"]
            assert series.t0 == t0
            times = t0 + np.arange(series.data.size) / 16.0
            np.testing.assert_array_equal((np.asarray(series.data) & 0b11) == 0b11, observed(detector, times))
            strain = read_hdf5(tmp_path / "output" / "noise" / f"E-{detector}_STRAIN-{int(t0)}-4096.hdf5")
            strain = np.asarray(strain[f"{detector}:STRAIN"].data)
            strain_times = t0 + np.arange(strain.size) / 512.0
            np.testing.assert_array_equal(np.isnan(strain), ~_observed(observing[detector], strain_times))
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
    raw = duty_cycle.realization.observing["ET1_EMR"]
    # The first observing interval with a fractional start: a time in its first fraction of a second is
    # observing but outside the science segment, so it is flagged as not analysable.
    edge = next(row for row in raw if row[0] % 1 != 0 and math.floor(row[1]) > math.ceil(row[0]))
    inside, fraction, end = math.ceil(edge[0]) + 0.5, (edge[0] + math.ceil(edge[0])) / 2, math.floor(edge[1])
    rows = [
        {"event_id": 0, "parameters": {"coa_time": inside}},
        {"event_id": 1, "parameters": {"coa_time": fraction}},
        {"event_id": 2, "parameters": {"coa_time": end}},
        {"event_id": 3, "parameters": {"frequency": 100.0}},
    ]
    flagged = orchestrator._flag_observing_injections(rows)
    assert duty_cycle.realization.is_observing("ET1_EMR", np.array([fraction]))[0]
    assert flagged[0]["observing"]["ET1_EMR"] is True
    assert flagged[1]["observing"]["ET1_EMR"] is False
    assert flagged[2]["observing"]["ET1_EMR"] is False  # a segment is [start, end)
    assert set(flagged[0]["observing"]) == set(TRIANGLE)
    assert "observing" not in flagged[3]


def _whole_seconds(intervals) -> np.ndarray:
    """The whole GPS seconds inside each interval: the science segments, stated independently of gwmock."""
    rows = [(math.ceil(start), math.floor(end)) for start, end in np.asarray(intervals, float).reshape(-1, 2)]
    return np.array([row for row in rows if row[1] > row[0]], dtype=float).reshape(-1, 2)


def _observed(intervals, times: np.ndarray) -> np.ndarray:
    """Whether each time lies in one of the ``[start, end)`` intervals, read independently of gwmock."""
    segments = np.asarray(intervals, dtype=float).reshape(-1, 2)
    index = np.searchsorted(segments[:, 0], times, side="right") - 1
    inside = index >= 0
    inside[inside] = times[inside] < segments[index[inside], 1]
    return inside


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


@pytest.mark.parametrize(
    "file_name", ["state-{{ detectors }}-{{ counter }}.npy", ["a-{{ counter }}.gwf", "b-{{ counter }}.hdf5", "c.hdf5"]]
)
def test_a_state_vector_format_gwmock_cannot_write_is_refused_before_anything_is_written(
    tmp_path: Path, file_name
) -> None:
    import typer

    from gwmock.utils.errors import DeterministicRefusalError

    with pytest.raises(typer.Exit) as excinfo:
        _simulate(_config(tmp_path, duty_cycle={"output": {"file_name": file_name}}))
    assert isinstance(excinfo.value.__cause__, DeterministicRefusalError)
    assert "must all end with .gwf" in str(excinfo.value.__cause__)
    assert _noise_files(tmp_path) == []


def test_a_replay_from_metadata_writes_each_frames_own_state_vectors(tmp_path: Path, monkeypatch) -> None:
    """Each batch's record names its own state-vector files; a replay writes those, not the first batch's."""
    from gwmock.cli.simulate import _simulate_impl

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _simulate(_config(run_dir))
    records = sorted((run_dir / "metadata").glob("orchestration-*.metadata.json"))
    assert len(records) == 3

    repro_dir = tmp_path / "repro"
    repro_dir.mkdir()
    for record in records:
        (repro_dir / record.name).write_text(record.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.chdir(repro_dir)
    _simulate_impl([str(repro_dir / record.name) for record in records], metadata=False)

    for index in range(3):
        for detector in TRIANGLE:
            name = f"duty-cycle-{detector}-{index}.hdf5"
            original = run_dir / "output" / "duty-cycle" / name
            replayed = repro_dir / "output" / "duty-cycle" / name
            assert replayed.read_bytes() == original.read_bytes() or _same_state(original, replayed, detector)


def _same_state(first: Path, second: Path, detector: str) -> bool:
    from gwmock_data.io import read_hdf5

    a = read_hdf5(first)[f"{detector}:STATE_VECTOR"]
    b = read_hdf5(second)[f"{detector}:STATE_VECTOR"]
    return a.t0 == b.t0 and np.array_equal(np.asarray(a.data), np.asarray(b.data))


def _ligolw_flags(path: Path) -> dict[tuple[str, str], dict[str, list[tuple[float, float]]]]:
    """The flags of a LIGO_LW XML segment file, read with the standard library only."""
    import xml.etree.ElementTree as ET

    tables = {}
    for table in ET.parse(path).getroot().iter("Table"):  # noqa: S314 -- a file this test just wrote
        columns = [column.get("Name") for column in table.iter("Column")]
        text = table.find("Stream").text.strip()
        rows = [] if not text else [line.strip().rstrip(",") for line in text.splitlines()]
        tables[table.get("Name")] = [dict(zip(columns, row.split(","), strict=True)) for row in rows]

    def time(row: dict, prefix: str) -> float:
        return int(row[f"{prefix}_time"]) + int(row[f"{prefix}_time_ns"]) * 1e-9

    flags = {}
    for row in tables["segment_definer:table"]:
        key = (row["ifos"].strip('"'), row["name"].strip('"'))
        flags[row["segment_def_id"]] = {"key": key, "known": [], "active": []}
    for row in tables["segment_summary:table"]:
        flags[row["segment_definer:segment_def_id"]]["known"].append((time(row, "start"), time(row, "end")))
    for row in tables["segment:table"]:
        flags[row["segment_definer:segment_def_id"]]["active"].append((time(row, "start"), time(row, "end")))
    return {flag["key"]: {"known": flag["known"], "active": flag["active"]} for flag in flags.values()}


@pytest.mark.parametrize("start", [START, START + 0.25])
def test_the_xml_segment_file_holds_each_interferometers_science_and_maintenance_flags(
    tmp_path: Path, start: float
) -> None:
    """``<ifo>:science`` is what PyCBC's offline search reads, in whole seconds; maintenance as scheduled."""
    duty_cycle = _duty_cycle(start=start)
    duty_cycle.write_segment_files(tmp_path)
    flags = _ligolw_flags(tmp_path / SEGMENTS_XML_FILE)
    assert sorted(flags) == sorted((d, name) for d in TRIANGLE for name in ("science", "maintenance"))
    realization = duty_cycle.realization
    for detector in TRIANGLE:
        science = {d: _whole_seconds(realization.observing[d]) for d in TRIANGLE}
        for name, intervals in (("science", science), ("maintenance", realization.maintenance)):
            flag = flags[(detector, name)]
            np.testing.assert_allclose(flag["known"], [(realization.start, realization.end)], rtol=0, atol=1e-9)
            assert len(flag["active"]) == len(intervals[detector])
            np.testing.assert_allclose(
                np.asarray(flag["active"]).reshape(-1, 2), intervals[detector], rtol=0, atol=1e-9
            )


def test_strain_is_written_in_full_when_masking_is_off(tmp_path: Path) -> None:
    from gwmock_data.io import read_hdf5

    _simulate(_config(tmp_path, duty_cycle={"mask-strain": False, "maintenance": {"strategy": "coincident"}}))
    for path in sorted((tmp_path / "output" / "noise").glob("*.hdf5")):
        for series in read_hdf5(path).values():
            assert np.isfinite(np.asarray(series.data)).all()
    metadata = json.loads((tmp_path / "metadata" / "orchestration-0.metadata.json").read_text(encoding="utf-8"))
    assert metadata["simulator_metadata"]["orchestration"]["duty_cycle"]["strain_masked"] is False


def test_signal_strain_is_masked_where_its_interferometer_does_not_observe(tmp_path: Path) -> None:
    """The signal writer masks too, so a signal-only run, and the signal half of a merge, carry the NaN."""
    from gwmock.cli.adapter_orchestration import AdapterOrchestrator
    from gwmock.cli.utils.config import DutyCycleConfig
    from gwmock.data import TimeSeries as MultichannelTimeSeries
    from tests.cli.test_cli_orchestration import _fake_sgwb_config

    config = _fake_sgwb_config(tmp_path)
    config.orchestration.duty_cycle = DutyCycleConfig.model_validate(
        {"seed": 5, "mean-uptime": 1.5, "mean-downtime": 1.5, "maintenance": {"strategy": "none"}}
    )
    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)
    start, rate = float(orchestrator.duty_cycle.start), 8.0
    n = int(orchestrator.duty_cycle.span * rate)
    data = MultichannelTimeSeries(np.ones((2, n)), start_time=start, sampling_frequency=rate)
    stack = orchestrator._build_signal_stack(data=data, channel_names=[None, None])
    times = start + np.arange(n) / rate
    for name, series in stack.to_dict().items():
        values = np.asarray(getattr(series, "value", series))
        observing = _observed(_whole_seconds(orchestrator.duty_cycle.realization.observing[name]), times)
        assert observing.any()
        assert not observing.all()
        np.testing.assert_array_equal(np.isnan(values), ~observing)


def test_a_signal_interferometer_the_duty_cycle_does_not_cover_is_refused(tmp_path: Path) -> None:
    """Its observing time would be unknown, and its strain unmasked beside masked noise."""
    from gwmock.cli.adapter_orchestration import AdapterOrchestrator
    from gwmock.cli.utils.config import DutyCycleConfig
    from gwmock.utils.errors import DeterministicRefusalError
    from tests.cli.test_cli_orchestration import _fake_orchestration_config

    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    config.orchestration.signal.detectors = ["H1", "L1"]
    config.orchestration.duty_cycle = DutyCycleConfig.model_validate({"maintenance": {"strategy": "none"}})
    with pytest.raises(DeterministicRefusalError, match="also written for L1"):
        AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)


@pytest.mark.parametrize(("t0", "rate"), [(START, 16.0), (START + 0.3, 16.0), (START + 0.03125, 4.0)])
def test_a_state_valid_sample_never_covers_a_masked_strain_sample(t0: float, rate: float) -> None:
    """A pipeline that analyses only state-valid samples never reads NaN, whatever the frame's alignment."""
    duty_cycle = _duty_cycle()
    duration = 8192.0
    state = duty_cycle.state_vector("ET1_EMR", t0, duration)
    strain_rate = 512.0
    strain_times = t0 + np.arange(int(duration * strain_rate)) / strain_rate
    masked = np.isnan(duty_cycle.mask("ET1_EMR", np.ones(strain_times.size), strain_times))
    assert masked.any()
    covering = np.floor((strain_times - t0) * duty_cycle.sampling_frequency).astype(int)
    valid = (state & 0b11) == 0b11
    assert not (masked & valid[covering]).any()
    # And the state vector is not needlessly strict: a sample whose interval lies in a science segment is valid.
    science = _whole_seconds(duty_cycle.realization.observing["ET1_EMR"])
    starts = t0 + np.arange(state.size) / duty_cycle.sampling_frequency
    stops = t0 + np.arange(1, state.size + 1) / duty_cycle.sampling_frequency
    inside = np.array([((science[:, 0] <= a) & (b <= science[:, 1])).any() for a, b in zip(starts, stops, strict=True)])
    np.testing.assert_array_equal(valid, inside)


@pytest.mark.parametrize("suffix", ["gwf", "npy"])
def test_noise_in_every_format_is_masked_outside_the_science_segments(tmp_path: Path, suffix: str) -> None:
    if suffix == "gwf" and TimeSeries is None:
        pytest.skip("GWpy is not installed")
    path = _config(tmp_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    noise = config["orchestration"]["noise"]
    noise["output"]["file_name"] = "E-{{ detectors }}_STRAIN-{{ start_time }}-{{ duration }}." + suffix
    if suffix == "npy":
        noise["output"].pop("arguments")
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    _simulate(path)
    record = json.loads((tmp_path / "output" / "duty-cycle" / SEGMENTS_FILE).read_text(encoding="utf-8"))
    for index in range(3):
        t0 = START + 4096.0 * index
        for detector in TRIANGLE:
            file = tmp_path / "output" / "noise" / f"E-{detector}_STRAIN-{int(t0)}-4096.{suffix}"
            data = np.load(file) if suffix == "npy" else TimeSeries.read(str(file), f"{detector}:STRAIN").value
            times = t0 + np.arange(data.size) / 512.0
            science = np.reshape(record["detectors"][detector]["science"], (-1, 2))
            np.testing.assert_array_equal(np.isnan(data), ~_observed(science, times))


def test_masking_changes_nothing_but_the_masked_samples(tmp_path: Path) -> None:
    """With and without the mask, the noise written in the science segments is bit-identical."""
    from gwmock_data.io import read_hdf5

    masked_dir, full_dir = tmp_path / "masked", tmp_path / "full"
    masked_dir.mkdir()
    full_dir.mkdir()
    _simulate(_config(masked_dir))
    _simulate(_config(full_dir, duty_cycle={"mask-strain": False, "maintenance": _MAINTENANCE}))
    n_masked = 0
    for path in sorted((masked_dir / "output" / "noise").glob("*.hdf5")):
        for name, series in read_hdf5(path).items():
            masked = np.asarray(series.data)
            full = np.asarray(read_hdf5(full_dir / "output" / "noise" / path.name)[name].data)
            assert np.isfinite(full).all()
            finite = np.isfinite(masked)
            n_masked += int((~finite).sum())
            np.testing.assert_array_equal(masked[finite], full[finite])
    assert n_masked > 0


@pytest.mark.parametrize(
    ("time", "expected"),
    [
        (1_000_000_000.0, (1_000_000_000, 0)),
        (1_000_000_000.25, (1_000_000_000, 250_000_000)),
        (0.0, (0, 0)),
        (0.9999999995, (1, 0)),  # rounds up to the next second: the carry
    ],
)
def test_a_gps_time_splits_into_seconds_and_nanoseconds(time: float, expected: tuple[int, int]) -> None:
    from gwmock.cli.duty_cycle import _gps_split

    assert _gps_split(time) == expected


def test_a_run_past_the_segment_files_32_bit_range_is_refused(tmp_path: Path) -> None:
    import typer

    from gwmock.utils.errors import DeterministicRefusalError

    path = _config(tmp_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["globals"]["simulator-arguments"]["start-time"] = 2**31 - 4096
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(typer.Exit) as excinfo:
        _simulate(path)
    assert isinstance(excinfo.value.__cause__, DeterministicRefusalError)
    assert "32-bit" in str(excinfo.value.__cause__)


def test_the_xml_segment_file_reads_with_gwpy(tmp_path: Path) -> None:
    """The reader the file exists for, when it is installed (it needs igwn-ligolw and lal)."""
    pytest.importorskip("gwpy")
    pytest.importorskip("igwn_ligolw")
    pytest.importorskip("lal")
    from gwpy.segments import DataQualityDict

    duty_cycle = _duty_cycle(start=START + 0.25)
    duty_cycle.write_segment_files(tmp_path)
    flags = DataQualityDict.read(str(tmp_path / SEGMENTS_XML_FILE), format="ligolw")
    for detector in TRIANGLE:
        science = flags[f"{detector}:science:1"]
        np.testing.assert_array_equal(
            [(float(a), float(b)) for a, b in science.active],
            _whole_seconds(duty_cycle.realization.observing[detector]),
        )
        assert [(float(a), float(b)) for a, b in science.known] == [(START + 0.25, START + 0.25 + 3 * 86400.0)]
