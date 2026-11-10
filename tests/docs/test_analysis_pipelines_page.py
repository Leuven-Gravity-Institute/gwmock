"""The analysis-pipelines page's examples must work on a gwmock run, with each pipeline's current release.

The page tells a user how PyCBC, cWB, bilby and gwpy read a run with a duty cycle. Its examples are run
here, as written, against a run made the way the page describes, so a change in gwmock, in a pipeline,
or on the page is caught here rather than by a user.

PyCBC and bilby are heavy and are not in the ``dev`` group: those tests are marked ``pipelines``,
which the default run excludes, and the ``pipelines`` CI job installs them and runs them. With
``GWMOCK_REQUIRE_PIPELINES=1`` a missing pipeline fails a test instead of skipping it, so that job
cannot pass by skipping everything. cWB is not on PyPI; its test uses a reimplementation of how cWB
reads data quality files (``tests/docs/cwb.py``).
"""

from __future__ import annotations

import configparser
import importlib
import os
import re
import shlex
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.docs import cwb

_PAGE = Path(__file__).resolve().parents[2] / "docs" / "user-guide" / "analysis-pipelines.md"
_EXAMPLE = re.compile(r"<!-- example: (?P<name>[\w-]+) -->\s*```(?P<language>\w+)\n(?P<code>.*?)```", re.S)

START = 1_000_000_000
FRAME = 1024
N_FRAMES = 3
TRIANGLE = ("ET1_EMR", "ET2_EMR", "ET3_EMR")


def _require(module: str):
    """Import ``module``, or skip -- unless the run requires the pipelines, when it fails."""
    if os.environ.get("GWMOCK_REQUIRE_PIPELINES") == "1":
        return importlib.import_module(module)
    return pytest.importorskip(module)


def _example(name: str) -> tuple[str, str]:
    """The page's example called ``name``: its language and its code."""
    examples = {match["name"]: (match["language"], match["code"]) for match in _EXAMPLE.finditer(_PAGE.read_text())}
    assert name in examples, f"the page no longer has the {name!r} example"
    return examples[name]


def _run_example(name: str, directory: Path, monkeypatch) -> None:
    language, code = _example(name)
    assert language == "python"
    # The examples check their own results; a test that only ran them would pass once a check was deleted.
    assert "assert " in code, f"the {name!r} example no longer checks its result"
    monkeypatch.chdir(directory)
    exec(compile(code, f"{_PAGE.name}:{name}", "exec"), {"__name__": "__example__"})  # noqa: S102


def _make_run(tmp_path_factory, *, mask_strain: bool) -> Path:
    """A three-frame run of the ET triangle with a duty cycle, merged as the page's merge loop does."""
    _require("gwpy")
    from gwmock.cli.merge import merge_command
    from gwmock.cli.simulate import simulate_command

    directory = tmp_path_factory.mktemp("analysis-pipelines")
    config = {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": 256,
                "duration": FRAME,
                "total-duration": FRAME * N_FRAMES,
                "start-time": START,
            },
            "working-directory": ".",
            "output-directory": "output",
            "metadata-directory": "metadata",
        },
        "orchestration": {
            "noise": {
                "arguments": {
                    "seed": 11,
                    "psd_file": "ET2026a_10km_psd",
                    "low_frequency_cutoff": 5.0,
                    "detectors": ["ET-Triangle-EMR"],
                },
                "output": {
                    "output_directory": "noise",
                    "file_name": "E-{{ detectors }}_STRAIN-{{ start_time }}-{{ duration }}.gwf",
                    "arguments": {"channel": "{{ detectors }}:STRAIN"},
                },
            },
            "duty-cycle": {
                "mask-strain": mask_strain,
                "mean-uptime": 1200,
                "mean-downtime": 300,
                "maintenance": {"strategy": "coincident", "period": 2048, "duration": 100, "offset": 0},
                "output": {
                    "output_directory": "duty-cycle",
                    "file_name": "E-{{ detectors }}_STATE-{{ start_time }}-{{ duration }}.gwf",
                },
            },
        },
    }
    (directory / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    cwd = Path.cwd()
    os.chdir(directory)
    try:
        simulate_command(["config.yaml"])
        Path("merged").mkdir()
        for index in range(N_FRAMES):
            t0 = START + FRAME * index
            for ifo in TRIANGLE:
                merge_command(
                    [
                        Path(f"output/noise/E-{ifo}_STRAIN-{t0}-{FRAME}.gwf"),
                        Path(f"output/duty-cycle/E-{ifo}_STATE-{t0}-{FRAME}.gwf"),
                    ],
                    metadata=[f"metadata/orchestration-{index}.metadata.json"],
                    channel=f"{ifo}:STRAIN",
                    copy_channel=[f"{ifo}:STATE_VECTOR"],
                    output=f"merged/E-{ifo}_STRAIN-{t0}-{FRAME}.gwf",
                )
    finally:
        os.chdir(cwd)
    # The examples only test something if the run has gaps, and segments long enough to analyse.
    for ifo in TRIANGLE:
        segments = _science(directory, ifo)
        assert segments.shape[0] >= 2
        assert (segments[:, 1] - segments[:, 0]).max() >= 132
    return directory


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> Path:
    """The run with the strain masked outside the science segments, the default."""
    return _make_run(tmp_path_factory, mask_strain=True)


@pytest.fixture(scope="module")
def unmasked_run(tmp_path_factory) -> Path:
    """The same run with the strain in full, as the page says PyCBC's low-latency search needs."""
    return _make_run(tmp_path_factory, mask_strain=False)


def _science(directory: Path, ifo: str) -> np.ndarray:
    return np.loadtxt(directory / "output" / "duty-cycle" / f"{ifo}-observing-segments.txt", ndmin=2)


def test_the_merge_loop_on_the_page_matches_the_run_the_tests_make() -> None:
    """The examples read the files the page's merge loop writes; the fixture writes the same names."""
    page = _PAGE.read_text()
    assert '--copy-channel "${ifo}:STATE_VECTOR"' in page
    assert '--output "merged/E-${ifo}_STRAIN-${t0}-1024.gwf"' in page
    assert '"output/duty-cycle/E-${ifo}_STATE-${t0}-1024.gwf"' in page


@pytest.mark.requires_gwpy
def test_the_cwb_example_reads_only_observing_data(run: Path) -> None:
    """cWB's data quality files, as the page writes them, give jobs whose data hold no NaN."""
    from gwpy.timeseries import TimeSeries

    language, code = _example("cwb")
    assert language == "cpp"
    dqfiles = cwb.parse_dqfiles(code)
    assert [dqf.ifo for dqf in dqfiles] == list(TRIANGLE)
    assert all(dqf.cat == cwb.CWB_CAT0 and not dqf.invert and not dqf.c4 for dqf in dqfiles)

    network = cwb.read_seg_lists(dqfiles, run)
    jobs = cwb.get_job_list(network)
    assert jobs, "the run gives cWB no job"
    for start, stop in cwb.read_windows(jobs):
        for ifo in TRIANGLE:
            assert any(a <= start and stop <= b for a, b in _science(run, ifo))
            frames = sorted(str(path) for path in (run / "merged").glob(f"E-{ifo}_STRAIN-*.gwf"))
            data = TimeSeries.read(frames, f"{ifo}:STRAIN", start=start, end=stop)
            assert np.isfinite(data.value).all()


def test_cwb_stops_on_the_file_of_an_interferometer_that_never_observes(tmp_path: Path) -> None:
    """The page tells the user to leave such an interferometer out; this is why."""
    (tmp_path / "never.txt").write_text("", encoding="utf-8")
    with pytest.raises(cwb.CWBExitError, match="is empty"):
        cwb.read_seg_list(cwb.DQFile("ET1_EMR", "never.txt", cwb.CWB_CAT0, 0.0, False, False), tmp_path)


@pytest.mark.pipelines
def test_the_gwpy_example_runs(run: Path, monkeypatch) -> None:
    _require("gwpy")
    _require("igwn_ligolw")
    _require("lal")
    _run_example("gwpy", run, monkeypatch)


@pytest.mark.pipelines
def test_the_pycbc_offline_example_runs(run: Path, monkeypatch) -> None:
    _require("pycbc")
    _run_example("pycbc-offline", run, monkeypatch)


@pytest.mark.pipelines
def test_the_pycbc_offline_configuration_gives_only_the_science_segments_a_file(run: Path) -> None:
    """In ``[workflow-segments-science]`` the file is the science segments' source, and only theirs."""
    _require("pycbc")
    from pycbc.types.config import InterpolatingConfigParser
    from pycbc.workflow.jobsetup import int_gps_time_to_str
    from pycbc.workflow.segment import SegFile

    language, ini = _example("pycbc-offline-ini")
    assert language == "ini"
    parser = InterpolatingConfigParser()
    parser.read_string("[workflow-segments]\nsegments-veto-definer-url = https://example.invalid/vdf.xml\n" + ini)
    assert parser.get_opt_tags("workflow-segments", "segments-source", ["science"]) == "file"
    assert parser.get_opt_tags("workflow-segments", "segments-science-file", ["science"])
    for tags in (["veto"], ["dq"]):
        with pytest.raises(configparser.Error):
            parser.get_opt_tags("workflow-segments", "segments-source", tags)
    # The workflow passes each bound through this, which refuses a fractional GPS time.
    segments = SegFile.from_segment_xml(str(run / "output" / "duty-cycle" / "duty-cycle-segments.xml")).segment_dict
    for ifo in TRIANGLE:
        for segment in segments[f"{ifo}:science"]:
            int_gps_time_to_str(segment[0])
            int_gps_time_to_str(segment[1])


@pytest.mark.pipelines
def test_pycbc_live_with_the_pages_options_analyses_exactly_the_science_segments(unmasked_run: Path) -> None:
    """``pycbc_live``'s own option parsing and state buffer, given the page's options and the run's frames."""
    _require("pycbc")
    import argparse

    from pycbc.frame import StatusBuffer, flag_names_to_bitmask
    from pycbc.types.optparse import MultiDetMultiColonOptionAction, MultiDetOptionAction

    language, command = _example("pycbc-live")
    assert language == "bash"
    arguments = shlex.split(command.replace("\\\n", " "))[1:]
    parser = argparse.ArgumentParser()
    # As pycbc_live declares them.
    parser.add_argument("--frame-src")
    parser.add_argument("--channel-name", action=MultiDetMultiColonOptionAction, nargs="+")
    parser.add_argument("--state-channel", action=MultiDetMultiColonOptionAction, nargs="+")
    parser.add_argument("--analyze-flags", action=MultiDetOptionAction, nargs="+")
    options = parser.parse_args(arguments)

    ifo = "ET1_EMR"
    state_channel = f"{ifo}:{options.state_channel[ifo]}"  # as pycbc_live joins them
    mask = flag_names_to_bitmask(options.analyze_flags[ifo].split(","))
    assert mask == 0b11
    frames = sorted(str(path) for path in (unmasked_run / "merged").glob(f"E-{ifo}_STRAIN-*.gwf"))
    state = StatusBuffer(frames, state_channel, START, max_buffer=64, valid_mask=mask, force_update_cache=False)
    science = _science(unmasked_run, ifo)
    block = 8
    analysed = np.array([bool(state.advance(block)) for _ in range(FRAME * N_FRAMES // block)])
    starts = START + block * np.arange(analysed.size)
    expected = np.array([((science[:, 0] <= t) & (t + block <= science[:, 1])).any() for t in starts])
    assert analysed.any()
    assert not analysed.all()
    np.testing.assert_array_equal(analysed, expected)


def _live_psds(directory: Path, ifo: str) -> list[np.ndarray]:
    """The PSDs ``pycbc_live``'s strain buffer estimates, block by block, over a run's merged frames."""
    from pycbc.strain.strain import StrainBuffer

    frames = sorted(str(path) for path in (directory / "merged").glob(f"E-{ifo}_STRAIN-*.gwf"))
    buffer = StrainBuffer(
        frames,
        f"{ifo}:STRAIN",
        START,
        state_channel=f"{ifo}:STATE_VECTOR",
        analyze_flags=["HOFT_OK", "SCIENCE_INTENT"],
        sample_rate=128,
        low_frequency_cutoff=15,
        highpass_frequency=15,
        psd_samples=30,
        psd_segment_length=4,
        psd_inverse_length=3.5,
        trim_padding=0.25,
        autogating_taper=0.1,
        max_buffer=512,
        force_update_cache=False,
    )
    psds = []
    for second in range(1, FRAME * N_FRAMES - 2):
        if not buffer.advance(1) or second < 600:  # a PSD once the buffer holds enough data
            continue
        buffer.recalculate_psd()
        psds.append(buffer.psd.numpy()[1:])
    return psds


@pytest.mark.pipelines
def test_pycbc_live_needs_the_strain_in_full(run: Path, unmasked_run: Path) -> None:
    """With the strain masked, its filter reaches NaN after a gap and the PSD turns NaN; in full, it does not.

    This is why the page tells a ``pycbc_live`` user to set ``mask-strain: false``.
    """
    _require("pycbc")
    ifo = "ET1_EMR"
    full = _live_psds(unmasked_run, ifo)
    assert full
    assert all(np.isfinite(psd).all() for psd in full)
    masked = _live_psds(run, ifo)
    assert any(not np.isfinite(psd).all() for psd in masked)


@pytest.mark.pipelines
def test_the_bilby_example_runs(run: Path, monkeypatch) -> None:
    _require("bilby")
    _run_example("bilby", run, monkeypatch)


@pytest.mark.pipelines
def test_bilby_pipe_skips_its_segment_server_check_by_default() -> None:
    """The page tells the user to leave the check at its default; the check would fail on these data."""
    _require("bilby_pipe")
    from bilby_pipe.parser import create_parser

    assert create_parser().get_default("ignore_gwpy_data_quality_check") is True
