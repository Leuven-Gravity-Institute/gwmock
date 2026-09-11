# Copyright (C) 2026 Leuven Gravity Institute
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

"""Injected glitches must be recorded, indexed and findable, as signals already are.

A run used to write the glitch *configuration* and a per-model count, so the frames could be
produced and not scored: a detection-efficiency curve, a classifier's labels and a veto study all
need per-event times and classes, and once a Gaussian background sits on top of the transients none
of that is recoverable from the strain. These tests hold the consumer-facing end of the fix -- the
record in the batch metadata, the ``glitch_index.yaml`` cache over it, the lookup that answers
"which frame holds glitch X", and the rule that a released blind data file carries none of it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from gwmock.cli.main import app
from gwmock.cli.simulate_utils import rebuild_glitch_index, update_glitch_index, update_signal_index
from gwmock.cli.utils.metadata import SCHEMA_VERSION, MetadataRecord, embeddable_metadata
from gwmock.cli.utils.signal_lookup import find_glitches, parse_param_filter

pytestmark = pytest.mark.unit

runner = CliRunner()

O3_EPOCH = 1256655618.0


def _glitch(event_id: str, gps_start_time: float, **overrides: Any) -> dict[str, Any]:
    """One catalogue row, shaped as the producer records it."""
    row = {
        "event_id": event_id,
        "detector": event_id.split("-", 1)[0],
        "model_index": 0,
        "kind": "deepextractor",
        "glitch_class": "Blip",
        "gps_start_time": gps_start_time,
        "gps_peak_time": gps_start_time + 0.97,
        "duration_seconds": 2.0,
        "n_samples": 8192,
        "segment_index": 0,
        "sample_index": 0,
        "target_snr": 8.0,
        "realized_snr": 8.0,
        "amplitude": 1.0,
    }
    row.update(overrides)
    return row


def _batch(index: int, glitches: list[dict[str, Any]], frames: list[str]) -> dict[str, Any]:
    """One batch metadata record injecting *glitches* into noise *frames*."""
    return {
        "noise": {"backend": "gwmock_noise.DefaultNoiseSimulator", "glitch_injections": glitches},
        "signal": {"injections": [{"event_id": index, "parameters": {"coa_time": 100.0 + index}}]},
        "outputs": [{"kind": "noise", "path": frame} for frame in frames]
        + [{"kind": "signal", "path": f"signal/signal-{index}.gwf"}],
    }


def _write_batch(directory: Path, index: int, metadata: dict[str, Any], *, update_index: bool = True) -> str:
    name = f"orchestration-{index}.metadata.json"
    (directory / name).write_text(json.dumps(metadata), encoding="utf-8")
    if update_index:
        update_glitch_index(directory, metadata, name)
        update_signal_index(directory, metadata, name)
    return name


def _populate(directory: Path, *, update_index: bool = True) -> None:
    """Write three batches: two with glitches and one with none.

    The glitch in batch 1 starts 1.2 s before its segment ends and runs for 2 s, so its samples
    reach batch 2's frame. It is recorded once, against the batch it starts in -- the case that
    separates "which frame holds glitch X" from "which frames does glitch X touch".
    """
    _write_batch(
        directory,
        0,
        _batch(0, [_glitch("H1-0-0", O3_EPOCH + 3.0), _glitch("L1-0-0", O3_EPOCH + 11.0)], ["noise/noise-0.gwf"]),
        update_index=update_index,
    )
    _write_batch(
        directory,
        1,
        _batch(
            1,
            [_glitch("H1-0-1", O3_EPOCH + 30.8, segment_index=1, glitch_class="Koi_Fish", realized_snr=16.0)],
            ["noise/noise-1.gwf"],
        ),
        update_index=update_index,
    )
    _write_batch(directory, 2, _batch(2, [], ["noise/noise-2.gwf"]), update_index=update_index)


def test_a_glitch_lookup_answers_which_frame_holds_it(tmp_path: Path) -> None:
    """The acceptance question, by id, through the fast index path."""
    _populate(tmp_path)

    matches = find_glitches(tmp_path, event_id="H1-0-1")

    assert matches == [
        {
            "event_id": "H1-0-1",
            "frames": ["noise/noise-1.gwf"],
            "metadata": ["orchestration-1.metadata.json"],
            "gps_start_time": O3_EPOCH + 30.8,
        }
    ]


def test_the_index_holds_every_injected_glitch_and_nothing_else(tmp_path: Path) -> None:
    """A batch that injected no glitches must not appear, and none may be missed."""
    _populate(tmp_path)

    index = yaml.safe_load((tmp_path / "glitch_index.yaml").read_text())

    assert set(index) == {"H1-0-0", "L1-0-0", "H1-0-1"}
    # Only the noise frames: the signal outputs of the same batch belong to the other index.
    assert index["H1-0-0"]["batches"] == [
        {"metadata": "orchestration-0.metadata.json", "frames": ["noise/noise-0.gwf"]}
    ]
    # The two indexes are separate files over separate catalogues, keyed differently.
    signal_index = yaml.safe_load((tmp_path / "signal_index.yaml").read_text())
    assert set(signal_index) == {"0", "1", "2"}


def test_a_boundary_crossing_glitch_is_recorded_once_at_its_true_time(tmp_path: Path) -> None:
    """Its samples reach the next frame; its row names the frame it starts in, at its start time."""
    _populate(tmp_path)

    matches = find_glitches(tmp_path, event_id="H1-0-1")

    assert len(matches) == 1
    match = matches[0]
    # Not two rows, and not the frame its tail spills into.
    assert match["metadata"] == ["orchestration-1.metadata.json"]
    assert match["frames"] == ["noise/noise-1.gwf"]
    assert match["gps_start_time"] == O3_EPOCH + 30.8
    # And the row itself says how far past that time the waveform runs, so a consumer can find the
    # spill without the index having to enumerate it.
    rows = json.loads((tmp_path / "orchestration-1.metadata.json").read_text())["noise"]["glitch_injections"]
    assert rows[0]["duration_seconds"] == 2.0
    assert rows[0]["gps_peak_time"] > rows[0]["gps_start_time"]


def test_glitches_are_findable_by_catalogue_column(tmp_path: Path) -> None:
    """A glitch row is flat, so a filter names a column: class, SNR, detector, time."""
    _populate(tmp_path)

    by_class = find_glitches(tmp_path, param_filters=[parse_param_filter("glitch_class==Koi_Fish")])
    assert [match["event_id"] for match in by_class] == ["H1-0-1"]
    assert by_class[0]["frames"] == ["noise/noise-1.gwf"]

    loud = find_glitches(tmp_path, param_filters=[parse_param_filter("realized_snr>=10")])
    assert [match["event_id"] for match in loud] == ["H1-0-1"]

    in_l1 = find_glitches(tmp_path, param_filters=[parse_param_filter("detector==L1")])
    assert [match["event_id"] for match in in_l1] == ["L1-0-0"]

    late = find_glitches(tmp_path, param_filters=[parse_param_filter(f"gps_start_time>={O3_EPOCH + 20.0}")])
    assert [match["event_id"] for match in late] == ["H1-0-1"]


def test_rebuild_reproduces_the_index_the_incremental_path_wrote(tmp_path: Path) -> None:
    """One schema, whichever path built it -- the same guarantee the signal index has.

    The rebuild is the repair for an index that lost entries, so a rebuilt entry a lookup does not
    understand would be a repair that corrupts. Both paths go through one builder precisely so this
    holds.
    """
    _populate(tmp_path)
    index_file = tmp_path / "glitch_index.yaml"
    incremental = yaml.safe_load(index_file.read_text())
    assert set(incremental) == {"H1-0-0", "L1-0-0", "H1-0-1"}, incremental

    index_file.unlink()
    rebuilt = rebuild_glitch_index(tmp_path)

    assert rebuilt.index_file == index_file
    assert rebuilt.batches == 2, "the batch injecting no glitches must not count as a contribution"
    assert rebuilt.events == 3
    assert yaml.safe_load(index_file.read_text()) == incremental


def test_reindex_rebuilds_both_indexes(tmp_path: Path) -> None:
    """``gwmock reindex`` repairs the glitch index as well as the signal one."""
    _populate(tmp_path)
    (tmp_path / "glitch_index.yaml").unlink()
    (tmp_path / "signal_index.yaml").unlink()

    result = runner.invoke(app, ["reindex", "--metadata-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "signal_index.yaml" in result.output
    assert "glitch_index.yaml" in result.output
    assert set(yaml.safe_load((tmp_path / "glitch_index.yaml").read_text())) == {"H1-0-0", "L1-0-0", "H1-0-1"}
    assert set(yaml.safe_load((tmp_path / "signal_index.yaml").read_text())) == {"0", "1", "2"}


def test_reindex_accepts_a_run_that_injected_no_glitches(tmp_path: Path) -> None:
    """No glitches is an ordinary run, unlike a directory holding no metadata at all."""
    _write_batch(tmp_path, 0, _batch(0, [], ["noise/noise-0.gwf"]))

    result = runner.invoke(app, ["reindex", "--metadata-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert yaml.safe_load((tmp_path / "glitch_index.yaml").read_text()) in (None, {})


def test_find_glitch_command_names_the_start_time(tmp_path: Path) -> None:
    """The CLI says which instant it is reporting, because the name alone does not."""
    _populate(tmp_path)

    result = runner.invoke(app, ["find-glitch", "--metadata-dir", str(tmp_path), "--id", "H1-0-1"])

    assert result.exit_code == 0, result.output
    assert "H1-0-1" in result.output
    assert "noise/noise-1.gwf" in result.output
    # "starts at", not a bare "gps=", which reads as the peak -- the reading that cuts the window
    # in the wrong place and loses the glitch.
    assert "starts at" in result.output


def test_find_glitch_command_filters_by_column(tmp_path: Path) -> None:
    """The parameter path works for glitches too, and reports no match rather than crashing."""
    _populate(tmp_path)

    matched = runner.invoke(
        app, ["find-glitch", "--metadata-dir", str(tmp_path), "--param", "glitch_class==Koi_Fish", "--json"]
    )
    assert matched.exit_code == 0, matched.output
    assert [entry["event_id"] for entry in json.loads(matched.output)] == ["H1-0-1"]

    missing = runner.invoke(app, ["find-glitch", "--metadata-dir", str(tmp_path), "--param", "glitch_class==Whistle"])
    assert missing.exit_code == 1
    assert "No matching glitches found." in missing.output


def test_find_glitch_requires_something_to_look_for(tmp_path: Path) -> None:
    """Neither an id nor a filter would match every glitch in the directory by accident."""
    result = runner.invoke(app, ["find-glitch", "--metadata-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "Provide --id" in result.output


def test_the_record_carries_the_catalogue_and_says_so_in_its_version(tmp_path: Path) -> None:
    """``noise.glitch_injections`` is part of the validated record, and the schema version moved."""
    assert SCHEMA_VERSION == "1.6.0"

    rows = [_glitch("H1-0-0", O3_EPOCH + 3.0)]
    record = MetadataRecord.model_validate(
        {
            "subpackage_versions": {},
            "config": {},
            "config_sha256": "0" * 64,
            "host": {"platform": "linux", "python": "3.13", "cpu": "x86_64"},
            "noise": {"backend": "gwmock_noise.DefaultNoiseSimulator", "glitch_injections": rows},
        }
    )

    assert record.noise is not None
    assert record.noise.glitch_injections == rows
    # A record from before the catalogue existed still validates: the key is optional, and an
    # absent key is not the same fact as a run that injected nothing.
    older = MetadataRecord.model_validate(
        {
            "schema_version": "1.5.0",
            "subpackage_versions": {},
            "config": {},
            "config_sha256": "0" * 64,
            "host": {"platform": "linux", "python": "3.13", "cpu": "x86_64"},
            "noise": {"backend": "gwmock_noise.DefaultNoiseSimulator"},
        }
    )
    assert older.noise is not None
    assert older.noise.glitch_injections == []


def test_a_blind_data_file_carries_no_glitch_truth() -> None:
    """Which transients a file holds is an answer, so it is withheld with the signal parameters.

    A blind mock data challenge is released as the data files alone, and finding or vetoing
    transients is the task. Embedding the glitch times, classes and SNRs would hand that over.
    """
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "noise": {
            "backend": "gwmock_noise.DefaultNoiseSimulator",
            "glitch_injections": [_glitch("H1-0-0", O3_EPOCH + 3.0)],
            "metadata": {
                # The configuration -- which models at which rates -- is not the truth and stays.
                "arguments": {"glitches": [{"kind": "deepextractor", "rate": 0.05}]},
                # A verbatim copy of what the orchestrator reported, which carries the rows again
                # -- under the same name, so one strip reaches both.
                "glitch_injections": [_glitch("H1-0-0", O3_EPOCH + 3.0)],
            },
        },
        "signal": {"injections": [{"event_id": 0, "parameters": {"mass_1": 30.0}}]},
    }

    blind = embeddable_metadata(metadata)
    document = json.dumps(blind)
    assert "glitch_injections" not in document
    assert "H1-0-0" not in document
    assert "injections" not in document
    # Stripped at every depth, including the orchestrator's second copy of the same rows.
    assert "glitch_injections" not in blind["noise"]["metadata"]
    # The configuration survives, so the file still describes how it was made.
    assert blind["noise"]["metadata"]["arguments"]["glitches"] == [{"kind": "deepextractor", "rate": 0.05}]

    opted_in = embeddable_metadata(metadata, include_injection_parameters=True)
    assert opted_in["noise"]["glitch_injections"][0]["event_id"] == "H1-0-0"
    assert opted_in["signal"]["injections"][0]["parameters"] == {"mass_1": 30.0}


def test_a_glitch_row_survives_the_json_normalisation_the_sidecar_applies(tmp_path: Path) -> None:
    """The rows reach the sidecar intact, numpy scalars and all.

    The producer builds them from numpy arrays, so the record holds ``np.float64`` where a reader
    expects a number. `save_metadata_record` normalises those, and a row that failed to survive it
    would make the whole batch unwritable.
    """
    from gwmock.cli.utils.metadata import save_metadata_record

    rows = [_glitch("H1-0-0", np.float64(O3_EPOCH + 3.0), n_samples=np.int64(8192), realized_snr=np.float64(8.0))]
    metadata_file = tmp_path / "orchestration-0.metadata.json"
    save_metadata_record(
        metadata={
            "subpackage_versions": {},
            "config": {},
            "config_sha256": "0" * 64,
            "host": {"platform": "linux", "python": "3.13", "cpu": "x86_64"},
            "noise": {"backend": "gwmock_noise.DefaultNoiseSimulator", "glitch_injections": rows},
        },
        metadata_file=metadata_file,
    )

    written = json.loads(metadata_file.read_text())["noise"]["glitch_injections"][0]
    assert written["gps_start_time"] == O3_EPOCH + 3.0
    assert written["n_samples"] == 8192
    assert written["realized_snr"] == 8.0
