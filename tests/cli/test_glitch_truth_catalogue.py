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
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from gwmock.cli.main import app
from gwmock.cli.simulate_utils import (
    IndexRebuildError,
    PartialIndexRebuildError,
    rebuild_glitch_index,
    rebuild_truth_indexes,
    update_glitch_index,
    update_signal_index,
)
from gwmock.cli.utils.metadata import SCHEMA_VERSION, MetadataRecord, embeddable_metadata
from gwmock.cli.utils.signal_lookup import find_glitches, parse_param_filter

pytestmark = pytest.mark.unit

runner = CliRunner()

O3_EPOCH = 1256655618.0

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_BOX = re.compile(r"[\u2500-\u257f]")


def _plain(output: str) -> str:
    """Flatten a Rich-rendered CLI message into one unstyled, unwrapped line.

    A plain ``"Provide --id" in result.output`` passed on a developer machine and failed
    on every CI job. The cause is not the wording: Rich highlights an option name inside
    the sentence, so with colour enabled -- which a CI runner has and an ordinary local
    run does not -- the rendered bytes are
    ``Provide \x1b[1;36m-\x1b[0m\x1b[1;36m-id\x1b[0m and/or ...`` and the phrase the test
    looked for does not occur in them at all. Reproduced locally with ``FORCE_COLOR=1``.

    Stripping the styling and the box drawing, then collapsing whitespace, also removes
    the second, latent version of the same trap: the panel's width follows the terminal,
    so a longer message word-wraps at a different point in each environment. Matching a
    fragment short enough to survive both would stop discriminating between this refusal
    and any other non-zero exit, which is what the assertion is for.
    """
    return " ".join(_BOX.sub(" ", _ANSI.sub("", output)).split())


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
    reported = _plain(result.output)
    assert "signal_index.yaml" in reported
    assert "glitch_index.yaml" in reported
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
    reported = _plain(result.output)
    assert "H1-0-1" in reported
    assert "noise/noise-1.gwf" in reported
    # "starts at", not a bare "gps=", which reads as the peak -- the reading that cuts the window
    # in the wrong place and loses the glitch.
    assert "waveform starts at" in reported


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
    assert "No matching glitches found." in _plain(missing.output)


def test_find_glitch_requires_something_to_look_for(tmp_path: Path) -> None:
    """Neither an id nor a filter would match every glitch in the directory by accident."""
    result = runner.invoke(app, ["find-glitch", "--metadata-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "Provide --id and/or at least one --param filter." in _plain(result.output)


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


def test_a_record_only_one_index_can_read_leaves_both_untouched(tmp_path: Path) -> None:
    """A rebuild of the pair must not replace one index and then refuse the other.

    Validation is per catalogue -- one index reads `signal.injections`, the other
    `noise.glitch_injections` -- so a record whose shape one accepts and the other refuses
    passes the first rebuild and stops the second. Rebuilding them in sequence left the
    signal index replaced and its digest re-recorded, with the glitch index still holding
    whatever needed repairing and nothing in the output saying so.
    """
    _populate(tmp_path)
    signal_index = tmp_path / "signal_index.yaml"
    glitch_index = tmp_path / "glitch_index.yaml"
    before = (signal_index.read_bytes(), glitch_index.read_bytes())

    # Readable JSON, a valid `signal` section, and a `noise.glitch_injections` that is a
    # string rather than a list -- accepted by the signal rebuild, refused by the glitch one.
    broken = _batch(3, [], ["noise/noise-3.gwf"])
    broken["noise"]["glitch_injections"] = "not a list"
    (tmp_path / "orchestration-3.metadata.json").write_text(json.dumps(broken), encoding="utf-8")

    with pytest.raises(IndexRebuildError, match=re.escape("orchestration-3.metadata.json")):
        rebuild_truth_indexes(tmp_path)

    # Neither file replaced: the refusal came before anything was published.
    assert (signal_index.read_bytes(), glitch_index.read_bytes()) == before


def test_reindex_refuses_such_a_directory_without_writing(tmp_path: Path) -> None:
    """The same through the command, which reports it rather than raising a traceback."""
    _populate(tmp_path)
    before = (tmp_path / "signal_index.yaml").read_bytes()
    broken = _batch(3, [], ["noise/noise-3.gwf"])
    broken["noise"]["glitch_injections"] = "not a list"
    (tmp_path / "orchestration-3.metadata.json").write_text(json.dumps(broken), encoding="utf-8")

    result = runner.invoke(app, ["reindex", "--metadata-dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "orchestration-3.metadata.json" in _plain(result.output)
    assert (tmp_path / "signal_index.yaml").read_bytes() == before


def test_a_write_that_fails_part_way_names_the_index_it_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two files cannot be replaced atomically, so the residual case is reported, not hidden.

    A full disk between the two writes is not preventable here. What is preventable is an
    operator being told only that the command failed, when in fact one index of the pair is
    now current and the other is not.
    """
    _populate(tmp_path)
    import gwmock.cli.simulate_utils as module

    real_rebuild = module._rebuild_index

    def _fail_on_the_glitch_index(spec, metadata_directory, encoding="utf-8"):
        if spec.index_file_name == "glitch_index.yaml":
            raise OSError("No space left on device")
        return real_rebuild(spec, metadata_directory, encoding)

    monkeypatch.setattr(module, "_rebuild_index", _fail_on_the_glitch_index)

    with pytest.raises(PartialIndexRebuildError) as raised:
        module.rebuild_truth_indexes(tmp_path)

    message = str(raised.value)
    assert "signal_index.yaml" in message
    assert "glitch_index.yaml" in message
    assert "No space left on device" in message
    # The indexes that were replaced are reported as data, not only inside the message, so a
    # caller can act on them.
    assert [result.index_file.name for result in raised.value.rebuilt] == ["signal_index.yaml"]

    result = runner.invoke(app, ["reindex", "--metadata-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "signal_index.yaml" in _plain(result.output)


def test_rebuild_truth_indexes_rebuilds_both_on_the_happy_path(tmp_path: Path) -> None:
    """The pre-validation pass must not change what a successful rebuild produces."""
    _populate(tmp_path)
    incremental = {
        name: yaml.safe_load((tmp_path / name).read_text()) for name in ("signal_index.yaml", "glitch_index.yaml")
    }
    for name in incremental:
        (tmp_path / name).unlink()

    rebuilt = rebuild_truth_indexes(tmp_path)

    assert [result.index_file.name for result in rebuilt] == ["signal_index.yaml", "glitch_index.yaml"]
    for name, expected in incremental.items():
        assert yaml.safe_load((tmp_path / name).read_text()) == expected


def test_a_digest_failure_after_the_write_is_not_reported_as_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An index that was replaced must not be described as still holding its old contents.

    The digest is recorded *after* the index is committed, so a digest failure leaves that
    index current and its sidecar behind — a state that refuses every later write until it is
    re-baselined. The first version of this reporting folded that in with "could not rebuild"
    and told the operator the file still held what it held before, which by then was gone.
    """
    _populate(tmp_path)
    import gwmock.cli.simulate_utils as module

    real_record = module._record_digest

    def _fail_for_the_signal_index(lock_file: Path, digest: str) -> None:
        if lock_file.name.startswith("signal_index"):
            raise module.IndexDigestNotRecordedError("could not be recorded: read-only sidecar")
        real_record(lock_file, digest)

    monkeypatch.setattr(module, "_record_digest", _fail_for_the_signal_index)

    with pytest.raises(PartialIndexRebuildError) as raised:
        module.rebuild_truth_indexes(tmp_path)

    message = str(raised.value)
    # The index that WAS replaced is named as replaced, and carried as data.
    assert "Replaced signal_index.yaml" in message
    assert raised.value.committed_without_digest == tmp_path / "signal_index.yaml"
    # It must not be described as still holding its previous contents.
    assert "signal_index.yaml still hold" not in message
    assert "Rebuilt" not in message
    # The index that genuinely was not touched is named as such, and is not in `rebuilt`.
    assert "glitch_index.yaml was not rebuilt at all" in message
    assert [result.index_file.name for result in raised.value.rebuilt] == []


def test_a_digest_failure_on_the_last_index_is_left_to_speak_for_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing was skipped, so the original error already describes the whole outcome."""
    _populate(tmp_path)
    import gwmock.cli.simulate_utils as module

    real_record = module._record_digest

    def _fail_for_the_glitch_index(lock_file: Path, digest: str) -> None:
        if lock_file.name.startswith("glitch_index"):
            raise module.IndexDigestNotRecordedError("could not be recorded: read-only sidecar")
        real_record(lock_file, digest)

    monkeypatch.setattr(module, "_record_digest", _fail_for_the_glitch_index)

    # Not wrapped: every index is current and only a sidecar is behind, which is exactly what
    # this error says, with the re-baselining advice a wrapper could only restate.
    with pytest.raises(module.IndexDigestNotRecordedError):
        module.rebuild_truth_indexes(tmp_path)


def test_a_malformed_record_does_not_stop_the_other_batches_answering(tmp_path: Path) -> None:
    """A lookup returns what it can find; one unreadable record must not abort it.

    Unlike a rebuild, which refuses the whole directory rather than silently dropping events.
    A truthy non-list event collection reached `.get` on a string and took the query down with
    it, so a single bad file made every glitch in the directory unfindable.
    """
    _populate(tmp_path)
    for name, payload in (
        ("orchestration-90.metadata.json", {"noise": {"glitch_injections": "not a list"}, "outputs": []}),
        ("orchestration-91.metadata.json", {"noise": {"glitch_injections": ["not a mapping"]}, "outputs": []}),
        ("orchestration-92.metadata.json", {"noise": {"glitch_injections": [_glitch("X1-0-0", 1.0)]}, "outputs": "x"}),
        ("orchestration-93.metadata.json", {"noise": "not a mapping", "outputs": []}),
        # The document itself, not just its sections: each of these parses as JSON and then
        # raises on `.get`. Guarding the sections and the items stopped one level short.
        ("orchestration-94.metadata.json", []),
        ("orchestration-95.metadata.json", "a string"),
        ("orchestration-96.metadata.json", None),
        ("orchestration-97.metadata.json", 7),
    ):
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

    found = find_glitches(tmp_path, param_filters=[parse_param_filter("detector==H1")])

    assert sorted(match["event_id"] for match in found) == ["H1-0-0", "H1-0-1"]
    # The record whose `outputs` was malformed still answers, with no frames rather than a crash.
    by_x1 = find_glitches(tmp_path, param_filters=[parse_param_filter("detector==X1")])
    assert [match["event_id"] for match in by_x1] == ["X1-0-0"]
    assert by_x1[0]["frames"] == []


def test_a_malformed_param_filter_is_a_usage_error_not_a_traceback(tmp_path: Path) -> None:
    """`parse_param_filter` raises ValueError, which reached the user as a traceback."""
    _populate(tmp_path)

    result = runner.invoke(app, ["find-glitch", "--metadata-dir", str(tmp_path), "--param", "glitch_class"])

    assert result.exit_code != 0
    # A usage error, the way every other input mistake in this command is reported.
    assert result.exception is None or isinstance(result.exception, SystemExit), result.exception
    reported = _plain(result.output)
    assert "--param" in reported
    assert "Invalid parameter filter" in reported


def test_an_output_with_a_null_path_is_not_reported_as_a_frame(tmp_path: Path) -> None:
    """A present-but-null path put `None` into `frames`, which the command could not print.

    `"path": null` satisfies `"path" in output`, so the row was taken and the command handed
    `None` to `", ".join(...)`. The rebuild already carries a note about the same shape
    putting `frames: [null]` into an index; this is the read side of it.
    """
    batch = _batch(0, [_glitch("H1-0-0", O3_EPOCH + 1.0)], [])
    batch["outputs"] = [{"kind": "noise", "path": None}, {"kind": "noise"}]
    (tmp_path / "orchestration-0.metadata.json").write_text(json.dumps(batch), encoding="utf-8")

    found = find_glitches(tmp_path, param_filters=[parse_param_filter("detector==H1")])

    assert [match["event_id"] for match in found] == ["H1-0-0"]
    assert found[0]["frames"] == []

    # And the command prints it rather than raising on the join.
    result = runner.invoke(app, ["find-glitch", "--metadata-dir", str(tmp_path), "--param", "detector==H1"])
    assert result.exit_code == 0, result.output
    assert "(no frame recorded)" in _plain(result.output)


def test_the_signal_lookup_survives_the_same_malformed_records(tmp_path: Path) -> None:
    """Both lookups share one implementation, so the guard has to hold for signals too.

    Worth asserting separately rather than trusting the shared code path: the reason this
    matters is that a single unreadable file used to make *every* event in the directory
    unfindable, and `find-signal` is the older, more used of the two commands.
    """
    _populate(tmp_path)
    for name, payload in (
        ("orchestration-94.metadata.json", []),
        ("orchestration-95.metadata.json", "a string"),
        ("orchestration-96.metadata.json", None),
    ):
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

    from gwmock.cli.utils.signal_lookup import find_signals

    found = find_signals(tmp_path, param_filters=[parse_param_filter("coa_time>=0")])

    assert sorted(match["event_id"] for match in found) == [0, 1, 2]


def test_the_documented_schema_version_matches_the_code(tmp_path: Path) -> None:
    """The user guide advertises the schema version, so it must not lag behind the bump.

    A consumer reads the documented version to decide whether it can parse a record; the
    guide said 1.5.0 and its example record carried `"schema_version": "1.5.0"` after the
    code had moved to 1.6.0.
    """
    guide = Path(__file__).resolve().parents[2] / "docs" / "user-guide" / "reproducibility.md"
    text = guide.read_text(encoding="utf-8")

    assert f"uses schema version `{SCHEMA_VERSION}`" in text
    assert f'"schema_version": "{SCHEMA_VERSION}"' in text
