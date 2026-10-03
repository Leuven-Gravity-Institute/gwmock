#
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
#
"""A signal is recorded against every frame it reaches, not only the one that generated it.

Measured before this was fixed, with a 1.6+1.4 binary from 30 Hz across 32 s segments: the waveform
spans three frames, ``gwmock find-signal --id 0`` named three of the nine files written, and the six
it omitted included the **merger** -- the named frame held a peak of 1.29e-23 against 7.28e-23 in one
it did not name. Continuous waves make it universal rather than occasional, since every pulsar is in
every frame of the run.

Two mechanisms had to be right, and they fail independently, so they are tested separately:

* a chunk carried into a later segment must still say which event it is, and the segment must record
  it -- otherwise the batch metadata, which is the documented source of truth, is itself lossy;
* ``signal_index.yaml`` must accumulate across batches rather than assign, or the id fast path keeps
  whichever batch wrote last.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from astropy.units.quantity import Quantity

from gwmock.cli.simulate_utils import update_signal_index
from gwmock.cli.utils.signal_lookup import find_signals
from gwmock.data.time_series.time_series import TimeSeries
from gwmock.mixin.time_series import _contributing_injections, _merge_injection_records


def _segment(start: float, duration: float = 4.0, sampling_frequency: float = 16.0) -> TimeSeries:
    return TimeSeries(
        data=np.zeros((1, int(duration * sampling_frequency))),
        start_time=Quantity(start, unit="s"),
        sampling_frequency=Quantity(sampling_frequency, unit="Hz"),
    )


def _chunk(start: float, duration: float, event_id: int | None = 7, sampling_frequency: float = 16.0) -> TimeSeries:
    chunk = TimeSeries(
        data=np.ones((1, int(duration * sampling_frequency))),
        start_time=Quantity(start, unit="s"),
        sampling_frequency=Quantity(sampling_frequency, unit="Hz"),
    )
    chunk.metadata.update({"injection_parameters": {"coa_time": start + duration}, "event_id": event_id})
    return chunk


class TestWhichChunksCountAsPresent:
    """The overlap rule, at the boundary, where an off-by-one costs exactly one frame per signal."""

    @pytest.mark.parametrize(
        ("chunk_start", "chunk_duration", "expected", "why"),
        [
            (100.0, 4.0, True, "exactly the segment"),
            (98.0, 4.0, True, "overlapping the start"),
            (102.0, 4.0, True, "overlapping the end"),
            (96.0, 4.0, False, "ends exactly at the segment start, so no sample is inside"),
            (104.0, 4.0, False, "starts exactly at the segment end, so no sample is inside"),
            (96.0, 4.0625, True, "ends one sample into the segment"),
            (103.9375, 4.0, True, "starts on the segment's last sample"),
            (90.0, 4.0, False, "entirely before"),
            (110.0, 4.0, False, "entirely after"),
            (90.0, 30.0, True, "spanning the whole segment and out both sides"),
        ],
    )
    def test_the_boundary_cases(self, chunk_start, chunk_duration, expected, why):
        """Enumerated across the boundary rather than argued in a comment.

        The endpoints are the cases a plausible implementation gets wrong: ``<=`` instead of ``<``
        cross-lists a signal against a frame it contributes nothing to, which is a silent over-claim
        that no downstream check would catch.
        """
        segment = _segment(100.0)
        assert segment.contributes_samples(_chunk(chunk_start, chunk_duration)) is expected, why


class TestOffGridChunksAgreeWithInjection:
    """For an off-grid chunk, presence is whether injection places a sample, not whether spans overlap.

    The chunk is resampled onto the segment's grid points within its own sample span, so one that
    overlaps an edge by less than a sample -- or a one-sample chunk between two grid points --
    places nothing, and claiming it in the provenance record would name a signal the frame lacks.
    """

    @pytest.mark.parametrize(
        ("first_sample", "samples", "expected", "why"),
        [
            (63.5, 4, False, "starts half a sample before the segment end"),
            (62.5, 4, True, "starts one and a half samples before the segment end"),
            (-3.5, 4, False, "last sample half a sample before the segment start"),
            (-2.5, 4, True, "last sample half a sample into the segment"),
            (10.5, 1, False, "a single sample between two grid points"),
            (10.5, 2, True, "two samples straddling a grid point"),
        ],
    )
    def test_presence_matches_what_injection_places(self, first_sample, samples, expected, why):
        segment = _segment(100.0)
        chunk = TimeSeries(
            data=np.ones((1, samples)),
            start_time=Quantity(100.0 + first_sample / 16.0, unit="s"),
            sampling_frequency=Quantity(16.0, unit="Hz"),
        )

        claimed = segment.contributes_samples(chunk)
        segment.inject(chunk)
        placed = bool(np.any(np.asarray(segment[0])))

        assert claimed is expected, why
        assert placed is expected, f"{why}: injection disagrees with the expectation"


class TestTheSegmentRecordsWhatReachesIt:
    """`_contributing_injections` and `_merge_injection_records`, the pieces `simulate` composes."""

    def test_a_carried_chunk_is_recorded_against_the_segment_it_reaches(self):
        """The whole point: an event generated earlier is still named by the frame it lands in."""
        segment = _segment(100.0)
        records = _contributing_injections(segment, [_chunk(96.0, 8.0, event_id=3)])
        assert [record["event_id"] for record in records] == [3]

    def test_a_chunk_that_does_not_reach_the_segment_is_not_recorded(self):
        """The converse, so the fix cannot be satisfied by recording everything unconditionally."""
        segment = _segment(100.0)
        assert _contributing_injections(segment, [_chunk(80.0, 4.0, event_id=3)]) == []

    def test_a_chunk_without_injection_parameters_is_not_recorded(self):
        """Noise and background chunks reach the segment too, and are not injections."""
        segment = _segment(100.0)
        plain = _chunk(100.0, 4.0)
        plain.metadata.clear()
        assert _contributing_injections(segment, [plain]) == []

    def test_one_event_arriving_on_several_chunks_is_recorded_once(self):
        """A multi-detector batch emits one chunk per detector carrying the same record.

        Without deduplication a three-detector network would list every injection three times, which
        reads as three injections rather than one.
        """
        segment = _segment(100.0)
        chunks = [_chunk(100.0, 4.0, event_id=5) for _ in range(3)]
        assert len(_contributing_injections(segment, chunks)) == 3
        assert len(_merge_injection_records(_contributing_injections(segment, chunks))) == 1

    def test_records_without_an_event_id_are_not_collapsed_together(self):
        """Absent ids are not evidence of sameness, so merging them would drop real injections."""
        segment = _segment(100.0)
        chunks = [_chunk(100.0, 4.0, event_id=None) for _ in range(3)]
        assert len(_merge_injection_records(_contributing_injections(segment, chunks))) == 3

    def test_the_generated_record_wins_and_order_is_stable(self):
        """Generated first, then carried, so the common single-segment case reads as it always did."""
        merged = _merge_injection_records(
            [{"event_id": 1, "parameters": {"source": "generated"}}],
            [{"event_id": 1, "parameters": {"source": "carried"}}, {"event_id": 2, "parameters": {}}],
        )
        assert [record["event_id"] for record in merged] == [1, 2]
        assert merged[0]["parameters"]["source"] == "generated"


def _write_batch(directory, name: str, frames: list[str], event_ids: list[int]) -> None:
    update_signal_index(
        directory,
        {
            "signal": {"injections": [{"event_id": i, "parameters": {"coa_time": 12.0}} for i in event_ids]},
            "outputs": [{"kind": "signal", "path": frame} for frame in frames],
        },
        name,
    )


class TestTheIndexAccumulatesAcrossBatches:
    """`signal_index.yaml`, which assigned where it had to append."""

    def test_one_event_reaching_three_batches_names_all_their_frames(self, tmp_path):
        """Reproduces the measured failure: three of nine frames, keeping whichever batch wrote last."""
        for index, name in enumerate(("a", "b", "c")):
            _write_batch(tmp_path, f"orchestration-{name}.metadata.json", [f"signal/{name}-{index}.gwf"], [0])

        matches = find_signals(tmp_path, event_id=0)

        assert len(matches) == 1
        assert matches[0]["frames"] == ["signal/a-0.gwf", "signal/b-1.gwf", "signal/c-2.gwf"]
        assert matches[0]["metadata"] == [
            "orchestration-a.metadata.json",
            "orchestration-b.metadata.json",
            "orchestration-c.metadata.json",
        ]

    def test_a_rerun_withdraws_only_its_own_contribution(self, tmp_path):
        """A re-run that now injects nothing must not leave its old frames behind, nor drop others'.

        The old code dropped whole entries whose ``metadata`` matched, which was equivalent only while
        an entry belonged to one batch. Once entries span batches, dropping the entry would discard
        the other batches' frames too -- so this asserts both halves.
        """
        _write_batch(tmp_path, "orchestration-a.metadata.json", ["signal/a.gwf"], [0])
        _write_batch(tmp_path, "orchestration-b.metadata.json", ["signal/b.gwf"], [0])

        _write_batch(tmp_path, "orchestration-a.metadata.json", ["signal/a-new.gwf"], [])

        matches = find_signals(tmp_path, event_id=0)
        assert matches[0]["frames"] == ["signal/b.gwf"], "the re-run's old frame survived, or b's was lost"

    def test_an_index_written_before_the_schema_change_still_answers(self, tmp_path):
        """An index is a rebuildable cache, so an upgrade must not make it look like data loss."""
        import yaml

        (tmp_path / "signal_index.yaml").write_text(
            yaml.safe_dump(
                {"0": {"frames": ["signal/old.gwf"], "metadata": "orchestration-old.metadata.json", "coa_time": 1.0}}
            )
        )

        matches = find_signals(tmp_path, event_id=0)

        assert matches[0]["frames"] == ["signal/old.gwf"]
        assert matches[0]["metadata"] == ["orchestration-old.metadata.json"]

    def test_a_new_batch_does_not_discard_a_pre_migration_entry(self, tmp_path):
        """Writing after an upgrade must migrate old entries, not quietly delete them.

        Separate from the read-side migration test above, and the reason is a measured gap: dropping
        the migration inside ``_withdraw_batch`` -- so every pre-1.5.0 entry vanishes on the next
        write -- left the whole suite green. Reading an old index was covered; *surviving a write*
        was not, and that is the path a real upgrade takes, since the next batch of a resumed run
        rewrites the index.
        """
        import yaml

        (tmp_path / "signal_index.yaml").write_text(
            yaml.safe_dump(
                {"9": {"frames": ["signal/old.gwf"], "metadata": "orchestration-old.metadata.json", "coa_time": 1.0}}
            )
        )

        _write_batch(tmp_path, "orchestration-new.metadata.json", ["signal/new.gwf"], [0])

        assert find_signals(tmp_path, event_id=9)[0]["frames"] == ["signal/old.gwf"], (
            "the pre-1.5.0 entry was discarded when an unrelated batch wrote the index"
        )
        assert find_signals(tmp_path, event_id=0)[0]["frames"] == ["signal/new.gwf"]

    def test_both_lookup_paths_report_metadata_the_same_way(self, tmp_path):
        """The id path and the parameter path must not differ in the *type* of what they return."""
        _write_batch(tmp_path, "orchestration-a.metadata.json", ["signal/a.gwf"], [0])
        (tmp_path / "orchestration-a.metadata.json").write_text(
            __import__("json").dumps(
                {
                    "signal": {"injections": [{"event_id": 0, "parameters": {"coa_time": 12.0}}]},
                    "outputs": [{"kind": "signal", "path": "signal/a.gwf"}],
                }
            )
        )

        by_id: list[dict[str, Any]] = find_signals(tmp_path, event_id=0)
        by_param = find_signals(tmp_path, param_filters=[("coa_time", "==", 12.0)])

        assert isinstance(by_id[0]["metadata"], list)
        assert isinstance(by_param[0]["metadata"], list)
