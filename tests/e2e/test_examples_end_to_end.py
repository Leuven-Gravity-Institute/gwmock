"""Drive each matrix example through the real CLI and check what it produced.

These are the tests the matrix in :mod:`tests.e2e.matrix` exists for. Each runs one example
configuration -- shortened by :mod:`tests.e2e.overlay`, never edited in place -- and asserts the
run completed and wrote data of the expected shape.

Marked ``e2e`` and therefore excluded from the default run: they generate data and take seconds
to minutes each. The ``e2e`` CI job runs them with the ``sgwb`` and ``jax`` extras installed (not ``cuda``:
the runner has no GPU).

Scope of what is checked here is deliberately structural. Whether the output *matches a stored
reference* comes later, and depends on first establishing that each path is reproducible at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwmock.cli.utils.hash import compute_content_hash

from .matrix import E2E_MATRIX, MatrixEntry
from .overlay import NONZERO_OUTPUT
from .runner import config_of, manifest, samples, skip_if_unavailable, written_files

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("entry", E2E_MATRIX, ids=lambda entry: entry.label)
class TestExampleRuns:
    """One run per matrix entry, with the checks that do not need a stored reference."""

    def test_the_run_completes_and_writes_data(self, entry: MatrixEntry, completed_run):
        """The example must run to completion and leave non-empty output behind.

        The weakest useful assertion, and the one that catches most breakage: an exception
        anywhere in orchestration, or a run that silently writes nothing.
        """
        skip_if_unavailable(entry)
        tmp_path = completed_run(entry)

        written = written_files(tmp_path)
        assert written, f"'{entry.label}' completed but wrote no output files"
        empty = [path.name for path in written if path.stat().st_size == 0]
        assert not empty, f"'{entry.label}' wrote empty files: {empty}"

    def test_every_declared_output_was_written(self, entry: MatrixEntry, completed_run):
        """The files on disk must be exactly the ones the run says it wrote.

        Checked both ways on purpose. Requiring only that *some* output exists lets a run pass
        having written one detector's frame and dropped the rest -- and a run that writes a file
        its own metadata does not mention is equally wrong, in a way a file count would miss.
        """
        skip_if_unavailable(entry)
        tmp_path = completed_run(entry)

        declared = {item["path"] for item in manifest(tmp_path)}
        assert declared, f"'{entry.label}' recorded no outputs in its metadata"

        found = {str(path.relative_to(tmp_path)) for path in written_files(tmp_path)}
        assert declared == found, (
            f"'{entry.label}' output does not match its own manifest: "
            f"declared but absent {sorted(declared - found)}, "
            f"present but undeclared {sorted(found - declared)}"
        )

    def test_every_output_decodes_and_is_finite(self, entry: MatrixEntry, completed_run):
        """*Every* written file must decode, carry samples, and contain no NaN or infinity.

        Previously this counted the files that decoded and required at least one, which let an
        unreadable or empty file pass as long as a sibling was fine. An unrecognised format now
        fails in ``_samples`` rather than being skipped.

        The channel list is cross-checked against the manifest as well, since a frame can decode
        while holding fewer channels than the run claims to have put in it.
        """
        skip_if_unavailable(entry)
        tmp_path = completed_run(entry)

        for item in manifest(tmp_path):
            path = tmp_path / item["path"]
            decoded = samples(path)
            assert decoded.size, f"'{item['path']}' decoded to no samples"
            assert np.all(np.isfinite(decoded)), f"'{item['path']}' contains non-finite samples"

            declared_hash = item.get("content_sha256")
            if declared_hash:
                # The run records a content hash of what it wrote. Recomputing it here checks the
                # pipeline's own bookkeeping against an independent calculation -- a recorded hash
                # that does not match the data would make every downstream provenance claim wrong,
                # and nothing else would notice.
                assert compute_content_hash(path) == declared_hash, (
                    f"'{item['path']}' does not match the content hash recorded for it in the run "
                    f"metadata, so the run's own provenance is wrong about its output"
                )

            declared_channels = item.get("channels")
            if declared_channels and path.suffix == ".gwf":
                from gwpy.io.gwf import get_channel_names

                assert sorted(get_channel_names(str(path))) == sorted(declared_channels), (
                    f"'{item['path']}' does not hold the channels its metadata declares"
                )

    def test_the_output_is_nonzero_where_expected(self, entry: MatrixEntry, completed_run):
        """A run whose span covers its events must not write all zeros where they belong.

        This is the assertion that distinguishes "the pipeline ran" from "the pipeline produced
        data". A configuration whose segment misses its events, or whose injector never fires,
        completes, writes correctly-shaped files, and contains nothing at all -- a trap hit for
        real while developing these tests, not a hypothetical one.
        """
        if entry.label not in NONZERO_OUTPUT:
            pytest.skip(f"'{entry.label}' is not expected to contain located content")
        skip_if_unavailable(entry)
        tmp_path = completed_run(entry)
        config = config_of(entry, tmp_path)
        section = NONZERO_OUTPUT[entry.label]

        # Only the named section's outputs. Counting every file instead lets a signal+noise
        # configuration pass on its noise alone -- verified: with the start time deliberately
        # misaligned, `signal/bbh` failed but `quick_start` still passed, because its noise is
        # non-zero whatever the signal did.
        directory = tmp_path / "output" / config["orchestration"][section]["output"]["output_directory"]
        files = [path for path in written_files(tmp_path) if directory in path.parents]
        assert files, f"'{entry.label}' wrote nothing under {directory}"

        occupied = sum(int(np.count_nonzero(samples(path))) for path in files)
        assert occupied > 0, (
            f"'{entry.label}' wrote only zeros to {directory.name}/, so nothing reached its "
            f"{section} output -- most likely the segment misses the population's events, or no "
            f"glitch was injected."
        )
