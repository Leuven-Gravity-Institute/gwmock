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
"""A checkpoint must say which configuration wrote it, and a resume must check.

Measured before this existed: two configs run from one working directory, the first interrupted
after two of three batches. The second **loaded the first's checkpoint, skipped those batches as
already complete, and wrote one frame where a clean run writes three** — exit code 0, no warning.
Silent data loss from an ordinary action, since every shipped example puts `.gwmock_checkpoints` and
`metadata/` relative to the working directory.

`_batch_outputs_present` does not catch it, and cannot: it looks up
``{simulator_name}-{batch_index}.metadata.json`` — a name with nothing config-specific in it — and
then verifies the outputs *that file* records. The first run's metadata records the first run's
frames, which exist, so the check passes on one run's evidence while a different run is executing.

The refusal is deliberate rather than a silent fresh start: ignoring the foreign checkpoint would fix
the data loss and hide its cause, leaving a stale checkpoint in place and an unexplained full re-run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gwmock.cli.utils.checkpoint import (
    CheckpointManager,
    ForeignCheckpointError,
    require_matching_config,
    run_fingerprint,
)

#: Only ever formatted into an error message -- nothing here opens it -- so any path serves. Named
#: rather than written inline so it does not read as a real location a test might depend on.
_ANY_CHECKPOINT = Path("checkpoints") / "simulation.checkpoint.json"


class TestTheGuardItself:
    """`require_matching_config`, isolated from everything that has to call it."""

    def test_a_different_configuration_is_refused(self):
        with pytest.raises(ForeignCheckpointError, match="different configuration"):
            require_matching_config("a" * 64, "b" * 64, _ANY_CHECKPOINT)

    def test_the_message_names_both_hashes_and_the_file(self):
        """The user has to be able to act on it: which two collided, and what to move."""
        with pytest.raises(ForeignCheckpointError) as raised:
            require_matching_config("a" * 64, "b" * 64, Path("/runs/x/.gwmock_checkpoints/simulation.checkpoint.json"))

        message = str(raised.value)
        assert "aaaaaaaaaaaa" in message
        assert "bbbbbbbbbbbb" in message
        assert "/runs/x/.gwmock_checkpoints/simulation.checkpoint.json" in message

    def test_the_same_configuration_is_allowed(self):
        """The case that must not break: an ordinary resume of the run that wrote the checkpoint."""
        require_matching_config("a" * 64, "a" * 64, _ANY_CHECKPOINT)

    def test_a_checkpoint_without_a_hash_is_refused(self):
        """The last way the silent skip stays reachable: a checkpoint from before the field existed.

        It was accepted with a warning while a mid-run upgrade could still produce one, which is a
        window one interrupted run wide and long since past. A warning was never enough on its own:
        it is emitted where the data loss is invisible, and the run goes on to skip the batches
        anyway.
        """
        with pytest.raises(ForeignCheckpointError, match="predates configuration fingerprinting"):
            require_matching_config(None, "b" * 64, _ANY_CHECKPOINT)

    def test_a_checkpoint_without_a_hash_is_refused_even_with_no_plan_hash(self):
        """Nothing to compare against does not make an unidentifiable checkpoint safe to believe."""
        with pytest.raises(ForeignCheckpointError, match="predates configuration fingerprinting"):
            require_matching_config(None, None, _ANY_CHECKPOINT)

    def test_the_pre_fingerprint_message_names_the_file_and_the_way_out(self):
        """Refusing is a dead end unless the message says what to move and which flag continues."""
        with pytest.raises(ForeignCheckpointError) as raised:
            require_matching_config(None, "b" * 64, Path("/runs/x/.gwmock_checkpoints/simulation.checkpoint.json"))

        message = str(raised.value)
        assert "/runs/x/.gwmock_checkpoints/simulation.checkpoint.json" in message
        assert "--ignore-checkpoint" in message

    def test_an_unknown_plan_hash_is_allowed(self):
        """Nothing to compare against is not evidence of a mismatch."""
        require_matching_config("a" * 64, None, _ANY_CHECKPOINT)


class TestTheFingerprintIsTheRunNotTheConfigFile:
    """The config hash alone is not run identity, and treating it as one leaves the bug reachable.

    Found in review after the first version of this fix shipped the narrower identity: the same
    config file with a different ``--output-dir`` hashes identically, so the checkpoint is accepted
    and its batches skipped -- measured at 2 frames where a clean run writes 3.
    """

    def test_the_output_directory_changes_it(self, tmp_path):
        first = run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta")
        second = run_fingerprint(["a" * 64], tmp_path / "out2", tmp_path / "meta")
        assert first != second

    def test_the_metadata_directory_changes_it(self, tmp_path):
        first = run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta")
        second = run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta2")
        assert first != second

    def test_an_equivalent_path_does_not(self, tmp_path):
        """Two spellings of one directory are one run -- a false refusal is still a bug.

        Uses ``out/../out`` rather than ``out/.``: `Path` collapses a lone ``.`` at construction, so
        that spelling never reaches `resolve()` and the assertion compared a value with itself. ``..``
        survives construction, so this exercises the normalization it claims to. The premise is
        asserted rather than assumed, because that is precisely what went wrong the first time.
        """
        (tmp_path / "out").mkdir()
        (tmp_path / "meta").mkdir()
        roundabout = tmp_path / "out" / ".." / "out"
        assert str(roundabout) != str(tmp_path / "out"), "the spellings collapsed before resolve(); this proves nothing"

        first = run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta")
        second = run_fingerprint(["a" * 64], roundabout, tmp_path / "meta")
        assert first == second

    def test_every_batch_hash_counts_not_just_the_first(self, tmp_path):
        """A plan assembled from several metadata records can mix configs; one of them is not enough."""
        first = run_fingerprint(["a" * 64, "b" * 64], tmp_path / "out", tmp_path / "meta")
        second = run_fingerprint(["a" * 64, "c" * 64], tmp_path / "out", tmp_path / "meta")
        assert first != second

    def test_an_unhashed_batch_is_not_the_same_as_no_batch(self, tmp_path):
        """`None` is kept as a marker; dropping it would collapse two different plans onto one id."""
        first = run_fingerprint(["a" * 64, None], tmp_path / "out", tmp_path / "meta")
        second = run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta")
        assert first != second

    def test_the_same_run_fingerprints_the_same(self, tmp_path):
        """The property every resume depends on: unchanged inputs give an unchanged identity."""
        assert run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta") == run_fingerprint(
            ["a" * 64], tmp_path / "out", tmp_path / "meta"
        )


class TestTheEscapeHatchCannotBeBoundByAccident:
    """`ignore_checkpoint` skips the guard, so nothing may reach it positionally.

    It was inserted before `max_retries`, which is an int: a positional call passing a retry count
    would bind it here, and any non-zero count is truthy. The checkpoint would be skipped silently,
    which is the exact failure this change exists to prevent -- reintroduced by an argument order.
    """

    @pytest.mark.parametrize(
        ("module", "name"),
        [("gwmock.cli.simulate_utils", "execute_plan"), ("gwmock.cli.simulate", "_simulate_impl")],
    )
    def test_it_is_keyword_only(self, module, name):
        import importlib
        import inspect

        parameter = inspect.signature(getattr(importlib.import_module(module), name)).parameters["ignore_checkpoint"]

        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{name} accepts ignore_checkpoint positionally, so a caller passing a later argument "
            f"by position can switch off the checkpoint guard without meaning to"
        )


class TestTheFingerprintSurvivesTheCheckpoint:
    """It is only a guard if it is written, and written checkpoints are what a resume reads."""

    def test_the_hash_is_saved_and_read_back(self, tmp_path):
        manager = CheckpointManager(tmp_path)
        manager.save_checkpoint([0], "orchestration", 0, {"counter": 1}, config_sha256="c" * 64)

        assert (manager.load_checkpoint() or {}).get("config_sha256") == "c" * 64

    def test_a_checkpoint_saved_without_one_reads_as_none(self, tmp_path):
        """Not as a missing key: the guard distinguishes "unknown" from "known and different"."""
        manager = CheckpointManager(tmp_path)
        manager.save_checkpoint([0], "orchestration", 0, {"counter": 1})

        assert (manager.load_checkpoint() or {}).get("config_sha256") is None


@pytest.mark.integration
def test_a_second_configuration_in_the_same_directory_is_refused_end_to_end(tmp_path):
    """The whole failure, through the CLI, because the unit tests above cannot reach the wiring.

    Two configs differing only in their population and output names, run from one directory. Before
    the guard the second wrote 1 frame of 3 and exited 0; it must now refuse and write none.

    Deterministic: the interrupt waits for the checkpoint file to appear rather than for a delay, so
    it lands after a batch has committed however slow the machine is.
    """
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    import json
    import shutil
    import signal as signal_module
    import subprocess
    import time

    executable = shutil.which("gwmock")
    if executable is None:
        pytest.skip("the gwmock console script is not on PATH")

    (tmp_path / "pop_a.csv").write_text(_POPULATION.format(mass_1="1.6", mass_2="1.4"))
    (tmp_path / "pop_b.csv").write_text(_POPULATION.format(mass_1="30.0", mass_2="25.0"))
    (tmp_path / "a.yaml").write_text(_CONFIG.format(population="pop_a.csv", tag="A"))
    (tmp_path / "b.yaml").write_text(_CONFIG.format(population="pop_b.csv", tag="B"))

    checkpoint = tmp_path / ".gwmock_checkpoints" / "simulation.checkpoint.json"
    first = subprocess.Popen(  # noqa: S603 - absolute path from `shutil.which`
        [executable, "simulate", "a.yaml"], cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        deadline = time.monotonic() + 600.0
        while not checkpoint.exists() and first.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert checkpoint.exists(), "no checkpoint was written, so there is nothing for the second run to find"
        completed = set(json.loads(checkpoint.read_text()).get("completed_batch_indices") or [])
        assert completed, "the checkpoint records no completed batch, so the second run would skip nothing"
        first.send_signal(signal_module.SIGINT)
        first.wait(timeout=120)
    finally:
        if first.poll() is None:  # pragma: no cover - only on an unexpected hang
            first.kill()

    second = subprocess.run(  # noqa: S603 - absolute path from `shutil.which`
        [executable, "simulate", "b.yaml"], cwd=tmp_path, capture_output=True, text=True, timeout=900, check=False
    )

    assert second.returncode != 0, "the second configuration resumed from the first's checkpoint instead of refusing"
    assert "different configuration" in (second.stdout + second.stderr)
    written = list((tmp_path / "output" / "signal").glob("*B-*.gwf"))
    assert not written, f"the refused run still wrote {[p.name for p in written]}"


@pytest.mark.integration
def test_a_checkpoint_from_before_the_fingerprint_is_refused_end_to_end(tmp_path):
    """The residual case, through the CLI: a checkpoint on disk from before the field existed.

    It carries no ``config_sha256``, so nothing says which configuration wrote it, and it was once
    resumed from anyway -- with a warning, and with the batches it recorded skipped. That left every
    such file exactly as exposed to the silent skip as before the guard shipped, and interrupted runs
    are precisely the population that resumes.

    The file is a real one with the field removed rather than one written by hand, so the resume it
    is offered to is the resume it would otherwise have completed: the same configuration, in the
    same directory, which is refused here only because the checkpoint cannot be attributed to it.
    """
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    import json
    import shutil
    import signal as signal_module
    import subprocess
    import time

    executable = shutil.which("gwmock")
    if executable is None:
        pytest.skip("the gwmock console script is not on PATH")

    (tmp_path / "pop_a.csv").write_text(_POPULATION.format(mass_1="1.6", mass_2="1.4"))
    (tmp_path / "a.yaml").write_text(_CONFIG.format(population="pop_a.csv", tag="A"))

    checkpoint = tmp_path / ".gwmock_checkpoints" / "simulation.checkpoint.json"
    first = subprocess.Popen(  # noqa: S603 - absolute path from `shutil.which`
        [executable, "simulate", "a.yaml"], cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        deadline = time.monotonic() + 600.0
        while not checkpoint.exists() and first.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert checkpoint.exists(), "no checkpoint was written, so there is nothing to age backwards"
        first.send_signal(signal_module.SIGINT)
        first.wait(timeout=120)
    finally:
        if first.poll() is None:  # pragma: no cover - only on an unexpected hang
            first.kill()

    aged = json.loads(checkpoint.read_text())
    assert aged.pop("config_sha256", None), "the interrupted run wrote no fingerprint to remove"
    assert aged.get("completed_batch_indices"), "the checkpoint records no completed batch, so nothing is skipped"
    checkpoint.write_text(json.dumps(aged))

    resumed = subprocess.run(  # noqa: S603 - absolute path from `shutil.which`
        [executable, "simulate", "a.yaml"], cwd=tmp_path, capture_output=True, text=True, timeout=900, check=False
    )

    assert resumed.returncode != 0, "a checkpoint naming no configuration was resumed from instead of refused"
    output = resumed.stdout + resumed.stderr
    assert "predates configuration fingerprinting" in output
    assert "--ignore-checkpoint" in output, "the refusal has to say how to get past it"


@pytest.mark.integration
def test_ignore_checkpoint_gets_past_the_refusal(tmp_path):
    """The refusal must have a way out that is not "delete the file by hand".

    Without one it is a dead end for anything that cannot answer a prompt: an automated campaign
    that knows the checkpoint is stale would fail on it with no forward path. Raised in review, and
    the reason the default stays a refusal rather than a warning -- the escape is explicit, so
    nothing is skipped by accident.
    """
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    import json
    import shutil
    import signal as signal_module
    import subprocess
    import time

    executable = shutil.which("gwmock")
    if executable is None:
        pytest.skip("the gwmock console script is not on PATH")

    (tmp_path / "pop_a.csv").write_text(_POPULATION.format(mass_1="1.6", mass_2="1.4"))
    (tmp_path / "a.yaml").write_text(_CONFIG.format(population="pop_a.csv", tag="A"))

    checkpoint = tmp_path / ".gwmock_checkpoints" / "simulation.checkpoint.json"
    first = subprocess.Popen(  # noqa: S603 - absolute path from `shutil.which`
        [executable, "simulate", "a.yaml"], cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        deadline = time.monotonic() + 600.0
        while not checkpoint.exists() and first.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert checkpoint.exists()
        assert json.loads(checkpoint.read_text()).get("completed_batch_indices")
        first.send_signal(signal_module.SIGINT)
        first.wait(timeout=120)
    finally:
        if first.poll() is None:  # pragma: no cover - only on an unexpected hang
            first.kill()

    # A different output directory is a different run, so this would be refused without the flag --
    # which the test above pins. Here it must go through and produce a complete dataset.
    completed = subprocess.run(  # noqa: S603 - absolute path from `shutil.which`
        [executable, "simulate", "a.yaml", "--output-dir", "elsewhere", "--ignore-checkpoint"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "different configuration" not in (completed.stdout + completed.stderr)
    frames = sorted((tmp_path / "elsewhere" / "signal").glob("*ET1*.gwf"))
    assert len(frames) == 3, f"ignoring the checkpoint should give a complete run, got {[f.name for f in frames]}"


_POPULATION = (
    "detector_frame_mass_1,detector_frame_mass_2,coa_time,distance,"
    "declination,right_ascension,polarization_angle,inclination\n"
    "{mass_1},{mass_2},1000000610.0,100.0,0.3,1.1,0.2,0.5\n"
)

_CONFIG = """
globals:
    simulator-arguments:
        sampling-frequency: 1024
        duration: 32
        total-duration: 96
        start-time: 1000000540
    working-directory: .
    output-directory: output
    metadata-directory: metadata

orchestration:
    population:
        backend: FilePopulationLoader
        source-type: bns
        arguments:
            path: {population}
    signal:
        waveform-model: IMRPhenomXPHM
        minimum-frequency: 30
        effects:
            earth-rotation: true
        detectors:
            - ET-Triangle-Sardinia
        output:
            output_directory: signal
            file_name: 'E-{{{{ detectors }}}}_STRAIN_{tag}-{{{{ start_time }}}}-{{{{ duration }}}}.gwf'
            arguments:
                channel: '{{{{ detectors }}}}:STRAIN'
"""
