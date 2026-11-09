"""Resolved-config metadata layer and replay preference (A of the A+B design).

Covers the gwmock-core half of dataset-version reproducibility: runtime-resolved
values (e.g. a pinned Hugging Face dataset revision) are folded into a
``resolved_config`` metadata layer that replay prefers over the raw input
``config``, so an *unpinned* run still replays to the exact resources it used.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from gwmock.cli.simulate_utils import (
    _build_resolved_config,
    _deep_merge,
    _unresolved_external_inputs,
)
from gwmock.cli.utils.simulation_plan import (
    create_batch_metadata,
    create_plan_from_metadata_files,
)
from gwmock.noise.adapter import NoiseAdapter

COMMIT_SHA = "0123456789abcdef0123456789abcdef01234567"


class _FakeGlitchModel:
    """Minimal stand-in for a dataset-backed glitch model.

    Mirrors the gwmock-noise contract used by the resolved-config producer:
    ``resolve()`` pins an external version and ``serialize()`` emits a
    config-shaped dict carrying the resolved ``revision``.
    """

    def __init__(self, *, resolved: str | None) -> None:
        self._resolved = resolved
        self.resolve_calls = 0

    def resolve(self) -> str | None:
        self.resolve_calls += 1
        return self._resolved

    def serialize(self) -> dict[str, Any]:
        return {"kind": "deepextractor", "rate": 0.5, "revision": self._resolved}


# --- NoiseAdapter.resolved_config() ---------------------------------------- #


def test_noise_adapter_resolved_config_resolves_and_serializes_glitches() -> None:
    """resolved_config() pins every glitch model and returns their serialized form."""
    adapter = NoiseAdapter.from_backend()
    model = _FakeGlitchModel(resolved=COMMIT_SHA)
    adapter._glitch_models = [model]

    resolved = adapter.resolved_config()

    assert model.resolve_calls == 1
    assert resolved == {"glitches": [{"kind": "deepextractor", "rate": 0.5, "revision": COMMIT_SHA}]}


def test_noise_adapter_resolved_config_is_memoized_per_stream() -> None:
    """Repeated calls reuse one resolution; reconfiguring the stream invalidates it."""
    adapter = NoiseAdapter.from_backend()
    model = _FakeGlitchModel(resolved=COMMIT_SHA)
    adapter._glitch_models = [model]

    first = adapter.resolved_config()
    second = adapter.resolved_config()

    assert first is second  # same cached object, no recomputation
    assert model.resolve_calls == 1

    # Reconfiguring a stream clears the cache (mirrors _configure_default_stream_backend).
    adapter._glitch_models = None
    adapter._resolved_config_cache = None
    assert adapter.resolved_config() == {}


def test_noise_adapter_resolved_config_empty_without_glitches() -> None:
    """A stream with no glitches resolves to an empty mapping, not an error."""
    adapter = NoiseAdapter.from_backend()
    assert adapter.resolved_config() == {}


def test_deepextractor_resolved_config_pins_commit_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real, unpinned DeepExtractor glitch resolves to a concrete SHA via the adapter.

    Exercises the full producer path with the Hugging Face Hub mocked: open a
    stream with an unpinned DeepExtractor glitch, then read resolved_config() and
    confirm it records the concrete commit the cache resolved to — the exact
    value replay would refetch.
    """
    from types import SimpleNamespace

    import numpy as np
    from gwmock_noise.glitches import deepextractor as de

    # Synthetic dataset laid out under the Hub cache's snapshots/<sha>/ path.
    snapshot_dir = tmp_path / "datasets--tomdooney--x" / "snapshots" / COMMIT_SHA
    snapshot_dir.mkdir(parents=True)
    label_order = np.array(list(reversed(de.GLITCH_CLASS_NAMES)))
    n_rows = 2 * label_order.size
    rng = np.random.default_rng(0)
    labels = np.zeros((n_rows, label_order.size))
    for row in range(n_rows):
        labels[row, row % label_order.size] = 1.0
    np.save(snapshot_dir / de.SAMPLES_FILENAME, rng.normal(size=(n_rows, 256)))
    np.save(snapshot_dir / de.LABELS_FILENAME, labels)
    np.save(snapshot_dir / de.LABEL_ORDER_FILENAME, label_order)

    psd_file = tmp_path / "psd.txt"
    freqs = np.linspace(0.0, 4096.0, 129)
    np.savetxt(psd_file, np.column_stack((freqs, np.ones_like(freqs))))

    def fake_download(*, repo_id, filename, repo_type, revision=None, local_files_only=False):
        return str(snapshot_dir / filename)

    monkeypatch.setattr(de, "_load_hf_hub", lambda: SimpleNamespace(hf_hub_download=fake_download))

    adapter = NoiseAdapter.from_backend()
    glitch = {
        "kind": "deepextractor",
        "rate": 0.5,
        "amplitude_distribution": {"distribution": "lognormal", "mean": 1.0, "std": 0.0},
        "psd_file": str(psd_file),
        "snr": 10.0,
        # Deliberately unpinned: reproducibility must still capture the commit.
    }

    adapter.open_stream(chunk_duration=8.0, sampling_frequency=256.0, detectors=["H1"], seed=11, glitches=[glitch])

    resolved = adapter.resolved_config()
    assert resolved["glitches"][0]["kind"] == "deepextractor"
    assert resolved["glitches"][0]["revision"] == COMMIT_SHA
    assert _unresolved_external_inputs({"noise": {"arguments": resolved}}) == []


def test_reusing_adapter_without_glitches_clears_stale_models() -> None:
    """A later glitch-free stream must not leave a previous stream's models visible."""
    adapter = NoiseAdapter.from_backend()
    blip = {
        "kind": "blip",
        "rate": 1.0,
        "amplitude_distribution": {"distribution": "lognormal", "mean": 1e-21, "std": 0.0},
        "width": 0.01,
    }

    adapter.open_stream(chunk_duration=8.0, sampling_frequency=256.0, detectors=["H1"], seed=11, glitches=[blip])
    assert adapter.resolved_config() != {}

    # Reopen with a component but no glitches; the earlier models must be gone.
    adapter.open_stream(
        chunk_duration=8.0,
        sampling_frequency=256.0,
        detectors=["H1"],
        seed=11,
        spectral_lines=[{"frequency": 60.0, "amplitude": 1e-23}],
    )
    assert adapter.resolved_config() == {}


def test_open_stream_captures_real_glitch_models_for_resolved_config() -> None:
    """Opening a stream with glitches captures the built models for resolved_config().

    Uses a parametric blip (no external dataset) to exercise the real capture and
    serialization wiring without a network dependency.
    """
    adapter = NoiseAdapter.from_backend()
    blip = {
        "kind": "blip",
        "rate": 1.0,
        "amplitude_distribution": {"distribution": "lognormal", "mean": 1e-21, "std": 0.0},
        "width": 0.01,
    }

    adapter.open_stream(chunk_duration=8.0, sampling_frequency=256.0, detectors=["H1"], seed=11, glitches=[blip])

    resolved = adapter.resolved_config()
    assert resolved["glitches"][0]["kind"] == "blip"
    # A parametric model has no external version, so nothing is flagged unresolved.
    assert _unresolved_external_inputs({"noise": {"arguments": resolved}}) == []


# --- helpers ---------------------------------------------------------------- #


def test_deep_merge_replaces_lists_and_merges_dicts() -> None:
    """Nested dicts merge key-by-key; lists replace wholesale."""
    base = {"a": {"x": 1, "y": 2}, "list": [1, 2, 3]}
    _deep_merge(base, {"a": {"y": 20, "z": 30}, "list": [9]})
    assert base == {"a": {"x": 1, "y": 20, "z": 30}, "list": [9]}


def test_unresolved_external_inputs_requires_a_full_commit_sha() -> None:
    """Only a full commit SHA pins the run; None and symbolic refs are unresolved."""
    parametric = {"noise": {"arguments": {"glitches": [{"kind": "blip"}]}}}

    def _fragment(revision: Any) -> dict[str, Any]:
        return {"noise": {"arguments": {"glitches": [{"kind": "deepextractor", "revision": revision}]}}}

    # A 40-char hex commit SHA is immutable -> pinned.
    assert _unresolved_external_inputs(_fragment(COMMIT_SHA)) == []
    # None (resolution failed) and symbolic refs (branch/tag, still move) are not.
    assert _unresolved_external_inputs(_fragment(None)) == ["glitch:deepextractor"]
    assert _unresolved_external_inputs(_fragment("main")) == ["glitch:deepextractor"]
    assert _unresolved_external_inputs(_fragment("v1.0")) == ["glitch:deepextractor"]
    # A parametric model has no external revision to pin.
    assert _unresolved_external_inputs(parametric) == []


def test_unresolved_external_inputs_reports_a_remote_glitch_file() -> None:
    """A URL a replay would fetch again is not pinned, even beside a pinned dataset revision."""

    def _fragment(**files: str) -> dict[str, Any]:
        entry = {"kind": "deepextractor", "revision": COMMIT_SHA, **files}
        return {"noise": {"arguments": {"glitches": [entry]}}}

    assert _unresolved_external_inputs(_fragment(psd_file="https://example.org/psd.txt")) == ["glitch:deepextractor"]
    assert _unresolved_external_inputs(_fragment(population_file="http://example.org/p.hdf5")) == [
        "glitch:deepextractor"
    ]
    assert _unresolved_external_inputs(_fragment(psd_file="ET2026a_10km_psd")) == []
    assert _unresolved_external_inputs(_fragment(psd_file="/data/psd.txt")) == []


# --- _build_resolved_config ------------------------------------------------- #


class _StubSimulator:
    """Simulator exposing only the resolved_config() contract."""

    def __init__(self, fragment: dict[str, Any]) -> None:
        self._fragment = fragment

    def resolved_config(self) -> dict[str, Any]:
        return self._fragment


def _input_payload(revision: Any) -> dict[str, Any]:
    return {
        "globals": {},
        "orchestration": {
            "noise": {
                "backend": "DefaultNoiseSimulator",
                "arguments": {
                    "detectors": ["H1"],
                    "seed": 7,
                    "glitches": [{"kind": "deepextractor", "rate": 0.5, "revision": revision}],
                },
            }
        },
    }


def test_build_resolved_config_folds_in_pinned_revision() -> None:
    """The resolved payload overlays the pinned revision, preserving other args."""
    fragment = {"noise": {"arguments": {"glitches": [{"kind": "deepextractor", "rate": 0.5, "revision": COMMIT_SHA}]}}}
    payload = _input_payload(revision=None)

    resolved, replayable = _build_resolved_config(_StubSimulator(fragment), payload)

    assert replayable is True
    assert resolved is not None
    args = resolved["orchestration"]["noise"]["arguments"]
    assert args["glitches"][0]["revision"] == COMMIT_SHA
    # Untouched arguments survive the merge.
    assert args["seed"] == 7
    assert args["detectors"] == ["H1"]
    # The input payload is not mutated.
    assert payload["orchestration"]["noise"]["arguments"]["glitches"][0]["revision"] is None


def test_build_resolved_config_marks_unresolved_non_replayable(caplog: pytest.LogCaptureFixture) -> None:
    """An external input that cannot be pinned is flagged non-replayable, with a warning."""
    fragment = {"noise": {"arguments": {"glitches": [{"kind": "deepextractor", "rate": 0.5, "revision": None}]}}}
    payload = _input_payload(revision=None)

    with caplog.at_level(logging.WARNING):
        resolved, replayable = _build_resolved_config(_StubSimulator(fragment), payload)

    assert replayable is False
    assert resolved is not None
    assert "non-replayable" in caplog.text


def test_build_resolved_config_none_when_nothing_resolves() -> None:
    """A simulator with no resolved_config(), or an empty fragment, records no layer."""

    class _NoContract:
        pass

    assert _build_resolved_config(_NoContract(), _input_payload(revision=None)) == (None, True)
    assert _build_resolved_config(_StubSimulator({}), _input_payload(revision=None)) == (None, True)


# --- replay prefers resolved_config ----------------------------------------- #


def _write_metadata(path: Path, *, config_revision: Any, resolved_revision: Any, replayable: bool) -> Path:
    """Write a valid batch metadata file whose config and resolved_config diverge."""
    from gwmock.cli.utils.config import GlobalsConfig, OrchestrationConfig

    def _orchestration(revision: Any) -> dict[str, Any]:
        return {
            "globals": {},
            "orchestration": {
                "noise": {
                    "backend": "DefaultNoiseSimulator",
                    "arguments": {
                        "detectors": ["H1"],
                        "seed": 7,
                        "glitches": [{"kind": "deepextractor", "rate": 0.5, "revision": revision}],
                    },
                }
            },
        }

    simulator_config = OrchestrationConfig(
        noise={"backend": "DefaultNoiseSimulator", "arguments": {"detectors": ["H1"], "seed": 7}}
    )
    metadata = create_batch_metadata(
        simulator_name="noise",
        batch_index=0,
        simulator_config=simulator_config,
        globals_config=GlobalsConfig(),
        config_payload=_orchestration(config_revision),
        resolved_config=_orchestration(resolved_revision) if resolved_revision is not None else None,
        replayable=replayable,
    )
    file = path / "noise-0.metadata.json"
    file.write_text(json.dumps(metadata))
    return file


def test_replay_prefers_resolved_config(tmp_path: Path) -> None:
    """Replay reconstructs the plan from resolved_config, not the unpinned input config."""
    metadata_file = _write_metadata(tmp_path, config_revision=None, resolved_revision=COMMIT_SHA, replayable=True)

    plan = create_plan_from_metadata_files([metadata_file], tmp_path / "ckpt")

    glitches = plan.batches[0].simulator_config.noise.arguments["glitches"]
    assert glitches[0]["revision"] == COMMIT_SHA


def test_replay_falls_back_to_config_without_resolved_layer(tmp_path: Path) -> None:
    """With no resolved_config, replay uses the input config (legacy behavior)."""
    metadata_file = _write_metadata(tmp_path, config_revision="main", resolved_revision=None, replayable=True)

    plan = create_plan_from_metadata_files([metadata_file], tmp_path / "ckpt")

    glitches = plan.batches[0].simulator_config.noise.arguments["glitches"]
    assert glitches[0]["revision"] == "main"


def test_replay_warns_when_non_replayable(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Replaying a run marked non-replayable warns that it is not bit-for-bit."""
    metadata_file = _write_metadata(tmp_path, config_revision=None, resolved_revision=None, replayable=False)

    with caplog.at_level(logging.WARNING):
        create_plan_from_metadata_files([metadata_file], tmp_path / "ckpt")

    assert "non-replayable" in caplog.text


# --- schema ----------------------------------------------------------------- #


def test_create_batch_metadata_records_resolved_layer_and_flag() -> None:
    """create_batch_metadata surfaces resolved_config and replayable in the schema."""
    from gwmock.cli.utils.config import GlobalsConfig, OrchestrationConfig

    globals_config = GlobalsConfig()
    simulator_config = OrchestrationConfig(
        noise={"backend": "DefaultNoiseSimulator", "arguments": {"detectors": ["H1"]}}
    )
    resolved = {
        "orchestration": {"noise": {"arguments": {"glitches": [{"kind": "deepextractor", "revision": COMMIT_SHA}]}}}
    }

    metadata = create_batch_metadata(
        simulator_name="noise",
        batch_index=0,
        simulator_config=simulator_config,
        globals_config=globals_config,
        resolved_config=resolved,
        replayable=False,
    )

    assert metadata["resolved_config"] == resolved
    assert metadata["replayable"] is False
