"""Slow end-to-end tests for SGWB orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from gwmock_data import hdf5

from gwmock.cli.simulate import _simulate_impl
from gwmock.simulator.seeds import derive_seed

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _write_config(path: Path) -> None:
    """Write a compact real SGWB + noise orchestration config."""
    config = {
        "globals": {
            "working-directory": str(path.parent),
            "output-directory": "output",
            "metadata-directory": "metadata",
            "simulator-arguments": {
                "sampling-frequency": 128,
                "duration": 4,
                "total-duration": 4,
                "start-time": 1000000000,
                "seed": 2026,
            },
        },
        "orchestration": {
            "signal": {
                "source-type": "sgwb",
                "detectors": ["H1", "L1"],
                "minimum-frequency": 4,
                "parameters": {
                    "omega_ref": 1.0e30,
                    "spectral_index": 0.0,
                    "reference_frequency": 25.0,
                },
                "output": {
                    "output_directory": "signal",
                    "file_name": "sgwb-{{ counter }}.hdf5",
                },
            },
            "noise": {
                "arguments": {
                    "detectors": ["H1", "L1"],
                },
                "output": {
                    "output_directory": "noise",
                    "file_name": "noise-{{ counter }}-{{ detectors }}.npy",
                },
            },
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def _read_signal_hdf5(path: Path) -> dict[str, np.ndarray]:
    """Read detector arrays from one DetectorStrainStack HDF5 artifact."""
    return {name: dataset.data for name, dataset in hdf5.read(path).members.items()}


def test_cli_generates_valid_sgwb_signal_and_noise_outputs(tmp_path: Path) -> None:
    """Run the real CLI orchestration path and validate generated artifacts."""
    config_path = tmp_path / "sgwb.yaml"
    _write_config(config_path)

    _simulate_impl(str(config_path), overwrite=True, metadata=True)

    output_dir = tmp_path / "output"
    metadata_dir = tmp_path / "metadata"
    signal_path = output_dir / "signal" / "sgwb-0.hdf5"
    noise_h1_path = output_dir / "noise" / "noise-0-H1.npy"
    noise_l1_path = output_dir / "noise" / "noise-0-L1.npy"
    metadata_path = metadata_dir / "orchestration-0.metadata.json"

    assert signal_path.exists()
    assert noise_h1_path.exists()
    assert noise_l1_path.exists()
    assert metadata_path.exists()

    signal = _read_signal_hdf5(signal_path)
    assert set(signal) == {"H1", "L1"}
    assert signal["H1"].shape == (512,)
    assert signal["L1"].shape == (512,)
    assert np.all(np.isfinite(signal["H1"]))
    assert np.all(np.isfinite(signal["L1"]))
    assert np.std(signal["H1"]) > 0.0
    assert np.std(signal["L1"]) > 0.0

    noise_h1 = np.load(noise_h1_path)
    noise_l1 = np.load(noise_l1_path)
    assert noise_h1.shape == (512,)
    assert noise_l1.shape == (512,)
    assert np.all(np.isfinite(noise_h1))
    assert np.all(np.isfinite(noise_l1))
    assert np.std(noise_h1) > 0.0
    assert np.std(noise_l1) > 0.0

    metadata = json.loads(metadata_path.read_text())
    assert metadata["simulator_metadata"]["orchestration"]["source_type"] == "sgwb"
    assert metadata["simulator_metadata"]["orchestration"]["signal"]["parameters"]["omega_ref"] == 1.0e30
    assert metadata["simulator_metadata"]["orchestration"]["signal"]["segment_seed"] == derive_seed(2026, "signal", 0)
    assert metadata["simulator_metadata"]["orchestration"]["noise"]["stream_seed"] == derive_seed(
        2026,
        "noise",
        "stream",
    )
    output_kinds = {entry["kind"] for entry in metadata["outputs"]}
    assert output_kinds == {"signal", "noise"}
