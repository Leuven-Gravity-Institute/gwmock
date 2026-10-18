"""Unit tests for the simulate command and related functionality."""

from __future__ import annotations

import json
import tempfile
import time
from importlib import metadata as stdlib_metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest
import yaml

from gwmock.cli.simulate import (
    _simulate_impl,
)
from gwmock.cli.simulate_utils import (
    _get_host_metadata,
    _get_source_tree_git_sha,
    execute_plan,
    instantiate_simulator,
    process_batch,
    reconcile_completed_batches,
    restore_batch_state,
    retry_with_backoff,
    save_batch_metadata,
    update_metadata_index,
    validate_plan,
)
from gwmock.cli.utils.checkpoint import CheckpointManager, run_fingerprint
from gwmock.cli.utils.config import (
    Config,
    GlobalsConfig,
    SimulatorConfig,
    SimulatorOutputConfig,
)
from gwmock.cli.utils.simulation_plan import (
    SimulationBatch,
    SimulationPlan,
    create_plan_from_config,
    parse_batch_metadata,
)
from gwmock.mixin.randomness import RandomnessMixin
from gwmock.simulator.base import Simulator

# Constants for test assertions to avoid magic values
SEED = 42
SEED_77 = 77
MAX_SAMPLES_3 = 3
MAX_SAMPLES_2 = 2
MAX_SAMPLES_100 = 100
MAX_SAMPLES_50 = 50
COUNTER_5 = 5
COUNTER_10 = 10
NUM_FILES_2 = 2
NUM_FILES_3 = 3
RETRY_COUNT_3 = 3
RETRY_COUNT_2 = 2
TIME_THRESHOLD = 0.14


class MockSimulator(RandomnessMixin, Simulator):
    """Mock simulator for testing, inheriting from Simulator base class.

    This generates simple integer data that increments with each call.
    Useful for testing state management and reproducibility.
    """

    def __init__(self, seed: int = 42, max_samples: int | None = None, **kwargs):
        """Initialize mock simulator with a seed for reproducibility.

        Args:
            seed: Random seed
            max_samples: Maximum number of samples to generate
            **kwargs: Additional arguments (absorbed by base class)
        """

        super().__init__(max_samples=max_samples, **kwargs)
        self.seed = seed
        self._generated_data = []

    def simulate(self) -> int:
        """Generate a mock sample (random integer).

        Returns:
            A random integer
        """
        value = int(self.rng.random() * 100)
        self._generated_data.append(value)
        return value

    def _save_data(self, data, file_name: str | Path, **kwargs) -> None:
        """Save mock data to a JSON file.

        Args:
            data: Data to save
            file_name: Output file path
            **kwargs: Additional arguments
        """
        file_name = Path(file_name)
        file_name.parent.mkdir(parents=True, exist_ok=True)
        with file_name.open("w") as f:
            json.dump({"data": data, "counter": self.counter}, f)


class TestMockSimulator:
    """Test MockSimulator to verify it works correctly."""

    def test_mock_simulator_instantiation(self):
        """Test that MockSimulator can be instantiated."""
        sim = MockSimulator(seed=42)
        assert sim.seed == SEED
        assert sim.counter == 0

    def test_mock_simulator_generate_samples(self):
        """Test that MockSimulator can generate samples."""
        sim = MockSimulator(seed=42, max_samples=3)
        samples = list(sim)
        assert len(samples) == MAX_SAMPLES_3
        assert all(isinstance(s, int) for s in samples)
        assert sim.counter == MAX_SAMPLES_3

    def test_mock_simulator_state_persistence(self):
        """Test that MockSimulator state persists across generations."""
        sim = MockSimulator(seed=42)
        _sample1 = next(sim)
        state_after_1 = sim.state.copy()

        _sample2 = next(sim)
        state_after_2 = sim.state.copy()

        # Counter should increment
        assert state_after_2["counter"] > state_after_1["counter"]

    def test_mock_simulator_save_data(self):
        """Test that MockSimulator can save data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = MockSimulator(seed=42)
            output_file = Path(tmpdir) / "output.json"

            sim.save_data(42, output_file)

            assert output_file.exists()
            with output_file.open() as f:
                data = json.load(f)
            assert data["data"] == SEED

    def test_mock_simulator_reproducibility_with_seed(self):
        """Test that same seed produces same sequence."""
        sim1 = MockSimulator(seed=42, max_samples=5)
        samples1 = list(sim1)

        sim2 = MockSimulator(seed=42, max_samples=5)
        samples2 = list(sim2)

        assert samples1 == samples2


class TestInstantiateSimulator:
    """Test instantiate_simulator function."""

    def test_instantiate_mock_simulator(self):
        """Test instantiating MockSimulator from config."""
        config = SimulatorConfig(
            class_="tests.cli.test_cli_simulate.MockSimulator",
            arguments={"seed": 42},
        )
        sim = instantiate_simulator(config)
        assert isinstance(sim, MockSimulator)
        assert sim.seed == SEED

    def test_instantiate_simulator_invalid_class(self):
        """Test instantiating with invalid class raises error."""
        config = SimulatorConfig(
            class_="nonexistent.Class",
            arguments={},
        )
        with pytest.raises((ImportError, AttributeError)):
            instantiate_simulator(config)

    def test_instantiate_simulator_with_global_arguments(self):
        """Test that global simulator arguments are merged with simulator arguments."""
        global_args = {"seed": 99, "max_samples": 100}
        simulator_args = {"seed": 42}  # Override seed, use global max_samples

        config = SimulatorConfig(
            class_="tests.cli.test_cli_simulate.MockSimulator",
            arguments=simulator_args,
        )

        sim = instantiate_simulator(config, global_simulator_arguments=global_args)

        # Simulator-specific seed should override global
        assert sim.seed == SEED
        # Max samples from global should be used since not in simulator args
        assert sim.max_samples == MAX_SAMPLES_100

    def test_instantiate_simulator_global_only(self):
        """Test simulator instantiation with only global arguments."""
        global_args = {"seed": 77, "max_samples": 50}
        config = SimulatorConfig(
            class_="tests.cli.test_cli_simulate.MockSimulator",
            arguments={},  # No simulator-specific arguments
        )

        sim = instantiate_simulator(config, global_simulator_arguments=global_args)

        # All arguments should come from global
        assert sim.seed == SEED_77
        assert sim.max_samples == MAX_SAMPLES_50


class TestRestoreBatchState:
    """Test restore_batch_state function."""

    def test_restore_state_with_snapshot_priority(self):
        """Test that snapshot is used even if checkpoint state is available."""
        sim = MockSimulator(seed=42, max_samples=10)
        next(sim)  # counter = 1

        snapshot_state = {"counter": 5}
        checkpoint_state = {"counter": 10}

        batch = SimulationBatch(
            simulator_name="mock",
            simulator_config=SimulatorConfig(class_="MockSimulator"),
            globals_config=GlobalsConfig(),
            batch_index=0,
            pre_batch_state=snapshot_state,
        )

        restore_batch_state(sim, batch, last_simulator_state=checkpoint_state)
        assert sim.counter == COUNTER_5, "Snapshot state should take priority over checkpoint state"
        assert sim.counter != COUNTER_10

    def test_restore_state_from_checkpoint_when_no_snapshot(self):
        """Test restoration from checkpoint state when no snapshot is present."""
        sim = MockSimulator(seed=42)
        next(sim)  # counter = 1

        checkpoint_state = {"counter": 10}

        batch = SimulationBatch(
            simulator_name="mock",
            simulator_config=SimulatorConfig(class_="MockSimulator"),
            globals_config=GlobalsConfig(),
            batch_index=10,  # matches the counter in checkpoint state
            pre_batch_state=None,
        )

        restore_batch_state(sim, batch, last_simulator_state=checkpoint_state)
        assert sim.counter == COUNTER_10, "Should restore from checkpoint state when snapshot is missing"

    def test_restore_state_from_checkpoint_wrong_batch_index(self):
        """Test that checkpoint state is NOT used if batch_index doesn't match counter."""
        sim = MockSimulator(seed=42)
        next(sim)  # counter = 1

        checkpoint_state = {"counter": 10}

        batch = SimulationBatch(
            simulator_name="mock",
            simulator_config=SimulatorConfig(class_="MockSimulator"),
            globals_config=GlobalsConfig(),
            batch_index=5,  # does NOT match checkpoint counter
            pre_batch_state=None,
        )

        restore_batch_state(sim, batch, last_simulator_state=checkpoint_state)
        assert sim.counter == 1, "Checkpoint state should NOT be used when batch_index != checkpoint counter"

    def test_restore_state_no_snapshot_no_checkpoint(self):
        """Test that state remains unchanged when neither snapshot nor checkpoint is available."""
        sim = MockSimulator(seed=42)
        next(sim)  # counter = 1

        batch = SimulationBatch(
            simulator_name="mock",
            simulator_config=SimulatorConfig(class_="MockSimulator"),
            globals_config=GlobalsConfig(),
            batch_index=0,
            pre_batch_state=None,
        )

        restore_batch_state(sim, batch, last_simulator_state=None)
        assert sim.counter == 1, "State should remain unchanged when no restoration data is available"

    def test_restore_state_invalid_state_raises_value_error(self):
        """Test that invalid state dict raises ValueError."""
        sim = MockSimulator(seed=42)

        batch = SimulationBatch(
            simulator_name="mock",
            simulator_config=SimulatorConfig(class_="MockSimulator"),
            globals_config=GlobalsConfig(),
            batch_index=0,
            pre_batch_state={"invalid": "structure"},  # missing required keys or wrong format
        )

        with pytest.raises(ValueError, match="Failed to restore state for batch 0"):
            restore_batch_state(sim, batch)


class TestUpdateMetadataIndex:
    """Test update_metadata_index function."""

    def test_create_new_index(self):
        """Test creating a new metadata index file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_dir = Path(tmpdir)
            output_files = [
                Path(tmpdir) / "H1-1234567890-1024.gwf",
                Path(tmpdir) / "L1-1234567890-1024.gwf",
            ]

            update_metadata_index(metadata_dir, output_files, "signal-0.metadata.json")

            index_file = metadata_dir / "index.yaml"
            assert index_file.exists()

            with index_file.open() as f:
                index = yaml.safe_load(f)

            assert "H1-1234567890-1024.gwf" in index
            assert "L1-1234567890-1024.gwf" in index
            assert index["H1-1234567890-1024.gwf"] == "signal-0.metadata.json"
            assert index["L1-1234567890-1024.gwf"] == "signal-0.metadata.json"

    def test_update_existing_index(self):
        """Test updating an existing metadata index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_dir = Path(tmpdir)

            # Create initial index
            output_files_1 = [Path(tmpdir) / "H1-batch0.gwf"]
            update_metadata_index(metadata_dir, output_files_1, "signal-0.metadata.json")

            # Update with more files
            output_files_2 = [Path(tmpdir) / "L1-batch1.gwf"]
            update_metadata_index(metadata_dir, output_files_2, "signal-1.metadata.json")

            # Verify both entries exist
            index_file = metadata_dir / "index.yaml"
            with index_file.open() as f:
                index = yaml.safe_load(f)

            assert len(index) == NUM_FILES_2
            assert index["H1-batch0.gwf"] == "signal-0.metadata.json"
            assert index["L1-batch1.gwf"] == "signal-1.metadata.json"

    def test_index_enables_quick_lookup(self):
        """Test that the index enables quick metadata lookup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_dir = Path(tmpdir)

            # Generate metadata for multiple batches
            output_files_batch0 = [
                Path(tmpdir) / "H1-0.gwf",
                Path(tmpdir) / "L1-0.gwf",
                Path(tmpdir) / "V1-0.gwf",
            ]
            update_metadata_index(metadata_dir, output_files_batch0, "detector-0.metadata.json")

            output_files_batch1 = [
                Path(tmpdir) / "H1-1.gwf",
                Path(tmpdir) / "L1-1.gwf",
                Path(tmpdir) / "V1-1.gwf",
            ]
            update_metadata_index(metadata_dir, output_files_batch1, "detector-1.metadata.json")

            # Load index and verify quick lookup
            index_file = metadata_dir / "index.yaml"
            with index_file.open() as f:
                index = yaml.safe_load(f)

            # Should be able to find metadata for any data file in O(1)
            assert index["H1-0.gwf"] == "detector-0.metadata.json"
            assert index["L1-1.gwf"] == "detector-1.metadata.json"
            assert index["V1-0.gwf"] == "detector-0.metadata.json"


class TestSaveBatchMetadata:
    """Test save_batch_metadata function."""

    def test_save_batch_metadata_creates_file(self):
        """Test that metadata file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = MockSimulator(seed=42)
            next(sim)

            batch = SimulationBatch(
                simulator_name="mock",
                simulator_config=SimulatorConfig(
                    class_="MockSimulator",
                    arguments={"seed": 42},
                ),
                globals_config=GlobalsConfig(),
                batch_index=0,
            )

            metadata_dir = Path(tmpdir)
            output_files = [Path(tmpdir) / "output.json"]
            output_files[0].write_text("{}")
            save_batch_metadata(sim, batch, metadata_dir, batch_data=0, output_files=output_files)

            metadata_file = metadata_dir / "mock-0.metadata.json"
            assert metadata_file.exists()

    @pytest.mark.requires_gwpy
    def test_save_batch_metadata_contains_state_and_files(self):
        """Test that metadata contains simulator state and output file list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = MockSimulator(seed=42)
            next(sim)

            batch = SimulationBatch(
                simulator_name="test",
                simulator_config=SimulatorConfig(
                    class_="MockSimulator",
                    arguments={"seed": 42},
                ),
                globals_config=GlobalsConfig(),
                batch_index=0,
            )

            metadata_dir = Path(tmpdir)
            output_files = [
                Path(tmpdir) / "H1-1234567890-1024.gwf",
                Path(tmpdir) / "L1-1234567890-1024.gwf",
                Path(tmpdir) / "V1-1234567890-1024.gwf",
            ]
            for output_file in output_files:
                output_file.write_text("test")
            save_batch_metadata(sim, batch, metadata_dir, batch_data=0, output_files=output_files)

            metadata_file = metadata_dir / "test-0.metadata.json"
            with metadata_file.open() as f:
                metadata = yaml.safe_load(f)

            assert "pre_batch_state" in metadata
            assert metadata["simulator_name"] == "test"
            assert metadata["batch_index"] == 0
            assert "output_files" in metadata
            assert len(metadata["output_files"]) == NUM_FILES_3
            assert "H1-1234567890-1024.gwf" in metadata["output_files"]
            assert "L1-1234567890-1024.gwf" in metadata["output_files"]
            assert "V1-1234567890-1024.gwf" in metadata["output_files"]

    @pytest.mark.requires_gwpy
    def test_save_batch_metadata_updates_index(self):
        """Test that save_batch_metadata updates the metadata index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = MockSimulator(seed=42)
            next(sim)

            batch = SimulationBatch(
                simulator_name="test",
                simulator_config=SimulatorConfig(
                    class_="MockSimulator",
                    arguments={"seed": 42},
                ),
                globals_config=GlobalsConfig(),
                batch_index=0,
            )

            metadata_dir = Path(tmpdir)
            output_files = [
                Path(tmpdir) / "H1-1234567890-1024.gwf",
                Path(tmpdir) / "L1-1234567890-1024.gwf",
            ]
            for output_file in output_files:
                output_file.write_text("test")
            save_batch_metadata(sim, batch, metadata_dir, batch_data=0, output_files=output_files)

            # Verify index was created and contains entries
            index_file = metadata_dir / "index.yaml"
            assert index_file.exists()

            with index_file.open() as f:
                index = yaml.safe_load(f)

            assert index["H1-1234567890-1024.gwf"] == "test-0.metadata.json"
            assert index["L1-1234567890-1024.gwf"] == "test-0.metadata.json"


class TestRetryWithBackoff:
    """Test retry_with_backoff function."""

    def test_retry_success_first_attempt(self):
        """Test successful execution on first attempt."""
        call_count = 0

        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = retry_with_backoff(success_func, max_retries=3)
        assert result == "success"
        assert call_count == 1

    def test_retry_success_after_failures(self):
        """Test successful execution after retries."""
        call_count = 0

        def retry_func():
            nonlocal call_count
            call_count += 1
            if call_count < RETRY_COUNT_3:
                raise OSError("Simulated I/O error")
            return "success"

        result = retry_with_backoff(retry_func, max_retries=3, initial_delay=0.01)
        assert result == "success"
        assert call_count == RETRY_COUNT_3

    def test_retry_all_attempts_fail(self):
        """Test that exception is raised after all retries fail."""
        call_count = 0

        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            retry_with_backoff(always_fails, max_retries=2, initial_delay=0.01)

        # Should attempt 3 times (initial + 2 retries)
        assert call_count == RETRY_COUNT_3

    def test_retry_exponential_backoff(self):
        """Test that backoff delays increase exponentially."""
        call_times = []

        def track_calls():
            call_times.append(time.time())
            if len(call_times) < RETRY_COUNT_3:
                raise OSError("Retrying")
            return "success"

        start = time.time()
        result = retry_with_backoff(track_calls, max_retries=2, initial_delay=0.05, backoff_factor=2.0)
        total_time = time.time() - start

        assert result == "success"
        assert len(call_times) == RETRY_COUNT_3
        # Total time should be at least: 0.05 + 0.1 = 0.15 seconds
        assert total_time >= TIME_THRESHOLD  # Allow some margin for execution time

    def test_retry_with_state_restoration(self):
        """Test that state restoration function is called before retries."""
        call_count = 0
        restore_count = 0
        state = {"value": 0}

        def state_func():
            nonlocal call_count, state
            call_count += 1
            if call_count < RETRY_COUNT_2:
                # First attempt: fail and modify state
                state["value"] = 999
                raise RuntimeError("First attempt fails")
            # Second attempt: state should have been restored
            return state["value"]

        def restore_func():
            nonlocal restore_count
            restore_count += 1
            state["value"] = 0

        result = retry_with_backoff(state_func, max_retries=1, initial_delay=0.01, state_restore_func=restore_func)

        assert result == 0  # State was restored
        assert call_count == RETRY_COUNT_2
        assert restore_count == 1

    def test_retry_state_restoration_failure_raises_error(self):
        """Test that failure to restore state raises error."""
        call_count = 0

        def state_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("First attempt fails")
            return "success"

        def bad_restore_func():
            raise ValueError("Cannot restore state")

        with pytest.raises(RuntimeError, match=r"Cannot retry.*failed to restore state"):
            retry_with_backoff(state_func, max_retries=1, initial_delay=0.01, state_restore_func=bad_restore_func)

    def test_retry_state_restoration_not_called_on_success(self):
        """Test that state restoration is not called if first attempt succeeds."""
        restore_count = 0

        def success_func():
            return "success"

        def restore_func():
            nonlocal restore_count
            restore_count += 1

        result = retry_with_backoff(success_func, max_retries=3, state_restore_func=restore_func)

        assert result == "success"
        assert restore_count == 0  # Should never be called


class TestValidatePlan:
    """Test validate_plan function."""

    def test_validate_empty_plan_raises_error(self):
        """Test that empty plan fails validation."""
        plan = SimulationPlan()
        with pytest.raises(ValueError, match="no batches"):
            validate_plan(plan)

    def test_validate_plan_with_valid_batch(self):
        """Test that valid plan passes validation."""
        batch = SimulationBatch(
            simulator_name="test",
            simulator_config=SimulatorConfig(
                class_="MockSimulator",
                output=SimulatorOutputConfig(file_name="output.json"),
            ),
            globals_config=GlobalsConfig(),
            batch_index=0,
        )
        plan = SimulationPlan()
        plan.add_batch(batch)

        # Should not raise
        validate_plan(plan)

    def test_validate_plan_missing_file_name(self):
        """Test that batch without file_name fails validation."""
        batch = SimulationBatch(
            simulator_name="test",
            simulator_config=SimulatorConfig(
                class_="MockSimulator",
                output=SimulatorOutputConfig(file_name=""),
            ),
            globals_config=GlobalsConfig(),
            batch_index=0,
        )
        plan = SimulationPlan()
        plan.add_batch(batch)

        with pytest.raises(ValueError, match="file_name"):
            validate_plan(plan)


class TestProcessBatch:
    """Test process_batch function."""

    def test_process_batch_saves_data(self):
        """Test that process_batch saves data file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = MockSimulator(seed=42)
            output_dir = Path(tmpdir)

            batch = SimulationBatch(
                simulator_name="test",
                simulator_config=SimulatorConfig(
                    class_="MockSimulator",
                    output=SimulatorOutputConfig(file_name="output.json"),
                ),
                globals_config=GlobalsConfig(),
                batch_index=0,
            )

            batch_data = 42
            output_files = process_batch(sim, batch_data, batch, output_dir, overwrite=True)

            assert len(output_files) == 1
            assert output_files[0].exists()

    def test_process_batch_respects_overwrite_flag(self):
        """Test that overwrite flag is respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = MockSimulator(seed=42)
            output_dir = Path(tmpdir)

            batch = SimulationBatch(
                simulator_name="test",
                simulator_config=SimulatorConfig(
                    class_="MockSimulator",
                    output=SimulatorOutputConfig(file_name="output.json"),
                ),
                globals_config=GlobalsConfig(),
                batch_index=0,
            )

            # First save
            process_batch(sim, 1, batch, output_dir, overwrite=True)

            # Second save without overwrite should raise
            with pytest.raises(FileExistsError):
                process_batch(sim, 2, batch, output_dir, overwrite=False)

    def test_process_batch_returns_list_of_paths(self):
        """Test that process_batch returns list of Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = MockSimulator(seed=42)
            output_dir = Path(tmpdir)

            batch = SimulationBatch(
                simulator_name="test",
                simulator_config=SimulatorConfig(
                    class_="MockSimulator",
                    output=SimulatorOutputConfig(file_name="output.json"),
                ),
                globals_config=GlobalsConfig(),
                batch_index=0,
            )

            output_files = process_batch(sim, 42, batch, output_dir, overwrite=True)

            assert isinstance(output_files, list)
            assert all(isinstance(f, Path) for f in output_files)


class TestExecutePlan:
    """Test execute_plan function."""

    def test_execute_plan_single_simulator(self):
        """Test executing a plan with one simulator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            metadata_dir = Path(tmpdir) / "metadata"

            batch = SimulationBatch(
                simulator_name="mock",
                simulator_config=SimulatorConfig(
                    class_="tests.cli.test_cli_simulate.MockSimulator",
                    arguments={"seed": 42},
                    output=SimulatorOutputConfig(file_name="data.json"),
                ),
                globals_config=GlobalsConfig(),
                batch_index=0,
            )

            plan = SimulationPlan(checkpoint_directory=Path(tmpdir) / "checkpoints")
            plan.add_batch(batch)

            execute_plan(plan, output_dir, metadata_dir, overwrite=True)

            # Verify output file exists
            assert (output_dir / "data.json").exists()
            # Verify metadata file exists
            assert (metadata_dir / "mock-0.metadata.json").exists()

    def test_execute_plan_multiple_batches_same_simulator(self):
        """Test executing multiple batches for same simulator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            metadata_dir = Path(tmpdir) / "metadata"

            plan = SimulationPlan(checkpoint_directory=Path(tmpdir) / "checkpoints")
            for i in range(3):
                batch = SimulationBatch(
                    simulator_name="mock",
                    simulator_config=SimulatorConfig(
                        class_="tests.cli.test_cli_simulate.MockSimulator",
                        arguments={"seed": 42},
                        output=SimulatorOutputConfig(file_name=f"batch_{i}.json"),
                    ),
                    globals_config=GlobalsConfig(),
                    batch_index=i,
                )
                plan.add_batch(batch)

            execute_plan(plan, output_dir, metadata_dir, overwrite=True)

            # Verify all output files exist
            for i in range(3):
                assert (output_dir / f"batch_{i}.json").exists()

            # Verify all metadata files exist
            for i in range(3):
                assert (metadata_dir / f"mock-{i}.metadata.json").exists()

    def test_execute_plan_maintains_simulator_state(self):
        """Test that simulator state persists across batches.

        This is critical: the simulator should be created once and
        generate multiple batches, with state accumulating.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            metadata_dir = Path(tmpdir) / "metadata"

            # Keep the checkpoint inside the temporary directory: the default is a
            # relative "checkpoints/", which writes into the working directory and
            # makes this test fail against a checkpoint left by an earlier run.
            plan = SimulationPlan(checkpoint_directory=Path(tmpdir) / "checkpoints")
            for i in range(2):
                batch = SimulationBatch(
                    simulator_name="mock",
                    simulator_config=SimulatorConfig(
                        class_="tests.cli.test_cli_simulate.MockSimulator",
                        arguments={"seed": 42},
                        output=SimulatorOutputConfig(file_name=f"batch_{i}.json"),
                    ),
                    globals_config=GlobalsConfig(),
                    batch_index=i,
                )
                plan.add_batch(batch)

            execute_plan(plan, output_dir, metadata_dir, overwrite=True)

            # Load metadata files and check counter progression
            metadata_0 = yaml.safe_load((metadata_dir / "mock-0.metadata.json").open())
            metadata_1 = yaml.safe_load((metadata_dir / "mock-1.metadata.json").open())

            # Batch 1 should have higher counter than batch 0
            counter_0 = metadata_0["pre_batch_state"]["counter"]
            counter_1 = metadata_1["pre_batch_state"]["counter"]
            assert counter_1 > counter_0


class TestCheckpointReconciliation:
    """Resume must reconcile the checkpoint against the filesystem (ISS-013).

    A batch may be recorded as completed in the checkpoint while its output is
    missing (partial write at interrupt, external move/backup, fs hiccup). Trusting
    the checkpoint blindly silently drops files; an orphan output from a not-yet
    checkpointed batch must not abort resume with FileExistsError.
    """

    @staticmethod
    def _build_plan(checkpoint_dir: Path, n_batches: int) -> SimulationPlan:
        plan = SimulationPlan(checkpoint_directory=checkpoint_dir)
        for i in range(n_batches):
            plan.add_batch(
                SimulationBatch(
                    simulator_name="mock",
                    simulator_config=SimulatorConfig(
                        class_="tests.cli.test_cli_simulate.MockSimulator",
                        arguments={"seed": 42},
                        output=SimulatorOutputConfig(file_name=f"batch_{i}.json"),
                    ),
                    globals_config=GlobalsConfig(),
                    batch_index=i,
                )
            )
        return plan

    @staticmethod
    def _fingerprint_of(plan: SimulationPlan, output_dir: Path, metadata_dir: Path) -> str:
        """The identity `execute_plan` computes for this run, so a planted checkpoint is its own.

        A resume refuses any checkpoint it cannot attribute to the configuration being run --
        including one carrying no fingerprint at all -- so a checkpoint planted to reach the
        reconciliation these tests are about has to carry this run's. Taken from the same function
        the run itself uses rather than written as a literal, so it follows a change in what
        identity means instead of turning it into an unrelated refusal here. No referenced inputs:
        these plans name no population file.
        """
        return run_fingerprint([batch.config_sha256 for batch in plan.batches], output_dir, metadata_dir)

    @staticmethod
    def _stepped_state(plan: SimulationPlan, steps: int) -> dict[str, Any]:
        """Return a valid simulator state advanced ``steps`` times (counter == steps)."""
        simulator = instantiate_simulator(plan.batches[0].simulator_config, "mock", {})
        for _ in range(steps):
            simulator.simulate()
            simulator.update_state()
        return simulator.state

    def test_reconcile_truncates_at_first_missing_artifact(self):
        """reconcile_completed_batches keeps the valid contiguous prefix only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            metadata_dir = Path(tmpdir) / "metadata"
            plan = self._build_plan(Path(tmpdir) / "ckpt", n_batches=3)

            execute_plan(plan, output_dir, metadata_dir, overwrite=True)

            # All outputs present -> whole set is valid.
            assert reconcile_completed_batches(plan, metadata_dir, {0, 1, 2}) == {0, 1, 2}

            # A missing output truncates at that batch (it and later batches re-run).
            (output_dir / "batch_1.json").unlink()
            assert reconcile_completed_batches(plan, metadata_dir, {0, 1, 2}) == {0}

            # A missing metadata file truncates there too.
            (metadata_dir / "mock-0.metadata.json").unlink()
            assert reconcile_completed_batches(plan, metadata_dir, {0, 1, 2}) == set()

    def test_resume_regenerates_when_checkpointed_output_missing(self):
        """User scenario: checkpoint says batch done but its output is gone -> regenerate.

        Before the fix this silently reported success with fewer files than configured.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            metadata_dir = Path(tmpdir) / "metadata"
            checkpoint_dir = Path(tmpdir) / "ckpt"
            plan = self._build_plan(checkpoint_dir, n_batches=3)

            # Produce a complete run, then plant a stale checkpoint claiming 0,1,2 done.
            execute_plan(plan, output_dir, metadata_dir, overwrite=True)
            CheckpointManager(checkpoint_dir).save_checkpoint(
                completed_batch_indices=[0, 1, 2],
                last_simulator_name="mock",
                last_completed_batch_index=2,
                last_simulator_state=self._stepped_state(plan, 3),
                config_sha256=self._fingerprint_of(plan, output_dir, metadata_dir),
            )

            # The first batch's output goes missing while the checkpoint persists.
            (output_dir / "batch_0.json").unlink()

            execute_plan(plan, output_dir, metadata_dir, overwrite=True)

            # Every configured batch output exists again (no silent drop).
            for i in range(3):
                assert (output_dir / f"batch_{i}.json").exists()

    def test_resume_tolerates_orphan_output_without_overwrite_flag(self):
        """An orphan output for a not-checkpointed batch must not abort resume.

        Before the fix this raised FileExistsError (overwrite=False) and resume could
        not proceed at all.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            metadata_dir = Path(tmpdir) / "metadata"
            checkpoint_dir = Path(tmpdir) / "ckpt"
            plan = self._build_plan(checkpoint_dir, n_batches=3)

            # All outputs on disk, but the checkpoint only records batch 0 as complete,
            # so batch_1.json / batch_2.json are orphan leftovers relative to it.
            execute_plan(plan, output_dir, metadata_dir, overwrite=True)
            CheckpointManager(checkpoint_dir).save_checkpoint(
                completed_batch_indices=[0],
                last_simulator_name="mock",
                last_completed_batch_index=0,
                last_simulator_state=self._stepped_state(plan, 1),
                config_sha256=self._fingerprint_of(plan, output_dir, metadata_dir),
            )

            # Resume WITHOUT --overwrite must succeed (orphans are unverified leftovers).
            execute_plan(plan, output_dir, metadata_dir, overwrite=False)

            for i in range(3):
                assert (output_dir / f"batch_{i}.json").exists()

    def test_resume_skips_completed_batches_with_present_outputs(self):
        """Happy path: validated-complete batches are skipped, not regenerated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            metadata_dir = Path(tmpdir) / "metadata"
            checkpoint_dir = Path(tmpdir) / "ckpt"
            plan = self._build_plan(checkpoint_dir, n_batches=3)

            execute_plan(plan, output_dir, metadata_dir, overwrite=True)
            CheckpointManager(checkpoint_dir).save_checkpoint(
                completed_batch_indices=[0, 1],
                last_simulator_name="mock",
                last_completed_batch_index=1,
                last_simulator_state=self._stepped_state(plan, 2),
                config_sha256=self._fingerprint_of(plan, output_dir, metadata_dir),
            )

            # Sentinel content on the completed batches: untouched iff they are skipped.
            sentinel = '{"sentinel": true}'
            (output_dir / "batch_0.json").write_text(sentinel)
            (output_dir / "batch_1.json").write_text(sentinel)

            execute_plan(plan, output_dir, metadata_dir, overwrite=True)

            assert (output_dir / "batch_0.json").read_text() == sentinel
            assert (output_dir / "batch_1.json").read_text() == sentinel
            assert (output_dir / "batch_2.json").exists()


@pytest.mark.skip(reason="Legacy simulator configs removed; orchestration tests cover this path.")
class TestCreateSimulationPlanFromConfig:
    """Test creating simulation plans from configs."""

    def test_create_plan_single_simulator(self):
        """Test creating a plan from config with one simulator."""
        config = Config(
            globals=GlobalsConfig(
                working_directory=".",
                simulator_arguments={"sampling_frequency": 4096},
            ),
            simulators={
                "mock": SimulatorConfig(
                    class_="tests.cli.test_cli_simulate.MockSimulator",
                    arguments={"seed": 42},
                    output=SimulatorOutputConfig(file_name="output.json"),
                )
            },
        )

        plan = create_plan_from_config(config, Path("checkpoints"))

        assert plan.total_batches == 1
        assert len(plan.batches) == 1
        assert plan.batches[0].simulator_name == "mock"

    def test_create_plan_multiple_simulators(self):
        """Test creating a plan with multiple simulators."""
        config = Config(
            globals=GlobalsConfig(),
            simulators={
                "mock1": SimulatorConfig(
                    class_="tests.cli.test_cli_simulate.MockSimulator",
                    arguments={"seed": 1},
                    output=SimulatorOutputConfig(file_name="out1.json"),
                ),
                "mock2": SimulatorConfig(
                    class_="tests.cli.test_cli_simulate.MockSimulator",
                    arguments={"seed": 2},
                    output=SimulatorOutputConfig(file_name="out2.json"),
                ),
            },
        )

        plan = create_plan_from_config(config, Path("checkpoints"))

        assert plan.total_batches == NUM_FILES_2
        simulator_names = {b.simulator_name for b in plan.batches}
        assert simulator_names == {"mock1", "mock2"}


@pytest.mark.skip(reason="Legacy simulator configs removed; orchestration tests cover this path.")
class TestSimulateCommandIntegration:
    """Integration tests for the simulate_command."""

    def test_simulate_command_with_config_file(self):
        """Test simulate command with a real config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create config file
            config = Config(
                globals=GlobalsConfig(
                    working_directory=str(tmpdir_path),
                    output_directory="output",
                    metadata_directory="metadata",
                ),
                simulators={
                    "mock": SimulatorConfig(
                        class_="tests.cli.test_cli_simulate.MockSimulator",
                        arguments={"seed": 42},
                        output=SimulatorOutputConfig(file_name="data.json"),
                    )
                },
            )

            config_file = tmpdir_path / "config.yaml"
            with config_file.open("w") as f:
                # Convert config to dict with aliases for YAML
                config_dict = config.model_dump(by_alias=True, exclude_none=True)
                yaml.safe_dump(config_dict, f)

            # Run simulate command
            _simulate_impl(str(config_file), overwrite=True, metadata=True)

            # Verify output structure
            # Paths should be resolved relative to working_directory
            output_dir = tmpdir_path / "output"
            metadata_dir = tmpdir_path / "metadata"

            assert (output_dir / "data.json").exists(), f"Output file not found at {output_dir / 'data.json'}"
            assert (metadata_dir / "mock-0.metadata.json").exists(), (
                f"Metadata file not found at {metadata_dir / 'mock-0.metadata.json'}"
            )
            assert (metadata_dir / "index.yaml").exists(), f"Index file not found at {metadata_dir / 'index.yaml'}"

    def test_simulate_command_data_correctness(self):
        """Test that simulate command produces correct data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            config = Config(
                globals=GlobalsConfig(
                    working_directory=str(tmpdir_path),
                    output_directory="output",
                ),
                simulators={
                    "mock": SimulatorConfig(
                        class_="tests.cli.test_cli_simulate.MockSimulator",
                        arguments={"seed": 42, "max_samples": 1},
                        output=SimulatorOutputConfig(file_name="data.json"),
                    )
                },
            )

            config_file = tmpdir_path / "config.yaml"
            with config_file.open("w") as f:
                config_dict = config.model_dump(by_alias=True, exclude_none=True)
                yaml.safe_dump(config_dict, f)

            _simulate_impl(str(config_file), overwrite=True, metadata=False)

            # Verify data format
            output_file = tmpdir_path / "output" / "data.json"
            with output_file.open() as f:
                data = json.load(f)

            assert "data" in data
            assert "counter" in data

    def test_simulate_command_with_hyphenated_yaml_keys(self):
        """Test simulate command properly handles hyphenated YAML keys (e.g., 'sampling-frequency')."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create config using dict and dump to YAML (more reliable than f-strings)
            config_dict = {
                "globals": {
                    "working-directory": str(tmpdir_path),
                    "simulator-arguments": {
                        "sampling-frequency": 2048,
                        "max-samples": 3,
                    },
                    "output-directory": "output",
                    "metadata-directory": "metadata",
                },
                "simulators": {
                    "mock": {
                        "class": "tests.cli.test_cli_simulate.MockSimulator",
                        "arguments": {
                            "seed": 42,
                        },
                        "output": {
                            "file_name": "data.json",
                        },
                    }
                },
            }

            config_file = tmpdir_path / "config.yaml"
            with config_file.open("w") as f:
                yaml.safe_dump(config_dict, f)

            # This should not raise any errors about unused kwargs with hyphens
            _simulate_impl(str(config_file), overwrite=True, metadata=True)

            # Verify output was created successfully
            output_file = tmpdir_path / "output" / "data.json"
            assert output_file.exists(), "Output file not created - hyphenated key conversion may have failed"

            # Verify metadata was created - should have 3 files (one per batch)
            metadata_dir = tmpdir_path / "metadata"
            assert metadata_dir.exists(), "Metadata directory not created"

            # Check for all 3 batch metadata files
            metadata_files = list(metadata_dir.glob("mock-*.metadata.json"))
            assert len(metadata_files) == NUM_FILES_3, (
                f"Expected 3 batch metadata files (max-samples: 3), but found {len(metadata_files)}: {metadata_files}"
            )

            assert (metadata_dir / "index.yaml").exists(), "Index file not created"

    def test_simulate_command_reproduce_from_metadata(self):
        """Test that the CLI can reproduce batches using a metadata directory.

        This test demonstrates the user workflow for metadata-based reproduction:
        1. Run initial simulation with max-samples=3, saving metadata
        2. Run CLI again with the metadata directory to reproduce those batches
        3. Verify the reproduced batches match the original execution

        This is the realistic user scenario for exact reproducibility.
        The CLI automatically detects whether the argument is a config file or
        metadata directory, making the interface intuitive.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # ===== STEP 1: Initial simulation with multiple batches =====
            config_dict = {
                "globals": {
                    "working-directory": str(tmpdir_path),
                    "simulator-arguments": {
                        "sampling-frequency": 2048,
                        "max-samples": 3,
                    },
                    "output-directory": "output",
                    "metadata-directory": "metadata",
                },
                "simulators": {
                    "mock": {
                        "class": "tests.cli.test_cli_simulate.MockSimulator",
                        "arguments": {
                            "seed": 42,
                        },
                        "output": {
                            "file_name": "batch_{{counter}}.json",
                        },
                    }
                },
            }

            config_file = tmpdir_path / "config.yaml"
            with config_file.open("w") as f:
                yaml.safe_dump(config_dict, f)

            # Run initial simulation
            _simulate_impl(str(config_file), overwrite=True, metadata=True)

            # Verify initial metadata was created
            metadata_dir = tmpdir_path / "metadata"
            initial_metadata_files = sorted(metadata_dir.glob("mock-*.metadata.json"))
            assert len(initial_metadata_files) == NUM_FILES_3, (
                f"Expected 3 metadata files from initial run, but found {len(initial_metadata_files)}"
            )

            # Load metadata for batch 1 (the middle batch)
            batch_1_metadata_file = metadata_dir / "mock-1.metadata.json"
            batch_1_metadata = parse_batch_metadata(batch_1_metadata_file)

            # Verify metadata has pre-batch state
            assert "pre_batch_state" in batch_1_metadata, (
                "Metadata must contain pre_batch_state for exact reproducibility"
            )
            batch_1_initial_counter = batch_1_metadata["pre_batch_state"].get("counter")
            assert batch_1_initial_counter is not None, "pre_batch_state should contain counter value"

            # ===== STEP 2: Use CLI to reproduce from metadata directory =====
            # User runs: gwmock simulate metadata/
            # The CLI automatically detects this is a metadata directory and reproduces
            _simulate_impl(
                str(metadata_dir),  # Pass metadata directory instead of config file
                overwrite=True,
                metadata=False,  # Don't create new metadata during reproduction
            )

            # ===== STEP 3: Verify reproduction worked =====
            # After reproduction, the output files should exist in the default output dir
            reproduced_files = list((tmpdir_path / "output").glob("batch_*.json"))
            assert len(reproduced_files) > 0, (
                "Reproduced batches should create output files"
            )  # Verify each reproduced batch file contains counter information
            for batch_file in reproduced_files:
                with batch_file.open() as f:
                    batch_data = json.load(f)
                assert "counter" in batch_data, f"Reproduced batch {batch_file.name} should contain counter"
                assert "data" in batch_data, f"Reproduced batch {batch_file.name} should contain data"

            # ===== STEP 4: Verify reproducibility is exact =====
            # Load the reproduced batch 1 output
            reproduced_batch_1_file = tmpdir_path / "output" / "batch_1.json"
            assert reproduced_batch_1_file.exists(), "Batch 1 output file should exist after reproduction"

            # Load the reproduced batch 1 metadata to check the pre_batch_state
            reproduced_batch_1_metadata_file = metadata_dir / "mock-1.metadata.json"
            reproduced_batch_1_metadata = parse_batch_metadata(reproduced_batch_1_metadata_file)
            reproduced_pre_batch_state = reproduced_batch_1_metadata.get("pre_batch_state", {})
            reproduced_counter = reproduced_pre_batch_state.get("counter")

            # The reproduced counter should match the pre_batch_state counter
            # since pre_batch_state is captured at the start of batch generation
            assert reproduced_counter == batch_1_initial_counter, (
                f"Reproduced batch 1 counter in metadata should be {batch_1_initial_counter}, "
                f"but got {reproduced_counter}. This indicates exact reproducibility "
                f"from metadata FAILED."
            )

    def test_simulate_command_batch_reproducibility_via_metadata(self):
        """Test that individual batches can be reproduced exactly using saved metadata.

        This test verifies the reproducibility mechanism:
        1. Run initial simulation with max-samples=3, saving metadata
        2. Verify that batch 1 (second batch) metadata contains pre-batch state
        3. Load and re-run batch 1 in isolation, verifying state restoration works
        4. Confirm that running batch 1 again produces identical results

        This simulates a user workflow where they might:
        - Run an initial MDC generation
        - Need to reproduce/verify a specific batch
        - Use metadata to exactly recreate that batch
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # ===== STEP 1: Initial simulation with multiple batches =====
            config_dict = {
                "globals": {
                    "working-directory": str(tmpdir_path),
                    "simulator-arguments": {
                        "sampling-frequency": 2048,
                        "max-samples": 3,
                    },
                    "output-directory": "output",
                    "metadata-directory": "metadata",
                },
                "simulators": {
                    "mock": {
                        "class": "tests.cli.test_cli_simulate.MockSimulator",
                        "arguments": {
                            "seed": 42,
                        },
                        "output": {
                            "file_name": "batch_{{counter}}.json",
                        },
                    }
                },
            }

            config_file = tmpdir_path / "config.yaml"
            with config_file.open("w") as f:
                yaml.safe_dump(config_dict, f)

            # Run initial simulation
            _simulate_impl(str(config_file), overwrite=True, metadata=True)

            # ===== STEP 2: Verify metadata for batch 1 =====
            metadata_dir = tmpdir_path / "metadata"
            batch_1_metadata_file = metadata_dir / "mock-1.metadata.json"
            assert batch_1_metadata_file.exists(), "Batch 1 metadata file not created"

            # Load batch 1's metadata

            metadata = parse_batch_metadata(batch_1_metadata_file)

            # Verify metadata has the required reproducibility info
            assert "pre_batch_state" in metadata, "Metadata must contain pre_batch_state for exact reproducibility"
            assert metadata.get("source") == "config", (
                "Initial simulation should have source='config' (not a recovery from checkpoint)"
            )

            # Store the counter value from batch 1's metadata
            # This represents the RNG/state position before batch 1 was generated
            batch_1_initial_counter = metadata["pre_batch_state"].get("counter")
            assert batch_1_initial_counter is not None, "pre_batch_state should contain counter value"

            # ===== STEP 3: Reproduce batch 1 independently =====
            # Instantiate a fresh simulator with the same configuration
            simulator_config = SimulatorConfig(**metadata["simulator_config"])
            globals_config = GlobalsConfig(**metadata["globals_config"])

            # Normalize keys for simulator instantiation (this is what happens in simulate.py)
            global_sim_args = {k.replace("-", "_"): v for k, v in globals_config.simulator_arguments.items()}
            local_sim_args = {k.replace("-", "_"): v for k, v in simulator_config.arguments.items()}
            merged_args = {**global_sim_args, **local_sim_args}

            # Create simulator with same seed and arguments
            sim = MockSimulator(**merged_args)

            # Restore state to exactly what it was before batch 1
            sim.counter = metadata["pre_batch_state"].get("counter", 0)

            # Verify RNG can be restored if state was saved
            if "rng_state" in metadata["pre_batch_state"]:
                state_dict = metadata["pre_batch_state"]["rng_state"]
                sim.rng.bit_generator.state = state_dict

            # Generate one sample (simulating batch 1 generation)
            sim.simulate()

            # ===== STEP 4: Verify reproducibility =====
            # After restoration, counter should match the pre_batch_state
            # since pre_batch_state is the starting state before batch increments
            assert sim.counter == batch_1_initial_counter, (
                f"After state restoration, "
                f"counter should be {batch_1_initial_counter}, "
                f"but got {sim.counter}. This indicates state restoration failed."
            )

            # ===== STEP 5: Verify multiple batches can be reproduced independently =====
            # Test batch 2 (third batch) as well
            batch_2_metadata_file = metadata_dir / "mock-2.metadata.json"
            assert batch_2_metadata_file.exists(), "Batch 2 metadata file not created"

            metadata_batch2 = parse_batch_metadata(batch_2_metadata_file)
            batch_2_initial_counter = metadata_batch2["pre_batch_state"].get("counter")

            # Create fresh simulator for batch 2
            sim2 = MockSimulator(**merged_args)
            sim2.counter = batch_2_initial_counter
            sim2.simulate()

            # Verify batch 2 can also be reproduced
            assert sim2.counter == batch_2_initial_counter, (
                f"Batch 2 reproducibility failed: counter should be {batch_2_initial_counter}, but got {sim2.counter}"
            )

            # Verify batches have different initial states
            # (batch 1 counter != batch 2 counter because RNG advances between batches)
            assert batch_1_initial_counter != batch_2_initial_counter, (
                "Batch 1 and batch 2 should have different initial counters, indicating correct state progression"
            )

    def test_simulate_command_dry_run_mode(self):
        """Test that dry-run mode validates the plan without executing or creating files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create config file
            config = Config(
                globals=GlobalsConfig(
                    working_directory=str(tmpdir_path),
                    output_directory="output",
                    metadata_directory="metadata",
                ),
                simulators={
                    "mock": SimulatorConfig(
                        class_="tests.cli.test_cli_simulate.MockSimulator",
                        arguments={"seed": 42},
                        output=SimulatorOutputConfig(file_name="data.json"),
                    )
                },
            )

            config_file = tmpdir_path / "config.yaml"
            with config_file.open("w") as f:
                config_dict = config.model_dump(by_alias=True, exclude_none=True)
                yaml.safe_dump(config_dict, f)

            # Run simulate command in dry-run mode
            with patch("logging.getLogger") as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger
                _simulate_impl(str(config_file), dry_run=True, overwrite=True, metadata=False)

            # Verify dry-run log message
            mock_logger.info.assert_called_with("Dry run mode: Simulation plan validated but not executed")

            # Verify no output files were created
            output_dir = tmpdir_path / "output"
            assert not output_dir.exists() or not list(output_dir.glob("*")), (
                "No output files should be created in dry-run mode"
            )

            # Verify no metadata files were created
            metadata_dir = tmpdir_path / "metadata"
            assert not metadata_dir.exists() or not list(metadata_dir.glob("*")), (
                "No metadata files should be created in dry-run mode"
            )


@pytest.mark.skip(reason="Legacy simulator configs removed; orchestration tests cover checkpoint behavior.")
class TestSimulateCommandCheckpoint:
    """Test checkpoint functionality for simulation resumption."""

    def test_checkpoint_created_after_batch_completion(self):
        """Test that checkpoint is created after each batch completes successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            checkpoint_dir = tmpdir_path / ".gwmock_checkpoints"

            config_dict = {
                "globals": {
                    "working-directory": str(tmpdir_path),
                    "simulator-arguments": {
                        "max-samples": 2,
                    },
                    "output-directory": "output",
                    "metadata-directory": "metadata",
                },
                "simulators": {
                    "mock": {
                        "class": "tests.cli.test_cli_simulate.MockSimulator",
                        "arguments": {
                            "seed": 42,
                        },
                        "output": {
                            "file_name": "batch_{{counter}}.json",
                        },
                    }
                },
            }

            config_file = tmpdir_path / "config.yaml"
            with config_file.open("w") as f:
                yaml.safe_dump(config_dict, f)

            # Run simulation
            _simulate_impl(str(config_file), overwrite=True, metadata=True)

            # Verify checkpoint was created and then cleaned up (successful completion)
            assert not checkpoint_dir.exists() or not list(checkpoint_dir.glob("simulation.checkpoint.json*")), (
                "Checkpoint files should be cleaned up after successful completion"
            )

    def test_checkpoint_skips_already_completed_batches(self):
        """Test that checkpoint allows resumption by skipping completed batches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            config_dict = {
                "globals": {
                    "working-directory": str(tmpdir_path),
                    "simulator-arguments": {
                        "max-samples": 3,
                    },
                    "output-directory": "output",
                    "metadata-directory": "metadata",
                },
                "simulators": {
                    "mock": {
                        "class": "tests.cli.test_cli_simulate.MockSimulator",
                        "arguments": {
                            "seed": 42,
                        },
                        "output": {
                            "file_name": "batch_{{counter}}.json",
                        },
                    }
                },
            }

            config_file = tmpdir_path / "config.yaml"
            with config_file.open("w") as f:
                yaml.safe_dump(config_dict, f)

            # First run - should complete 3 batches
            _simulate_impl(str(config_file), overwrite=True, metadata=True)

            # Verify output files exist
            output_dir = tmpdir_path / "output"
            initial_files = sorted(output_dir.glob("batch_*.json"))
            assert len(initial_files) == NUM_FILES_3, f"Expected 3 output files, got {len(initial_files)}"

            # Delete one output file to simulate partial failure
            initial_files[1].unlink()

            # Run again - should skip completed batches and regenerate only the missing one
            # This verifies that checkpoint restoration works
            _simulate_impl(str(config_file), overwrite=True, metadata=True)

            # Verify files are regenerated
            final_files = sorted(output_dir.glob("batch_*.json"))
            assert len(final_files) == NUM_FILES_3, "All batches should be re-generated or skipped correctly"

    def test_checkpoint_contains_simulator_state(self):
        """Test that checkpoint saves and restores simulator state correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            checkpoint_dir = tmpdir_path / ".gwmock_checkpoints"

            config_dict = {
                "globals": {
                    "working-directory": str(tmpdir_path),
                    "simulator-arguments": {
                        "max-samples": 2,
                    },
                    "output-directory": "output",
                    "metadata-directory": "metadata",
                },
                "simulators": {
                    "mock": {
                        "class": "tests.cli.test_cli_simulate.MockSimulator",
                        "arguments": {
                            "seed": 42,
                        },
                        "output": {
                            "file_name": "batch_{{counter}}.json",
                        },
                    }
                },
            }

            config_file = tmpdir_path / "config.yaml"
            with config_file.open("w") as f:
                yaml.safe_dump(config_dict, f)

            # Run simulation that completes successfully
            _simulate_impl(str(config_file), overwrite=True, metadata=True)

            # Verify checkpoint was cleaned up (no residual files)
            checkpoint_files = (
                list(checkpoint_dir.glob("simulation.checkpoint.json*")) if checkpoint_dir.exists() else []
            )
            assert len(checkpoint_files) == 0, "Checkpoint files should be deleted after successful completion"

    def test_checkpoint_recovery_from_interrupt(self):
        """Test that checkpoint persists during execution and is cleaned up after success.

        This test verifies that:
        1. Checkpoint is created during batch execution
        2. Checkpoint is cleaned up after all batches complete successfully
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            checkpoint_dir = tmpdir_path / ".gwmock_checkpoints"

            config_dict = {
                "globals": {
                    "working-directory": str(tmpdir_path),
                    "simulator-arguments": {
                        "max-samples": 2,
                    },
                    "output-directory": "output",
                    "metadata-directory": "metadata",
                },
                "simulators": {
                    "mock": {
                        "class": "tests.cli.test_cli_simulate.MockSimulator",
                        "arguments": {
                            "seed": 42,
                        },
                        "output": {
                            "file_name": "batch_{{counter}}.json",
                        },
                    }
                },
            }

            config_file = tmpdir_path / "config.yaml"
            with config_file.open("w") as f:
                yaml.safe_dump(config_dict, f)

            # Run simulation normally
            _simulate_impl(str(config_file), overwrite=True, metadata=True)

            # Verify output files were created
            output_dir = tmpdir_path / "output"
            output_files = sorted(output_dir.glob("batch_*.json"))
            assert len(output_files) == MAX_SAMPLES_2, f"Expected 2 output files, got {len(output_files)}"

            # Checkpoint should be cleaned up after successful completion
            checkpoint_files = (
                list(checkpoint_dir.glob("simulation.checkpoint.json*")) if checkpoint_dir.exists() else []
            )
            assert len(checkpoint_files) == 0, (
                f"Checkpoint should be cleaned up after completion, but found: {checkpoint_files}"
            )

    def test_multiple_simulators_with_checkpoint(self):
        """Test checkpoint behavior with multiple simulators."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            checkpoint_dir = tmpdir_path / ".gwmock_checkpoints"

            config_dict = {
                "globals": {
                    "working-directory": str(tmpdir_path),
                    "output-directory": "output",
                    "metadata-directory": "metadata",
                },
                "simulators": {
                    "mock1": {
                        "class": "tests.cli.test_cli_simulate.MockSimulator",
                        "arguments": {
                            "seed": 1,
                            "max-samples": 2,
                        },
                        "output": {
                            "file_name": "mock1_{{counter}}.json",
                        },
                    },
                    "mock2": {
                        "class": "tests.cli.test_cli_simulate.MockSimulator",
                        "arguments": {
                            "seed": 2,
                            "max-samples": 2,
                        },
                        "output": {
                            "file_name": "mock2_{{counter}}.json",
                        },
                    },
                },
            }

            config_file = tmpdir_path / "config.yaml"
            with config_file.open("w") as f:
                yaml.safe_dump(config_dict, f)

            # Run simulation with multiple simulators
            _simulate_impl(str(config_file), overwrite=True, metadata=True)

            # Verify both simulators produced output
            output_dir = tmpdir_path / "output"
            mock1_files = sorted(output_dir.glob("mock1_*.json"))
            mock2_files = sorted(output_dir.glob("mock2_*.json"))
            assert len(mock1_files) == MAX_SAMPLES_2, f"mock1 should produce 2 files, got {len(mock1_files)}"
            assert len(mock2_files) == MAX_SAMPLES_2, f"mock2 should produce 2 files, got {len(mock2_files)}"

            # Checkpoint should be cleaned up
            checkpoint_files = (
                list(checkpoint_dir.glob("simulation.checkpoint.json*")) if checkpoint_dir.exists() else []
            )
            assert len(checkpoint_files) == 0, "Checkpoint should be cleaned up after successful completion"


class TestHostMetadata:
    """Tests for distribution-backed host provenance metadata."""

    def test_get_host_metadata_reads_git_sha_from_direct_url_json(self):
        """Host metadata should use the gwmock distribution Direct URL commit ID."""

        class _DistributionStub:
            metadata: ClassVar[dict[str, Any]] = {}

            @staticmethod
            def read_text(path: str) -> str | None:
                if path == "direct_url.json":
                    return '{"url":"https://example.invalid/repo","vcs_info":{"vcs":"git","commit_id":"abc123def"}}'
                return None

        with patch("gwmock.cli.simulate_utils.importlib_metadata.distribution", return_value=_DistributionStub()):
            host = _get_host_metadata()

        assert host["git_sha"] == "abc123def"

    def test_get_host_metadata_sets_git_sha_none_when_distribution_missing(self):
        """With no distribution VCS metadata and no source repo, git_sha is None."""
        with (
            patch(
                "gwmock.cli.simulate_utils.importlib_metadata.distribution",
                side_effect=stdlib_metadata.PackageNotFoundError,
            ),
            patch("gwmock.cli.simulate_utils._get_source_tree_git_sha", return_value=None),
        ):
            host = _get_host_metadata()

        assert host["git_sha"] is None

    def test_get_host_metadata_falls_back_to_source_tree_git_sha(self):
        """When the distribution carries no VCS metadata, the source-tree commit is used."""
        with (
            patch(
                "gwmock.cli.simulate_utils.importlib_metadata.distribution",
                side_effect=stdlib_metadata.PackageNotFoundError,
            ),
            patch("gwmock.cli.simulate_utils._get_source_tree_git_sha", return_value="deadbeef-dirty"),
        ):
            host = _get_host_metadata()

        assert host["git_sha"] == "deadbeef-dirty"

    def test_get_host_metadata_prefers_distribution_over_source_tree(self):
        """A distribution VCS commit wins; the source-tree fallback is not consulted."""

        class _DistributionStub:
            metadata: ClassVar[dict[str, Any]] = {}

            @staticmethod
            def read_text(path: str) -> str | None:
                if path == "direct_url.json":
                    return '{"url":"https://example.invalid/repo","vcs_info":{"vcs":"git","commit_id":"abc123def"}}'
                return None

        with (
            patch("gwmock.cli.simulate_utils.importlib_metadata.distribution", return_value=_DistributionStub()),
            patch("gwmock.cli.simulate_utils._get_source_tree_git_sha") as mock_source,
        ):
            host = _get_host_metadata()

        assert host["git_sha"] == "abc123def"
        mock_source.assert_not_called()


class TestSourceTreeGitSha:
    """Tests for the source-checkout git-commit fallback."""

    def test_returns_none_when_git_missing(self):
        """No git executable on PATH yields None rather than raising."""
        with patch("gwmock.cli.simulate_utils.shutil.which", return_value=None):
            assert _get_source_tree_git_sha() is None

    def test_returns_commit_from_clean_tree(self):
        """A clean working tree returns the bare commit hash."""

        def _run(cmd, **kwargs):
            out = "cafe1234\n" if "rev-parse" in cmd else ""
            return SimpleNamespace(returncode=0, stdout=out, stderr="")

        with (
            patch("gwmock.cli.simulate_utils.shutil.which", return_value="/usr/bin/git"),
            patch("gwmock.cli.simulate_utils.subprocess.run", side_effect=_run),
        ):
            assert _get_source_tree_git_sha() == "cafe1234"

    def test_marks_dirty_tree(self):
        """Uncommitted changes append a -dirty suffix."""

        def _run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return SimpleNamespace(returncode=0, stdout="cafe1234\n", stderr="")
            return SimpleNamespace(returncode=0, stdout=" M src/gwmock/foo.py\n", stderr="")

        with (
            patch("gwmock.cli.simulate_utils.shutil.which", return_value="/usr/bin/git"),
            patch("gwmock.cli.simulate_utils.subprocess.run", side_effect=_run),
        ):
            assert _get_source_tree_git_sha() == "cafe1234-dirty"

    def test_returns_none_when_not_a_repository(self):
        """A non-repository source directory (rev-parse fails) yields None."""

        def _run(cmd, **kwargs):
            return SimpleNamespace(returncode=128, stdout="", stderr="not a git repository")

        with (
            patch("gwmock.cli.simulate_utils.shutil.which", return_value="/usr/bin/git"),
            patch("gwmock.cli.simulate_utils.subprocess.run", side_effect=_run),
        ):
            assert _get_source_tree_git_sha() is None

    def test_returns_none_on_subprocess_error(self):
        """A subprocess failure (e.g. timeout) is swallowed to None."""
        with (
            patch("gwmock.cli.simulate_utils.shutil.which", return_value="/usr/bin/git"),
            patch("gwmock.cli.simulate_utils.subprocess.run", side_effect=OSError("boom")),
        ):
            assert _get_source_tree_git_sha() is None
