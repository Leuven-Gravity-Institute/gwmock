"""Custom build hook to include the example configurations and their data in gwmock/examples/."""

from __future__ import annotations

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Copy the example YAML files, and the CSV catalogues they reference, into gwmock/examples/.

    The catalogues are small and ship so that an example copied with ``gwmock config --get`` runs
    without network access.
    """

    #: File patterns copied from examples/ into the wheel.
    PATTERNS = ("*.yaml", "*.csv")

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:  # pylint: disable=unused-argument
        """Run during wheel build to copy the example files."""
        if self.target_name != "wheel":
            return  # Only apply to wheel builds

        examples_src = Path(self.root) / "examples"
        if not examples_src.exists():
            return

        # Force-include will be handled by copying files to the build directory
        # We need to add files to the wheel's force_include dynamically
        force_include = build_data.setdefault("force_include", {})

        num_files = 0
        for pattern in self.PATTERNS:
            for example_file in examples_src.rglob(pattern):
                rel_path = example_file.relative_to(examples_src)
                # Map source file to destination in gwmock/examples/
                force_include[str(example_file)] = f"gwmock/examples/{rel_path.as_posix()}"
                num_files += 1

        if num_files > 0:
            print(f"✓ Including {num_files} example file(s) in gwmock/examples/")
