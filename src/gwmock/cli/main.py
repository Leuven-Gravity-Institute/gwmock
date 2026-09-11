# ruff: noqa PLC0415

"""
Main command line tool to generate mock data.
"""

from __future__ import annotations

import enum
from typing import Annotated

import typer


class LoggingLevel(str, enum.Enum):
    """Logging levels for the CLI."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Create the main Typer app
app = typer.Typer(
    name="gwmock",
    help="Gravitational Wave Simulation Data Simulator",
    rich_markup_mode="rich",
)


def setup_logging(level: LoggingLevel = LoggingLevel.INFO) -> None:
    """Set up logging with Rich handler."""
    import logging  # pylint: disable=import-outside-toplevel

    from rich.console import Console  # pylint: disable=import-outside-toplevel
    from rich.logging import RichHandler  # pylint: disable=import-outside-toplevel

    logger = logging.getLogger("gwmock")

    logger.setLevel(level.value)

    console = Console()

    # Remove any existing handlers to ensure RichHandler is used
    for h in logger.handlers[:]:  # Use slice copy to avoid modification during iteration
        logger.removeHandler(h)
    # Add the RichHandler
    if not logger.handlers:
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_time=True,
            show_level=True,  # Keep level (e.g., DEBUG, INFO) for clarity
            markup=True,  # Enable Rich markup in messages for styling
            level=level.value,  # Ensure handler respects the level
            omit_repeated_times=False,
            log_time_format="%H:%M",
        )
        handler.setLevel(level.value)
        logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate output
    logger.propagate = False


@app.callback()
def main(
    verbose: Annotated[
        LoggingLevel,
        typer.Option("--verbose", "-v", help="Set verbosity level"),
    ] = LoggingLevel.INFO,
) -> None:
    """Gravitational Wave Simulation Data Simulator.

    This command-line tool provides functionality for generating
    gravitational wave detector simulation data.
    """
    setup_logging(verbose)


# Import and register commands after app is created
def register_commands() -> None:
    """Register all CLI commands."""

    # Fast imports
    from gwmock.cli.batch import batch_command
    from gwmock.cli.config import config_command
    from gwmock.cli.find_glitch import find_glitch_command
    from gwmock.cli.find_signal import find_signal_command
    from gwmock.cli.merge import merge_command
    from gwmock.cli.reindex import reindex_command
    from gwmock.cli.repository.main import repository_app
    from gwmock.cli.simulate import simulate_command
    from gwmock.cli.validate import validate_command

    app.command("simulate")(simulate_command)
    app.command("merge")(merge_command)
    app.command("config")(config_command)
    app.command("validate")(validate_command)
    app.command("batch")(batch_command)
    app.command("find-signal")(find_signal_command)
    app.command("find-glitch")(find_glitch_command)
    app.command("reindex")(reindex_command)

    app.add_typer(repository_app, name="repository", help="Manage Zenodo repositories")


register_commands()
