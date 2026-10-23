"""
Utility functions used in the command line tools.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gwmock.utils.errors import UnknownTemplateVariableError

logger = logging.getLogger("gwmock")


def get_file_name_from_template_with_dict(
    template: str, values: dict[str, Any], exclude: set[str] | None = None
) -> str:
    """Get the file name from a template string.
    The template string should use a double curly bracket to indicate the placeholder.
    For example, in '{{ x }}-{{ y }}.txt', x and y are interpreted as placeholders,
    and the values are retrieved from the values dictionary.

    Args:
        template (str): A template string.
        values (dict[str, Any]): A dictionary of values.
        exclude (set[str] | None): Set of attribute names to exclude from expansion. Defaults to None.

    Returns:
        str: The file name with the placeholders substituted by the values from the dictionary.
    """
    if exclude is None:
        exclude = set()

    def replace(matched):
        label = matched.group(1).strip()
        if label in exclude:
            return matched.group(0)  # Return the original placeholder unchanged
        try:
            return str(values[label])
        except KeyError as e:
            raise ValueError(f"Key '{label}' not found in values dictionary") from e

    return re.sub(r"\{\{\s*(\w+)\s*\}\}", replace, template)


def get_file_name_from_template(template: str, instance: object, exclude: set[str] | None = None) -> str:
    """Get the file name from a template string.
    The template string should use a double curly bracket to indicate the placeholder.
    For example, in '{{ x }}-{{ y }}.txt', x and y are interpreted as placeholders,
    and the values are retrieved from the instance.

    Args:
        template (str): A template string.
        instance (object): An instance.
        exclude (set[str] | None): Set of attribute names to exclude from expansion. Defaults to None.

    Returns:
        str: The file name with the placeholders substituted by the values of the attributes of the instance.
    """
    if exclude is None:
        exclude = set()

    def replace(matched):
        label = matched.group(1).strip()
        if label in exclude:
            return matched.group(0)  # Return the original placeholder unchanged
        try:
            return str(getattr(instance, label))
        except AttributeError as e:
            raise UnknownTemplateVariableError(
                f"Attribute '{label}' not found in instance of type {type(instance).__name__}"
            ) from e

    return re.sub(r"\{\{\s*(\w+)\s*\}\}", replace, template)


def handle_signal(cleanup_fn: Callable) -> Callable:
    """A factory to create a signal handler from a clean-up function.

    Args:
        cleanup_fn (Callable): A clean-up function to be called when the signal is received.

    Returns:
        Callable: A signal handler.
    """

    def handler(sig_num, _frame):
        logger.error("Received signal %s, exiting...", sig_num)
        cleanup_fn()
        sys.exit(1)

    return handler


def save_file_safely(file_name: str | Path, backup_file_name: str | Path, save_function: Callable, **kwargs) -> None:  # pylint: disable=duplicate-code
    """A helper function to save file safely by first creating a backup.

    This function is designed for saving a checkpoint file that has a fixed file name.
    If an existing `file_name` is detected, it is first renamed to `backup_file_name`
    before calling `save_function`.

    `save_function` needs to have an argument `file_name` to define the name of the output file.
    Additional arguments can be provided through **kwargs.

    Args:
        file_name (str | Path): File name of the output.
        backup_file_name (str | Path): File name of the backup.
        save_function (Callable): A callable to perform the saving.
    """
    file_name = Path(file_name)
    backup_file_name = Path(backup_file_name)

    if file_name.is_file():
        file_name.rename(backup_file_name)
        logger.debug("Existing file backed up to: %s", backup_file_name)

    # Try to call save_function to save to file.
    try:
        save_function(file_name=file_name, **kwargs)

        if backup_file_name.is_file():
            backup_file_name.unlink()
            logger.debug("Backup file deleted after successful save.")
    except (OSError, PermissionError, ValueError) as e:
        logger.error("Failed to save file: %s", e)

        if backup_file_name.is_file():
            try:
                backup_file_name.rename(file_name)
                logger.warning("Restored file from backup due to a failure.")
            except (OSError, PermissionError) as restore_error:
                logger.error("Failed to restore backup file: %s", restore_error)
        raise
