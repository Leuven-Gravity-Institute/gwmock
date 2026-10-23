"""Template validation utilities for gwmock CLI."""

from __future__ import annotations

import itertools
import logging
import re
from typing import Any

from astropy.units import Quantity

from gwmock.utils.errors import UnknownTemplateVariableError

logger = logging.getLogger("gwmock")


class TemplateValidator:
    """Validate template strings for simulators."""

    @staticmethod
    def validate_template(template: str, simulator_name: str) -> tuple[bool, list[str]]:
        """Validate template and return (is_valid, errors)."""
        errors = []

        try:
            # Extract all placeholder fields from template
            # template_fields = TemplateValidator._extract_template_fields(template)

            # Try to format with dummy data to catch syntax errors
            dummy_state = TemplateValidator._create_dummy_state()
            template.format(**dummy_state)

            logger.debug("Template validation passed for %s: %s", simulator_name, template)

        except KeyError as e:
            errors.append(f"Missing template field: {e}")
        except ValueError as e:
            errors.append(f"Template formatting error: {e}")
        except (AttributeError, TypeError) as e:
            errors.append(f"Template validation error: {e}")

        return len(errors) == 0, errors

    @staticmethod
    def extract_template_fields(template: str) -> set[str]:
        """Extract field names from template string."""
        # Find all {field_name} patterns, excluding format specs
        fields = re.findall(r"\{([^}:]+)", template)
        return set(fields)

    @staticmethod
    def _create_dummy_state() -> dict:
        """Create dummy state data for validation."""
        return {
            "counter": 1,
            "start_time": 1696291200,
            "duration": 4096,
            "detector": "H1",
            "batch_id": "test",
            "sample_rate": 4096,
            "end_time": 1696295296,
        }


def expand_template_variables(
    value: Any,
    simulator_instance: Any,
) -> Any:
    """Recursively expand template variables in config values.

    Supports template syntax: {{ variable_name }} where variable_name is an
    attribute of the simulator instance or global variables.

    If a template contains iterable attributes, multiple values are returned
    as a Cartesian product (similar to get_file_name_from_template).

    Template expansion happens at runtime, allowing dynamic values that change
    across iterations (e.g., simulator.counter, simulator.detector).

    Args:
        value: Value to expand (str, dict, list, or other)
        simulator_instance: Simulator instance for attribute lookup

    Returns:
        Value with expanded templates. If template contains iterables, returns
        a list of expanded values (Cartesian product).

    Raises:
        UnknownTemplateVariableError: If template variable not found in simulator or globals (an
            ``AttributeError`` too).

    Example:
        >>> class MockSimulator:
        ...     detector = "H1"
        ...     detectors = ["H1", "L1"]
        >>> sim = MockSimulator()
        >>> expand_template_variables("{{ detector }}:STRAIN", sim)
        "H1:STRAIN"
        >>> expand_template_variables("{{ detectors }}:STRAIN", sim)
        ["H1:STRAIN", "L1:STRAIN"]
    """
    # Handle strings: expand {{ variable }} patterns
    if isinstance(value, str):
        return _expand_string_template(value, simulator_instance)

    # Handle dicts: recursively expand values
    if isinstance(value, dict):
        return {key: expand_template_variables(val, simulator_instance) for key, val in value.items()}

    # Handle lists: recursively expand items
    if isinstance(value, list):
        return [expand_template_variables(item, simulator_instance) for item in value]

    # Return other types as-is
    return value


def _expand_string_template(  # pylint: disable=duplicate-code
    template_str: str,
    simulator_instance: Any,
) -> str | list[str]:
    """Expand {{ variable }} patterns in a string.

    If template contains iterable attributes, returns multiple expanded strings
    as a Cartesian product (similar to get_file_name_from_template).

    Args:
        template_str: String potentially containing {{ variable }} patterns
        simulator_instance: Simulator instance for attribute lookup

    Returns:
        String with expanded variables, or list of strings if iterables present

    Raises:
        UnknownTemplateVariableError: If variable not found in simulator or globals. An
            ``AttributeError`` as well, and a refusal the batch retry does not repeat.
    """
    # Find all unique placeholders in the template
    placeholders = list(dict.fromkeys(re.findall(r"\{\{\s*(\w+)\s*\}\}", template_str)))

    # Collect values for each placeholder
    values_dict = {}
    for var_name in placeholders:
        # Try simulator instance first
        if hasattr(simulator_instance, var_name):
            value = getattr(simulator_instance, var_name)
        else:
            raise UnknownTemplateVariableError(f"Template variable '{var_name}' not found in simulator")

        # Convert value to list of strings (similar to get_file_name_from_template)
        values_dict[var_name] = _value_to_string_list(value)

    # Prepare lists for Cartesian product
    product_lists = [values_dict[var_name] for var_name in placeholders]

    # Generate all combinations
    combinations = list(itertools.product(*product_lists))

    # Helper function for substitution
    def substitute_template(combo_dict: dict) -> str:
        def replace(matched):
            label = matched.group(1).strip()
            return str(combo_dict[label])

        return re.sub(r"\{\{\s*(\w+)\s*\}\}", replace, template_str)

    # Substitute for each combination
    results = [substitute_template(dict(zip(placeholders, combo, strict=False))) for combo in combinations]

    # Reshape into nested list if there are array placeholders (matching io.py behavior)
    array_placeholders = [p for p in placeholders if len(values_dict[p]) > 1]
    if array_placeholders:
        lengths = [len(values_dict[p]) for p in array_placeholders]

        def reshape_to_nested(flat_list: list, lengths: list[int]):
            if not lengths:
                return flat_list[0] if flat_list else ""
            size = lengths[0]
            sub_size = len(flat_list) // size
            nested = []
            for i in range(size):
                start = i * sub_size
                end = (i + 1) * sub_size
                sub_list = flat_list[start:end]
                nested.append(reshape_to_nested(sub_list, lengths[1:]))
            return nested

        return reshape_to_nested(results, lengths)

    # No arrays: return single string
    return results[0] if results else ""


def _value_to_string_list(value: Any) -> list[str]:
    """Convert a value to a list of strings for Cartesian product expansion.

    Handles special types like astropy.units.Quantity similar to get_file_name_from_template.

    Args:
        value: Value to convert

    Returns:
        List of string representations (single item for scalars, multiple for iterables)
    """
    # Handle astropy Quantity
    if isinstance(value, Quantity):
        x = value.value
        if x.is_integer():
            x = int(x)
        return [str(x)]

    # Handle iterables (list, tuple, or other iterables but not str)
    # Return list of stringified elements for Cartesian product
    if isinstance(value, (list, tuple)) or (hasattr(value, "__iter__") and not isinstance(value, str)):
        return [str(ele) for ele in list(value)]

    # Handle native types: convert to single-item list
    return [str(value)]
