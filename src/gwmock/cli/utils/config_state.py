"""Pure state management for the interactive config editor.

ConfigState holds the current configuration as a plain dict, provides
section-aware getters/setters with type coercion, and validates via the
existing Pydantic Config model.  No I/O beyond YAML loading — the TUI
layer (config_editor.py) is responsible for display.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("gwmock")

# --------------------------------------------------------------------------- #
# nested-dict helpers
# --------------------------------------------------------------------------- #


def _get(data: dict, keys: list[str], default: Any = None) -> Any:
    for k in keys:
        if not isinstance(data, dict):
            return default
        data = data.get(k, default)
    return data


def _set(data: dict, keys: list[str], value: Any) -> None:
    for k in keys[:-1]:
        data = data.setdefault(k, {})
    data[keys[-1]] = value


def _delete(data: dict, keys: list[str]) -> None:
    for k in keys[:-1]:
        if not isinstance(data, dict) or k not in data:
            return
        data = data[k]
    if isinstance(data, dict) and keys[-1] in data:
        del data[keys[-1]]


def _clean_empty(data: Any) -> Any:
    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for k, v in data.items():
            cleaned = _clean_empty(v)
            if isinstance(cleaned, dict) and not cleaned:
                continue
            if isinstance(cleaned, list) and not cleaned:
                continue
            out[k] = cleaned
        return out
    if isinstance(data, list):
        return [_clean_empty(v) for v in data]
    return data


def _clean_none(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _clean_none(v) for k, v in data.items() if v is not None}
    if isinstance(data, list):
        return [_clean_none(v) for v in data]
    return data


# --------------------------------------------------------------------------- #
# key-path and type mappings
# --------------------------------------------------------------------------- #

# Mapping: user-facing key → internal dict path (list of keys)

NOISE_KEYS: dict[str, list[str]] = {
    "psd": ["orchestration", "noise", "arguments", "psd_file"],
    "seed": ["orchestration", "noise", "arguments", "seed"],
    "detectors": ["orchestration", "noise", "arguments", "detectors"],
    # The noise backend reads its floor as `low_frequency_cutoff`; any other spelling is rejected.
    "minimum-frequency": ["orchestration", "noise", "arguments", "low_frequency_cutoff"],
    "backend": ["orchestration", "noise", "backend"],
}

SIGNAL_KEYS: dict[str, list[str]] = {
    "source-type": ["orchestration", "signal", "source-type"],
    "waveform-model": ["orchestration", "signal", "waveform-model"],
    "detectors": ["orchestration", "signal", "detectors"],
    "minimum-frequency": ["orchestration", "signal", "minimum-frequency"],
    "earth-rotation": ["orchestration", "signal", "earth-rotation"],
    "backend": ["orchestration", "signal", "backend"],
}

POPULATION_KEYS: dict[str, list[str]] = {
    "backend": ["orchestration", "population", "backend"],
    "n-samples": ["orchestration", "population", "n-samples"],
    "source-type": ["orchestration", "population", "source-type"],
    "path": ["orchestration", "population", "arguments", "path"],
}

GLOBALS_KEYS: dict[str, list[str]] = {
    "working-directory": ["globals", "working-directory"],
    "output-directory": ["globals", "output-directory"],
    "metadata-directory": ["globals", "metadata-directory"],
    "sampling-frequency": ["globals", "simulator-arguments", "sampling-frequency"],
    "duration": ["globals", "simulator-arguments", "duration"],
    "start-time": ["globals", "simulator-arguments", "start-time"],
    "total-duration": ["globals", "simulator-arguments", "total-duration"],
    "seed": ["globals", "simulator-arguments", "seed"],
}

BATCH_KEYS: dict[str, list[str]] = {
    "scheduler": ["batch", "scheduler"],
    "job-name": ["batch", "job-name"],
    "chunks-enabled": ["batch", "chunks", "enabled"],
    "chunks-n-chunks": ["batch", "chunks", "n-chunks"],
    "chunks-parallel": ["batch", "chunks", "parallel"],
}

SECTION_KEYS: dict[str, dict[str, list[str]]] = {
    "noise": NOISE_KEYS,
    "signal": SIGNAL_KEYS,
    "population": POPULATION_KEYS,
    "globals": GLOBALS_KEYS,
    "batch": BATCH_KEYS,
}

# Type coercion per section/key
NOISE_TYPES: dict[str, type | str] = {
    "psd": str,
    "seed": int,
    "detectors": "list",
    "minimum-frequency": float,
    "backend": str,
}

SIGNAL_TYPES: dict[str, type | str] = {
    "source-type": str,
    "waveform-model": str,
    "detectors": "list",
    "minimum-frequency": float,
    "earth-rotation": bool,
    "backend": str,
}

POPULATION_TYPES: dict[str, type | str] = {
    "backend": str,
    "n-samples": int,
    "source-type": str,
    "path": str,
}

GLOBALS_TYPES: dict[str, type | str] = {
    "working-directory": str,
    "output-directory": str,
    "metadata-directory": str,
    "sampling-frequency": int,
    "duration": int,
    "start-time": int,
    "total-duration": str,
    "seed": int,
}

BATCH_TYPES: dict[str, type | str] = {
    "scheduler": str,
    "job-name": str,
    "chunks-enabled": bool,
    "chunks-n-chunks": int,
    "chunks-parallel": bool,
}

SECTION_TYPES: dict[str, dict[str, type | str]] = {
    "noise": NOISE_TYPES,
    "signal": SIGNAL_TYPES,
    "population": POPULATION_TYPES,
    "globals": GLOBALS_TYPES,
    "batch": BATCH_TYPES,
}

# Human-readable descriptions for each section's keys
NOISE_DESC: dict[str, str] = {
    "psd": "PSD file name (use /psds to see available)",
    "seed": "Random seed (integer)",
    "detectors": "Detector names, space-separated (use /geometries for network detectors)",
    "minimum-frequency": "Minimum frequency in Hz (float)",
    "backend": "Noise backend name or class path",
}

SIGNAL_DESC: dict[str, str] = {
    "source-type": "Source type (use /source-types to see available)",
    "waveform-model": "Waveform model name (use /waveforms to see available)",
    "detectors": "Detector names, space-separated (use /geometries for network detectors)",
    "minimum-frequency": "Minimum frequency in Hz (float)",
    "earth-rotation": "Use time-dependent detector response (true/false)",
    "backend": "Signal backend name or class path",
}

POPULATION_DESC: dict[str, str] = {
    "backend": "Population backend (use /presets to see available)",
    "n-samples": "Number of samples (integer)",
    "source-type": "Source type (use /source-types to see available)",
    "path": "Population file path",
}

GLOBALS_DESC: dict[str, str] = {
    "working-directory": "Base working directory (path)",
    "output-directory": "Default output directory (path)",
    "metadata-directory": "Default metadata directory (path)",
    "sampling-frequency": "Sampling frequency in Hz (integer)",
    "duration": "Duration in seconds (integer)",
    "start-time": "GPS start time (integer)",
    "total-duration": "Total duration (e.g. '1 day', '6 hours', or seconds)",
    "seed": "Global random seed (integer)",
}

BATCH_DESC: dict[str, str] = {
    "scheduler": "Scheduler name (default: slurm)",
    "job-name": "Job name",
    "chunks-enabled": "Enable chunking for parallel execution (true/false)",
    "chunks-n-chunks": "Number of chunks to split the simulation into",
    "chunks-parallel": "Run chunks in parallel (local) or as array job (SLURM)",
}

SECTION_DESC: dict[str, dict[str, str]] = {
    "noise": NOISE_DESC,
    "signal": SIGNAL_DESC,
    "population": POPULATION_DESC,
    "globals": GLOBALS_DESC,
    "batch": BATCH_DESC,
}

# Extra commands available per section (shown in help but not in key mappings)
NOISE_EXTRA: list[tuple[str, str]] = [
    ("glitch add <kind>", "Add a glitch model (use /glitches to see available)"),
    ("glitch remove <index>", "Remove a glitch by index"),
]

BATCH_EXTRA: list[tuple[str, str]] = [
    ("resources <key> <value>", "Set a batch resource (e.g. nodes, cpus_per_task, mem)"),
    ("submit <key> <value>", "Set a batch submit option (e.g. account, partition, time)"),
]

SECTION_EXTRA: dict[str, list[tuple[str, str]]] = {
    "noise": NOISE_EXTRA,
    "batch": BATCH_EXTRA,
}


# --------------------------------------------------------------------------- #
# type coercion
# --------------------------------------------------------------------------- #


def _coerce(raw: str, target: type | str) -> Any:
    """Coerce a string *raw* to *target* type."""
    if target == "list":
        return raw.split()
    if target is bool:
        low = raw.lower()
        if low in ("true", "yes", "1"):
            return True
        if low in ("false", "no", "0"):
            return False
        raise ValueError(f"Cannot parse '{raw}' as boolean (use true/false)")
    return target(raw)  # type: ignore[operator]


# Range constraints: key -> (min, max, error message)
_VALUE_CONSTRAINTS: dict[str, tuple[int | None, int | None, str]] = {
    "chunks-n-chunks": (1, None, "chunks-n-chunks must be at least 1"),
    "n-samples": (1, None, "n-samples must be at least 1"),
    "seed": (0, None, "seed must be non-negative"),
    "minimum-frequency": (0, None, "minimum-frequency must be non-negative"),
    "sampling-frequency": (1, None, "sampling-frequency must be at least 1"),
    "duration": (1, None, "duration must be at least 1"),
}


def _validate_value(key: str, value: Any) -> None:
    """Validate *value* against known constraints for *key*."""
    constraint = _VALUE_CONSTRAINTS.get(key)
    if constraint is None:
        return
    min_val, max_val, error_msg = constraint
    if not isinstance(value, (int, float)):
        return
    if min_val is not None and value < min_val:
        raise ValueError(error_msg)
    if max_val is not None and value > max_val:
        raise ValueError(error_msg)


# --------------------------------------------------------------------------- #
# ConfigState
# --------------------------------------------------------------------------- #


class ConfigState:
    """Mutable, section-aware wrapper around a gwmock config dict."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = data if data is not None else {}
        self.config_file: str | None = None

    # -- section access ----------------------------------------------------- #

    def set(self, section: str, key: str, raw_value: str) -> None:
        """Set *key* in *section* from a string *raw_value*."""
        keys_map = SECTION_KEYS.get(section)
        if keys_map is None:
            raise ValueError(f"Unknown section: {section}")
        path = keys_map.get(key)
        if path is None:
            valid = ", ".join(keys_map)
            raise ValueError(f"Unknown key '{key}' for /{section}. Valid keys: {valid}")
        types_map = SECTION_TYPES.get(section, {})
        target = types_map.get(key, str)
        value = _coerce(raw_value, target)
        _validate_value(key, value)
        _set(self._data, path, value)

    def get(self, section: str, key: str) -> Any:
        """Get *key* from *section*, or ``None`` if unset."""
        keys_map = SECTION_KEYS.get(section, {})
        path = keys_map.get(key)
        if path is None:
            return None
        return _get(self._data, path)

    def get_section(self, section: str) -> dict[str, Any] | None:
        """Return the raw dict for *section*, or ``None`` if empty."""
        mapping = {
            "noise": ["orchestration", "noise"],
            "signal": ["orchestration", "signal"],
            "population": ["orchestration", "population"],
            "globals": ["globals"],
            "batch": ["batch"],
        }
        path = mapping.get(section)
        if path is None:
            return None
        result = _get(self._data, path)
        if isinstance(result, dict) and result:
            return result
        return None

    def set_batch_resource(self, key: str, raw_value: str) -> None:
        _set(self._data, ["batch", "resources", key], raw_value)

    def set_batch_submit(self, key: str, raw_value: str) -> None:
        _set(self._data, ["batch", "submit", key], raw_value)

    # -- glitch helpers ---------------------------------------------------- #

    def add_glitch(self, kind: str) -> int:
        glitches = _get(self._data, ["orchestration", "noise", "arguments", "glitches"], [])
        if not isinstance(glitches, list):
            glitches = []
        glitches.append({"kind": kind})
        _set(self._data, ["orchestration", "noise", "arguments", "glitches"], glitches)
        return len(glitches) - 1

    def remove_glitch(self, index: int) -> dict[str, Any]:
        glitches = _get(self._data, ["orchestration", "noise", "arguments", "glitches"], [])
        if not isinstance(glitches, list) or not (0 <= index < len(glitches)):
            raise IndexError(f"Glitch index {index} out of range (have {len(glitches)} glitches)")
        removed = glitches.pop(index)
        if glitches:
            _set(self._data, ["orchestration", "noise", "arguments", "glitches"], glitches)
        else:
            _delete(self._data, ["orchestration", "noise", "arguments", "glitches"])
        return removed

    # -- reset ------------------------------------------------------------- #

    def reset(self, section: str | None = None) -> None:
        if section is None:
            self._data = {}
            return
        mapping = {
            "noise": ["orchestration", "noise"],
            "signal": ["orchestration", "signal"],
            "population": ["orchestration", "population"],
            "globals": ["globals"],
            "batch": ["batch"],
        }
        path = mapping.get(section)
        if path is None:
            raise ValueError(f"Unknown section: {section}")
        _delete(self._data, path)

    # -- serialisation ----------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        return _clean_empty(self._data)

    def load(self, path: Path) -> None:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError("Configuration file must contain a YAML mapping")
        self._data = _clean_none(data)

    def validate(self) -> tuple[bool, str]:
        from gwmock.cli.utils.config import Config  # noqa: PLC0415

        try:
            Config(**self.to_dict())
            return True, ""
        except Exception as e:
            return False, str(e)

    @property
    def is_empty(self) -> bool:
        return not self._data or not _clean_empty(self._data)
