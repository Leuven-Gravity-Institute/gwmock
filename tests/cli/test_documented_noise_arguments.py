"""The noise arguments in the documentation's YAML snippets must be ones the default backend reads.

A snippet is copied into a config as written, so a key the adapter does not read (``csd_file``
where the adapter reads ``csd_files``) either fails the copied config or, worse, configures
nothing while looking as if it did. The adapter already rejects unknown keys at load time; this
test applies the same check to the documentation, where nothing else would run it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from gwmock.cli.adapter_orchestration import _DEFAULT_NOISE_ARGUMENT_KEYS
from gwmock.noise.adapter import _parse_csd_file_map

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _REPO_ROOT / "docs"

_YAML_BLOCK = re.compile(r"^```ya?ml\n(.*?)^```", re.MULTILINE | re.DOTALL)
#: A pymdownx.snippets include line, which pulls a repository file into the block verbatim.
_SNIPPET_INCLUDE = re.compile(r'^--8<-- "([^"]+)"$', re.MULTILINE)


def _documented_noise_arguments() -> list[tuple[str, dict[str, Any]]]:
    """Return ``(location, noise.arguments)`` for every documented default-backend noise block."""
    found = []
    for page in sorted(_DOCS.rglob("*.md")):
        for index, match in enumerate(_YAML_BLOCK.finditer(page.read_text(encoding="utf-8"))):
            text = _SNIPPET_INCLUDE.sub(
                lambda include: (_REPO_ROOT / include.group(1)).read_text(encoding="utf-8"), match.group(1)
            )
            document = yaml.safe_load(text)
            if not isinstance(document, dict):
                continue
            noise = (document.get("orchestration") or {}).get("noise")
            if not isinstance(noise, dict) or noise.get("backend") is not None:
                continue
            arguments = noise.get("arguments")
            if isinstance(arguments, dict):
                found.append((f"{page.relative_to(_REPO_ROOT)} yaml block {index}", arguments))
    return found


def test_documented_noise_arguments_are_keys_the_default_backend_reads():
    blocks = _documented_noise_arguments()
    assert blocks, "no documented noise.arguments block was found, so nothing was checked"

    unread = {
        location: sorted(key for key in arguments if key.replace("-", "_") not in _DEFAULT_NOISE_ARGUMENT_KEYS)
        for location, arguments in blocks
    }
    assert {location: keys for location, keys in unread.items() if keys} == {}


def test_documented_csd_files_name_detector_pairs_with_a_psd_each():
    blocks = [
        (location, arguments) for location, arguments in _documented_noise_arguments() if "csd_files" in arguments
    ]
    assert blocks, "no documented snippet sets csd_files, so the correlated-noise path is undocumented"

    for location, arguments in blocks:
        pairs = _parse_csd_file_map(arguments["csd_files"])
        assert pairs, location
        psd_detectors = set(arguments.get("psd_files") or {})
        assert {detector for pair in pairs for detector in pair} <= psd_detectors, location
