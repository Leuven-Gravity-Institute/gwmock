"""Which signal settings each execution path actually honours, and refusing the rest.

Three bugs in the batched path shared one shape: a setting was read from the configuration, stored
on the orchestrator, and then never forwarded to the generator. ``waveform-backend-arguments`` were
discarded, ``waveform-options`` had nowhere to go, and ``waveform-arguments`` outside a fixed set
were dropped. In each case the run completed and produced plausible output.

They were all instances of the same default: **a setting nobody wired is ignored**. Fixing them one
at a time leaves the next one waiting. This inverts the default for the batched path -- a setting the
path does not declare is *refused* -- so a configuration key added later fails loudly there until
someone wires it, rather than passing silently.

The per-event path is treated differently on purpose. It is the default, it has users, and one of
its settings (``parameters``, which only the stochastic path reads) has been quietly ignored for CBC
runs since before this. Turning that into an error would break working configurations for a problem
they did not introduce, so it warns instead and the batched path is where the strict rule applies.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("gwmock")

#: Signal-config fields the batched execution path reads.
#:
#: Anything the user sets that is not here is refused, because the batched path would ignore it.
#: ``waveform_arguments`` is listed but only partially honoured -- the batched entry point reads a
#: fixed set of canonical parameters -- so it carries its own per-key check in
#: :func:`gwmock.signal.device_chunks.require_batched_parameters_supported`.
_BATCHED_HONOURED_FIELDS: frozenset[str] = frozenset(
    {
        "backend",
        "source_type",
        "waveform_model",
        "waveform_backend",
        "waveform_backend_arguments",
        "waveform_arguments",
        "detectors",
        "minimum_frequency",
        "earth_rotation",
        "execution",
        "output",
    }
)

#: Why each unhonoured field cannot be applied, so the error says what to do rather than only what
#: went wrong.
_BATCHED_REASONS: dict[str, str] = {
    "arguments": (
        "these configure the simulator's constructor, and the batched path calls gwmock-signal's "
        "batched entry point directly rather than going through the simulator"
    ),
    "parameters": (
        "these are fixed parameters for the simulator's simulate() call, which the batched path "
        "does not use; note they are also ignored by the per-event path for CBC sources"
    ),
    "projection_backend": (
        "the batched entry point projects on device unconditionally, so there is no host/device "
        "choice left for it to make; the setting exists for the per-event path, where the "
        "projection would otherwise run on the host"
    ),
    "waveform_options": (
        "the batched entry point has no equivalent parameter, so LAL dictionary options such as "
        "ModeArray cannot be applied"
    ),
}


def _alias_of(signal_config: Any, field: str) -> str:
    """Return the name the user writes in YAML for *field*, falling back to the field name."""
    info = type(signal_config).model_fields.get(field)
    return getattr(info, "alias", None) or field


def _unknown_keys(signal_config: Any) -> set[str]:
    """Return every unrecognised key under ``orchestration.signal``, nested ones included.

    Checking only the top level would leave an escape hatch: nested blocks such as ``output`` are
    themselves ``extra="allow"`` models, so ``output.typo`` validates, is never read, and would
    otherwise pass the strict batched rule because ``output`` as a whole is honoured.
    """
    unknown = set(getattr(signal_config, "model_extra", None) or {})
    for field in type(signal_config).model_fields:
        nested = getattr(signal_config, field, None)
        nested_extra = getattr(nested, "model_extra", None)
        if not nested_extra:
            continue
        unknown |= {f"{_alias_of(signal_config, field)}.{name}" for name in nested_extra}
    return unknown


def require_execution_supports_configuration(signal_config: Any, execution: str, source_type: str = "") -> None:
    """Refuse a configuration whose settings the chosen execution path would ignore.

    Args:
        signal_config: The validated ``orchestration.signal`` block.
        execution: The execution mode, ``"per-event"`` or ``"batched"``.
        source_type: The resolved source type. Only used to decide whether ``parameters`` is read,
            which it is for ``"sgwb"`` and is not for compact-binary sources.

    Raises:
        ValueError: If *execution* is ``"batched"`` and the user set a field it does not honour, or
            an unrecognised key.
    """
    # Only what the user actually wrote. Defaults are not "configured", and treating them as such
    # would refuse every configuration.
    configured = set(getattr(signal_config, "model_fields_set", set()))
    unknown = _unknown_keys(signal_config)

    if execution != "batched":
        # Unknown keys are ignored whatever the path does with them, so they are worth saying out
        # loud even here -- a misspelled setting is silently absent otherwise.
        for name in sorted(unknown):
            logger.warning(
                "orchestration.signal.%s is not a recognised setting and will be ignored. Check the "
                "spelling against the configuration reference.",
                name,
            )
        # The stochastic path does read `parameters`, so warning about it there would be wrong.
        if "parameters" in configured and source_type != "sgwb":
            logger.warning(
                "orchestration.signal.parameters is only read by the stochastic-background path. "
                "For %s sources it is ignored, and has been since before the batched path existed.",
                source_type or "compact-binary",
            )
        return

    ignored = sorted((configured - _BATCHED_HONOURED_FIELDS) | unknown)
    if not ignored:
        return

    lines = []
    for field in ignored:
        reason = _BATCHED_REASONS.get(field, "the batched path does not read it")
        lines.append(f"  {_alias_of(signal_config, field)}: {reason}")
    raise ValueError(
        "execution: batched would ignore settings this configuration sets:\n"
        + "\n".join(lines)
        + "\n\nA setting that is read but never applied produces output that looks correct, so the "
        "batched path refuses rather than continuing. Use execution: per-event, or remove them."
    )
