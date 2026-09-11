"""Which implementation projects the polarizations, and whether this environment can run it.

``projection-backend`` selects between gwmock-signal's two implementations of the rotating
projection. They compute the same thing -- agreement is ~1e-10 of peak -- but not at the same
speed, and projection is where a long transient spends almost all of its time: measured at 1024 s
and 8192 Hz across five ET detectors, it was 604 s of a 620 s single-event run, and the device
path did the same work in 225 s on the same CPU.

The constraints on the device path belong to gwmock-signal and are enforced there. They are
checked *here* as well, against the configuration, because the difference between the two
moments is a whole run: a configuration error found while parsing costs a second, and the same
error found from inside the projection arrives after the population is drawn, the noise is
generated and the first segments are written.
"""

from __future__ import annotations

import inspect
from typing import Any

#: The accepted names, and the longest span the device path is validated for, in seconds.
#:
#: Read from gwmock-signal so this module cannot drift from what the projection will actually
#: accept. The fallbacks are for an installed gwmock-signal that predates those exports: it also
#: predates the simulator argument this key is threaded through, so a configuration that reaches
#: them is refused by :func:`require_backend_accepts_projection_backend` regardless of the values
#: -- they exist so importing this module never depends on the version, not to stand in for the
#: real limits.
try:  # pragma: no cover - which branch runs depends on the installed gwmock-signal
    from gwmock_signal.projection.network import (
        MAX_LINEAR_SIDEREAL_SPAN_SECONDS as MAX_PROJECTION_SPAN_SECONDS,
    )
    from gwmock_signal.projection.network import (
        PROJECTION_BACKENDS,
    )
except ImportError:  # pragma: no cover - same
    PROJECTION_BACKENDS = frozenset({"numpy", "jax"})
    MAX_PROJECTION_SPAN_SECONDS = 86400.0


#: Parameter kinds a caller can actually pass ``projection_backend`` to *by name*.
#:
#: Positional-only is excluded deliberately: a parameter named ``projection_backend`` that can
#: only be given positionally cannot be reached by the keyword this module forwards, so its name
#: is not a promise about anything.
_NAMED_PARAMETER_KINDS = frozenset({inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY})


def validate_projection_backend_name(value: str) -> str:
    """Return *value* if it names a projection implementation, else raise.

    Args:
        value: The configured ``projection-backend``.

    Returns:
        *value* unchanged.

    Raises:
        ValueError: If *value* is not one of :data:`PROJECTION_BACKENDS`.
    """
    if value not in PROJECTION_BACKENDS:
        raise ValueError(f"'projection-backend' must be one of {sorted(PROJECTION_BACKENDS)}, got {value!r}")
    return value


def require_projection_backend_runnable(value: str) -> None:
    """Check that this environment can actually run *value*, and prepare it if it can.

    Only ``"jax"`` has anything to check. Two things can be wrong with it, and both are
    environment problems rather than configuration mistakes, so both say what to install or
    change rather than only what failed:

    * JAX is not installed. The projection needs JAX and nothing else -- not ripple, which
      ``execution: batched`` needs and this key does not.
    * JAX is not in 64-bit mode. The device projection refuses to run without it, because in
      32-bit mode the GPS times and sidereal angles lose the precision the delays depend on and
      the result is wrong by of order a percent of peak while still looking like strain. It is
      *enabled* here rather than reported, since it is a process-wide switch that has to be set
      before any array exists and parsing a configuration is the earliest such moment. That it is
      then verified rather than assumed is the point: an environment that forces it off would
      otherwise reach the projection and fail there.

    Args:
        value: The validated ``projection-backend`` name.

    Raises:
        ValueError: If JAX is absent, or 64-bit mode cannot be turned on.
    """
    if value != "jax":
        return

    try:
        import jax  # noqa: PLC0415 -- optional [jax] dep, and only this backend needs it
    except ImportError as exc:
        raise ValueError(
            "'projection-backend: jax' needs JAX, which is not installed. Install it with "
            "`pip install 'gwmock[jax]'`, or `pip install 'gwmock[cuda]'` to project on a GPU. "
            "Note this is JAX alone -- unlike 'execution: batched', it does not need ripple."
        ) from exc

    jax.config.update("jax_enable_x64", True)
    if not jax.config.jax_enable_x64:
        raise ValueError(
            "'projection-backend: jax' requires JAX in 64-bit mode, and enabling it here had no "
            "effect -- something in this environment is forcing it off, such as JAX_ENABLE_X64=0. "
            "In 32-bit mode the projection is wrong by of order a percent of peak while still "
            "looking like strain, so it refuses rather than degrading. Clear that setting, or use "
            "'projection-backend: numpy'."
        )


def require_backend_accepts_projection_backend(backend_class: type[Any], value: str) -> None:
    """Refuse a signal backend that cannot be *shown* to consume the setting.

    The setting is threaded to gwmock-signal as a constructor argument, so what happens to a
    backend that does not take one depends entirely on how its constructor is written, and one of
    the two outcomes is silent:

    * A constructor with named parameters raises ``TypeError`` about an unexpected keyword --
      loud, but naming neither the setting nor what to do about it.
    * A constructor ending in ``**kwargs`` accepts the keyword, discards it, and projects with
      its own default. The run then completes, the configuration and the run metadata both say
      ``jax`` was asked for, and the output is what the host path produced. Nothing anywhere
      says the request was dropped.

    So a **named** ``projection_backend`` parameter is required, and ``**kwargs`` does not
    substitute for one: accepting a keyword is not evidence of using it, and this is the only
    evidence available before the projection runs. A backend that takes its arguments through
    ``**kwargs`` and does honour the setting has to name it in its signature to say so -- which
    costs one parameter and is what makes the promise checkable.

    Args:
        backend_class: The resolved ``orchestration.signal.backend`` class.
        value: The configured ``projection-backend``.

    Raises:
        ValueError: If *backend_class* does not expose a named ``projection_backend`` parameter,
            or if its signature cannot be read at all.
    """
    try:
        parameters = inspect.signature(backend_class).parameters
    except (TypeError, ValueError) as exc:
        # Refused rather than waved through. An unreadable signature is not evidence that the
        # setting is honoured, and the failure it would otherwise lead to is the silent one.
        raise ValueError(
            f"'projection-backend: {value}' cannot be applied: the signature of the signal "
            f"backend {getattr(backend_class, '__name__', backend_class)!r} cannot be read, so "
            "there is no way to tell whether it would use the setting or discard it. Wrap it in "
            "a class whose __init__ names 'projection_backend', or remove the setting."
        ) from exc

    named = parameters.get("projection_backend")
    if named is not None and named.kind in _NAMED_PARAMETER_KINDS:
        return

    takes_any_keyword = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    if takes_any_keyword:
        raise ValueError(
            f"'projection-backend: {value}' cannot be applied: the signal backend "
            f"{backend_class.__name__} takes arbitrary keyword arguments and does not name "
            "'projection_backend' among them, so it would accept the setting and could silently "
            "discard it -- leaving a run that reports the implementation it was asked for and "
            "produces the output of the other one. Name 'projection_backend' in its __init__ if "
            "it honours the setting, or remove the setting to leave the backend on its own "
            "default."
        )
    raise ValueError(
        f"'projection-backend: {value}' cannot be applied: the signal backend "
        f"{backend_class.__name__} takes no 'projection_backend' argument. A gwmock-signal older "
        "than the release that added it is the usual cause -- upgrade it, or remove the setting "
        "to project on the host as before."
    )
