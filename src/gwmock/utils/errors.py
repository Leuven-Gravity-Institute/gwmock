"""Exception types shared across gwmock."""

from __future__ import annotations


class DeterministicRefusalError(ValueError):
    """A refusal decided by the configuration and inputs alone, which no retry can change.

    Raised for configuration and catalogue validation inside a batch: a waveform argument or
    population column the waveform does not read, a placement query or generation call the signal
    simulator refuses (a ``ValueError`` or ``TypeError`` from a pure function of its inputs), an
    alias clash, a setting the batched path cannot apply, a noise configuration the stream refuses,
    and output arguments or file-name templates that cannot be written. A batch retry re-runs the same configuration on
    the same inputs, so it would fail the same way after a backoff, repeating the log; the retry
    loop raises these at once instead (see ``gwmock.cli.simulate_utils.retry_with_backoff``).

    A subclass of :class:`ValueError`, which is what these refusals were raised as before the type
    existed, so a caller catching ``ValueError`` still catches them.
    """


class TransientDownloadError(OSError):
    """A download that failed for a reason a later attempt may not meet.

    Raised for a connection failure, a timeout, a server error (5xx), a rate limit or request
    timeout (429, 408), and for waiting too long on another process's download lock. An
    :class:`OSError`, so a batch retry treats it as the I/O failure it is rather than as a
    refusal of the configuration. A client error such as 404 is not transient and stays a
    :class:`ValueError`.
    """


def wraps_transient_failure(error: BaseException) -> bool:
    """Return whether *error* was raised from an I/O or network failure, at any depth.

    A library that turns a connection error into a ``ValueError`` -- explicitly chained or raised
    while handling it -- has hidden a failure a retry may get past. ``requests``, ``urllib`` and
    ``filelock`` failures are all :class:`OSError` subclasses, so one check covers them. Used where a
    ``ValueError`` or ``TypeError`` from a whole subsystem is marked final, so such a wrapping stays
    retryable.

    Args:
        error: The exception to inspect.

    Returns:
        Whether an :class:`OSError` appears in its cause or context chain.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, OSError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


class UnknownTemplateVariableError(DeterministicRefusalError, AttributeError):
    """A ``{{ variable }}`` in a file-name or channel template that the simulator does not have.

    Decided by the configuration alone, so it is not retried. Also an :class:`AttributeError`, which
    is what template expansion raised for it before, so a caller catching that still catches it.
    """
