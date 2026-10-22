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


#: HTTP client errors a later attempt may not meet: a request timeout, too early, rate limited.
_RETRYABLE_CLIENT_STATUSES = frozenset({408, 425, 429})
_CLIENT_ERROR_RANGE = range(400, 500)


def is_permanent_http_failure(error: BaseException) -> bool:
    """Return whether *error* carries an HTTP response no repeat of the request would change.

    That is a client error -- the URL is wrong, unauthorised or forbidden (401, 403, 404, ...) --
    other than a request timeout, "too early" or a rate limit (408, 425, 429). Read from the
    ``response.status_code`` that ``requests.HTTPError`` carries, or the ``code`` that
    ``urllib.error.HTTPError`` carries, so neither library need be imported here.

    Args:
        error: The exception to inspect.

    Returns:
        Whether it is a permanent HTTP failure.
    """
    status = getattr(getattr(error, "response", None), "status_code", None)
    if not isinstance(status, int) and isinstance(error, OSError):
        status = getattr(error, "code", None)
    return isinstance(status, int) and status in _CLIENT_ERROR_RANGE and status not in _RETRYABLE_CLIENT_STATUSES


def wraps_transient_failure(error: BaseException) -> bool:
    """Return whether *error* was raised from an I/O or network failure a retry may get past.

    A library that turns a connection error into a ``ValueError`` has hidden a failure a retry may
    get past. ``requests``, ``urllib`` and ``filelock`` failures are all :class:`OSError`
    subclasses, so one check covers them. Used where a ``ValueError`` or ``TypeError`` from a whole
    subsystem is marked final, so such a wrapping stays retryable.

    The whole chain is walked from *error* outwards before deciding. A
    :class:`DeterministicRefusalError` or a permanent HTTP failure (see
    :func:`is_permanent_http_failure`) anywhere in it wins -- *not transient* -- because something
    along the chain already shows the failure is final, however many ``OSError`` wrappers sit
    above it. Otherwise the answer is whether any :class:`OSError` was seen. A cycle in the chain
    ends the walk.

    **Which links are followed.** ``__cause__`` always: ``raise X from Y`` says ``Y`` is why. The
    implicit ``__context__`` only when nothing suppresses it -- ``raise X from None`` sets
    ``__suppress_context__`` and is the author saying the earlier failure is not the reason. The
    context is followed at all because a library may wrap a network failure without chaining it
    explicitly, and that must stay retryable.

    **Residual risk.** Following ``__context__`` can still misread a deterministic error raised
    while an :class:`OSError` was being handled -- a configuration check failing inside an
    ``except FileNotFoundError`` fallback, say -- as transient. The cost of that misreading is a
    bounded number of wasted retries before the same error is reported, while the opposite mistake
    would make a real network failure final; the rule is chosen for the cheaper error.

    Args:
        error: The exception to inspect.

    Returns:
        Whether the followed chain holds an :class:`OSError` and no refusal or permanent HTTP failure.
    """
    seen: set[int] = set()
    saw_io_failure = False
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, DeterministicRefusalError) or is_permanent_http_failure(current):
            return False
        saw_io_failure = saw_io_failure or isinstance(current, OSError)
        seen.add(id(current))
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    return saw_io_failure


class UnknownTemplateVariableError(DeterministicRefusalError, AttributeError):
    """A ``{{ variable }}`` in a file-name or channel template that the simulator does not have.

    Decided by the configuration alone, so it is not retried. Also an :class:`AttributeError`, which
    is what template expansion raised for it before, so a caller catching that still catches it.
    """
