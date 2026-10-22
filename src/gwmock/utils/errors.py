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
