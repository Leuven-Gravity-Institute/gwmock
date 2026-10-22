"""Exception types shared across gwmock."""

from __future__ import annotations


class DeterministicRefusalError(ValueError):
    """A refusal decided by the configuration and inputs alone, which no retry can change.

    Raised for a waveform argument or population column the waveform does not read, and for a
    placement query the signal simulator refuses. A batch retry re-runs the same configuration on
    the same inputs, so it would fail the same way after a backoff, repeating the log; the retry
    loop raises these at once instead (see ``gwmock.cli.simulate_utils.retry_with_backoff``).

    A subclass of :class:`ValueError`, which is what these refusals were raised as before the type
    existed, so a caller catching ``ValueError`` still catches them.
    """
