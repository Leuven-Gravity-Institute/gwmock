"""Which truth catalogue a lookup index is built from, and how its entries are shaped.

A run records two kinds of injection truth, and each gets an index from event id to the frame
file(s) holding it: signals in ``signal_index.yaml``, glitches in ``glitch_index.yaml``. The two
share every line of the machinery that writes them -- the cross-process lock, the staleness guard,
the atomic replacement -- and every line of the machinery that reads them. What differs is which
section of a batch metadata record the events come from, which outputs they are attributed to, and
what an entry summarises. That difference lives here, as data, so the code is written once.

Kept in its own module, apart from both the writer and the reader, because the reader is
deliberately light: ``gwmock find-signal`` and ``gwmock find-glitch`` open a YAML file and some
JSON, and should not pull in the simulation stack to do it.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class IndexSpec(NamedTuple):
    """One truth index: where its events come from and how its entries are shaped.

    Attributes:
        index_file_name: File name of the index inside the metadata directory.
        described_as: How the index is named in a message to whoever is holding the terminal.
        events_described_as: Plural noun for what the index holds, for the same purpose.
        section: Metadata-record section holding the events (``signal`` or ``noise``).
        events_key: Key within that section holding the event list.
        output_kind: ``outputs[].kind`` of the artifacts an event is recorded against.
        time_key: Parameter summarised onto the index entry, so a lookup can report an event's
            time without opening a metadata file.
        parameters_key: Sub-mapping of an event holding its filterable parameters, or ``None``
            when the event row is itself that mapping. A signal record separates its identity
            from its source parameters; a glitch row is flat.
    """

    index_file_name: str
    described_as: str
    events_described_as: str
    section: str
    events_key: str
    output_kind: str
    time_key: str
    parameters_key: str | None

    @property
    def events_path(self) -> str:
        """Return the dotted path to the event list, for use in messages."""
        return f"{self.section}.{self.events_key}"

    def events(self, metadata: dict[str, Any]) -> Any:
        """Return the event list a batch metadata record holds, exactly as it holds it.

        Not normalised to a list: a rebuild has to be able to tell an absent key from a malformed
        one, and that judgement belongs to the caller doing the validating.
        """
        section = metadata.get(self.section)
        if not isinstance(section, dict):
            return None
        return section.get(self.events_key)

    def parameters(self, event: dict[str, Any]) -> dict[str, Any]:
        """Return the filterable parameters of one event."""
        if self.parameters_key is None:
            return event
        nested = event.get(self.parameters_key)
        return nested if isinstance(nested, dict) else {}

    def event_time(self, event: dict[str, Any]) -> Any:
        """Return the time summarised onto this event's index entry."""
        return self.parameters(event).get(self.time_key)


#: The signal index: which frame file(s) hold a given injected signal.
SIGNAL_INDEX = IndexSpec(
    index_file_name="signal_index.yaml",
    described_as="signal index",
    events_described_as="signals",
    section="signal",
    events_key="injections",
    output_kind="signal",
    time_key="coa_time",
    parameters_key="parameters",
)

#: The glitch index: which noise frame file(s) hold a given injected glitch. The counterpart of the
#: signal index for the other producer, so a glitch frame set can be scored the way a signal one
#: already could.
GLITCH_INDEX = IndexSpec(
    index_file_name="glitch_index.yaml",
    described_as="glitch index",
    events_described_as="glitches",
    section="noise",
    events_key="glitch_injections",
    output_kind="noise",
    # The GPS time of the injected waveform's FIRST SAMPLE, not its peak -- which is what the
    # producer's catalogue records under this name, and what a consumer cutting a window around a
    # glitch has to know before it uses the number.
    time_key="gps_start_time",
    parameters_key=None,
)
