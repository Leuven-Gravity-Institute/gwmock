# Reproducibility

`gwmock` writes one versioned JSON provenance record per generated batch as
`*.metadata.json`.

## Schema

Each record is validated at write time and uses schema version `1.5.0`.
Consumers must reject unknown major versions.

```json
{
    "schema_version": "1.5.0",
    "gwmock_version": "x.y.z",
    "subpackage_versions": {
        "gwmock_signal": "x.y.z",
        "gwmock_noise": "x.y.z",
        "gwmock_pop": "x.y.z"
    },
    "config": {},
    "resolved_config": {},
    "replayable": true,
    "config_sha256": "...",
    "seed": 42,
    "segment_seeds": [123456789, 987654321],
    "population": {
        "backend": "module:Class",
        "source_type": "bbh",
        "n_events": 128,
        "parameter_names": [],
        "metadata": {}
    },
    "signal": {
        "backend": "module:Class",
        "waveform_model": "IMRPhenomXPHM",
        "detector_network": ["ET1_SARD", "ET2_SARD", "ET3_SARD"],
        "injections": [
            {
                "event_id": 0,
                "parameters": { "mass_1": 30.0, "coa_time": 1577491218.5 }
            }
        ],
        "metadata": {}
    },
    "noise": {
        "backend": "module:Class",
        "psd": "ET_10_full_cryo_psd",
        "metadata": {}
    },
    "outputs": [
        {
            "kind": "signal",
            "path": "output/signal/E-ET1_SARD_STRAIN_BBH-1577491218-1024.hdf5",
            "channels": ["ET1_SARD:STRAIN"],
            "t0": 1577491218,
            "duration": 1024,
            "sha256": "..."
        }
    ],
    "host": {
        "platform": "...",
        "python": "3.12.x",
        "cpu": "...",
        "git_sha": "..."
    },
    "environment": {
        "python": "3.12.5",
        "python_implementation": "CPython",
        "packages": { "numpy": "2.0.0", "gwmock": "x.y.z", "...": "..." }
    }
}
```

`config` stores the input configuration snapshot for that run (template
variables expanded). `resolved_config` stores the same config with every
runtime-resolved external value folded in — for example a `DeepExtractorGlitch`
whose dataset was downloaded at the repository default is recorded here pinned
to the concrete Hugging Face commit it actually used. It is `null` when nothing
needed resolving (a purely parametric run). **Replay prefers `resolved_config`
over `config`**, so a run that did not explicitly pin its external inputs still
reproduces the exact resources it used.

`replayable` is `true` unless a declared external-mutable input could not be
pinned to an immutable version (e.g. an offline dataset with no local cache); a
`false` run is not bit-for-bit reproducible from its metadata, and replaying it
emits a warning.

`segment_seeds` stores the deterministic per-segment seeds that `gwmock` derives
locally. Adapter-backed noise now consumes one shared
`gwmock_noise.open_stream(...)` iterator per run, so the top-level `seed` is
recorded once and noise continuation no longer appears as one derived seed per
batch. The subpackage `metadata` objects are preserved as JSON objects without
gwmock rewriting their internal structure.

`signal.injections` records the source parameters of the signals attributed to
that batch's frame(s), in injection order. Each entry is
`{"event_id": <index in the population>, "parameters": {...}}`. A signal is
listed against **every frame its samples reach**, so a long inspiral crossing a
segment boundary appears in each frame it spans, and a continuous wave appears
in all of them. This changed in schema 1.5.0: a signal used to be listed only
under the frame it was generated for, which for a 48 s inspiral across 32 s
segments meant one frame out of three -- and not the one holding the merger. The
frame a signal is _generated_ for is the one its waveform **starts** in (schema
1.4.0; before that, the one its `coa_time` fell in). `event_id` is the event's
index in the population as ordered for the run (by `coa_time` under the default
ordering), so it is stable for a fixed configuration. Stationary/SGWB segments
have no discrete events and record an empty list.

`environment` is a full freeze of the environment that produced the run — the
Python version and the version of every installed distribution (direct and
transitive) — recorded so the run can be reproduced against exactly those
dependencies. It is `null` for records written before this field existed.

A copy of the record is written into each HDF5 output as well, at the file root,
so a data file that reaches a consumer without its sidecar still describes the
run. It is the same record with three omissions: the file hashes (a file cannot
carry its own digest), `pre_batch_state` (stored as separate `.npy` files an
embedded copy could not point at), and `signal.injections` unless the run set
`orchestration.include-injection-parameters: true`. The sidecar described above
is always complete.

That copy carries a timestamp, the host and the environment freeze, so **two
identical runs no longer write byte-identical HDF5 files** even though they hold
identical samples. What "reproducible" is measured against here is the _content_
hash — the decoded samples plus their timing, recorded as `content_sha256` and
checked by `gwmock validate` separately from the byte hash — which the embedded
record does not affect. GWF output has never been byte-reproducible, for the
same kind of reason: a frame records its write time.

For the config shape that feeds this record, see
[Orchestration](orchestration.md) and [Protocol Contracts](protocols.md).

## Exact-dependency reproduction (`--isolate`)

By default, reproducing from metadata runs in your current environment and warns
if the recorded package versions differ. For bit-for-bit reproduction against
the exact dependencies of the original run, add `--isolate`:

```bash
gwmock simulate metadata/ --isolate
```

This reads the recorded `environment` freeze, builds a cached, isolated
[uv](https://docs.astral.sh/uv/) virtualenv pinned to those versions (matching
the recorded Python `major.minor`), and re-runs the reproduction inside it. If
the current environment already matches, it runs in place; if no environment was
recorded (older metadata), it warns and runs in place.

Requirements and limits:

- `uv` must be installed, and the recorded package versions must be resolvable
  from your package index — a run made with editable/dev installs (versions not
  published to an index) cannot be recreated this way and will fail loudly
  rather than run in the wrong environment.
- Environments are cached under `~/.cache/gwmock/reproduction-envs` (override
  with `GWMOCK_ENV_CACHE`) and keyed by the version set, so repeated
  reproductions of the same run skip reinstalling.
- **Earth-orientation data has a shelf life, and it is not pinned by a package
  version.** Anything using sidereal time — every projection with
  `earth-rotation: true` — depends on the IERS table Astropy loads. Pinning
  `astropy-iers-data` recreates the table _bundled_ with that release, but
  Astropy's `iers.conf.auto_download` is `True` by default and its
  `auto_max_age` is 30 days: once the pinned release is older than that, Astropy
  fetches the current table from the IERS server instead, and no recorded
  package version captures which one it got. A run reproduced within a month of
  its dependencies' release matches; reproduced a year later, the sidereal time
  can differ.

    To pin it properly, set `iers.conf.auto_download = False` before simulating,
    so the packaged table is used regardless of age — at the cost of using
    Earth-orientation data as old as the pinned release. Measured scale: one
    weekly table release moved a strain peak by 1.6e-06 relative.

## Finding which frame contains a signal

Alongside the per-file `index.yaml`, a run writes `signal_index.yaml` mapping
each signal's `event_id` to the frame file(s) that contain it. The entry records
one contribution per batch, because a signal reaching several segments is
written by several batches; `find-signal` flattens them, and reports `metadata`
as a **list** of batch metadata files on both lookup paths. An index written
before schema 1.5.0 is still read. Use `gwmock find-signal` to resolve a signal
to its frame:

```bash
# By id (fast path via signal_index.yaml)
gwmock find-signal --metadata-dir metadata/ --id 42

# By parameter filters (scans the recorded injections); combine with AND
gwmock find-signal --metadata-dir metadata/ --param mass_1>=30 --param coa_time<1577491300

# Machine-readable
gwmock find-signal --metadata-dir metadata/ --id 42 --json
```

Filters accept `==`, `!=`, `>`, `<`, `>=`, `<=`; numeric values are compared
numerically. The command exits non-zero when no signal matches.

## Rebuilding the signal index

`signal_index.yaml` is a cache. The `*.metadata.json` files are the source of
truth — they record every injection and every frame the batch wrote — so the
index can always be derived again from them:

```bash
gwmock reindex --metadata-dir metadata/
```

Reach for it when the id fast path disagrees with the frames: `find-signal --id`
reports nothing (or too little) for a signal whose samples are in the data, or a
run refuses to write the index because it "is not the one last committed". Both
are symptoms of a lost or hand-edited index, and neither needs the simulation
rerunning.

Updates to the index are serialised by an exclusive lock on a sidecar file, so
concurrent runs sharing one metadata directory on one host do not overwrite each
other. Two cases fall outside that and are what this command is for:

- **A filesystem without working advisory locks.** Where `flock` is unavailable
  or rejected, gwmock warns once and writes unsynchronised — the behaviour that
  loses one writer's entries.
- **Writers on different hosts.** Taking the lock revalidates the sidecar, not
  the index, so a client with a cached view can hold the lock and still read a
  stale index. gwmock refuses that write rather than letting it discard entries,
  which keeps the index correct at the cost of stopping the run.

On **Windows**, renaming the new index into place is refused while another
process holds `signal_index.yaml` open — and a reader is enough, so a
`gwmock find-signal` lookup running alongside a simulation is the ordinary way
to meet it. gwmock retries the rename on a bounded backoff, a few tenths of a
second in total, which covers a lookup that opens and closes the file. A
consumer that keeps the index open for longer than that is not waited out: the
update fails, saying so, with the previous index and its recorded digest
untouched and nothing to repair — close the consumer and run the batch again.

A **local POSIX filesystem** permits the rename while readers hold the
destination open, so a lookup running alongside a simulation never reaches the
retry there. Two POSIX cases do reach it: a metadata directory on a CIFS/SMB
mount, where the server's sharing semantics travel to the client, and a rename
refused for an ordinary reason such as the directory's permissions — which is
now reported after the bounded backoff has been spent rather than immediately.

Rebuilding takes the same lock a running batch does and re-records the digest
beside the index, so a directory that was refusing writes accepts them again
afterwards — there is no need to delete the sidecar by hand.

Two things it will not do. It refuses a directory holding no `*.metadata.json`
files, rather than replacing a good index with an empty one on a mistyped path;
and it refuses if any batch metadata file cannot be used — unreadable, not valid
JSON, or valid JSON that is not a batch metadata record — rather than writing an
index that silently omits that batch's events. On a shared filesystem, stop
writers on other hosts before rebuilding: the rebuild indexes the metadata files
it can list, and a client may not yet be listing a file another host has just
written.

## Reproducing a run

For deterministic reproduction, pin `gwmock`, `gwmock-signal`, `gwmock-noise`,
and `gwmock-pop` to the same versions used originally, then rerun the same
config file. The seed is stored in the config itself:

```bash
gwmock simulate config.yaml
```

In batch reproduction workflows, pass the generated `*.metadata.json` files
directly to `gwmock simulate`. Each metadata file carries the exact config
snapshot and per-segment seeds needed to reproduce that batch independently.
Because replay reads `resolved_config`, any downloaded dataset (e.g. a
DeepExtractor glitch dataset) is refetched at the exact version the original run
used, even if the config never pinned it and the upstream dataset has since
moved:

```bash
# Reproduce specific batches from their metadata files
gwmock simulate metadata/orchestration-0.metadata.json metadata/orchestration-1.metadata.json

# Or reproduce everything from a metadata directory
gwmock simulate metadata/
```
