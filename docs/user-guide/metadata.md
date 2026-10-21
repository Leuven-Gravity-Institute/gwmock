# Metadata Files

Provenance records are written alongside each simulated dataset and use the
`.metadata.json` suffix by default. The file format is determined by the suffix:
files ending in `.json` are written as JSON; files ending in any other suffix
(such as `.yaml`) are written as YAML. Both formats can be read back by
`gwmock simulate` for reproduction.

The `gwmock merge` command writes merged metadata with a `.metadata.yaml`
suffix.

A copy of the same record is also written **inside** each HDF5 output, at the
file root, so a data file separated from its sidecar still describes its run.
The copy omits the file hashes -- a file cannot carry its own digest -- and, by
default, the injection parameters; see
[Injection parameters in the data files](configuration.md#injection-parameters-in-the-data-files)
for the flag that includes them and
[The record inside the file](reading-data.md#the-record-inside-the-file) for
reading it back. The embedded copy is the shareable record described below; the
sidecar keeps the injection parameters and hashes the embedded copy omits.

## Shareable and private metadata

A record describes the run, but a complete one also describes who ran it and on
which machine: the user name, the absolute directories (which often contain a
home directory), and the exact kernel build. A data set is usually shared
together with its metadata, so by default gwmock writes every record as two
files:

| File                                                        | Holds                                                     | Share it |
| ----------------------------------------------------------- | --------------------------------------------------------- | -------- |
| `orchestration-<k>.metadata.json`, `<merged>.metadata.yaml` | the shareable record                                      | yes      |
| `orchestration-<k>.private.json`, `<merged>.private.yaml`   | the private details: what the shareable record leaves out | no       |

The shareable record keeps the name records always had, so everything that reads
records -- `gwmock merge`, `gwmock validate`, `gwmock reindex`,
`gwmock find-signal`, `gwmock find-glitch`, replay with `gwmock simulate`, and
checkpoint resume -- finds it, and all of them work from it alone. The private
file never matches the `*.metadata.json` pattern those readers look for.

What goes where:

| Field                                                                  | Shareable record                                                       | Private details file                              |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------- |
| `author`, `email`                                                      | absent                                                                 | `author`, `email`                                 |
| `config.globals.working-directory`, `globals_config.working-directory` | `.`                                                                    | the absolute directory                            |
| any other absolute path (a value, a mapping key, or inside text)       | relative to the working directory; its file name if it lies outside it | the original, at its location in the record       |
| `outputs[].path`, `file_hashes`, `content_hashes`, `output_files`      | unchanged: already relative paths and file names                       | --                                                |
| `host.platform`                                                        | `<os>-<architecture>`, e.g. `Linux-x86_64`                             | the full string, e.g. `Linux-6.8.0-...-glibc2.39` |
| `host.os`, `host.architecture`, `host.cpu_model`                       | added; `cpu_model` is `null` when the platform does not report it      | --                                                |
| `host.python`, `host.cpu`, `host.git_sha`                              | unchanged                                                              | --                                                |
| host name                                                              | absent                                                                 | `hostname`                                        |
| `private_metadata`                                                     | `{"file": <private file name>, "sha256": <digest of its bytes>}`       | --                                                |

The private file is JSON (or YAML, beside a YAML record) with the author, the
host name, the full platform string and the working directory at its top level,
for a person to read, and under `values` every original value at a JSON pointer
into the shareable record (`/config/globals/working-directory`, `/author`,
`/host`, ...), with any renamed mapping key under `renamed_keys`. Together the
two files rebuild the combined record exactly:

```python
import json

from gwmock.cli.utils.metadata import combine_metadata_records

shareable = json.load(open("metadata/orchestration-0.metadata.json"))
private = json.load(open("metadata/orchestration-0.private.json"))
complete = combine_metadata_records(shareable, private)
```

The private file also carries a random `nonce`. The shareable record publishes
the private file's SHA-256, and what the private file holds -- a user name, a
host name, a directory layout -- is often few enough possibilities to guess; the
nonce stops the published digest from confirming a guess.

**On the machine that wrote the run**, gwmock's readers use the private details
when they are beside the record and their SHA-256 matches: replay and checkpoint
resume then use the original absolute paths, exactly as from a combined record.
**Anywhere else** -- a data set received without its private files -- they work
from the shareable record: hashes are keyed by file name, outputs are located
relative to the record's own directory, and replay resolves paths against the
directory it is started in (see
[Reproducing a run](reproducibility.md#reproducing-a-run)).

**The copy embedded in each HDF5 output is the shareable record**, never the
private details, so a data file released on its own names no person, machine or
directory either. The signal and glitch indexes are built from it too.

`gwmock merge` writes its record the same way: the merged record, with the paths
of its sources relative to the directory the merge runs in (or reduced to file
names), and `file_hashes` keyed by the merged file's name; the author and the
original paths go to `<merged>.private.yaml`. The source records it nests are
split too, so a merge of records written with the split turned off still yields
a shareable merged record.

### Keeping the combined layout

Set `globals.metadata.split: false` in the configuration (and pass
`--no-split-metadata` to `gwmock merge`) to write one combined record per batch,
with the same keys as before the split existed -- including `author`, the
absolute paths and the full `host.platform` -- and no private file. That record
is also the one embedded in the outputs. See
[Configuration](configuration.md#metadata-shareable-and-private-records).

### Before sharing a data set

Share the `*.metadata.json` / `*.metadata.yaml` files, the indexes and the data;
keep the `*.private.*` files. Copying a whole metadata directory copies the
private files too, so exclude them, for example:

```bash
rsync -a --exclude '*.private.*' run/ destination/
```

Paths written _inside_ your configuration that are relative already are kept as
you wrote them, and free-text values from backends are scanned for absolute
paths, but nothing can recognise a user name typed into an unrelated
configuration value; review the `config` section of a record before a public
release.

See [Reproducibility](reproducibility.md) for the full schema, reproduction
recipe, and versioning rules.
