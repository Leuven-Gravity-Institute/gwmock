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
reading it back. The sidecar written here is always the complete record.

See [Reproducibility](reproducibility.md) for the full schema, reproduction
recipe, and versioning rules.
