# Quick Start

This guide will help you run your first gwmock simulation in just a few minutes.

## 1. Run Your First Simulation

Run the following command in your working directory:

```bash
# Copy quick-start configuration file to working directory
gwmock config --get noise/uncorrelated_gaussian/quick_start --output quick_start_config.yaml

# Run simulation
gwmock simulate quick_start_config.yaml
```

<!-- typos:off -->

The configuration file `quick_start_config.yaml` runs the full orchestration
pipeline — population sampling, CBC signal injection, and noise generation —
producing a 1024-second dataset for the ET Sardinia configuration.

Its one-event BBH population ships with gwmock: `gwmock config --get` copies it
next to the configuration as `bbh_population.csv` and points the configuration
at it, so the run needs no internet connection. Run `gwmock simulate` from the
directory you copied the configuration into, because the path is relative to it.

If copying fails, the command puts back whatever was in the directory before.
That holds against errors, not against the process being killed: an interrupted
copy can leave a `.gwmock-config-get.bbh_population.csv.*.backup` file holding
the catalogue that was there before, or a `.part` file. `gwmock config --get`
then refuses to run, naming them; rename the backup back if needed and delete
the rest.

<!-- typos:on -->

<!-- prettier-ignore -->
!!! note
    The quick-start needs about 200 MB of free disk space: the HDF5 output is uncompressed
    float64, so each run writes `duration × sampling-frequency × 8 bytes` per file,
    for every detector, once for signal and once for noise. Here that is
    1024 s × 4096 Hz × 8 B = 32 MiB per file, × 3 detectors × 2 = 192 MiB.

You should see output like:

```text
INFO Config mode: quick_start_config.yaml
INFO Configuration loaded and validated
INFO Created simulation plan with 1 batches
INFO Simulation plan validation passed
INFO Executing simulation plan: 1 batches
INFO Executing 1 simulators
Executing simulation plan:   0%|                                        | 0/1
Executing simulation plan: 100%|████████████████████████████████████████| 1/1
INFO Simulation completed successfully. Output written to output
```

## 2. Check the Output

Your working directory will contain:

```text
output/
├── noise/
│   ├── E-ET1_SARD_STRAIN_NOISE-1577491218-1024.hdf5
│   ├── E-ET2_SARD_STRAIN_NOISE-1577491218-1024.hdf5
│   └── E-ET3_SARD_STRAIN_NOISE-1577491218-1024.hdf5
└── signal/
    ├── E-ET1_SARD_STRAIN_BBH-1577491218-1024.hdf5
    ├── E-ET2_SARD_STRAIN_BBH-1577491218-1024.hdf5
    └── E-ET3_SARD_STRAIN_BBH-1577491218-1024.hdf5
metadata/
├── index.yaml
└── orchestration-0.metadata.json
resource_usage_summary.json
```

The `ET-Triangle-Sardinia` network alias expands to its three interferometers
(`ET1_SARD`, `ET2_SARD`, `ET3_SARD`), so one frame file per interferometer is
written for both noise and signal.

### Data Files

`output/noise/` contains the simulated Gaussian noise strain. `output/signal/`
contains the injected CBC signal strain. To produce a realistic data stream,
merge the two using GWpy (see [Reading Data](reading-data.md)).

### Metadata File

The `.metadata.json` file records the exact configuration, seeds, and package
versions needed to reproduce this run. For details on the schema and how to
reuse metadata files, see the [Metadata Files](metadata.md) and
[Reproducibility](reproducibility.md) pages.

## 3. Explore Different Simulators

To run any gwmock simulations, you only need to provide a `.yaml` configuration
file. A collection of ready-to-use configuration files is available in the
[`gwmock/examples`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples)
directory. You can use them directly or adapt them to suit your needs.

- For an overview of all available examples, see the [Examples](examples.md)
  page.
- For a more complete and user-friendly guide to writing your own configuration
  files, see the [Configuration Files](configuration.md) page.

## 4. Next Steps

- [Generating Data](generating-data.md) - Quick guide for generating datasets
  with gwmock, including Einstein Telescope (ET) examples.
- [Examples](examples.md) - Example configuration files for ET simulations.
- [Request New Features](../contributing.md) - How to request new features or
  improvements.
