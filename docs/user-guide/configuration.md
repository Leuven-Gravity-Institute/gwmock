# Configuration Files

This guide explains how to use and write configuration files to generate
datasets tailored to your needs.

## Verbosity

All gwmock commands accept a top-level `--verbose` / `-v` flag to control log
output:

```bash
gwmock --verbose DEBUG simulate config.yaml   # detailed debug output
gwmock --verbose WARNING simulate config.yaml # only warnings and errors
```

Supported levels: `NOTSET`, `DEBUG`, `INFO` (default), `WARNING`, `ERROR`,
`CRITICAL`.

## Command-Line Options

### Command [`simulate`](/reference/gwmock/cli/simulate/)

```bash
gwmock simulate config.yaml
```

This is the primary command used to generate mock data. It takes a `.yaml`
configuration file as input, which defines the simulation parameters.

#### Flag `--overwrite` (optional)

By default, gwmock does not overwrite existing output files. If a file already
exists, the tool will raise an error and halt execution. To force overwriting of
existing files, use the `--overwrite` flag:

```bash
gwmock simulate config.yaml --overwrite
```

#### Flag `--dry-run` (optional)

Test your configuration without generating data:

```bash
gwmock simulate config.yaml --dry-run
```

This validates the configuration and shows what would be generated without
actually creating files.

#### Flag `--output-dir` (optional)

Override the output directory from the command line without editing the config:

```bash
gwmock simulate config.yaml --output-dir /scratch/my_run/data
```

#### Flag `--metadata-dir` (optional)

Override the metadata directory from the command line (config mode only):

```bash
gwmock simulate config.yaml --metadata-dir /scratch/my_run/metadata
```

#### Flag `--metadata` (optional)

Generate metadata files along with the data (automatically enabled by default):

```bash
gwmock simulate config.yaml --metadata
```

Metadata files contain complete provenance information including:

- Simulator configuration
- Random number generator state
- Output file names
- Version information

#### Flags `--author` and `--email` (optional)

Include author information in the metadata files:

```bash
gwmock simulate config.yaml --author <your-name> --email <your-email>
```

### Command [`config`](/reference/gwmock/cli/config/)

```bash
gwmock config <flag>
```

This command is used to manage default and example configuration files. Exactly
one of the flags `--list`, `--get`, or `--init` must be provided.

#### Flag `--list`

List all the available example configuration files stored in the
[`examples`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples)
directory (see the [Examples](examples.md) page).

```bash
gwmock config --list
```

#### Flag `--get`

Copy one of the available example configuration files from the
[`examples`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples)
directory into the working directory. The `<example_label>` must be one of the
example names listed by the `gwmock config --list` command.

```bash
gwmock config --get <example_label>
```

#### Flag `--init`

Creates a default configuration file and saves it to the working directory.

```bash
gwmock config --init config.yaml
```

#### Flag `--overwrite` (optional)

By default, gwmock does not overwrite existing configuration files. If a file
already exists, the tool will raise an error and halt execution. To force
overwriting of existing files, use the `--overwrite` flag together with `--get`
or `--init`:

```bash
gwmock config --get noise/uncorrelated_gaussian/quick_start --overwrite
gwmock config --init config.yaml --overwrite
```

#### Flag `--output` (optional)

Specifies the directory where the configuration file will be saved. This flag
must be used together with `--get` or `--init`. If not provided, the working
directory is used by default.

```bash
gwmock config --get <label of the configuration file> --output <directory or file>
```

#### Flag `--interactive` (optional)

Launch an interactive terminal-based configuration editor with a live preview,
autocomplete, and guided workflows:

```bash
gwmock config --interactive
```

The interactive editor provides:

- **Live configuration preview**: See your configuration update in real-time as
  you build it
- **Autocomplete suggestions**: Type `/` to see available commands, then use Tab
  to complete
- **Command history**: Use Up/Down arrows to navigate through previously entered
  commands
- **Validation feedback**: Get immediate feedback on invalid values (e.g.,
  negative seeds, invalid chunk counts)
- **Templates**: Start with common configurations using `/template` commands
- **Script generation**: Generate SLURM job scripts or local execution scripts
  with `/generate-script`

**Common commands:**

| Command                             | Description                                             |
| ----------------------------------- | ------------------------------------------------------- |
| `/template <type>`                  | Load a preset (e.g., `noise`, `signal+noise`, `glitch`) |
| `/psds`                             | List available power spectral densities                 |
| `/geometries`                       | List available detector geometries                      |
| `/noise psd <value>`                | Set noise PSD                                           |
| `/noise detectors <list>`           | Set detector network                                    |
| `/batch chunks-enabled true`        | Enable chunking for parallel execution                  |
| `/batch chunks-n-chunks <n>`        | Set number of chunks                                    |
| `/save <filename>`                  | Save configuration to file                              |
| `/generate-script slurm <filename>` | Generate SLURM job script                               |
| `/generate-script local <filename>` | Generate local execution script                         |
| `/help`                             | Show all available commands                             |

**Example workflow:**

```bash
# Start interactive editor
gwmock config --interactive

# Inside the editor:
/template noise
/noise psd ET_10_full_cryo_psd
/noise detectors ET-Triangle-EMR
/batch chunks-enabled true
/batch chunks-n-chunks 4
/save my_config.yaml
/generate-script slurm submit.sh
```

#### Flag `--load` (optional)

Load an existing configuration file into the interactive editor for
modification:

```bash
gwmock config --interactive --load existing_config.yaml
```

This is useful for:

- Modifying existing configurations without manually editing YAML
- Exploring what settings are in a configuration file
- Generating job scripts for existing configurations

## Configuration File Structure

The configuration file uses YAML format. It consists of a shared `globals`
section plus the adapter-backed [orchestration](orchestration.md) schema.

### Globals

Top-level shared parameters used across all simulators:

```yaml
globals:
    working-directory: .
    output-directory: output
    metadata-directory: metadata
    simulator-arguments:
        sampling-frequency:
        duration:
        start-time:
        total-duration:
    output-arguments: {}
```

**Key parameters:**

- `working-directory`: Base directory for operations
- `output-directory`: Where to save generated data files
- `metadata-directory`: Where to save metadata files
- `sampling-frequency`: Sample rate in Hz
- `duration`: Duration of each segment in seconds
- `start-time`: GPS start time
- `total-duration`: Total duration of the dataset
- `output-arguments`: Additional global arguments passed to the file writer

### Orchestration

The `orchestration:` section is required and must contain at least one of
`population`, `signal`, or `noise`. CBC signal generation uses `population` plus
`signal`. SGWB signal generation can use `signal` without `population` when
`signal.source-type` is set.

```yaml
orchestration:
    population:
        backend: FilePopulationLoader # or any registered backend alias
        source-type: bbh
        n-samples: 128 # optional; omit to load the full catalogue
        arguments:
            path: population.h5

    signal:
        waveform-model: IMRPhenomXPHM
        minimum-frequency: 10
        detectors:
            - ET-Triangle-EMR
        output:
            file_name:
                'E-{{ detectors }}_STRAIN_BBH-{{ start_time }}-{{ duration
                }}.hdf5'
            arguments:
                channel: '{{ detectors }}:STRAIN'

    noise:
        arguments:
            psd_file: ET_10_full_cryo_psd
            seed: 42
            detectors:
                - ET-Triangle-EMR
        output:
            file_name:
                'E-{{ detectors }}_STRAIN_NOISE-{{ start_time }}-{{ duration
                }}.hdf5'
            arguments:
                channel: '{{ detectors }}:STRAIN'
```

### Choosing the waveform library

`signal.waveform-backend` selects which library generates the polarizations —
`lal` (the default), `pycbc`, `ripple`, or `gwsignal`. It also accepts an entry
point in the `gwmock.waveform` group or a `module:Class` reference, so a
third-party backend can be plugged in the same way. Such a backend is matched by
its public surface — `available_approximants` and `generate_td_waveform` — and
does not have to subclass gwmock-signal's `WaveformBackend`.

Constructor arguments for that backend go under
`signal.waveform-backend-arguments`. These are distinct from `signal.arguments`,
which is passed to the _simulator_ rather than to the waveform backend:

```yaml
signal:
    waveform-model: IMRPhenomD
    waveform-backend: ripple
    waveform-backend-arguments:
        taper_fraction: 0.05 # ripple-specific
```

Two things to be aware of:

- This selects a **library, not a compute device.** `ripple` is JAX-based, but
  on its own it runs through the same per-event path as LAL. The batched
  on-device entry point is selected separately, with
  [`signal.execution`](#choosing-the-execution-mode).
- The same approximant from two libraries agrees closely but not exactly, so the
  choice changes the data. It is recorded in the run metadata as
  `orchestration.signal.waveform_backend` for that reason.

`ripple` requires the extra: `pip install 'gwmock[jax]'`. It also JIT-compiles
each waveform model on first use — measured on a single 8-second segment,
`IMRPhenomD` took ~10 s end to end against ~0.7 s for LAL, and the precessing
`IMRPhenomXPHM` ~72 s. The cost is paid once per process, so it amortises over a
long run.

### Choosing the execution mode

`signal.execution` selects **how** a segment's events are computed,
independently of which library computes them:

| Value                 | Behaviour                                                                  |
| --------------------- | -------------------------------------------------------------------------- |
| `per-event` (default) | Loop over the segment's events, one waveform at a time.                    |
| `batched`             | Hand the whole segment to gwmock-signal's batched entry point in one call. |

```yaml
orchestration:
    signal:
        execution: batched
        waveform-model: IMRPhenomD
        waveform-backend: ripple
```

Batched is the **GPU-capable** path, but this key does not choose a device.
Whether it runs on a GPU depends only on the installed JAX backend:

- `pip install 'gwmock[jax]'` — batched, on the CPU.
- `pip install 'gwmock[cuda]'` — installs the CUDA backend (Linux x86_64, CUDA
  12); runs on a GPU when a compatible device and driver are present, and
  silently falls back to the CPU when they are not.

Nothing in the output distinguishes the two and no warning is raised for the CPU
case. Check with `python -c "import jax; print(jax.devices())"`.

Two constraints. The batched path always generates with `ripple` whatever
`waveform-backend` names — a different library is refused rather than silently
substituted. And it refuses any signal setting it cannot apply
(`waveform-options`, `signal.arguments`, `signal.parameters`), so a
configuration that reaches the generator unchanged is the only one that runs.

GPU and CPU results are **not** bit-identical. Measured on an RTX 2080 Ti, they
agree to ~4e-13 of peak — a sub-sample time shift, not an accuracy difference.
Neither is known to be more correct.

See `examples/signal/execution/batched` for a runnable configuration.

For SGWB studies, use `signal.source-type: sgwb`. Constructor options for the
SGWB backend belong under `signal.arguments`, while spectrum parameters passed
to `simulate(...)` belong under `signal.parameters`:

```yaml
orchestration:
    signal:
        source-type: sgwb
        detectors:
            - ET-Triangle-Sardinia
        minimum-frequency: 5
        parameters:
            omega_ref: 1.0e-9
            spectral_index: 0.0
            reference_frequency: 25.0
        output:
            file_name: sgwb-{{ counter }}.hdf5
```

For the full schema and backend registration options, see the
[Orchestration](orchestration.md) guide.

Transient glitches are configured on the noise side under
`orchestration.noise.arguments.glitches` using public `gwmock-noise` glitch
models. For example:

```yaml
orchestration:
    noise:
        arguments:
            glitches:
                - kind: gengli_blip
                  rate: 0.0011111111111111111
                  amplitude_distribution:
                      distribution: lognormal
                      mean: 1.0
                      std: 0.0
                  population_file: glitches.hdf5
                  psd_file: https://example.org/ET_10_full_cryo_psd.txt
```

`kind: deepextractor` requires the extra: `pip install 'gwmock[deepextractor]'`.
Its waveforms are reconstructions fetched from a HuggingFace dataset rather than
generated, so without the extra the model raises on the first glitch it is asked
for.

## Template Variables

You can use Jinja2-style templates in configuration values such as file names
and channel names:

```yaml
orchestration:
    noise:
        arguments:
            detectors:
                - E1_triangle_emr
                - E2_triangle_emr
                - E3_triangle_emr
        output:
            file_name:
                'E-{{ detectors }}_STRAIN_NOISE-{{ start_time }}-{{ duration
                }}.hdf5'
            arguments:
                channel: '{{ detectors }}:STRAIN'
```

In this example, `file_name` is automatically expanded for each detector being
processed.

**Common variables:**

- `{{ start_time }}`: GPS start time from globals
- `{{ duration }}`: Segment duration from globals
- `{{ detectors }}`: Current detector being processed. A network alias such as
  `ET-Triangle-EMR` expands to one file/channel per interferometer, with
  `{{ detectors }}` resolving to the per-interferometer token (`ET1_EMR`,
  `ET2_EMR`, `ET3_EMR`)

## Checkpointing

gwmock automatically creates checkpoints during long simulations. If a process
is interrupted:

1. A `.gwmock_checkpoint/simulation.checkpoint.json` file is saved in the
   working directory
2. Rerun the same command to resume from the last checkpoint
3. The tool automatically detects and continues from where it left off

```bash
# Start simulation
gwmock simulate config.yaml

# If interrupted (Ctrl+C, crash, etc.), resume with same command
gwmock simulate config.yaml
```

The checkpoint contains:

- Simulator state
- Progress information
- Already-generated file tracking
- The fingerprint of the run that wrote it — which configuration, which output
  and metadata directories, and, for a population file on local disk, that
  file's contents (a population fetched from a URL contributes its address only,
  since identifying a run does not download the catalogue again)

A resume only continues from a checkpoint it can attribute to the command being
run. It refuses when the fingerprints differ, and when the checkpoint carries no
fingerprint at all because it was written by a version from before the field
existed (`gwmock` 0.13.0 and earlier). Without that check, a second
configuration run from the same working directory resumes from the first's
checkpoint and skips the batches it recorded, so the outputs those batches would
have produced are never written and the run still exits successfully.

Both refusals name the checkpoint file, so it can be moved or deleted to start
fresh. To start fresh without touching it — for an automated run that cannot
answer a prompt — pass `--ignore-checkpoint`:

```bash
gwmock simulate config.yaml --ignore-checkpoint
```

## Resource Usage Summary

After every successful simulation, gwmock writes a `resource_usage_summary.json`
file to the working directory. This file records CPU time, peak memory usage,
and wall time for the run. It is always written (overwriting any previous
summary) and is not controlled by a flag.

## Best Practices

1. **Use templates**: Leverage Jinja2 templates for dynamic configuration
2. **Set seeds**: Always set `seed` for reproducibility
3. **Check space**: Ensure sufficient disk space before long runs
4. **Use dry-run**: Test configurations with `--dry-run` before full simulation
5. **Organize outputs**: Use descriptive `output-directory` and
   `metadata-directory` names
