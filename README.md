# gwmock

[![Python CI](https://github.com/Leuven-Gravity-Institute/gwmock/actions/workflows/ci.yml/badge.svg)](https://github.com/Leuven-Gravity-Institute/gwmock/actions/workflows/ci.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Leuven-Gravity-Institute/gwmock/main.svg)](https://results.pre-commit.ci/latest/github/Leuven-Gravity-Institute/gwmock/main)
[![Documentation Status](https://github.com/Leuven-Gravity-Institute/gwmock/actions/workflows/documentation.yml/badge.svg)](https://leuven-gravity-institute.github.io/gwmock)
[![codecov](https://codecov.io/gh/Leuven-Gravity-Institute/gwmock/graph/badge.svg?token=GLW2LEFKW7)](https://codecov.io/gh/Leuven-Gravity-Institute/gwmock)
[![PyPI Version](https://img.shields.io/pypi/v/gwmock)](https://pypi.org/project/gwmock/)
[![Python Versions](https://img.shields.io/pypi/pyversions/gwmock)](https://pypi.org/project/gwmock/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![DOI](https://zenodo.org/badge/1115995501.svg)](https://doi.org/10.5281/zenodo.17925458)

A Python package for generating Mock Data Challenge (MDC) datasets for the
gravitational-wave (GW) community. It simulates strain data for detectors like
Einstein Telescope, providing a unified interface for reproducible GW data
generation.

## Features

- **Modular Design**: Uses mixins for flexible simulator composition
- **Detector Support**: Built-in support for various GW detectors with custom
  configuration options
- **Waveform Generation**: Integrates with PyCBC and LALSuite for accurate
  signal simulation
- **Noise Models**: Colored and correlated noise, spectral lines, and detector
  glitches via the `gwmock-noise` package
- **Population Models**: Handles injection populations for signals and glitches
- **Data Formats**: Outputs in standard GW formats (GWF frames)
- **CLI**: Command-line tools for easy simulation workflows

## Installation

We recommend using `uv` to manage virtual environments for installing gwmock.
The commands below use `uv`, but gwmock is a regular PyPI package that also
installs fine with plain `pip` in any environment.

If you don't have `uv` installed, you can install it with pip. See the project
pages for more details:

- Install via pip: `pip install --upgrade pip && pip install uv`
- Project pages: [uv on PyPI](https://pypi.org/project/uv/) |
  [uv on GitHub](https://github.com/astral-sh/uv)
- Full documentation and usage guide: [uv docs](https://docs.astral.sh/uv/)

### Without uv (venv or Conda)

If you prefer not to use uv, install with plain `pip` in a standard virtual
environment:

```bash
# venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install gwmock
```

```bash
# Conda
conda create -n gwmock python=3.13
conda activate gwmock
pip install gwmock
```

**Note:** The package requires Python 3.13 or later and is built and tested
against Python 3.13–3.14. When creating a virtual environment with `uv`, specify
the Python version to ensure compatibility: `uv venv --python 3.13` (replace
`3.13` with your preferred supported version: 3.13 or 3.14). This avoids
potential issues with unsupported Python versions.

### From PyPI

```bash
# Create a virtual environment (recommended with uv)
uv venv --python 3.13
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install gwmock
```

### From Source

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwmock.git
cd gwmock
# Create a virtual environment (recommended with uv)
uv venv --python 3.13
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

## Quick Start

### Command Line

```bash
# Generate simulated data
gwmock simulate config.yaml
```

### Interactive Configuration Editor

For an easier configuration experience, use the interactive editor:

```bash
gwmock config --interactive
```

The interactive editor provides:

- Live configuration preview with real-time updates
- Autocomplete suggestions for commands and values
- Command history navigation (Up/Down arrows)
- Built-in templates for common simulation types
- Script generation for SLURM and local execution
- Immediate validation feedback

Use `/help` inside the editor to see all available commands, or `/template` to
start with a preset configuration. To modify an existing configuration:

```bash
gwmock config --interactive --load existing_config.yaml
```

## Configuration

gwmock uses YAML configuration files for reproducible simulations. New runs use
the adapter-backed `orchestration` surface, which keeps backend selection
explicit and lets third-party packages plug in through public protocols.

Key configuration sections:

| Section                    | Purpose                                                                                               |
| -------------------------- | ----------------------------------------------------------------------------------------------------- |
| `globals`                  | Shared orchestration parameters such as sampling rate, segment duration, start time, and output roots |
| `orchestration.population` | Public population backend, source type, sample count, and backend arguments                           |
| `orchestration.signal`     | Public signal backend, detector network, waveform model, and output settings                          |
| `orchestration.noise`      | Public noise backend, generation arguments, and output settings                                       |

Example:

```yaml
globals:
    working-directory: .
    output-directory: output
    metadata-directory: metadata
    simulator-arguments:
        sampling-frequency: 4096
        duration: 1024
        start-time: 1577491218
        total-duration: 5 hours

orchestration:
    population:
        backend: FilePopulationLoader
        source-type: bbh
        n-samples: 1
        arguments:
            path: https://raw.githubusercontent.com/Leuven-Gravity-Institute/gwmock/main/examples/signal/bbh_population.csv
    signal:
        detectors:
            - H1
        waveform-model: IMRPhenomXPHM
        minimum-frequency: 20
        output:
            output_directory: signal
            file_name: signal-{{ counter }}.gwf
            arguments:
                channel: H1:STRAIN
    noise:
        output:
            output_directory: noise
            file_name: noise-{{ counter }}.gwf
```

Third-party backends can be exposed through an entry point or referenced
directly as `module:Class`, as long as they satisfy the public protocol for the
relevant section. See `docs/user-guide/protocols.md`,
`docs/user-guide/orchestration.md`, and `docs/user-guide/extensibility.md` for
the protocol model and integration details.

## Documentation

Full documentation to be available at
[https://leuven-gravity-institute.github.io/gwmock](https://leuven-gravity-institute.github.io/gwmock).

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a merge request

### Release Schedule

Releases follow a fixed schedule: every Tuesday at 00:00 UTC, unless an emergent
bugfix is required. This ensures predictable updates while allowing flexibility
for critical issues. Users can view upcoming changes in the draft release on the
[GitHub Releases page](https://github.com/Leuven-Gravity-Institute/gwmock/releases).

## Testing

Run the test suite:

```bash
uv run pytest
```

## License

This project is licensed under **GPL-3.0-or-later**. See the [LICENSE](LICENSE)
file for the full license text.

> gwmock is free software: you can redistribute it and/or modify it under the
> terms of the GNU General Public License as published by the Free Software
> Foundation, either version 3 of the License, or (at your option) any later
> version.
>
> gwmock is distributed in the hope that it will be useful, but WITHOUT ANY
> WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
> A PARTICULAR PURPOSE. See the GNU General Public License for more details.
>
> You should have received a copy of the GNU General Public License along with
> gwmock. If not, see <https://www.gnu.org/licenses/>.

## Support

For questions or issues, please open an issue on
[GitHub](https://github.com/Leuven-Gravity-Institute/gwmock/issues/new) or
contact the maintainers.
