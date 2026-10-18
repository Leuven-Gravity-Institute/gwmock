# Installation

We recommend using `uv` to manage virtual environments for installing gwmock.

If you don't have `uv` installed, you can install it with pip. See the project
pages for more details:

- Install via pip: `pip install --upgrade pip && pip install uv`
- Project pages: [uv on PyPI](https://pypi.org/project/uv/) |
  [uv on GitHub](https://github.com/astral-sh/uv)
- Full documentation and usage guide: [uv docs](https://docs.astral.sh/uv/)

## Requirements

- Python 3.13 or later (3.13 and 3.14 are supported)
- Operating System: Linux, macOS, or Windows

<!-- prettier-ignore -->
!!!note
    The package requires Python 3.13 or later and is built and tested against Python 3.13–3.14. When creating a
    virtual environment with `uv`, specify the Python version to ensure compatibility: `uv venv --python 3.13`
    (replace `3.13` with your preferred supported version: 3.13 or 3.14). This avoids potential issues
    with unsupported Python versions.

## Install from PyPI

The recommended way to install gwmock is from PyPI:

```bash
# Create a virtual environment (recommended with uv)
uv venv --python 3.13
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install gwmock
```

## Install from Source

For the latest development version:

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwmock.git
cd gwmock
# Create a virtual environment (recommended with uv)
uv venv --python 3.13
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### Development Installation

To set up for development:

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwmock.git
cd gwmock

# Create a virtual environment (recommended with uv)
uv venv --python 3.13
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync --group dev

# Install prek hooks
uv run prek install
```

## Verify Installation

Check that gwmock is installed correctly:

```bash
gwmock --help
```

```bash
python -c "import gwmock; print(gwmock.__version__)"
```

## Getting Help

1. Check the [troubleshooting guide](../dev/troubleshooting.md)
2. Search existing
   [issues](https://github.com/Leuven-Gravity-Institute/gwmock/issues)
3. Create a new issue with:
    - Your operating system and Python version
    - Full error message
    - Steps to reproduce the problem
