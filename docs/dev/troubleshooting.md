# Troubleshooting

## Version Conflicts During Installation

If you encounter version conflicts or dependency errors while installing
`gwmock`, this is often due to conflicting package versions in your Python
environment. To resolve this:

<!-- prettier-ignore-start -->

1. **Use a Virtual Environment**: Always install `gwmock` in a dedicated virtual environment to isolate it from your
    system Python and other projects.
    We recommend using `uv` for this:

    ```bash
    uv venv --python 3.13  # Replace 3.13 with your preferred supported version (3.13 or 3.14)
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    uv pip install gwmock
    ```

2. **Check Python Version**: Ensure you're using Python 3.13 or later. The package is built and
    tested against Python 3.13–3.14. Versions before 3.13 are not supported:

    ```bash
    python --version  # Should show 3.13.x or later
    ```

3. **Update Dependencies**: If conflicts persist, try updating `pip` and `uv`:

    ```bash
    pip install --upgrade pip uv
    uv pip install --upgrade gwmock
    ```

4. **Clean Install**: If issues continue, remove the virtual environment and recreate it:

    ```bash
    rm -rf .venv  # Or delete the folder
    uv venv --python 3.13
    source .venv/bin/activate
    uv pip install gwmock
    ```

<!-- prettier-ignore-end -->

For more details on installation, see the
[Installation Guide](../user-guide/installation.md). If problems persist, check
the [GitHub issues](https://github.com/Leuven-Gravity-Institute/gwmock/issues)
or create a new issue with your Python version and full error output.
