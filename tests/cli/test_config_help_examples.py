"""The examples in ``gwmock config --help`` must run.

The help once quoted ``gwmock config --get basic.yaml``, a label that never existed, and named a
``--force`` flag "if added later" for what is ``--overwrite``. A reader copies these lines, so every
label the help passes to ``--get`` must be one ``--list`` prints, and every flag it names must exist.
"""

from __future__ import annotations

import re

from typer.testing import CliRunner

from gwmock.cli.main import app

_RUNNER = CliRunner()
_WIDE = {"COLUMNS": "400", "TERMINAL_WIDTH": "400"}
# Rich colours its output when it detects a CI service such as GitHub Actions, and the escape
# sequences then split the quoted command lines the regexes look for.
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _output(*arguments: str) -> str:
    result = _RUNNER.invoke(app, ["config", *arguments], env=_WIDE)
    assert result.exit_code == 0, result.output
    return _ANSI_ESCAPE.sub("", result.output)


def test_every_label_the_help_quotes_is_listed():
    quoted = set(re.findall(r"gwmock config --get (\S+)", _output("--help")))
    assert quoted, "the help no longer shows how to copy an example, so nothing was checked"

    listed = set(re.findall(r"-\s+(\S+/\S+)", _output("--list")))
    assert quoted <= listed, f"quoted in --help but not offered by --list: {sorted(quoted - listed)}"


def test_every_flag_the_help_names_exists():
    help_text = _output("--help")
    named = set(re.findall(r"(?<![\w-])(--[a-z][a-z-]*)", help_text))
    options = {"--output", "--get", "--init", "--list", "--overwrite", "--interactive", "--load", "--help"}
    assert named <= options, f"the help names flags the command does not have: {sorted(named - options)}"
