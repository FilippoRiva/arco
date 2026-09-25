"""Interactive configuration assistant for the ARCO project."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import tool

from arco.core import Workflow
from arco.core.graph import END

if TYPE_CHECKING:
    from arco.core import Config, Graph


_PROJECT_ROOT = Path.cwd().resolve()
_CONFIG_ROOT = (_PROJECT_ROOT / "config").resolve()
_MAX_READ_CHARS = 50000
_MAX_COMMAND_OUTPUT = 20000
_ALLOWED_COMMANDS = {"find", "grep", "ls", "pwd", "tree"}
_FORBIDDEN_SHELL_CHARS = set(";&|><`$()")


def _config_path(path: str) -> Path:
    """Resolve a path and ensure it stays inside the project's config folder."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = _PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(_CONFIG_ROOT)
    except ValueError as exc:
        raise ValueError(
            "Only files inside the project config folder are allowed"
        ) from exc
    return resolved


@tool
def read_config_file(path: str) -> str:
    """Read a text file inside the project's config folder.

    Use repository-relative paths such as ``config/catalog.yaml``. Files are
    read-only through this tool and large files are truncated.
    """
    try:
        resolved = _config_path(path)
        if not resolved.is_file():
            return f"Config file not found: {path}"
        content = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as exc:
        return f"Could not read config file {path!r}: {exc}"

    if len(content) > _MAX_READ_CHARS:
        return content[:_MAX_READ_CHARS] + "\n… <file truncated>"
    return content


@tool
def write_config_file(path: str, content: str) -> str:
    """Create a new text file inside the project's config folder.

    Existing files are never overwritten by this tool. Use
    ``edit_config_file`` for changes to an existing file.
    """
    try:
        resolved = _config_path(path)
        if resolved.exists():
            return f"Config file already exists: {path}. Use edit_config_file instead."
        if len(content) > _MAX_READ_CHARS * 2:
            return "Config write failed: content is too large."
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as exc:
        return f"Could not create config file {path!r}: {exc}"
    return f"Created {resolved.relative_to(_PROJECT_ROOT)}"


@tool
def edit_config_file(path: str, old_text: str, new_text: str) -> str:
    """Replace one exact unique text block in a config file.

    The target must be inside ``config/``. The replacement is refused when
    ``old_text`` is absent or occurs more than once, preventing accidental
    broad edits. Read the file first and preserve its existing format.
    """
    if not old_text:
        return "Config edit failed: old_text cannot be empty."

    try:
        resolved = _config_path(path)
        if not resolved.is_file():
            return f"Config file not found: {path}"
        content = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as exc:
        return f"Could not read config file {path!r}: {exc}"

    occurrences = content.count(old_text)
    if occurrences == 0:
        return "Config edit failed: old_text was not found. Read the file and retry."
    if occurrences > 1:
        return (
            f"Config edit failed: old_text occurs {occurrences} times. "
            "Provide a larger unique block."
        )

    try:
        resolved.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return f"Could not write config file {path!r}: {exc}"
    return f"Updated {resolved.relative_to(_PROJECT_ROOT)}"


@tool
def project_shell(command: str) -> str:
    """Run a safe, read-only project inspection command.

    Allowed commands are ``pwd``, ``ls``, ``tree``, ``find``, and ``grep``.
    Commands run from the project root, cannot use shell operators, and are
    limited to a short timeout and bounded output. This tool cannot edit files.
    """
    command = command.strip()
    if not command:
        return "Project command failed: command cannot be empty."
    if any(character in command for character in _FORBIDDEN_SHELL_CHARS):
        return "Project command rejected: shell operators are not allowed."

    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return f"Project command failed: invalid command syntax: {exc}"
    if not parts or parts[0] not in _ALLOWED_COMMANDS:
        return (
            "Project command rejected: allowed commands are "
            "pwd, ls, tree, find, and grep."
        )

    try:
        result = subprocess.run(
            parts,
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"Project command failed: {exc}"

    output = (result.stdout + result.stderr).strip()
    if len(output) > _MAX_COMMAND_OUTPUT:
        output = output[:_MAX_COMMAND_OUTPUT] + "\n… <command output truncated>"
    return output or f"Command exited with status {result.returncode}."


class ArcoConfigAssistant(Workflow):
    """Tool-using assistant for creating and updating ARCO configuration."""

    workflow_id = "arco_config_assistant"
    description = "Interactively inspect and edit ARCO configuration files."
    default_config_path = "config/arco/run/arco_config_assistant.yaml"

    def initialize(self, config: Config, graph: Graph) -> None:
        from arco.agents import ToolUseAgent

        assistant = ToolUseAgent(
            agent_name="ConfigAssistant",
            role="""You are the ARCO project configuration assistant.

Help the user create and update benchmark generation configs, benchmark configs,
workflow configs, and entries in config/catalog.yaml.

You may only modify files under the config/ folder. You may inspect the project
with the read-only project_shell tool, but never claim to have changed a file
unless write_config_file or edit_config_file confirms it.

Use write_config_file only for new files. It refuses to overwrite existing
files. Use edit_config_file for precise changes to existing files.

Before editing:
1. Inspect the shared schemas at config/run.schema.json and config/bench.schema.json
1. Inspect the relevant files with read_config_file.
2. Inspect nearby project structure with project_shell when useful.
3. Make the smallest exact edit needed with edit_config_file.
4. Re-read edited files and report what changed.

Preserve YAML formatting and existing catalog conventions. When adding a
benchmark experiment, check that its workflow, generation config, prompts,
ground-truth output, benchmark config, and output directory are consistent.
Ask for clarification in your final response when the user's request omits a
model, provider, workflow, prompt dataset, or output path. Do not edit files
outside config/ and do not execute arbitrary shell commands.""",
            tools=[
                read_config_file,
                write_config_file,
                edit_config_file,
                project_shell,
            ],
            max_tool_iterations=12,
        )
        graph.add_agent(assistant)
        graph.set_entry_agent(assistant)
        graph.add_agent_edge(assistant, END)


__all__ = [
    "ArcoConfigAssistant",
    "edit_config_file",
    "project_shell",
    "read_config_file",
    "write_config_file",
]
