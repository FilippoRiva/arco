"""Interactive browser for persisted workflow states."""

from __future__ import annotations

import json
import shutil
import sys
import termios
import tty
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import RenderableType

if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace, _SubParsersAction

    from arco.core import State


@contextmanager
def _raw_terminal() -> Iterator[None]:
    """Temporarily read single key presses from an interactive terminal."""
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def register(subparsers: _SubParsersAction[ArgumentParser]) -> ArgumentParser:
    parser = subparsers.add_parser(
        "storage", help="Browse saved workflow states interactively"
    )
    parser.add_argument(
        "--storage-dir",
        default="./output/storage",
        help="Directory containing saved state JSON files",
    )
    parser.add_argument(
        "--run-id",
        help="Open one saved state directly instead of showing the selector",
    )
    return parser


def _load_states(storage_dir: Path) -> list[tuple[Path, State]]:
    from arco.core import State

    states: list[tuple[Path, State]] = []
    for path in storage_dir.glob("*.json"):
        try:
            states.append((path, State.from_dict(json.loads(path.read_text()))))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # A malformed artifact should not make the whole browser unusable.
            print(f"Skipping {path}: {exc}", file=sys.stderr)
    return sorted(states, key=lambda item: item[0].stat().st_mtime, reverse=True)


def _short_prompt(prompt: str, width: int = 58) -> str:
    prompt = " ".join(prompt.split())
    return prompt if len(prompt) <= width else prompt[: width - 1] + "…"


def _selector_view(states: list[tuple[Path, State]], selected: int):
    from rich.console import Group
    from rich.table import Table

    table = Table(box=None, padding=(0, 1), expand=False)
    table.add_column("", width=2)
    table.add_column("Saved", no_wrap=True)
    table.add_column("Run ID", no_wrap=True)
    table.add_column("Prompt")

    for index, (path, state) in enumerate(states):
        marker = "[bold cyan]❯[/bold cyan]" if index == selected else " "
        timestamp = (
            datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            .astimezone()
            .strftime("%Y-%m-%d %H:%M")
        )
        table.add_row(
            marker,
            timestamp,
            state.run_id,
            _short_prompt(state.prompt),
        )
    return Group(
        "[bold cyan]Saved workflow states[/bold cyan]",
        "[dim]↑/↓ select · Enter open · d delete · D delete all · q quit[/dim]",
        table,
    )


def _state_view(path: Path, state: State):
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    from arco.cli.viz.panels import render_answer_verbose

    metadata = Text.assemble(
        ("Saved workflow state", "bold cyan"),
        ("\nRun ID  ", "dim"),
        (state.run_id, "bold"),
        ("\nSaved   ", "dim"),
        (
            datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            .astimezone()
            .strftime("%Y-%m-%d %H:%M:%S"),
            "",
        ),
        ("\nPrompt  ", "dim"),
        (state.prompt, ""),
    )
    sections: list[RenderableType] = [metadata, Text("")]
    if not state.answers:
        sections.append(Panel("No agent answers stored.", border_style="dim"))
    else:
        for answer in state.answers:
            sections.extend([render_answer_verbose(answer)])
    sections.append(Text("←/Backspace return · q quit", style="dim"))
    return Group(*sections)


def _delete_storage_contents(storage_dir: Path) -> int:
    deleted = 0
    for child in storage_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
        deleted += 1
    return deleted


def _open_interactively(states: list[tuple[Path, State]], storage_dir: Path) -> None:
    from rich.live import Live

    from arco.cli.console import console

    selected = 0
    view = "list"
    detail: tuple[Path, State] | None = None
    live: Live | None = Live(
        _selector_view(states, selected),
        console=console,
        refresh_per_second=12,
        transient=True,
    )
    live.start()

    with _raw_terminal():
        try:
            while True:
                key = sys.stdin.read(1)

                if view == "detail":
                    if key in {"q", "Q"}:
                        return
                    if key in {"b", "B", "\x7f"} or (
                        key == "\x1b" and sys.stdin.read(2) == "[D"
                    ):
                        view = "list"
                        live = Live(
                            _selector_view(states, selected),
                            console=console,
                            refresh_per_second=12,
                            transient=True,
                        )
                        live.start()
                    continue

                if key in {"q", "Q"}:
                    return
                if key in {"d", "D"}:
                    live.stop()
                    if key == "d":
                        path, _ = states[selected]
                        console.print(
                            f"Delete [bold]{path.name}[/bold]? [y/N] ", end=""
                        )
                        confirmed = sys.stdin.read(1).lower() == "y"
                        console.print()
                        if confirmed:
                            path.unlink()
                            states.pop(selected)
                            if not states:
                                console.print("[green]✓[/green] Storage is empty")
                                return
                            selected = min(selected, len(states) - 1)
                    else:
                        console.print(
                            f"Delete all contents of [bold]{storage_dir}[/bold]? [y/N] ",
                            end="",
                        )
                        confirmed = sys.stdin.read(1).lower() == "y"
                        console.print()
                        if confirmed:
                            deleted = _delete_storage_contents(storage_dir)
                            console.print(
                                f"[green]✓[/green] Removed {deleted} storage item(s)"
                            )
                            return
                    live = Live(
                        _selector_view(states, selected),
                        console=console,
                        refresh_per_second=12,
                        transient=True,
                    )
                    live.start()
                    continue
                if key in {"\r", "\n"}:
                    detail = states[selected]
                    live.stop()
                    console.print(_state_view(*detail))
                    view = "detail"
                    continue
                if key == "\x1b":
                    sequence = sys.stdin.read(2)
                    if sequence == "[A":
                        selected = (selected - 1) % len(states)
                        live.update(_selector_view(states, selected), refresh=True)
                    elif sequence == "[B":
                        selected = (selected + 1) % len(states)
                        live.update(_selector_view(states, selected), refresh=True)
        finally:
            if live is not None:
                live.stop()


def handle(args: Namespace, parser: ArgumentParser) -> None:
    from arco.cli.console import console

    storage_dir = Path(args.storage_dir).expanduser()
    if not storage_dir.is_dir():
        parser.error(f"Storage directory not found: {storage_dir}")

    states = _load_states(storage_dir)
    if args.run_id:
        matches = [
            (path, state) for path, state in states if state.run_id == args.run_id
        ]
        if not matches:
            parser.error(f"Saved state not found: {args.run_id}")
        console.print(_state_view(*matches[0]))
        return

    if not states:
        console.print(f"[dim]No saved states in {storage_dir}[/dim]")
        return

    if not sys.stdin.isatty():
        from arco.cli.console import console

        console.print(_selector_view(states, 0))
        return

    _open_interactively(states, storage_dir)
