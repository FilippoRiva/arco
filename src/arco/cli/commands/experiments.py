"""CLI browser for catalog-managed experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace, _SubParsersAction


def register(subparsers: _SubParsersAction[ArgumentParser]) -> ArgumentParser:
    parser = subparsers.add_parser(
        "experiments", help="List or inspect experiments from the catalog"
    )
    parser.add_argument(
        "experiment_id",
        nargs="?",
        help="Show a summary for this experiment ID instead of listing all experiments",
    )
    parser.add_argument(
        "--catalog",
        default="config/catalog.yaml",
        help="Path to the experiment catalog YAML",
    )
    return parser


def _display_path(path: Path, root_dir: Path) -> str:
    try:
        return path.resolve().relative_to(root_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def _format_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _flatten_mapping(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    """Flatten nested config mappings into compact dotted-key rows."""
    if isinstance(value, dict):
        if not value:
            return [(prefix or "Configuration", "{}")]
        rows: list[tuple[str, str]] = []
        for key, nested_value in value.items():
            child_key = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_mapping(nested_value, child_key))
        return rows
    return [(prefix or "Configuration", _format_value(value))]


def _new_table(Table: Any, title: str, columns: list[str]) -> Any:
    table = Table(
        title=title,
        title_style="bold cyan",
        box=None,
        padding=(0, 1),
        header_style="bold cyan",
        show_lines=False,
        expand=False,
    )
    for column in columns:
        table.add_column(column, no_wrap=column in {"Run", "Item"})
    return table


def _print_yaml_config(
    console: Any,
    Table: Any,
    *,
    title: str,
    path: Path,
    root_dir: Path,
    sections: tuple[str, ...] | None = None,
) -> None:
    console.print("\n")
    table = _new_table(Table, title, ["Setting", "Value"])
    table.add_row("File", _display_path(path, root_dir))
    if not path.is_file():
        table.add_row("Status", "Does not exist")
        console.print(table)
        return

    import yaml

    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        table.add_row("Status", f"Could not read: {exc}")
        console.print(table)
        return

    if sections is not None:
        config = {
            section: config[section]
            for section in sections
            if isinstance(config, dict) and section in config
        }
    rows = _flatten_mapping(config)
    if rows == [("Configuration", "{}")]:
        table.add_row("Status", "No configuration specified")
    else:
        for key, value in rows:
            table.add_row(key, value)
    console.print(table)


def _print_json_counts(
    console: Any,
    Table: Any,
    *,
    title: str,
    path: Path,
    root_dir: Path,
    prompts: bool,
) -> None:
    console.print("\n")
    table = _new_table(Table, title, ["Item", "Value"])
    table.add_row("File", _display_path(path, root_dir))
    if not path.is_file():
        table.add_row("Status", "Does not exist")
        console.print(table)
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        table.add_row("Status", f"Could not read: {exc}")
        console.print(table)
        return

    if not isinstance(data, list):
        table.add_row("Status", "Invalid format (expected a JSON list)")
    elif prompts:
        table.add_row("Prompts", str(len(data)))
    else:
        entries = [entry for entry in data if isinstance(entry, dict)]
        trace_steps = sum(
            len(entry.get("trace", []))
            for entry in entries
            if isinstance(entry.get("trace", []), list)
        )
        table.add_row("Traces / entries", str(len(entries)))
        table.add_row("Trace steps", str(trace_steps))
    console.print(table)


def _format_run_changes(changes: Any) -> str:
    if not isinstance(changes, dict) or not changes:
        return "No changes"

    agent_changes = []
    for agent, parameters in changes.items():
        if isinstance(parameters, dict):
            details = (
                ", ".join(
                    f"{key}={_format_value(value)}" for key, value in parameters.items()
                )
                or "No changes"
            )
        else:
            details = _format_value(parameters)
        agent_changes.append(f"{agent}: {details}")
    return "\n".join(agent_changes)


def _print_run_changes(console: Any, Table: Any, *, path: Path) -> None:
    console.print("\n")
    table = _new_table(
        Table, "Benchmark runs and changes", ["Run", "Description", "Changes"]
    )
    if not path.is_file():
        table.add_row("Status", "Does not exist", "")
        console.print(table)
        return

    import yaml

    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        table.add_row("Status", f"Could not read: {exc}", "")
        console.print(table)
        return

    runs = config.get("runs", []) if isinstance(config, dict) else []
    if not isinstance(runs, list) or not runs:
        table.add_row("Status", "No runs specified", "")
    else:
        for run in runs:
            if not isinstance(run, dict):
                continue
            table.add_row(
                str(run.get("name", "Unnamed run")),
                str(run.get("description") or "—"),
                _format_run_changes(run.get("changes", {})),
            )
    console.print(table)


def _print_output_status(console: Any, Table: Any, experiment: Any) -> None:
    table = _new_table(Table, "Output status", ["Output", "Status", "Path"])
    dataset_path = experiment.ground_truth_path
    table.add_row(
        "Generated dataset",
        "Produced" if dataset_path.is_file() else "Not produced",
        _display_path(dataset_path, experiment.root_dir),
    )

    output_dir = experiment.output_dir_path
    benchmark_produced = (
        (output_dir / "bench_metadata.json").is_file()
        or bool(list((output_dir / "runs").glob("*.csv")))
        if output_dir.is_dir()
        else False
    )
    table.add_row(
        "Benchmark results",
        "Produced" if benchmark_produced else "Not produced",
        _display_path(output_dir, experiment.root_dir),
    )
    console.print(table)


def _print_experiment_summary(console: Any, Table: Any, experiment: Any) -> None:
    from rich.text import Text

    console.print(f"\n[bold cyan]{experiment.id}[/bold cyan]")
    console.print(Text(f"Description: {experiment.description or '—'}"))
    console.print(Text(f"Workflow: {experiment.workflow}"))
    console.print()

    _print_yaml_config(
        console,
        Table,
        title="Generation config",
        path=experiment.generation_config_path,
        root_dir=experiment.root_dir,
    )
    _print_json_counts(
        console,
        Table,
        title="Prompts",
        path=experiment.prompts_path,
        root_dir=experiment.root_dir,
        prompts=True,
    )
    _print_json_counts(
        console,
        Table,
        title="Dataset traces",
        path=experiment.ground_truth_path,
        root_dir=experiment.root_dir,
        prompts=False,
    )
    _print_yaml_config(
        console,
        Table,
        title="Benchmark default configuration",
        path=experiment.benchmark_config_path,
        root_dir=experiment.root_dir,
        sections=("global", "defaults"),
    )
    _print_run_changes(console, Table, path=experiment.benchmark_config_path)
    _print_output_status(console, Table, experiment)


def handle(args: Namespace, parser: ArgumentParser) -> None:
    import yaml
    from rich.table import Table

    from arco.cli.console import console
    from arco.core.experiment import ExperimentCatalog

    try:
        catalog = ExperimentCatalog.load(args.catalog)
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.error(f"Could not load experiment catalog: {exc}")

    if args.experiment_id:
        try:
            experiment = catalog.get(args.experiment_id)
        except KeyError as exc:
            parser.error(str(exc.args[0]))
        _print_experiment_summary(console, Table, experiment)
        return

    table = Table(
        title="Available experiments",
        title_style="bold cyan",
        expand=False,
        box=None,
        padding=(0, 1),
    )
    table.add_column("ID", style="bold cyan", no_wrap=True)
    table.add_column("Workflow", style="green", no_wrap=True)
    table.add_column("Description")
    table.add_column("Tags", style="dim")

    for experiment_id, experiment in sorted(catalog.all().items()):
        table.add_row(
            experiment_id,
            experiment.workflow,
            experiment.description or "—",
            ", ".join(experiment.tags) or "—",
        )

    console.print(table)
