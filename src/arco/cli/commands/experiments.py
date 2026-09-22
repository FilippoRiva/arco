"""CLI command for browsing catalog-managed experiments."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace, _SubParsersAction


def register(subparsers: _SubParsersAction[ArgumentParser]) -> ArgumentParser:
    parser = subparsers.add_parser(
        "experiments", help="List experiments from the experiment catalog"
    )
    parser.add_argument(
        "--catalog",
        default="config/catalog.yaml",
        help="Path to the experiment catalog YAML",
    )
    return parser


def handle(args: Namespace, parser: ArgumentParser) -> None:
    import yaml
    from rich.table import Table

    from arco.cli.console import console
    from arco.core import ExperimentCatalog

    try:
        catalog = ExperimentCatalog.load(args.catalog)
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.error(f"Could not load experiment catalog: {exc}")

    table = Table(title="Available experiments", expand=False)
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
