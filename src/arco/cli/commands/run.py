from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace


# ---------------------------------------------------------------------------
# Script Parser Registration
# ---------------------------------------------------------------------------
def register(subparsers: ArgumentParser) -> ArgumentParser:
    parser = subparsers.add_parser(
        "run", help="Invokes the agent given an arco configuration file"
    )

    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to config YAML"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Whether if the agent's configuration and other metrics should be shown after each execution",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level for arco internals (default: INFO). Libraries always log at WARNING+.",
    )
    return parser


# Parameters the user can override, with (key, type, description).
_GLOBAL_PARAMS = [
    ("prompt", str, "Natural language query"),
    ("visualization_goal", str, "Chart description (empty to skip)"),
    ("run_id", str, "Run ID (empty = auto-generate)"),
]


def handle(args: Namespace, parser: ArgumentParser) -> None:
    from arco.cli.console import console

    status = console.status("[bold cyan]Loading run[/bold cyan]", spinner="dots")
    status.start()

    import os
    import sys

    from arco.tools import run_from_config

    from ..viz import display, printer

    if not os.path.isfile(args.config):
        console.print(
            f"[bold red]Error[/bold red]: config file not found at [bold cyan]{args.config}[/bold cyan]"
        )
        parser.print_help()
        sys.exit(1)

    generator = run_from_config(yaml_path=args.config)
    config = next(generator)
    workflow = next(generator)

    printer.print_config_table(config, verbose=args.verbose)
    printer.print_workflow_graph(workflow)

    status.stop()

    # runs the agent with a visualization logic in rich
    display.display_workflow(generator, verbose=args.verbose)
