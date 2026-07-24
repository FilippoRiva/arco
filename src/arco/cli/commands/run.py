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
    return parser


# Parameters the user can override, with (key, type, description).
_GLOBAL_PARAMS = [
    ("prompt", str, "Natural language query"),
    ("visualization_goal", str, "Chart description (empty to skip)"),
    ("run_id", str, "Run ID (empty = auto-generate)"),
]


def handle(args: Namespace, parser: ArgumentParser) -> None:
    from arco.cli.console import console
    from arco.cli.viz import display, printer
    from arco.workflows.workflow import WorkflowFactory

    status = console.status("[bold cyan]Loading run[/bold cyan]", spinner="dots")
    status.start()
    import os
    import sys

    console.print("[green]✓[/green] Built-in modules loaded")
    console.print("[green]✓[/green] Visualization tools loaded")
    from arco.workflows.workflow_executor import WorkflowExecutor

    console.print("[green]✓[/green] ARCO dependencies loaded")
    status.stop()

    ## Initialization
    # Load config from YAML
    if not os.path.isfile(args.config):
        console.print(
            f"[bold red]Error[/bold red]: config file not found at [bold cyan]{args.config}[/bold cyan]"
        )
        parser.print_help()
        sys.exit(1)

    console.print(f"Loading configuration from: [bold cyan]{args.config}[/bold cyan]")
    workflow, config = WorkflowFactory.get_from_config(args.config)

    printer.print_config_table(config, verbose=args.verbose)
    printer.print_workflow_graph(workflow)

    ## Get the agent
    executor = WorkflowExecutor(
        config=config,
        workflow=workflow,
    )

    # runs the agent with a visualization logic in rich
    display.display_workflow(executor.stream(), verbose=args.verbose)
