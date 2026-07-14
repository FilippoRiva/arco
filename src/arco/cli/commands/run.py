from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace


# ---------------------------------------------------------------------------
# Script Parser Registration
# ---------------------------------------------------------------------------
def register(subparsers: ArgumentParser) -> ArgumentParser:
    parser = subparsers.add_parser(
        "run",
        help="Invokes the agent given an arco configuration file"
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to config YAML"
    )
    parser.add_argument(
        "--verbose", "-v",
        action='store_true',
        help="Whether if the agent's configuration and other metrics should be shown after each execution"
    )
    return parser


# Parameters the user can override, with (key, type, description).
_GLOBAL_PARAMS = [
    ("prompt", str, "Natural language query"),
    ("visualization_goal", str, "Chart description (empty to skip)"),
    ("run_id", str, "Run ID (empty = auto-generate)"),
]


def handle(args: Namespace, parser: ArgumentParser) -> None:
    # Dependencies
    from arco.cli.console import console
    status = console.status("[bold cyan]Loading run[/bold cyan]", spinner="dots")
    status.start()
    import os, sys
    console.print("[green]✓[/green] Built-in modules loaded")
    from rich.rule import Rule
    from arco.cli import viz
    console.print("[green]✓[/green] Visualization tools loaded")
    from arco.workflow import SalesDataWorkflow
    from arco.core import ArcoConfig
    console.print("[green]✓[/green] ARCO dependencies loaded")
    status.stop()

    ## Initialization
    # Load config from YAML
    if not os.path.isfile(args.config):
        console.print(f"[bold red]Error[/bold red]: config file not found at [bold cyan]{args.config}[/bold cyan]")
        parser.print_help()
        sys.exit(1)

    console.print(f"Loading configuration from: [bold cyan]{args.config}[/bold cyan]")
    config = ArcoConfig.from_yaml(args.config)

    viz.print_config_table(config, verbose=args.verbose)
    console.print(Rule(title="[bold green]Running the Agent[/bold green]"))

    ## Run the agent
    agent = SalesDataWorkflow(
        config=config
    )

    # runs the agent with a visualization logic in rich
    # viz.agent_events_visualizer(agent.stream(), verbose=args.verbose, show_plot=True)
    viz.streaming_agent_visualizer(agent.stream(), verbose=args.verbose, show_plot=True)
