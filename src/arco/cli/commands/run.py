from argparse import ArgumentParser, Namespace

from arco.cli.console import console

import os
from arco.cli import viz
from rich.rule import Rule


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
        "--interactive", "-i",
        action='store_true',
        help="Whether if an interactive run is needed",
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


def _interactive_configure(config: ArcoConfig) -> ArcoConfig:
    """Show YAML defaults and let the user override them one by one.

    Returns the (possibly modified) config
    """

    def _prompt_value(name, current_value, param_type):
        """Prompt for a single value; return current_value on empty input."""
        while True:
            raw = input(f"  {name} [{current_value}]: ").strip()
            if not raw:
                return current_value
            try:
                if param_type is bool:
                    if raw.lower() in ("true", "1", "yes", "y"):
                        return True
                    if raw.lower() in ("false", "0", "no", "n"):
                        return False
                    raise ValueError
                if param_type is str:
                    if raw.lower() in ("none", "null", ""):
                        return None
                    return raw
                return param_type(raw)
            except (ValueError, TypeError):
                print(f"    Invalid value for {name} (expected {param_type.__name__}). Try again.")

    if not sys.stdin.isatty():
        return config

    agent_display = [(key, getattr(config, key)) for key, _, _ in _GLOBAL_PARAMS]
    viz.print_config_table(agent_display)

    choice = Prompt.ask("Accept [bold cyan]Global Settings[/bold cyan]? [Y/n]: ").strip().lower()
    if choice in ("n", "no"):
        choices = {}
        for key, ptype, _ in _GLOBAL_PARAMS:
            current = getattr(config, key)
            new_val = _prompt_value(key, current, ptype)
            choices.update({key: new_val})
        return replace(config, **choices)

    return config


def handle(args: Namespace, parser: ArgumentParser) -> None:
    # Load the required dependencies
    with console.status("[bold cyan]Loading Arco...[/bold cyan]", spinner="dots"):
        from arco.workflow import SalesDataWorkflow
        from arco.core import ArcoConfig

    ## Initialization
    # Load config from YAML
    if not os.path.isfile(args.config):
        console.print(f"[bold red]Error[/bold red]: config file not found at [bold cyan]{args.config}[/bold cyan]")
        parser.print_help()
        sys.exit(1)

    console.print(f"Loading configuration from: [bold cyan]{args.config}[/bold cyan]")
    config = ArcoConfig.from_yaml(args.config)

    # Interactive YAML config change if interactive mode is selected
    if args.interactive:
        config = _interactive_configure(config)

    configs_to_show = {f.name: getattr(config, f.name) for f in config.__dataclass_fields__.values()}
    configs_to_show.pop("schema")
    configs_to_show.pop("agent_configs")

    # Visualize run configuration
    run_config_data = [
        *[(key, value) for key, value in configs_to_show.items()],
        ("verbose", args.verbose),
        ("interactive", args.interactive)
    ]
    viz.print_config_table(run_config_data)
    console.print(Rule(title="[bold green]Running the Agent[/bold green]"))

    ## Run the agent
    agent = SalesDataWorkflow(
        config=config
    )

    # runs the agent with a visualization logic in rich
    viz.agent_events_visualizer(agent.run(), verbose=args.verbose)
