from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace, _SubParsersAction


# ---------------------------------------------------------------------------
# Script Parser Registration
# ---------------------------------------------------------------------------
def register(subparsers: _SubParsersAction[ArgumentParser]) -> ArgumentParser:
    parser = subparsers.add_parser("run", help="Invokes a workflow")

    parser.add_argument("--config", "-c", type=str, help="Path to config YAML")
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

    status = console.status("[bold cyan]Loading ARCO[/bold cyan]", spinner="dots")
    status.start()

    import os
    import sys

    from arco.tools.run import (
        get_workflow_descriptions,
        get_workflow_list,
        initialize_workflow,
        run,
    )

    from ..viz import display, printer

    status.stop()

    if args.config:
        if not os.path.isfile(args.config):
            console.print(
                f"[bold red]Error[/bold red]: config file not found at [bold cyan]{args.config}[/bold cyan]"
            )
            parser.print_help()
            sys.exit(1)
        config, workflow = initialize_workflow(yaml_path=args.config)
    else:
        available_workflows = get_workflow_list()
        if len(available_workflows) == 0:
            console.print(
                "No workflow available, please define a workflow or install an optional workflows module"
            )
            sys.exit(1)
        workflow_descriptions = get_workflow_descriptions()
        console.print("\n[bold cyan]Available workflows[/bold cyan]\n")
        for workflow_id in available_workflows:
            description = workflow_descriptions.get(
                workflow_id, "No description available."
            )
            console.print(
                f"  [bold cyan]{workflow_id}[/bold cyan]"
                f"  [dim]— {description}[/dim]"
            )
        console.print()
        user_input = console.input("[bold cyan]Workflow    >[/bold cyan] ")
        if user_input not in available_workflows:
            console.print(
                f"[bold red]Error[/bold red]: The selected workflow is not available : '[bold cyan]{user_input}[/bold cyan]'"
            )
            sys.exit(1)
        config, workflow = initialize_workflow(workflow_name=user_input)

    if config.prompt is None:
        console.print()
        user_input = console.input("[bold cyan]User prompt >[/bold cyan] ")
        console.print()
        config = config.update_prompt(user_input)
        workflow.config = config
    else:
        console.print()
        console.print(f"[bold cyan]User prompt >[/bold cyan] {config.prompt}")
        console.print()

    if args.verbose:
        printer.print_run_overview(config, workflow, verbose=args.verbose)
        console.print()

    # runs the agent with a visualization logic in rich
    display.display_workflow(
        run(config=config, workflow=workflow, log_level=args.log), verbose=args.verbose
    )
