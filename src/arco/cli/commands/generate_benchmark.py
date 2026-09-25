import time
from typing import TYPE_CHECKING

from arco.core import workflow

if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace, _SubParsersAction


def register(subparsers: _SubParsersAction[ArgumentParser]) -> ArgumentParser:
    parser = subparsers.add_parser(
        "generate-benchmark",
        help="Produce a benchmark dataset given a list of prompts",
    )
    parser.add_argument("--config", "-c", help="Path to run_config YAML")
    parser.add_argument(
        "--prompts", "-p", help="Path to prompts JSON (list of strings)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path where the benchmark JSON will be saved",
    )
    parser.add_argument(
        "--experiment",
        help="Experiment ID from config/catalog.yaml",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Show detailed agent output",
    )
    return parser


def handle(args: Namespace, parser: ArgumentParser) -> None:
    from rich.live import Live

    from arco.cli.console import console
    from arco.cli.viz.status import RunStatusPanel
    from arco.core import ExperimentCatalog
    from arco.tools.generate_benchmark import generate_benchmark

    console.print("\n[bold cyan]Generate benchmark dataset[/bold cyan]\n")

    if args.experiment:
        experiment = ExperimentCatalog.load().get(args.experiment)
        config_path = str(experiment.generation_config_path)
        prompts_path = str(experiment.prompts_path)
        output_path = str(experiment.ground_truth_path)
    else:
        if not args.config or not args.prompts or not args.output:
            parser.error(
                "--config, --prompts, and --output are required unless --experiment is used"
            )
        config_path = args.config
        prompts_path = args.prompts
        output_path = args.output

    if args.experiment:
        console.print(f"  Experiment  [cyan]{args.experiment}[/cyan]")
    console.print(f"  Config      [dim]{config_path}[/dim]")
    console.print(f"  Prompts     [dim]{prompts_path}[/dim]")
    console.print(f"  Output      [dim]{output_path}[/dim]")
    console.print()

    workflow_status = None
    workflow_live = None
    for event in generate_benchmark(
        config_path=config_path,
        prompts_path=prompts_path,
        save_path=output_path,
    ):
        e = event["event"]
        if e == "started":
            console.print(f"[dim]Loaded {event['total']} prompt(s)[/dim]\n")
        elif e == "prompt_start":
            if event["index"] > 0:
                console.print()
            prompt = event["prompt"][:72]
            suffix = "..." if len(event["prompt"]) > 72 else ""
            console.print(
                f"  [cyan]▶[/cyan] [{event['index'] + 1}/{event['total']}] "
                f"{prompt}{suffix}"
            )
            console.print()
        elif e == "workflow_event":
            update = event["workflow_event"]
            workflow_event = update["event"]
            if workflow_event == "started":
                if workflow_live is not None and workflow_status is not None:
                    workflow_status.stop()
                    workflow_live.__exit__(None, None, None)
                workflow_status = RunStatusPanel(compact=True)
                workflow_live = Live(
                    workflow_status,
                    refresh_per_second=8,
                    screen=False,
                    transient=True,
                )
                workflow_live.__enter__()
                workflow_status.set("    Starting run")
            elif workflow_event in {
                "check_connection",
                "node_started",
                "node_finished",
            }:
                if workflow_live is None:
                    workflow_status = RunStatusPanel(compact=True)
                    workflow_live = Live(
                        workflow_status,
                        refresh_per_second=8,
                        screen=False,
                        transient=True,
                    )
                    workflow_live.__enter__()
                if workflow_event == "check_connection" and workflow_status is not None:
                    workflow_status.set("    Checking models")
                elif workflow_event == "node_started" and workflow_status is not None:
                    node = update.get("node", "agent")
                    workflow_status.set("    " + str(node), time.time())
                    workflow_status.active_node = str(node)
                else:
                    state = update.get("state")
                    answer = state.get_last_answer() if state is not None else None
                    if answer is not None and workflow_status is not None:
                        workflow_status.set(f"    {answer.agent_id} completed")
            elif workflow_event == "error":
                console.print(f"[red]✗[/red] {update.get('message', 'Workflow error')}")
                if workflow_live is not None and workflow_status is not None:
                    workflow_status.stop()
                    workflow_live.__exit__(None, None, None)
                    workflow_status = workflow_live = None
            elif (
                workflow_event == "completed"
                and workflow_live is not None
                and workflow_status is not None
            ):
                workflow_status.stop()
                workflow_live.__exit__(None, None, None)
                workflow_status = workflow_live = None
        elif e == "prompt_done":
            console.print(
                f"    [green]✓[/green] trace completed · {event['trace_len']} step(s)"
            )
        elif e == "prompt_error":
            console.print(f"    [red]✗[/red] {event['message']}")
        elif e == "completed":
            console.print()
            console.print(
                f"[bold green]✓ Complete[/bold green]  {event['entries']} entries"
            )
            console.print(f"\n  Saved to [dim]{event['path']}[/dim]")

    if workflow_live is not None and workflow_status is not None:
        workflow_status.stop()
        workflow_live.__exit__(None, None, None)
