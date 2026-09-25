from typing import TYPE_CHECKING

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
    from arco.cli.console import console
    from arco.core import ExperimentCatalog
    from arco.tools.generate_benchmark import generate_benchmark

    console.print()
    console.print("[bold cyan]Generate benchmark dataset[/bold cyan]")
    console.print()

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

    for event in generate_benchmark(
        config_path=config_path,
        prompts_path=prompts_path,
        save_path=output_path,
    ):
        e = event["event"]
        if e == "started":
            console.print(f"[dim]Loaded {event['total']} prompt(s)[/dim]")
        elif e == "prompt_start":
            if event["index"] > 0:
                console.print()
            console.print(
                f"  [cyan]▶[/cyan] [{event['index'] + 1}/{event['total']}] "
                f"{event['prompt'][:72]}" + "..."
                if len(event["prompt"]) > 72
                else ""
            )
        elif e == "workflow_event":
            from arco.cli.viz.display import display_workflow_event

            display_workflow_event(event["workflow_event"], verbose=args.verbose)
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
            console.print(f"  Saved to [dim]{event['path']}[/dim]")
