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

    console.print("[bold]Generating benchmark dataset[/bold]")

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

    visualization_logic = _collect_state
    if args.verbose:
        from functools import partial

        from arco.cli.viz.display import display_workflow

        visualization_logic = partial(display_workflow, verbose=True)

    if args.experiment:
        console.print(f"Experiment: {args.experiment}")
    console.print(f"Configs from {config_path}")
    console.print(f"Prompts from {prompts_path}")
    console.print(f"Saved to {output_path}")

    for event in generate_benchmark(
        config_path=config_path,
        prompts_path=prompts_path,
        save_path=output_path,
        run_visualization_logic=visualization_logic,
    ):
        e = event["event"]
        if e == "started":
            console.print(f"  {event['total']} prompt(s) loaded")
        elif e == "prompt_start":
            console.print(
                f"  [{event['index'] + 1}/{event['total']}] [id: {event['id']}] {event['prompt']}..."
            )
        elif e == "prompt_done":
            console.print(f"    -> trace: {event['trace_len']} step(s)")
        elif e == "prompt_error":
            console.print(f"    [red]error: {event['message']}[/red]")
        elif e == "completed":
            console.print(
                f"\n[green]Done[/green] — {event['entries']} entries saved to {event['path']}"
            )


def _collect_state(events):
    state = None
    for event in events:
        if event.get("event") == "completed":
            state = event.get("state")
    return state
