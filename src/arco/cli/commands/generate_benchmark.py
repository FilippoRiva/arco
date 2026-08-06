from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace


def register(subparsers: ArgumentParser) -> ArgumentParser:
    parser = subparsers.add_parser(
        "generate-benchmark",
        help="Produce a benchmark dataset given a list of prompts",
    )
    parser.add_argument("--config", "-c", required=True, help="Path to run_config YAML")
    parser.add_argument(
        "--prompts", "-p", required=True, help="Path to prompts JSON (list of strings)"
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path where the benchmark JSON will be saved",
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
    from arco.tools.generate_benchmark import generate_benchmark

    console.print("[bold]Generating benchmark dataset[/bold]")

    visualization_logic = _collect_state
    if args.verbose:
        from functools import partial

        from arco.cli.viz.display import display_workflow

        visualization_logic = partial(display_workflow, verbose=True)

    console.print(f"Configs from {args.config}")
    console.print(f"Prompts from {args.prompts}")
    console.print(f"Saved to {args.output}")

    for event in generate_benchmark(
        config_path=args.config,
        prompts_path=args.prompts,
        save_path=args.output,
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
