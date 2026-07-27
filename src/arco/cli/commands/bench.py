from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace


# ---------------------------------------------------------------------------
# Script Parser Registration
# ---------------------------------------------------------------------------
def register(subparsers: ArgumentParser) -> ArgumentParser:
    parser = subparsers.add_parser(
        "benchmark", help="Runs a given set of configurations against a GT dataset"
    )
    parser.add_argument(
        "--dataset", "-d", required=True, help="Path to benchmark dataset JSON"
    )
    parser.add_argument(
        "--config", "-c", required=True, help="Path to benchmark_config.yaml"
    )
    parser.add_argument(
        "--save-dir", default="./output/benchmarks", help="Output directory"
    )
    parser.add_argument("--id", type=str, default=None, help="ID of this benchmark")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Whether if all the agent output should be shown",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level for arco internals (default: INFO). Libraries always log at WARNING+.",
    )
    return parser


# ---------------------------------------------------------------------------
# Script Handler
# ---------------------------------------------------------------------------
def handle(args: Namespace, parser: ArgumentParser) -> None:
    from arco.cli.console import console

    status = console.status("[bold cyan]Loading benchmark[/bold cyan]")
    status.start()
    from functools import partial

    from rich.rule import Rule

    from arco.cli.viz import display, printer
    from arco.tools import benchmark_from_config

    console.print("[green]✓[/green] Benchmark loaded")
    status.stop()

    if args.verbose:
        visualization_logic = partial(display.display_workflow, verbose=True)
    else:
        visualization_logic = display.display_workflow_compact

    generator = benchmark_from_config(
        config_path=args.config,
        dataset_path=args.dataset,
        id=args.id,
        save_dir=args.save_dir,
        run_visualization_logic=visualization_logic,
        logging_level=args.log,
    )

    for _event in generator:
        event = _event["event"]
        if event == "run_configs_loaded":
            console.print("[green]✓[/green] Run configurations loaded")
        elif event == "benchmark_already_exists":
            console.print(
                f"[yellow]![/yellow] Benchmark already exists, skipping execution. Path: '{_event['path']}'"
            )
        elif event == "benchmark_start":
            printer.print_benchmark_header(
                name=_event["name"],
                description=_event["description"],
                changes=_event["changes"],
            )
        elif event == "benchmark_run_save":
            console.print(f"[green]✓[/green] Benchmark saved. Path: '{_event['path']}'")
        elif event == "test_case_start":
            console.print(
                Rule(
                    f"[bold blue]Test Case {_event['iteration']}/{_event['max_iteration']}[/bold blue]"
                )
            )
        elif event == "test_case_stop":
            printer.print_benchmark_summary(_event["evaluation_summary"])
        elif event == "test_case_evaluation_start":
            status = console.status("Evaluating result")
            status.start()
        elif event == "test_case_evaluation_stop":
            status.stop()
