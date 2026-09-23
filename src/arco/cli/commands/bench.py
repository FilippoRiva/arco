from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace, _SubParsersAction


# ---------------------------------------------------------------------------
# Script Parser Registration
# ---------------------------------------------------------------------------
def register(subparsers: _SubParsersAction[ArgumentParser]) -> ArgumentParser:
    parser = subparsers.add_parser(
        "benchmark", help="Benchmarks a workflow on a benchmark dataset"
    )
    parser.add_argument(
        "--dataset", "-d", help="Path to benchmark dataset JSON"
    )
    parser.add_argument(
        "--config", "-c", help="Path to benchmark_config.yaml"
    )
    parser.add_argument("--experiment", help="Experiment ID from config/catalog.yaml")
    parser.add_argument(
        "--save-dir", default=None, help="Output directory"
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

    from arco.cli.viz import display, printer
    from arco.core import ExperimentCatalog
    from arco.data.benchmark_dataset import BenchmarkSummary
    from arco.tools.bench import benchmark_from_config

    status.stop()

    if args.experiment:
        experiment = ExperimentCatalog.load().get(args.experiment)
        config_path = str(experiment.benchmark_config_path)
        dataset_path = str(experiment.ground_truth_path)
        benchmark_id = args.id or experiment.output_dir_path.name
        save_dir = args.save_dir or str(experiment.output_dir_path.parent)
        experiment_metadata = experiment.metadata()
    else:
        if not args.config or not args.dataset:
            parser.error(
                "--config and --dataset are required unless --experiment is used"
            )
        config_path = args.config
        dataset_path = args.dataset
        benchmark_id = args.id
        save_dir = args.save_dir or "./output/benchmarks"
        experiment_metadata = None

    console.print()
    console.print("[bold cyan]Benchmark[/bold cyan]")
    if args.experiment:
        console.print(f"  Experiment  [cyan]{args.experiment}[/cyan]")
    console.print(f"  Config      [dim]{config_path}[/dim]")
    console.print(f"  Dataset     [dim]{dataset_path}[/dim]")
    console.print(f"  Output      [dim]{save_dir}[/dim]")
    console.print()

    if args.verbose:
        visualization_logic = partial(display.display_workflow, verbose=True)
    else:
        visualization_logic = display.display_workflow_compact

    generator = benchmark_from_config(
        config_path=config_path,
        dataset_path=dataset_path,
        id=benchmark_id,
        save_dir=save_dir,
        run_visualization_logic=visualization_logic,
        logging_level=args.log,
        experiment_metadata=experiment_metadata,
    )

    for _event in generator:
        event: str = _event["event"]
        if event == "run_configs_loaded":
            console.print("[green]✓[/green] Run configurations loaded")
        elif event == "benchmark_already_exists":
            console.print(
                f"[yellow]![/yellow] Skipping existing run: [dim]{_event['path']}[/dim]"
            )
        elif event == "benchmark_start":
            if isinstance(_event["changes"], dict):
                changes: dict[str, Any] = _event["changes"]
                printer.print_benchmark_header(
                    name=_event["name"],
                    description=_event["description"],
                    changes=changes,
                )
            else:
                raise ValueError("The passed changes are not in a dictionary format")
        elif event == "benchmark_run_save":
            console.print(f"[green]✓[/green] Run saved  [dim]{_event['path']}[/dim]")
        elif event == "test_case_start":
            console.print()
            console.print(
                f"[bold blue]Test case {_event['iteration']}/{_event['max_iteration']}[/bold blue]"
            )
        elif event == "test_case_stop":
            if isinstance(_event["evaluation_summary"], BenchmarkSummary):
                printer.print_benchmark_summary(_event["evaluation_summary"])
            else:
                raise ValueError(
                    f"The passed BenchmarkSummary is instead a {type(_event['evaluation_summary'])}"
                )
            console.print()
        elif event == "test_case_evaluation_start":
            status = console.status("Evaluating result")
            status.start()
        elif event == "error":
            console.print(f"[red]✗[/red] {_event['message']}")
        elif event == "test_case_evaluation_stop":
            status.stop()
