from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace, _SubParsersAction


def register(subparsers: _SubParsersAction[ArgumentParser]) -> ArgumentParser:
    parser = subparsers.add_parser(
        "analyze-benchmark",
        help="Analyze benchmark outputs producing html visuals",
    )
    parser.add_argument(
        "benchmark_dir",
        nargs="?",
        help="Path to the benchmark output directory (contains bench_metadata.json)",
    )
    parser.add_argument(
        "--experiment",
        help="Experiment ID from config/catalog.yaml",
    )
    return parser


def handle(args: Namespace, parser: ArgumentParser) -> None:
    from arco.cli.console import console
    from arco.core import ExperimentCatalog
    from arco.tools.analyze_benchmark import analyze_benchmark

    if args.experiment:
        benchmark_dir = str(
            ExperimentCatalog.load().get(args.experiment).output_dir_path
        )
    elif args.benchmark_dir:
        benchmark_dir = args.benchmark_dir
    else:
        parser.error("benchmark_dir or --experiment is required")

    console.print(f"[bold]Analysing benchmark:[/bold] {benchmark_dir}")
    try:
        analyze_benchmark(benchmark_dir)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise
