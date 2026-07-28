from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace


def register(subparsers: ArgumentParser) -> ArgumentParser:
    parser = subparsers.add_parser(
        "analyze-benchmark",
        help="Analyse a completed benchmark run: export tables and plots",
    )
    parser.add_argument(
        "benchmark_dir",
        help="Path to the benchmark output directory (contains bench_metadata.json)",
    )
    return parser


def handle(args: Namespace, parser: ArgumentParser) -> None:
    from arco.cli.console import console
    from arco.tools.analyze_benchmark import analyze_benchmark

    console.print(f"[bold]Analysing benchmark:[/bold] {args.benchmark_dir}")
    try:
        analyze_benchmark(args.benchmark_dir)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise
