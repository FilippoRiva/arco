from pathlib import Path
from typing import TYPE_CHECKING

from rich.text import Text

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

    console.print()
    console.print("[bold cyan]Analyze benchmark[/bold cyan]")
    console.print(f"  Source  [dim]{benchmark_dir}[/dim]")
    console.print()
    try:
        analyze_benchmark(benchmark_dir)
        dashboard_path = (
            Path(benchmark_dir).expanduser().resolve()
            / "analysis"
            / "dashboard.html"
        )
        console.print()
        console.print("[bold green]✓ Analysis complete[/bold green]")
        console.print(
            Text(
                f"  Open dashboard  {dashboard_path}",
                style=f"bold cyan underline link {dashboard_path.as_uri()}",
            )
        )
    except Exception as e:
        console.print(f"[bold red]✗ Analysis failed[/bold red]  {e}")
        raise
