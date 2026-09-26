"""Public entry point for benchmark analysis."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from .analysis.benchmark_result import BenchmarkResult
from .analysis.dashboard import build_dashboard
from .analysis.plots import (
    plot_emissions,
    plot_energy_consumption,
    plot_error_rate,
    plot_mean_energy_by_run,
    plot_per_agent_perplexity,
    plot_per_agent_scores,
    plot_prompt_score_heatmap,
    plot_score_improvement_vs_baseline,
    plot_score_latency_pareto,
    plot_score_vs_energy,
    plot_score_vs_perplexity,
    plot_timing_breakdown,
    plot_trace_exact_match,
    plot_trace_transitions,
)

console = Console()


def analyze_benchmark(benchmark_dir: str) -> BenchmarkResult:
    """Load benchmark data, generate plots and dashboard.

    Returns the :class:`BenchmarkResult` for programmatic use (e.g. from a notebook).
    """
    result = BenchmarkResult.load(benchmark_dir)
    meta = result.metadata

    console.print(
        f"[bold cyan]Benchmark:[/bold cyan] {Path(meta['benchmark_run']).name}"
    )
    console.print(f"  Runtime  {meta['total_runtime']:.1f}s")
    if meta.get("dataset_path"):
        console.print(f"  Dataset  [dim]{meta['dataset_path']}[/dim]")
    if meta.get("experiment"):
        console.print(f"  Experiment  [cyan]{meta['experiment'].get('id')}[/cyan]")
    console.print()

    out = result.benchmark_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    dashboard_html = build_dashboard(result)
    dashboard_path = out / "dashboard.html"
    dashboard_path.write_text(dashboard_html)
    console.print("  [green]✓[/green] dashboard.html")

    return result


__all__ = [
    "BenchmarkResult",
    "analyze_benchmark",
    "build_dashboard",
    "plot_emissions",
    "plot_energy_consumption",
    "plot_error_rate",
    "plot_mean_energy_by_run",
    "plot_per_agent_perplexity",
    "plot_per_agent_scores",
    "plot_prompt_score_heatmap",
    "plot_score_improvement_vs_baseline",
    "plot_score_latency_pareto",
    "plot_score_vs_energy",
    "plot_score_vs_perplexity",
    "plot_timing_breakdown",
    "plot_trace_exact_match",
    "plot_trace_transitions",
]
