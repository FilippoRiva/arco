import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from arco.core import State
from arco.data import BenchmarkDataset


@dataclass(slots=True)
class BenchmarkResult:
    """Loaded artifacts from a completed benchmark run."""

    metadata: dict
    benchmark_dir: Path
    summary_df: pd.DataFrame
    runs: dict[str, pd.DataFrame] = field(default_factory=dict)
    states: dict[str, dict[str, State]] = field(default_factory=dict)
    dataset: BenchmarkDataset | None = None

    @classmethod
    def load(cls, benchmark_dir: str) -> BenchmarkResult:
        bdir = Path(benchmark_dir)

        with open(bdir / "bench_metadata.json") as f:
            metadata = json.load(f)

        summary_df = pd.read_csv(bdir / "summary.csv")

        runs_dir = bdir / "runs"
        runs: dict[str, pd.DataFrame] = {}
        states: dict[str, dict[str, State]] = {}
        if runs_dir.exists():
            for run_dir in runs_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                run_name = run_dir.name
                csv_path = run_dir / f"{run_name}.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    df["trace"] = df["execution_trace"].apply(json.loads)
                    runs[run_name] = df

                run_states: dict[str, State] = {}
                for state_file in sorted(run_dir.glob("*.json")):
                    if state_file.stem == run_name:
                        continue
                    with open(state_file) as f:
                        run_states[state_file.stem] = State.from_dict(json.load(f))
                if run_states:
                    states[run_name] = run_states

        dataset_path = metadata.get("dataset_path")
        dataset = None
        if dataset_path and Path(dataset_path).exists():
            dataset = BenchmarkDataset.from_json(dataset_path)

        return cls(
            metadata=metadata,
            benchmark_dir=bdir,
            summary_df=summary_df,
            runs=runs,
            states=states,
            dataset=dataset,
        )


# ── Terminal output ──────────────────────────────────────────────────────


def _rich_table(title: str, columns: list[str], rows: list[tuple]) -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(title=title)
    for col in columns:
        table.add_column(col, header_style="bold")
    for row in rows:
        table.add_row(*[str(c) for c in row])
    Console().print(table)


def print_overview(result: BenchmarkResult) -> None:
    df = result.summary_df
    rows = []
    for _, r in df.iterrows():
        metrics = json.loads(r["metrics_by_agent"])
        all_scores = [
            m["evaluation_gt"] for m in metrics.values() if "evaluation_gt" in m
        ]
        all_ppl = [m["perplexity"] for m in metrics.values() if "perplexity" in m]
        all_time = [m["total_time"] for m in metrics.values() if "total_time" in m]
        rows.append(
            (
                r["name"],
                f"{sum(all_scores) / len(all_scores):.3f}" if all_scores else "\u2014",
                f"{sum(all_ppl) / len(all_ppl):.3f}" if all_ppl else "\u2014",
                f"{sum(all_time):.2f}s" if all_time else "\u2014",
            )
        )
    _rich_table(
        "Benchmark Overview", ["Run", "Avg Score", "Avg PPL", "Total Time"], rows
    )


def print_agent_breakdown(result: BenchmarkResult) -> None:
    df = result.summary_df
    agent_metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for _, r in df.iterrows():
        metrics = json.loads(r["metrics_by_agent"])
        for agent, m in metrics.items():
            for key in ("evaluation_gt", "perplexity", "total_time", "llm_time"):
                if key in m:
                    agent_metrics[agent][key].append(m[key])

    rows = []
    for agent in sorted(agent_metrics):
        m = agent_metrics[agent]
        score = (
            f"{sum(m['evaluation_gt']) / len(m['evaluation_gt']):.3f}"
            if "evaluation_gt" in m
            else "\u2014"
        )
        ppl = (
            f"{sum(m['perplexity']) / len(m['perplexity']):.3f}"
            if "perplexity" in m
            else "\u2014"
        )
        t = (
            f"{sum(m['total_time']) / len(m['total_time']):.2f}s"
            if "total_time" in m
            else "\u2014"
        )
        lt = (
            f"{sum(m['llm_time']) / len(m['llm_time']):.2f}s"
            if "llm_time" in m
            else "\u2014"
        )
        rows.append((agent, score, ppl, t, lt))

    _rich_table(
        "Per-Agent Averages", ["Agent", "Score", "PPL", "Total Time", "LLM Time"], rows
    )


def print_trace_analysis(result: BenchmarkResult) -> None:
    if result.dataset is None:
        print("  No dataset in metadata \u2014 skipping trace analysis")
        return

    dataset = result.dataset
    mismatches = []
    for run_name, df in result.runs.items():
        for _, row in df.iterrows():
            entry_id = row["entry_id"]
            if entry_id >= len(dataset.entries):
                continue
            entry = dataset.entries[entry_id]
            actual_agents = [a["agent_type"] for a in row["trace"]["answers"]]
            expected_agents = [te for te in entry.trace]

            divergence = None
            for i, (actual, expected) in enumerate(zip(actual_agents, expected_agents)):
                if actual != expected:
                    divergence = (i, expected, actual)
                    break
            if divergence is None:
                if len(actual_agents) < len(expected_agents):
                    divergence = (
                        len(actual_agents),
                        expected_agents[len(actual_agents)],
                        "(missing)",
                    )
                elif len(actual_agents) > len(expected_agents):
                    divergence = (
                        len(expected_agents),
                        "(end)",
                        actual_agents[len(expected_agents)],
                    )

            if divergence:
                mismatches.append(
                    (run_name, entry_id, divergence[0], divergence[1], divergence[2])
                )

    if not mismatches:
        print("  All traces match ground truth.")
        return

    _rich_table(
        "Trace Divergences", ["Run", "Entry", "Step", "Expected", "Got"], mismatches
    )


# ── Plotly plots ─────────────────────────────────────────────────────────


def _output_dir(result: BenchmarkResult) -> Path:
    out = result.benchmark_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _flatten_traces(runs: dict[str, pd.DataFrame]) -> list[dict]:
    records = []
    for run_name, df in runs.items():
        for _, row in df.iterrows():
            for answer in row["trace"]["answers"]:
                records.append(
                    {
                        "run": run_name,
                        "entry": row["entry_id"],
                        "agent": answer["agent_type"],
                        "score": answer.get("evaluation_gt"),
                        "ppl": answer.get("perplexity"),
                        "total_time": answer.get("total_time"),
                        "llm_time": answer.get("llm_time"),
                        "energy": answer.get("energy_consumed_kwh"),
                        "cpu_energy": answer.get("cpu_energy_kwh"),
                        "ram_energy": answer.get("ram_energy_kwh"),
                        "emissions": answer.get("emissions_kg_co2"),
                    }
                )
    return records


_COLOR_SEQ = ["#0984e3", "#00b894", "#e17055", "#6c5ce7", "#fdcb6e", "#d63031"]


def _save_fig(fig: go.Figure, path: Path) -> None:
    fig.update_layout(height=400)
    fig.write_html(path, include_plotlyjs="cdn", config={"displayModeBar": False})
    png_path = path.with_suffix(".png")
    try:
        fig.write_image(png_path, scale=2)
    except Exception:  # noqa BLE001 - fine
        pass  # kaleido/Chrome not available — PNG skipped


def plot_per_agent_scores(
    result: BenchmarkResult, save: bool = True
) -> go.Figure | None:
    records = _flatten_traces(result.runs)
    df = pd.DataFrame(records)
    if df.empty or df["score"].isna().all():
        return None
    fig = px.box(
        df,
        x="agent",
        y="score",
        color="agent",
        color_discrete_sequence=_COLOR_SEQ,
        title="Per-Agent Score Distribution",
    )
    fig.update_layout(
        xaxis_title="Agent", yaxis_title="Ground-truth score", showlegend=False
    )
    if save:
        _save_fig(fig, _output_dir(result) / "per_agent_scores.html")
    return fig


def plot_per_agent_perplexity(
    result: BenchmarkResult, save: bool = True
) -> go.Figure | None:
    records = _flatten_traces(result.runs)
    df = pd.DataFrame(records)
    if df.empty or df["ppl"].isna().all():
        return None
    fig = px.box(
        df,
        x="agent",
        y="ppl",
        color="agent",
        color_discrete_sequence=_COLOR_SEQ,
        title="Per-Agent Perplexity Distribution",
    )
    fig.update_layout(xaxis_title="Agent", yaxis_title="Perplexity", showlegend=False)
    if save:
        _save_fig(fig, _output_dir(result) / "per_agent_perplexity.html")
    return fig


def plot_timing_breakdown(
    result: BenchmarkResult, save: bool = True
) -> go.Figure | None:
    records = _flatten_traces(result.runs)
    df = pd.DataFrame(records)
    if df.empty or df["total_time"].isna().all():
        return None
    agg = df.groupby("agent")[["total_time", "llm_time"]].mean().reset_index()
    melted = agg.melt(
        id_vars=["agent"],
        value_vars=["total_time", "llm_time"],
        var_name="metric",
        value_name="seconds",
    )
    fig = px.bar(
        melted,
        x="agent",
        y="seconds",
        color="metric",
        barmode="group",
        color_discrete_map={"total_time": "#74b9ff", "llm_time": "#0984e3"},
        title="Mean Timing Breakdown per Agent",
    )
    fig.update_layout(xaxis_title="Agent", yaxis_title="Mean time (s)")
    if save:
        _save_fig(fig, _output_dir(result) / "timing_breakdown.html")
    return fig


def plot_energy_consumption(
    result: BenchmarkResult, save: bool = True
) -> go.Figure | None:
    records = _flatten_traces(result.runs)
    df = pd.DataFrame(records)
    has_any = any(
        df.get(k).notna().any() for k in ("cpu_energy", "ram_energy", "emissions")
    )
    agg = (
        df.groupby("agent")[["cpu_energy", "ram_energy", "emissions"]]
        .mean()
        .reset_index()
    )
    melted = agg.melt(
        id_vars=["agent"],
        value_vars=["cpu_energy", "ram_energy", "emissions"],
        var_name="metric",
        value_name="value",
    )
    melted = melted.dropna(subset=["value"])
    fig = px.bar(
        melted,
        x="agent",
        y="value",
        color="metric",
        barmode="group",
        color_discrete_map={
            "cpu_energy": "#00b894",
            "ram_energy": "#0984e3",
            "emissions": "#e17055",
        },
        title="Mean Energy Consumption per Agent",
    )
    if not has_any:
        fig.add_annotation(
            text="No energy data collected<br>(enable_codecarbon: false)",
            showarrow=False,
            font={"size": 14, "color": "#636e72"},
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
    fig.update_layout(xaxis_title="Agent", yaxis_title="Mean value")
    if save:
        _save_fig(fig, _output_dir(result) / "energy_consumption.html")
    return fig


def plot_score_vs_energy(
    result: BenchmarkResult, save: bool = True
) -> go.Figure | None:
    records = _flatten_traces(result.runs)
    df = pd.DataFrame(records)
    plot_df = df.dropna(subset=["score", "energy"])
    fig = px.scatter(
        plot_df,
        x="energy",
        y="score",
        color="agent",
        hover_data=["run", "entry"],
        color_discrete_sequence=_COLOR_SEQ,
        title="Score vs Energy Consumption",
    )
    if plot_df.empty:
        fig.add_annotation(
            text="No energy data collected",
            showarrow=False,
            font={"size": 14, "color": "#636e72"},
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
    fig.update_layout(
        xaxis_title="Energy consumed (kWh)", yaxis_title="Ground-truth score"
    )
    if save:
        _save_fig(fig, _output_dir(result) / "score_vs_energy.html")
    return fig


def plot_trace_completion(
    result: BenchmarkResult, save: bool = True
) -> go.Figure | None:
    dataset = result.dataset
    if dataset is None:
        return None

    entry_ids: list[int] = []
    completions: list[float] = []
    for entry in dataset.entries:
        entry_ids.append(entry.id)
        expected = [te for te in entry.trace]
        fractions = []
        for run_df in result.runs.values():
            row = run_df[run_df["entry_id"] == entry.id]
            if row.empty:
                continue
            actual = [a["agent_type"] for a in row.iloc[0]["trace"]["answers"]]
            correct = 0
            for exp, act in zip(expected, actual):
                if exp == act:
                    correct += 1
                else:
                    break
            fractions.append(correct / len(expected))
        completions.append(sum(fractions) / len(fractions) if fractions else 0)

    colors = ["#00b894" if c == 1.0 else "#e17055" for c in completions]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=entry_ids, y=completions, marker_color=colors))
    fig.update_layout(
        title="Trace Completion per Test Case",
        xaxis_title="Entry ID",
        yaxis_title="Avg trace completion rate",
        yaxis_range=[0, 1.1],
    )
    if save:
        _save_fig(fig, _output_dir(result) / "trace_completion.html")
    return fig


def plot_run_comparison(result: BenchmarkResult, save: bool = True) -> go.Figure | None:
    df = result.summary_df
    if len(df) < 2:
        return None

    run_names: list[str] = []
    run_data: list[dict] = []
    for _, r in df.iterrows():
        metrics = json.loads(r["metrics_by_agent"])
        run_names.append(r["name"])
        run_data.append(metrics)

    agents = sorted({a for m in run_data for a in m})
    rows = []
    for name, metrics in zip(run_names, run_data):
        for agent in agents:
            rows.append(
                {
                    "run": name,
                    "agent": agent,
                    "score": metrics.get(agent, {}).get("evaluation_gt", 0),
                }
            )
    plot_df = pd.DataFrame(rows)

    fig = px.bar(
        plot_df,
        x="agent",
        y="score",
        color="run",
        barmode="group",
        color_discrete_sequence=_COLOR_SEQ,
        title="Per-Agent Score Comparison Across Runs",
    )
    fig.update_layout(xaxis_title="Agent", yaxis_title="Mean evaluation_gt")
    if save:
        _save_fig(fig, _output_dir(result) / "run_comparison.html")
    return fig


# ── Dashboard ────────────────────────────────────────────────────────────


_DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benchmark Analysis — {title}</title>
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js" charset="utf-8"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          background: #f5f6fa; color: #2d3436; padding: 24px; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 4px; }}
  .meta {{ color: #636e72; font-size: 0.9rem; margin-bottom: 20px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .full {{ grid-column: 1 / -1; }}
  .card {{ background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
           padding: 8px; overflow: hidden; }}
  .card h2 {{ font-size: 1rem; padding: 8px 12px 0; color: #636e72; }}
  .card .js-plotly-plot {{ min-height: 380px !important; }}
  @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
  <h1>Benchmark: {title}</h1>
  <div class="meta">Runtime: {runtime:.1f}s &middot; {dataset_line}</div>
  <div class="grid">
{plots}
  </div>
</body>
</html>"""


def build_dashboard(result: BenchmarkResult) -> str:
    meta = result.metadata
    title = Path(meta["benchmark_run"]).name
    dataset_line = f"Dataset: {meta.get('dataset_path', 'not available')}"

    plots: list[tuple[str, go.Figure | None, bool]] = [
        ("Scores", plot_per_agent_scores(result, save=False), False),
        ("Perplexity", plot_per_agent_perplexity(result, save=False), False),
        ("Timing Breakdown", plot_timing_breakdown(result, save=False), False),
        ("Energy Consumption", plot_energy_consumption(result, save=False), False),
        ("Score vs Energy", plot_score_vs_energy(result, save=False), False),
        (
            "Trace Completion",
            plot_trace_completion(result, save=False),
            result.dataset is not None,
        ),
    ]

    run_cmp = plot_run_comparison(result, save=False)
    if run_cmp is not None:
        plots.append(("Run Comparison", run_cmp, True))

    plot_divs = []
    for name, fig, _ in plots:
        if fig is None:
            continue
        div = fig.to_html(
            full_html=False, include_plotlyjs=False, config={"displayModeBar": False}
        )
        plot_divs.append(f'    <div class="card"><h2>{name}</h2>{div}</div>')

    html = _DASHBOARD_TEMPLATE.format(
        title=title,
        runtime=meta["total_runtime"],
        dataset_line=dataset_line,
        plots="\n".join(plot_divs),
    )
    return html


# ── Main entry point ─────────────────────────────────────────────────────


def analyze_benchmark(benchmark_dir: str) -> BenchmarkResult:
    """Load benchmark data, print tables, generate plots and dashboard.

    Returns the :class:`BenchmarkResult` for programmatic use (e.g. from a notebook).
    """
    result = BenchmarkResult.load(benchmark_dir)
    meta = result.metadata

    print(f"Benchmark: {Path(meta['benchmark_run']).name}")
    print(f"  Runtime: {meta['total_runtime']:.1f}s")
    if meta.get("dataset_path"):
        print(f"  Dataset: {meta['dataset_path']}")
    print()

    print_overview(result)
    print()
    print_agent_breakdown(result)
    print()
    print_trace_analysis(result)

    out_dir = _output_dir(result)
    print(f"\n  Saving plots to {out_dir}")

    plot_fns = [
        ("per_agent_scores", plot_per_agent_scores),
        ("per_agent_perplexity", plot_per_agent_perplexity),
        ("timing_breakdown", plot_timing_breakdown),
        ("energy_consumption", plot_energy_consumption),
        ("score_vs_energy", plot_score_vs_energy),
        ("trace_completion", plot_trace_completion),
        ("run_comparison", plot_run_comparison),
    ]
    for name, fn in plot_fns:
        fig = fn(result)
        if fig is not None:
            print(f"  \u2713 {name}.html + {name}.png")
        else:
            print(f"  - {name}.html (skipped: no data)")

    dashboard_html = build_dashboard(result)
    dashboard_path = out_dir / "dashboard.html"
    dashboard_path.write_text(dashboard_html)
    print("  \u2713 dashboard.html")

    return result


__all__ = [
    "BenchmarkResult",
    "analyze_benchmark",
    "build_dashboard",
    "plot_energy_consumption",
    "plot_per_agent_perplexity",
    "plot_per_agent_scores",
    "plot_run_comparison",
    "plot_score_vs_energy",
    "plot_timing_breakdown",
    "plot_trace_completion",
    "print_agent_breakdown",
    "print_overview",
    "print_trace_analysis",
]
