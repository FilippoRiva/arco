import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from rich.console import Console
from rich.table import Table

from arco.core import State
from arco.data import BenchmarkDataset

logger = logging.getLogger(__name__)


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
            nested_run_dirs = [path for path in runs_dir.iterdir() if path.is_dir()]
            if nested_run_dirs:
                raise ValueError(
                    "Per-run subdirectories are not supported; expected flat CSV "
                    f"files directly under {runs_dir}"
                )
            run_csv_paths = sorted(runs_dir.glob("*.csv"))
            if not run_csv_paths:
                raise ValueError(f"No per-run CSV files found under {runs_dir}")
            for csv_path in run_csv_paths:
                run_name = csv_path.stem
                df = pd.read_csv(csv_path)
                required_columns = {
                    "entry_id",
                    "run_id",
                    "run_fingerprint",
                    "state",
                    "execution_trace",
                }
                missing_columns = required_columns.difference(df.columns)
                if missing_columns:
                    raise ValueError(
                        f"Unsupported benchmark CSV {csv_path}; missing columns: "
                        f"{', '.join(sorted(missing_columns))}"
                    )

                df["trace"] = df["execution_trace"].apply(json.loads)
                runs[run_name] = df
                run_states: dict[str, State] = {}
                for _, row in df.iterrows():
                    state_data = json.loads(row["state"])
                    run_states[str(row["run_id"])] = State.from_dict(state_data)
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


console = Console()


def _rich_table(title: str, columns: list[str], rows: list[tuple]) -> None:
    """Print a compact, borderless table matching the CLI run output."""
    table = Table(
        title=f"[bold cyan]{title}[/bold cyan]",
        title_justify="left",
        box=None,
        padding=(0, 1),
        header_style="bold cyan",
        show_lines=False,
        expand=False,
    )
    for col in columns:
        table.add_column(col, no_wrap=col in {"Run", "Agent", "Entry"})
    for row in rows:
        table.add_row(*[str(c) for c in row])
    console.print(table)


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
        console.print("[dim]No dataset in metadata — skipping trace analysis[/dim]")
        return

    dataset = result.dataset
    entries_by_id = {entry.id: entry for entry in dataset.entries}
    mismatches = []
    for run_name, df in result.runs.items():
        for _, row in df.iterrows():
            # Benchmark entries are identified by entry_id, not necessarily by
            # their position in the dataset list.
            raw_entry_id = row["entry_id"]
            try:
                entry_id = int(raw_entry_id)
            except (TypeError, ValueError):
                continue

            entry = entries_by_id.get(entry_id)
            if entry is None:
                continue

            actual_agents = [
                str(answer["agent_type"])
                for answer in row["trace"]["answers"]
            ]
            # Trace iteration yields TraceElement objects. Compare their
            # agent_type values rather than comparing strings to objects.
            expected_agents = [str(trace_element.agent_type) for trace_element in entry.trace]

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
        console.print("[green]✓[/green] All traces match ground truth.")
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
                        "gpu_energy": answer.get("gpu_energy_kwh"),
                        "ram_energy": answer.get("ram_energy_kwh"),
                        "emissions": answer.get("emissions_kg_co2"),
                        "error": answer.get("error"),
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
        points="all",
        hover_data=["run", "entry"],
        title="Per-Agent Score Distribution",
    )
    fig.update_layout(
        xaxis_title="Agent",
        yaxis_title="Ground-truth score",
        yaxis_range=[0, 1.05],
        showlegend=False,
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

    grouped = df.groupby("agent")
    rows: list[dict] = []
    for agent, values in grouped:
        for metric in ("total_time", "llm_time"):
            series = values[metric].dropna()
            if series.empty:
                continue
            rows.append(
                {
                    "agent": agent,
                    "metric": metric,
                    "median": series.median(),
                    "p90": series.quantile(0.9),
                    "p90_error": max(series.quantile(0.9) - series.median(), 0),
                }
            )
    if not rows:
        return None

    plot_df = pd.DataFrame(rows)
    fig = px.bar(
        plot_df,
        x="agent",
        y="median",
        color="metric",
        barmode="group",
        error_y="p90_error",
        color_discrete_map={"total_time": "#74b9ff", "llm_time": "#0984e3"},
        title="Median Timing Breakdown per Agent (p90 error bars)",
        hover_data={"median": ":.3f", "p90": ":.3f"},
    )
    fig.update_layout(xaxis_title="Agent", yaxis_title="Seconds")
    if save:
        _save_fig(fig, _output_dir(result) / "timing_breakdown.html")
    return fig


def plot_energy_consumption(
    result: BenchmarkResult, save: bool = True
) -> go.Figure | None:
    records = _flatten_traces(result.runs)
    df = pd.DataFrame(records)
    energy_columns = ["cpu_energy", "gpu_energy", "ram_energy"]
    if df.empty or not any(
        column in df and df[column].notna().any() for column in energy_columns
    ):
        return None

    agg = df.groupby("agent")[energy_columns].mean().reset_index()
    melted = agg.melt(
        id_vars=["agent"],
        value_vars=energy_columns,
        var_name="metric",
        value_name="kwh",
    ).dropna(subset=["kwh"])
    melted["wh"] = melted["kwh"] * 1000
    fig = px.bar(
        melted,
        x="agent",
        y="wh",
        color="metric",
        barmode="group",
        color_discrete_map={
            "cpu_energy": "#00b894",
            "gpu_energy": "#e17055",
            "ram_energy": "#0984e3",
        },
        title="Mean Energy Consumption per Agent",
        hover_data={"wh": ":.4f", "kwh": ":.6f"},
    )
    fig.update_layout(xaxis_title="Agent", yaxis_title="Mean energy (Wh)")
    if save:
        _save_fig(fig, _output_dir(result) / "energy_consumption.html")
    return fig


def plot_emissions(result: BenchmarkResult, save: bool = True) -> go.Figure | None:
    records = _flatten_traces(result.runs)
    df = pd.DataFrame(records)
    if df.empty or df["emissions"].isna().all():
        return None
    agg = df.groupby("agent", as_index=False)["emissions"].mean()
    agg["grams_co2"] = agg["emissions"] * 1000
    fig = px.bar(
        agg,
        x="agent",
        y="grams_co2",
        color="agent",
        color_discrete_sequence=_COLOR_SEQ,
        title="Mean CO₂ Emissions per Agent",
        hover_data={"grams_co2": ":.4f", "emissions": ":.6f"},
    )
    fig.update_layout(xaxis_title="Agent", yaxis_title="Mean emissions (g CO₂)", showlegend=False)
    if save:
        _save_fig(fig, _output_dir(result) / "emissions.html")
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
        hover_data=["run", "entry", "agent", "total_time", "emissions"],
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
    fig.update_yaxes(range=[0, 1.05])
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
        expected = [str(element.agent_type) for element in entry.trace]
        fractions = []
        for run_df in result.runs.values():
            row = run_df[run_df["entry_id"] == entry.id]
            if row.empty:
                continue
            actual = [
                str(answer["agent_type"])
                for answer in row.iloc[0]["trace"]["answers"]
            ]
            correct = 0
            for exp, act in zip(expected, actual):
                if exp == act:
                    correct += 1
                else:
                    break
            fractions.append(correct / len(expected) if expected else 1.0)
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
            score = metrics.get(agent, {}).get("evaluation_gt")
            if score is None:
                continue
            rows.append(
                {
                    "run": name,
                    "agent": agent,
                    "score": score,
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


def plot_score_vs_latency(
    result: BenchmarkResult, save: bool = True
) -> go.Figure | None:
    records = _flatten_traces(result.runs)
    df = pd.DataFrame(records).dropna(subset=["score", "total_time"])
    if df.empty:
        return None
    fig = px.scatter(
        df,
        x="total_time",
        y="score",
        color="agent",
        symbol="run",
        hover_data=["entry", "run", "ppl", "energy"],
        color_discrete_sequence=_COLOR_SEQ,
        title="Score vs Latency",
    )
    fig.update_layout(xaxis_title="Agent time (s)", yaxis_title="Ground-truth score")
    fig.update_yaxes(range=[0, 1.05])
    if save:
        _save_fig(fig, _output_dir(result) / "score_vs_latency.html")
    return fig


def plot_score_vs_perplexity(
    result: BenchmarkResult, save: bool = True
) -> go.Figure | None:
    records = _flatten_traces(result.runs)
    df = pd.DataFrame(records).dropna(subset=["score", "ppl"])
    if df.empty:
        return None
    fig = px.scatter(
        df,
        x="ppl",
        y="score",
        color="agent",
        symbol="run",
        hover_data=["entry", "run", "total_time"],
        color_discrete_sequence=_COLOR_SEQ,
        title="Score vs Perplexity",
    )
    fig.update_layout(xaxis_title="Perplexity", yaxis_title="Ground-truth score")
    fig.update_yaxes(range=[0, 1.05])
    if save:
        _save_fig(fig, _output_dir(result) / "score_vs_perplexity.html")
    return fig


def plot_prompt_score_heatmap(
    result: BenchmarkResult, save: bool = True
) -> go.Figure | None:
    records = _flatten_traces(result.runs)
    df = pd.DataFrame(records).dropna(subset=["score"])
    if df.empty:
        return None
    scores = (
        df.groupby(["run", "entry"], as_index=False)["score"]
        .mean()
        .pivot(index="run", columns="entry", values="score")
    )
    observed_min = float(scores.min().min())
    # Scale the palette to the observed worst score through the perfect score
    # so differences in a high-performing benchmark remain visible.
    color_min = observed_min if observed_min < 1 else 0.99
    fig = px.imshow(
        scores,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        zmin=color_min,
        zmax=1,
        labels={"x": "Prompt entry", "y": "Run", "color": "Score"},
        title="Per-Prompt Score Heatmap",
    )
    if save:
        _save_fig(fig, _output_dir(result) / "prompt_score_heatmap.html")
    return fig


def plot_score_latency_pareto(
    result: BenchmarkResult, save: bool = True
) -> go.Figure | None:
    records = _flatten_traces(result.runs)
    df = pd.DataFrame(records).dropna(subset=["score", "total_time"])
    if df.empty:
        return None
    grouped = (
        df.groupby(["run", "agent"], as_index=False)[["score", "total_time", "energy"]]
        .mean()
    )
    grouped["pareto"] = False
    for agent, agent_rows in grouped.groupby("agent"):
        pareto_indices = []
        for index, row in agent_rows.iterrows():
            dominated = (
                (agent_rows["score"] >= row["score"])
                & (agent_rows["total_time"] <= row["total_time"])
                & (
                    (agent_rows["score"] > row["score"])
                    | (agent_rows["total_time"] < row["total_time"])
                )
            ).any()
            if not dominated:
                pareto_indices.append(index)
        grouped.loc[pareto_indices, "pareto"] = True

    fig = px.scatter(
        grouped,
        x="total_time",
        y="score",
        color="agent",
        symbol="pareto",
        hover_data=["run", "energy"],
        color_discrete_sequence=_COLOR_SEQ,
        title="Quality / Latency Pareto Frontier per Agent",
    )
    for color_index, (agent, agent_rows) in enumerate(grouped.groupby("agent")):
        frontier = agent_rows[agent_rows["pareto"]].sort_values("total_time")
        if len(frontier) < 2:
            continue
        fig.add_trace(
            go.Scatter(
                x=frontier["total_time"],
                y=frontier["score"],
                mode="lines",
                line={"color": _COLOR_SEQ[color_index % len(_COLOR_SEQ)], "width": 2},
                name=f"{agent} frontier",
                hoverinfo="skip",
                showlegend=True,
            )
        )
    fig.update_layout(xaxis_title="Mean agent time (s)", yaxis_title="Mean score")
    fig.update_yaxes(range=[0, 1.05])
    if save:
        _save_fig(fig, _output_dir(result) / "score_latency_pareto.html")
    return fig


def plot_trace_exact_match(
    result: BenchmarkResult, save: bool = True
) -> go.Figure | None:
    dataset = result.dataset
    if dataset is None:
        return None
    expected_by_id = {
        entry.id: [str(element.agent_type) for element in entry.trace]
        for entry in dataset.entries
    }
    rows = []
    for run_name, run_df in result.runs.items():
        for _, row in run_df.iterrows():
            expected = expected_by_id.get(int(row["entry_id"]))
            if expected is None:
                continue
            actual = [str(answer["agent_type"]) for answer in row["trace"]["answers"]]
            rows.append({"run": run_name, "exact": actual == expected})
    if not rows:
        return None
    summary = pd.DataFrame(rows).groupby("run", as_index=False)["exact"].mean()
    summary["percent"] = summary["exact"] * 100
    observed_min = float(summary["percent"].min())
    color_min = observed_min if observed_min < 100 else 99
    fig = px.bar(
        summary,
        x="run",
        y="percent",
        color="percent",
        range_color=[color_min, 100],
        color_continuous_scale="RdYlGn",
        title="Exact Workflow Trace Match Rate",
        hover_data={"percent": ":.1f"},
    )
    fig.update_layout(xaxis_title="Run", yaxis_title="Exact matches (%)")
    fig.update_yaxes(range=[0, 100])
    if save:
        _save_fig(fig, _output_dir(result) / "trace_exact_match.html")
    return fig


def plot_trace_transitions(
    result: BenchmarkResult, save: bool = True
) -> go.Figure | None:
    transitions: dict[tuple[str, str], int] = defaultdict(int)
    for run_df in result.runs.values():
        for _, row in run_df.iterrows():
            agents = [str(answer["agent_type"]) for answer in row["trace"]["answers"]]
            path = ["START", *agents, "END"]
            for source, target in pairwise(path):
                transitions[(source, target)] += 1
    if not transitions:
        return None
    labels = sorted({node for transition in transitions for node in transition})
    index = {label: position for position, label in enumerate(labels)}
    fig = go.Figure(
        go.Sankey(
            node={"label": labels, "pad": 18, "thickness": 18},
            link={
                "source": [index[source] for source, _ in transitions],
                "target": [index[target] for _, target in transitions],
                "value": list(transitions.values()),
            },
        )
    )
    fig.update_layout(title="Observed Workflow Transitions")
    if save:
        _save_fig(fig, _output_dir(result) / "trace_transitions.html")
    return fig


def plot_error_rate(result: BenchmarkResult, save: bool = True) -> go.Figure | None:
    records = _flatten_traces(result.runs)
    df = pd.DataFrame(records)
    if df.empty or "error" not in df:
        return None
    df["failed"] = df["error"].notna() & df["error"].ne("")
    summary = df.groupby("agent", as_index=False)["failed"].mean()
    summary["percent"] = summary["failed"] * 100
    if summary["percent"].max() == 0:
        return None
    fig = px.bar(
        summary,
        x="agent",
        y="percent",
        color="agent",
        color_discrete_sequence=_COLOR_SEQ,
        title="Agent Error Rate",
        hover_data={"percent": ":.1f"},
    )
    fig.update_layout(xaxis_title="Agent", yaxis_title="Errors (%)", showlegend=False)
    if save:
        _save_fig(fig, _output_dir(result) / "error_rate.html")
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
  .meta {{ color: #636e72; font-size: 0.9rem; margin-bottom: 28px; }}
  .section {{ margin-bottom: 32px; }}
  .section-title {{ font-size: 1.2rem; margin-bottom: 12px; padding-bottom: 8px;
                    border-bottom: 2px solid #dfe6e9; color: #2d3436; }}
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
{sections}
</body>
</html>"""


def build_dashboard(result: BenchmarkResult) -> str:
    meta = result.metadata
    title = Path(meta["benchmark_run"]).name
    dataset_line = f"Dataset: {meta.get('dataset_path', 'not available')}"

    sections: list[tuple[str, list[tuple[str, go.Figure | None, bool]]]] = [
        (
            "Quality",
            [
                ("Scores", plot_per_agent_scores(result, save=False), False),
                ("Perplexity", plot_per_agent_perplexity(result, save=False), False),
                ("Prompt Scores", plot_prompt_score_heatmap(result, save=False), True),
                ("Score vs Perplexity", plot_score_vs_perplexity(result, save=False), False),
            ],
        ),
        (
            "Efficiency",
            [
                ("Timing Breakdown", plot_timing_breakdown(result, save=False), False),
                ("Score vs Latency", plot_score_vs_latency(result, save=False), False),
                ("Quality / Latency Pareto", plot_score_latency_pareto(result, save=False), False),
                ("Run Comparison", plot_run_comparison(result, save=False), True),
            ],
        ),
        (
            "Sustainability",
            [
                ("Energy Consumption", plot_energy_consumption(result, save=False), False),
                ("CO₂ Emissions", plot_emissions(result, save=False), False),
                ("Score vs Energy", plot_score_vs_energy(result, save=False), False),
            ],
        ),
        (
            "Workflow behavior",
            [
                (
                    "Trace Completion",
                    plot_trace_completion(result, save=False),
                    False,
                ),
                (
                    "Exact Trace Match",
                    plot_trace_exact_match(result, save=False),
                    False,
                ),
                ("Workflow Transitions", plot_trace_transitions(result, save=False), True),
                ("Error Rate", plot_error_rate(result, save=False), False),
            ],
        ),
    ]

    section_html: list[str] = []
    for section_name, plots in sections:
        cards: list[str] = []
        for name, fig, full_width in plots:
            if fig is None:
                continue
            div = fig.to_html(
                full_html=False,
                include_plotlyjs=False,
                config={"displayModeBar": False},
            )
            card_class = "card full" if full_width else "card"
            cards.append(
                f'      <div class="{card_class}"><h2>{name}</h2>{div}</div>'
            )
        if cards:
            section_html.append(
                f'  <section class="section"><h2 class="section-title">{section_name}</h2>'
                f'<div class="grid">{"\n".join(cards)}</div></section>'
            )

    html = _DASHBOARD_TEMPLATE.format(
        title=title,
        runtime=meta["total_runtime"],
        dataset_line=dataset_line,
        sections="\n".join(section_html),
    )
    return html


# ── Main entry point ─────────────────────────────────────────────────────


def analyze_benchmark(benchmark_dir: str) -> BenchmarkResult:
    """Load benchmark data, print tables, generate plots and dashboard.

    Returns the :class:`BenchmarkResult` for programmatic use (e.g. from a notebook).
    """
    result = BenchmarkResult.load(benchmark_dir)
    meta = result.metadata

    console.print(f"[bold cyan]Benchmark:[/bold cyan] {Path(meta['benchmark_run']).name}")
    console.print(f"  Runtime  {meta['total_runtime']:.1f}s")
    if meta.get("dataset_path"):
        console.print(f"  Dataset  [dim]{meta['dataset_path']}[/dim]")
    if meta.get("experiment"):
        console.print(f"  Experiment  [cyan]{meta['experiment'].get('id')}[/cyan]")
    console.print()

    print_overview(result)
    console.print()
    print_agent_breakdown(result)
    console.print()
    print_trace_analysis(result)

    out_dir = _output_dir(result)
    console.print(f"\n[bold cyan]Generated plots[/bold cyan]  [dim]{out_dir}[/dim]")

    plot_fns = [
        ("per_agent_scores", plot_per_agent_scores),
        ("per_agent_perplexity", plot_per_agent_perplexity),
        ("timing_breakdown", plot_timing_breakdown),
        ("energy_consumption", plot_energy_consumption),
        ("emissions", plot_emissions),
        ("score_vs_energy", plot_score_vs_energy),
        ("score_vs_latency", plot_score_vs_latency),
        ("score_vs_perplexity", plot_score_vs_perplexity),
        ("prompt_score_heatmap", plot_prompt_score_heatmap),
        ("score_latency_pareto", plot_score_latency_pareto),
        ("trace_completion", plot_trace_completion),
        ("trace_exact_match", plot_trace_exact_match),
        ("trace_transitions", plot_trace_transitions),
        ("error_rate", plot_error_rate),
        ("run_comparison", plot_run_comparison),
    ]
    for name, fn in plot_fns:
        fig = fn(result)
        if fig is not None:
            console.print(f"  [green]✓[/green] {name}.html + {name}.png")
        else:
            console.print(f"  [dim]– {name}.html (skipped: no data)[/dim]")

    dashboard_html = build_dashboard(result)
    dashboard_path = out_dir / "dashboard.html"
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
    "plot_per_agent_perplexity",
    "plot_per_agent_scores",
    "plot_prompt_score_heatmap",
    "plot_run_comparison",
    "plot_score_latency_pareto",
    "plot_score_vs_energy",
    "plot_score_vs_latency",
    "plot_score_vs_perplexity",
    "plot_timing_breakdown",
    "plot_trace_completion",
    "plot_trace_exact_match",
    "plot_trace_transitions",
    "print_agent_breakdown",
    "print_overview",
    "print_trace_analysis",
]
