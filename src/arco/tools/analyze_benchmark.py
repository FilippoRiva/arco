import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from html import escape
from itertools import pairwise
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
from rich.console import Console
from rich.table import Table

from arco.core import Answer, State
from arco.data import BenchmarkDataset

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BenchmarkResult:
    """Loaded artifacts from a completed benchmark run."""

    metadata: dict
    benchmark_dir: Path
    answer_analysis_df: pd.DataFrame
    run_analysis_df: pd.DataFrame
    _states: dict[str, dict[str, State]]
    benchmark_dataset: BenchmarkDataset | None = None

    @classmethod
    def load(cls, benchmark_dir: str) -> BenchmarkResult:
        bdir = Path(benchmark_dir)

        with open(bdir / "bench_metadata.json") as f:
            metadata = json.load(f)

        run_names = _load_run_names(bdir, metadata)
        runs_dir = bdir / "runs"

        # Check folder
        if not runs_dir.exists():
            raise ValueError("No runs directory available for this benchmark")
        run_csv_paths = sorted(runs_dir.glob("*.csv"))
        if not run_csv_paths:
            raise ValueError(f"No per-run CSV files found under {runs_dir}")

        # Build the answer_analysis_df and extract states
        answer_analysis_records: list[dict] = []
        states: dict[str, dict[str, State]] = {}
        for csv_path in run_csv_paths:
            # check csv
            file_run_name = csv_path.stem
            run_name = run_names.get(file_run_name, file_run_name)
            df = pd.read_csv(csv_path)
            required_columns = {"entry_id", "run_id", "state", "changes"}
            missing_columns = required_columns.difference(df.columns)
            if missing_columns:
                raise ValueError(
                    f"Unsupported benchmark CSV {csv_path}; missing columns: "
                    f"{', '.join(sorted(missing_columns))}"
                )

            # get run info and build analysis_df rows
            run_states: dict[str, State] = {}
            for _, row in df.iterrows():
                entry_id = int(row["entry_id"])  # pyright: ignore
                run_id = str(row["run_id"])
                # state
                state = State.from_dict(_load_json_dict(row.get("state"), default={}))
                # changes
                changes = _load_json_dict(row.get("changes"), default={})

                for trace_index, answer in enumerate(
                    state.answers
                ):  # one entry per answer
                    answer_analysis_records.append(
                        _answer_analysis_record(
                            entry_id=entry_id,
                            run_id=run_id,
                            trace_index=trace_index,
                            answer=answer,
                            changes=changes,
                            run_name=run_name,
                        )
                    )

                # store each state in a convenient dict
                run_states[str(row["run_id"])] = state

            if run_states:
                states[run_name] = run_states

        answer_analysis_df = pd.DataFrame(answer_analysis_records)
        analysis_dir = bdir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = analysis_dir / "benchmark.parquet"
        answer_analysis_df.to_parquet(parquet_path, index=False)
        answer_analysis_df = pd.read_parquet(parquet_path)

        # Build the run_analysis_df
        run_analysis_df = _build_run_analysis_df(answer_analysis_df)

        # Load the dataset
        dataset_path = metadata.get("dataset_path")
        dataset = None
        if dataset_path:
            dataset_file = Path(dataset_path)
            if not dataset_file.is_absolute():
                dataset_file = bdir / dataset_file
            if not dataset_file.exists():
                dataset_file = bdir / "dataset.json"
            if dataset_file.exists():
                dataset = BenchmarkDataset.from_json(str(dataset_file))

        return cls(
            metadata=metadata,
            benchmark_dir=bdir,
            answer_analysis_df=answer_analysis_df,
            run_analysis_df=run_analysis_df,
            benchmark_dataset=dataset,
            _states=states,
        )


def _load_json_dict(value: Any, *, default: dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return default
    return parsed if isinstance(parsed, dict) else default


def _load_run_names(benchmark_dir: Path, metadata: dict) -> dict[str, str]:
    """Map safe flat CSV stems back to the names configured for each run."""
    candidates = [metadata.get("benchmark_config_snapshot"), "benchmark_config.yaml"]
    for candidate in candidates:
        if not candidate:
            continue
        config_path = Path(candidate)
        if not config_path.is_absolute():
            config_path = benchmark_dir / config_path
        if not config_path.exists():
            config_path = benchmark_dir / Path(candidate).name
        if not config_path.exists():
            continue
        with config_path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        return {
            str(run.get("name", "")).replace(" ", "_"): str(run.get("name", ""))
            for run in config.get("runs", [])
        }
    return {}


def _answer_analysis_record(
    *,
    answer: Answer,
    run_name: str,
    run_id: str,
    trace_index: int,
    entry_id: int,
    changes: dict,
) -> dict:
    """Create a flat, typed row suitable for downstream analysis in Parquet."""
    profile = answer.profiling_data

    return {
        "run_name": run_name,  # benchmark run name
        "entry_id": entry_id,  # prompt_id
        "run_id": run_id,  # specific run id
        "trace_index": trace_index,  # answer number
        "agent": answer.agent_id,
        "message": answer.message,
        "score": answer.gt_evaluation.score if answer.gt_evaluation else None,
        "score_success": answer.gt_evaluation.success
        if answer.gt_evaluation
        else False,
        "best_of_n_score": answer.evaluation.score if answer.evaluation else None,
        "best_of_n_success": answer.evaluation.success if answer.evaluation else False,
        "perplexity": answer.perplexity if answer.perplexity is not None else None,
        "thinking": answer.thinking,
        "error": answer.error,
        "agent_output": json.dumps(
            answer.agent_output, ensure_ascii=False, sort_keys=True
        ),
        "agent_config": json.dumps(
            asdict(answer.agent_config), ensure_ascii=False, sort_keys=True
        ),
        "changes": json.dumps(changes, ensure_ascii=False, sort_keys=True),
        **profile.as_dict(),
    }


def _build_run_analysis_df(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the unified Parquet data into the a run-level aggregated view."""
    summary_rows = []
    if analysis_df.empty:
        return pd.DataFrame(summary_rows, columns=["name", "metrics_by_agent"])

    aggregated_metrics = {
        "evaluation_gt": "score",
        "perplexity": "perplexity",
        "total_time": "total_time",
        "llm_time": "llm_time",
    }
    for run_name, run_df in analysis_df.groupby("run_name", sort=False):
        metrics_by_agent: dict[str, dict[str, float]] = defaultdict(dict)

        for agent, agent_df in run_df.groupby("agent", sort=False):
            for summary_metric, column in aggregated_metrics.items():
                values = agent_df[column].dropna()
                if not values.empty:
                    metrics_by_agent[str(agent)][summary_metric] = float(values.mean())

        summary_rows.append(
            {
                "name": run_name,
                "metrics_by_agent": json.dumps(dict(metrics_by_agent)),
            }
        )
    return pd.DataFrame(summary_rows, columns=["name", "metrics_by_agent"])


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


# ── Plotly plots ─────────────────────────────────────────────────────────


_COLOR_SEQ = ["#0984e3", "#00b894", "#e17055", "#6c5ce7", "#fdcb6e", "#d63031"]
_AGENT_LABEL_COLORS = ["#3f6b7a", "#5e7651", "#8a6744", "#725e83"]


def save_plot(fig: go.Figure, path: Path) -> go.Figure:
    fig.update_layout(height=400)
    html_dir = path.parent / "html"
    png_dir = path.parent / "png"
    html_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    html_path = html_dir / path.name
    png_path = png_dir / path.with_suffix(".png").name
    fig.write_html(html_path, include_plotlyjs="cdn", config={"displayModeBar": False})
    try:
        fig.write_image(png_path, scale=2)
    except Exception:  # noqa BLE001 - fine
        pass  # kaleido/Chrome not available — PNG skipped
    return fig


def plot_per_agent_scores(result: BenchmarkResult) -> go.Figure | None:
    if result.answer_analysis_df.empty:
        return None
    fig = px.box(
        result.answer_analysis_df,
        x="agent",
        y="score",
        color="run_name",
        color_discrete_sequence=_COLOR_SEQ,
        points="outliers",
        hover_data=["run_name", "entry_id"],
        title="Per-Agent Score Distribution by Run",
    )
    fig.update_layout(
        boxmode="group",
        xaxis_title="Agent",
        yaxis_title="Ground-truth score",
        yaxis_range=[0, 1.05],
    )
    return fig


def plot_per_agent_perplexity(result: BenchmarkResult) -> go.Figure | None:
    df = result.answer_analysis_df
    if df.empty or df["perplexity"].isna().all():
        return None
    fig = px.box(
        df,
        x="agent",
        y="perplexity",
        color="run_name",
        color_discrete_sequence=_COLOR_SEQ,
        points="outliers",
        hover_data=["run_name", "entry_id"],
        title="Per-Agent Perplexity Distribution by Run",
    )
    fig.update_layout(
        boxmode="group",
        xaxis_title="Agent",
        yaxis_title="Perplexity",
    )
    return fig


def plot_score_improvement_vs_baseline(result: BenchmarkResult) -> go.Figure:
    title = "Paired Score Improvement vs Baseline"
    df = result.answer_analysis_df
    if df.empty:
        df = pd.DataFrame(columns=["run_name", "agent", "entry_id", "score"])
    else:
        df = df.dropna(subset=["score"]).copy()
        df["run_name"] = df["run_name"].fillna("Unknown run").astype(str)
        df["agent"] = df["agent"].fillna("Unknown agent").astype(str)

    baseline_mask = df["run_name"].str.strip().str.casefold() == "baseline"
    if not baseline_mask.any():
        fig = go.Figure()
        fig.add_annotation(
            text="No Baseline run provided",
            showarrow=False,
            font={"size": 14, "color": "#636e72"},
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
        fig.update_layout(
            title=title,
            xaxis_title="Agent",
            yaxis_title="Score change vs baseline",
        )
        fig.update_yaxes(range=[-1.05, 1.05], zeroline=True)
        return fig

    df.loc[baseline_mask, "run_name"] = "Baseline"
    score_by_entry = df.groupby(
        ["run_name", "agent", "entry_id"], as_index=False, sort=False
    )["score"].mean()
    is_baseline = score_by_entry["run_name"] == "Baseline"
    baseline_scores = score_by_entry[is_baseline][
        ["agent", "entry_id", "score"]
    ].rename(columns={"score": "baseline_score"})
    candidate_scores = score_by_entry[~is_baseline]
    paired = candidate_scores.merge(
        baseline_scores,
        on=["agent", "entry_id"],
        how="inner",
    )
    if paired.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No paired score entries found",
            showarrow=False,
            font={"size": 14, "color": "#636e72"},
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
        fig.update_layout(
            title=title,
            xaxis_title="Agent",
            yaxis_title="Score change vs baseline",
        )
        fig.update_yaxes(range=[-1.05, 1.05], zeroline=True)
        return fig

    paired["score_delta"] = paired["score"] - paired["baseline_score"]
    summary = paired.groupby(["run_name", "agent"], as_index=False, sort=False).agg(
        mean_improvement=("score_delta", "mean"),
        paired_entry_count=("score_delta", "size"),
        run_mean_score=("score", "mean"),
        baseline_mean_score=("baseline_score", "mean"),
    )
    summary["label"] = summary["mean_improvement"].map(
        lambda value: "0.000" if value == 0 else f"{value:+.3f}"
    )
    fig = px.bar(
        summary,
        x="agent",
        y="mean_improvement",
        color="run_name",
        text="label",
        barmode="group",
        color_discrete_sequence=_COLOR_SEQ,
        hover_data={
            "run_name": True,
            "paired_entry_count": True,
            "run_mean_score": ":.3f",
            "baseline_mean_score": ":.3f",
            "mean_improvement": ":+.3f",
            "label": False,
        },
        title=title,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        barmode="group",
        xaxis_title="Agent",
        yaxis_title="Score change vs baseline",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#636e72")
    fig.update_yaxes(range=[-1.05, 1.05], zeroline=True)
    return fig


def plot_timing_breakdown(result: BenchmarkResult) -> go.Figure | None:
    df = result.answer_analysis_df
    if df.empty:
        return None

    rows: list[dict] = []
    for (run_name, agent), values in df.groupby(["run_name", "agent"], sort=False):
        for metric in ("total_time", "llm_time"):
            series = values[metric].dropna()
            if series.empty:
                continue
            median = series.median()
            p90 = series.quantile(0.9)
            rows.append(
                {
                    "run_name": str(run_name),
                    "agent": str(agent),
                    "metric": metric,
                    "median": median,
                    "p90": p90,
                    "p90_error": max(p90 - median, 0),
                }
            )
    if not rows:
        return None

    plot_df = pd.DataFrame(rows)
    fig = go.Figure()
    for metric, label, color in (
        ("total_time", "Total time", "#74b9ff"),
        ("llm_time", "LLM time", "#0984e3"),
    ):
        metric_df = plot_df[plot_df["metric"] == metric]
        if metric_df.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=[metric_df["run_name"].tolist(), metric_df["agent"].tolist()],
                y=metric_df["median"],
                name=label,
                marker_color=color,
                error_y={"type": "data", "array": metric_df["p90_error"]},
                customdata=metric_df[["run_name", "agent", "p90"]].to_numpy(),
                hovertemplate=(
                    "Run: %{customdata[0]}<br>"
                    "Agent: %{customdata[1]}<br>"
                    "Median: %{y:.3f} s<br>"
                    "p90: %{customdata[2]:.3f} s<extra>%{fullData.name}</extra>"
                ),
            )
        )
    fig.update_layout(
        barmode="group",
        title="Median Timing Breakdown per Run and Agent (p90 error bars)",
        xaxis_title="Run / Agent",
        yaxis_title="Seconds",
    )
    return fig


def plot_energy_consumption(result: BenchmarkResult) -> go.Figure | None:
    df = result.answer_analysis_df
    energy_columns = ["cpu_energy_kwh", "gpu_energy_kwh", "ram_energy_kwh"]
    if df.empty:
        return None

    energy_df = df.copy()
    energy_df["cumulative_energy_kwh"] = energy_df[energy_columns].sum(
        axis=1, min_count=1
    )
    energy_df = energy_df.dropna(subset=["cumulative_energy_kwh"])
    if energy_df.empty:
        return None

    energy_df["run_name"] = energy_df["run_name"].fillna("Unknown run").astype(str)
    means = energy_df.groupby(["run_name", "agent"], as_index=False, sort=False).agg(
        cumulative_energy_kwh=("cumulative_energy_kwh", "mean")
    )
    means["energy_wh"] = means["cumulative_energy_kwh"] * 1000
    fig = px.bar(
        means,
        x="agent",
        y="energy_wh",
        color="run_name",
        barmode="group",
        color_discrete_sequence=_COLOR_SEQ,
        title="Mean Cumulative Energy Consumption by Run and Agent",
        hover_data={
            "run_name": True,
            "cumulative_energy_kwh": ":.6f",
            "energy_wh": ":.4f",
        },
    )
    fig.update_layout(xaxis_title="Agent", yaxis_title="Mean cumulative energy (Wh)")
    return fig


def plot_mean_energy_by_run(result: BenchmarkResult) -> go.Figure | None:
    df = result.answer_analysis_df
    energy_columns = ["cpu_energy_kwh", "gpu_energy_kwh", "ram_energy_kwh"]
    if df.empty:
        return None

    energy_df = df.copy()
    energy_df["cumulative_energy_kwh"] = energy_df[energy_columns].sum(
        axis=1, min_count=1
    )
    energy_df = energy_df.dropna(subset=["cumulative_energy_kwh"])
    if energy_df.empty:
        return None

    energy_df["run_name"] = energy_df["run_name"].fillna("Unknown run").astype(str)
    energy_df["agent"] = energy_df["agent"].fillna("Unknown agent").astype(str)
    agent_means = energy_df.groupby(
        ["run_name", "agent"], as_index=False, sort=False
    ).agg(agent_mean_energy_kwh=("cumulative_energy_kwh", "mean"))
    means = agent_means.groupby("run_name", as_index=False, sort=False).agg(
        cumulative_energy_kwh=("agent_mean_energy_kwh", "sum"),
        agent_count=("agent", "nunique"),
    )
    means["energy_wh"] = means["cumulative_energy_kwh"] * 1000
    fig = px.bar(
        means,
        x="run_name",
        y="energy_wh",
        color="run_name",
        color_discrete_sequence=_COLOR_SEQ,
        title="Sum of Mean Agent Energy Consumption by Run",
        hover_data={
            "cumulative_energy_kwh": ":.6f",
            "energy_wh": ":.4f",
            "agent_count": True,
        },
    )
    fig.update_layout(
        xaxis_title="Run",
        yaxis_title="Sum of agent mean energy (Wh)",
        showlegend=False,
    )
    return fig


def plot_emissions(result: BenchmarkResult) -> go.Figure | None:
    df = result.answer_analysis_df
    if df.empty or df["emissions_kg_co2"].isna().all():
        return None
    emission_df = df.copy()
    emission_df["run_name"] = emission_df["run_name"].fillna("Unknown run").astype(str)
    agg = emission_df.groupby(["run_name", "agent"], as_index=False, sort=False)[
        "emissions_kg_co2"
    ].mean()
    agg["grams_co2"] = agg["emissions_kg_co2"] * 1000
    fig = px.bar(
        agg,
        x="agent",
        y="grams_co2",
        color="run_name",
        barmode="group",
        color_discrete_sequence=_COLOR_SEQ,
        title="Mean CO₂ Emissions by Run and Agent",
        hover_data={
            "run_name": True,
            "grams_co2": ":.4f",
            "emissions_kg_co2": ":.6f",
        },
    )
    fig.update_layout(xaxis_title="Agent", yaxis_title="Mean emissions (g CO₂)")
    return fig


def plot_score_vs_energy(result: BenchmarkResult) -> go.Figure:
    df = result.answer_analysis_df
    if df.empty:
        plot_df = pd.DataFrame(
            columns=["run_name", "agent", "score", "energy_consumed_kwh"]
        )
    else:
        plot_df = df.dropna(subset=["score", "energy_consumed_kwh"]).copy()
    if plot_df.empty:
        fig = go.Figure()
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
            title="Quality / Energy Pareto Frontier per Agent",
            xaxis_title="Mean energy consumed (kWh)",
            yaxis_title="Mean score",
        )
        fig.update_yaxes(range=[0, 1.05])
        return fig

    plot_df["run_name"] = plot_df["run_name"].fillna("Unknown run").astype(str)
    plot_df["agent"] = plot_df["agent"].fillna("Unknown agent").astype(str)
    grouped = plot_df.groupby(["run_name", "agent"], as_index=False)[
        ["score", "energy_consumed_kwh"]
    ].mean()
    grouped["pareto"] = False
    for agent, agent_rows in grouped.groupby("agent", sort=False):
        pareto_indices = []
        for index, row in agent_rows.iterrows():
            dominated = (
                (agent_rows["score"] >= row["score"])
                & (agent_rows["energy_consumed_kwh"] <= row["energy_consumed_kwh"])
                & (
                    (agent_rows["score"] > row["score"])
                    | (agent_rows["energy_consumed_kwh"] < row["energy_consumed_kwh"])
                )
            ).any()
            if not dominated:
                pareto_indices.append(index)
        grouped.loc[pareto_indices, "pareto"] = True

    fig = px.scatter(
        grouped,
        x="energy_consumed_kwh",
        y="score",
        color="agent",
        symbol="pareto",
        hover_data=["run_name"],
        color_discrete_sequence=_COLOR_SEQ,
        title="Quality / Energy Pareto Frontier per Agent",
    )
    for color_index, (agent, agent_rows) in enumerate(grouped.groupby("agent")):
        frontier = agent_rows[agent_rows["pareto"]].sort_values("energy_consumed_kwh")
        if len(frontier) < 2:
            continue
        fig.add_trace(
            go.Scatter(
                x=frontier["energy_consumed_kwh"],
                y=frontier["score"],
                mode="lines",
                line={"color": _COLOR_SEQ[color_index % len(_COLOR_SEQ)], "width": 2},
                name=f"{agent} frontier",
                hoverinfo="skip",
                showlegend=True,
            )
        )
    fig.update_layout(
        xaxis_title="Mean energy consumed (kWh)", yaxis_title="Mean score"
    )
    fig.update_yaxes(range=[0, 1.05])
    return fig


def plot_score_vs_perplexity(result: BenchmarkResult) -> go.Figure | None:
    if result.answer_analysis_df.empty:
        return None
    df = result.answer_analysis_df.dropna(subset=["score", "perplexity"]).copy()
    if df.empty:
        return None

    df["agent"] = df["agent"].fillna("Unknown agent").astype(str)
    df["run_name"] = df["run_name"].fillna("Unknown run").astype(str)
    run_df = df.groupby(["run_name", "agent"], as_index=False, sort=False).agg(
        avg_score=("score", "mean"),
        avg_perplexity=("perplexity", "mean"),
        entry_count=("entry_id", "nunique"),
    )
    run_symbols = [
        "circle",
        "square",
        "diamond",
        "cross",
        "x",
        "triangle-up",
        "triangle-down",
        "star",
        "hexagon",
        "pentagon",
    ]
    symbol_by_run = {
        run_name: run_symbols[index % len(run_symbols)]
        for index, run_name in enumerate(run_df["run_name"].drop_duplicates())
    }

    fig = go.Figure()
    for color_index, (agent, agent_df) in enumerate(
        run_df.groupby("agent", sort=False)
    ):
        customdata = agent_df[["run_name", "entry_count"]].to_numpy()
        fig.add_trace(
            go.Scatter(
                x=agent_df["avg_perplexity"],
                y=agent_df["avg_score"],
                mode="markers",
                name=agent,
                legendgroup=agent,
                marker={
                    "color": _COLOR_SEQ[color_index % len(_COLOR_SEQ)],
                    "symbol": [symbol_by_run[name] for name in agent_df["run_name"]],
                    "size": 12,
                    "opacity": 0.8,
                },
                customdata=customdata,
                hovertemplate=(
                    "Agent: %{fullData.name}<br>"
                    "Run: %{customdata[0]}<br>"
                    "Entries: %{customdata[1]}<br>"
                    "Avg perplexity: %{x:.4g}<br>"
                    "Avg score: %{y:.3f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Average Run Score vs Average Run Perplexity per Agent",
        xaxis_title="Average run perplexity",
        yaxis_title="Average run score",
        legend={
            "title": {"text": "Agent (click to toggle)"},
            "groupclick": "togglegroup",
        },
    )
    fig.update_yaxes(range=[0, 1.05])
    return fig


def plot_prompt_score_heatmap(result: BenchmarkResult) -> go.Figure | None:
    if result.answer_analysis_df.empty:
        return None
    df = result.answer_analysis_df.dropna(subset=["score"])
    if df.empty:
        return None
    scores = (
        df.groupby(["run_name", "entry_id"], as_index=False)["score"]
        .mean()
        .pivot(index="run_name", columns="entry_id", values="score")
    )
    fig = px.imshow(
        scores,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        zmin=0,
        zmax=1,
        labels={"x": "Prompt entry", "y": "Run", "color": "Score"},
        title="Per-Prompt Score Heatmap",
    )
    return fig


def plot_score_latency_pareto(result: BenchmarkResult) -> go.Figure | None:
    if result.answer_analysis_df.empty:
        return None
    df = result.answer_analysis_df.dropna(subset=["score", "total_time"])
    if df.empty:
        return None
    grouped = df.groupby(["run_name", "agent"], as_index=False)[
        ["score", "total_time", "energy_consumed_kwh"]
    ].mean()
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
        hover_data=["run_name", "energy_consumed_kwh"],
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
    return fig


def plot_trace_exact_match(result: BenchmarkResult) -> go.Figure | None:
    dataset = result.benchmark_dataset
    if dataset is None or result.answer_analysis_df.empty:
        return None
    expected_by_id = {
        entry.id: [str(element.agent_type) for element in entry.trace]
        for entry in dataset.entries
    }
    rows = []
    for (run_name, _run_id, entry_id), trace_df in result.answer_analysis_df.groupby(
        ["run_name", "run_id", "entry_id"], sort=False
    ):
        expected = expected_by_id.get(int(str(entry_id)))
        if expected is None:
            continue
        actual = trace_df.sort_values("trace_index")["agent"].astype(str).tolist()
        rows.append({"run_name": run_name, "exact": actual == expected})
    if not rows:
        return None
    summary = pd.DataFrame(rows).groupby("run_name", as_index=False)["exact"].mean()
    summary["percent"] = summary["exact"] * 100
    fig = px.bar(
        summary,
        x="run_name",
        y="percent",
        color="percent",
        range_color=[0, 100],
        color_continuous_scale="RdYlGn",
        title="Exact Workflow Trace Match Rate",
        hover_data={"percent": ":.1f"},
    )
    fig.update_layout(xaxis_title="Run", yaxis_title="Exact matches (%)")
    fig.update_yaxes(range=[0, 100])
    return fig


def plot_trace_transitions(result: BenchmarkResult) -> go.Figure | None:
    if result.answer_analysis_df.empty:
        return None
    transitions: dict[tuple[str, str], int] = defaultdict(int)
    for _, trace_df in result.answer_analysis_df.groupby(
        ["run_name", "run_id", "entry_id"], sort=False
    ):
        agents = trace_df.sort_values("trace_index")["agent"].astype(str).tolist()
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
    return fig


def plot_error_rate(result: BenchmarkResult) -> go.Figure | None:
    df = result.answer_analysis_df
    if df.empty or "error" not in df:
        return None

    error_df = df[["run_name", "agent", "error"]].copy()
    error_df["run_name"] = error_df["run_name"].fillna("Unknown run").astype(str)
    error_df["agent"] = error_df["agent"].fillna("Unknown agent").astype(str)
    error_df["failed"] = error_df["error"].notna() & error_df["error"].ne("")
    summary = error_df.groupby(["run_name", "agent"], as_index=False, sort=False).agg(
        failed_count=("failed", "sum"),
        answer_count=("failed", "size"),
    )
    summary["percent"] = summary["failed_count"] / summary["answer_count"] * 100
    summary["zero_label"] = summary["percent"].map(
        lambda value: "0%" if value == 0 else ""
    )
    y_max = max(float(summary["percent"].max()) * 1.15, 1.0)
    fig = px.bar(
        summary,
        x="agent",
        y="percent",
        color="run_name",
        barmode="group",
        text="zero_label",
        color_discrete_sequence=_COLOR_SEQ,
        title="Agent Error Rate by Run",
        hover_data={
            "run_name": True,
            "failed_count": True,
            "answer_count": True,
            "percent": ":.1f",
            "zero_label": False,
        },
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title="Agent", yaxis_title="Errors (%)")
    fig.update_yaxes(range=[0, y_max])
    return fig


def _display_dataset_path(dataset_path: str | None, benchmark_dir: Path) -> str:
    if not dataset_path:
        return "not available"

    path = Path(dataset_path).expanduser()
    if not path.is_absolute():
        benchmark_path = benchmark_dir / path
        working_path = Path.cwd() / path
        path = (
            benchmark_path
            if benchmark_path.exists() or not working_path.exists()
            else working_path
        )

    resolved_path = path.resolve()
    project_parent = Path.cwd().resolve().parent
    try:
        display_path = resolved_path.relative_to(project_parent)
    except ValueError:
        display_path = Path(os.path.relpath(resolved_path, start=project_parent))
    return display_path.as_posix()


def _run_changes_html(analysis_df: pd.DataFrame) -> str:
    if analysis_df.empty or "run_name" not in analysis_df:
        return ""

    parsed_runs: list[tuple[str, list[dict[str, Any]]]] = []
    all_agents: set[str] = set()
    for run_name, run_df in analysis_df.groupby("run_name", sort=False, dropna=False):
        display_name = "Unknown run" if pd.isna(run_name) else str(run_name)
        change_sets_by_json: dict[str, dict[str, Any]] = {}
        if "changes" in run_df:
            for value in run_df["changes"].dropna():
                changes = _load_json_dict(value, default={})
                key = json.dumps(
                    changes, sort_keys=True, ensure_ascii=False, default=str
                )
                change_sets_by_json[key] = changes
                all_agents.update(str(agent) for agent in changes)
        parsed_runs.append((display_name, list(change_sets_by_json.values()) or [{}]))

    agent_colors = {
        agent: _AGENT_LABEL_COLORS[index % len(_AGENT_LABEL_COLORS)]
        for index, agent in enumerate(sorted(all_agents))
    }
    run_items: list[str] = []
    for run_name, run_change_sets in parsed_runs:
        rendered_sets = []
        for index, changes in enumerate(run_change_sets, start=1):
            if not changes:
                rendered_sets.append('<p class="no-changes">No changes</p>')
                continue
            agent_items = []
            for agent, agent_changes in changes.items():
                if isinstance(agent_changes, dict):
                    parameters = "".join(
                        "<li>"
                        f'<span class="parameter-name">{escape(str(key))}</span>'
                        f" = {escape(str(value))}</li>"
                        for key, value in agent_changes.items()
                    )
                else:
                    parameters = f"<li>{escape(str(agent_changes))}</li>"
                agent_name = str(agent)
                agent_color = agent_colors[agent_name]
                agent_items.append(
                    f'<div class="agent-change"><h4 style="color:{agent_color}">'
                    f"{escape(agent_name)}</h4><ul>{parameters}</ul></div>"
                )
            set_label = (
                f'<h4 class="change-set-title">Change set {index}</h4>'
                if len(run_change_sets) > 1
                else ""
            )
            rendered_sets.append(f"{set_label}{''.join(agent_items)}")
        run_items.append(
            f'<article class="run-item"><h3>{escape(run_name)}</h3>'
            f"{''.join(rendered_sets)}</article>"
        )

    return (
        '<details class="section run-summary">'
        '<summary class="section-title">Runs and changes</summary>'
        f'<div class="run-list">{"".join(run_items)}</div></details>'
    )


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
  .meta {{ color: #636e72; font-size: 0.9rem; margin-bottom: 16px; }}
  .run-list {{ display: flex; flex-direction: column; gap: 8px; }}
  .run-item {{ min-width: 0; padding: 8px 0; border-bottom: 1px solid #dfe6e9; }}
  .run-item h3 {{ font-size: 0.95rem; margin-bottom: 4px; }}
  .agent-change {{ margin: 4px 0 0 12px; }}
  .agent-change h4 {{ font-size: 0.85rem; font-weight: 600; }}
  .agent-change ul {{ list-style: none; margin: 2px 0 0 12px; }}
  .agent-change li {{ font-size: 0.82rem; line-height: 1.4; }}
  .parameter-name {{ font-family: ui-monospace, monospace; }}
  .change-set-title {{ font-size: 0.85rem; margin: 4px 0; color: #636e72; }}
  .no-changes {{ margin-left: 12px; color: #636e72; font-style: italic; }}
  .section {{ margin-bottom: 32px; }}
  .section-title {{ font-size: 1.2rem; margin-bottom: 12px; padding-bottom: 8px;
                    border-bottom: 2px solid #dfe6e9; color: #2d3436; cursor: pointer; }}
  .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }}
  .card {{ background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
           padding: 8px; overflow: hidden; min-width: 0; position: relative; }}
  .card.expanded {{ grid-column: 1 / -1; }}
  .expand-button {{ position: absolute; top: 10px; right: 10px; z-index: 2;
                    width: 28px; height: 28px; display: grid; place-items: center;
                    cursor: pointer; border: 1px solid #dfe6e9; border-radius: 4px;
                    background: rgba(255,255,255,0.92); color: #2d3436; padding: 2px;
                    opacity: 0.75; pointer-events: auto;
                    transition: opacity 0.15s ease; }}
  .card:hover .expand-button, .card:focus-within .expand-button {{ opacity: 1; }}
  .expand-button svg {{ display: block; }}
  .expand-button:hover {{ background: #f5f6fa; }}
  .expand-button:focus-visible {{ outline: 2px solid #0984e3; outline-offset: 2px; }}
  .section-title {{ -webkit-user-select: none; user-select: none; }}
  .card .js-plotly-plot {{ min-height: 380px !important; }}
  @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
  <h1>Benchmark: {title}</h1>
  <div class="meta">Runtime: {runtime:.1f}s &middot; {dataset_line} &middot; {dataset_size_line}</div>
{run_summary}
{sections}
<script>
  document.querySelectorAll('details.section').forEach((section) => {{
    section.addEventListener('toggle', () => {{
      if (!section.open || !window.Plotly) return;
      section.querySelectorAll('.js-plotly-plot').forEach((plot) => {{
        requestAnimationFrame(() => Plotly.Plots.resize(plot));
      }});
    }});
  }});
  const plotWidthObserver = new ResizeObserver((entries) => {{
    entries.forEach((entry) => {{
      const width = Math.floor(entry.contentRect.width);
      if (width <= 0 || !window.Plotly) return;
      if (entry.target.dataset.plotWidth === String(width)) return;
      entry.target.dataset.plotWidth = String(width);
      entry.target.querySelectorAll('.js-plotly-plot').forEach((plot) => {{
        Plotly.relayout(plot, {{width, autosize: false}});
      }});
    }});
  }});
  document.querySelectorAll('.card').forEach((card) => {{
    plotWidthObserver.observe(card);
  }});
  document.querySelectorAll('.expand-button').forEach((button) => {{
    button.addEventListener('click', () => {{
      const card = button.closest('.card');
      const plot = card.querySelector('.js-plotly-plot');
      if (!plot || !window.Plotly) return;
      const expanded = !card.classList.contains('expanded');
      const normalHeight = Number(plot.dataset.normalHeight || plot.layout.height || 450);
      plot.dataset.normalHeight = String(normalHeight);
      card.classList.toggle('expanded', expanded);
      button.title = expanded ? 'Restore size' : 'Expand';
      button.setAttribute('aria-label', button.title);
      Plotly.relayout(plot, {{
        height: expanded ? normalHeight * 2 : normalHeight
      }}).then(() => requestAnimationFrame(() => Plotly.Plots.resize(plot)));
    }});
  }});
</script>
</body>
</html>"""


def build_dashboard(result: BenchmarkResult) -> str:
    meta = result.metadata
    title = Path(meta["benchmark_run"]).name
    dataset_path = _display_dataset_path(meta.get("dataset_path"), result.benchmark_dir)
    dataset_line = f"Dataset: {escape(dataset_path)}"
    if result.benchmark_dataset is None:
        dataset_size_line = "Dataset size: unavailable"
    else:
        entry_count = len(result.benchmark_dataset.entries)
        entry_label = "entry" if entry_count == 1 else "entries"
        dataset_size_line = f"Dataset size: {entry_count} {entry_label}"
    run_summary = _run_changes_html(result.answer_analysis_df)

    sections: list[tuple[str, list[tuple[str, go.Figure | None, bool]]]] = [
        (
            "Quality",
            [
                ("Scores", plot_per_agent_scores(result), True),
                (
                    "Paired Score Improvement",
                    plot_score_improvement_vs_baseline(result),
                    True,
                ),
                ("Perplexity", plot_per_agent_perplexity(result), True),
                ("Prompt Scores", plot_prompt_score_heatmap(result), True),
                (
                    "Score vs Perplexity",
                    plot_score_vs_perplexity(result),
                    True,
                ),
            ],
        ),
        (
            "Latency",
            [
                ("Timing Breakdown", plot_timing_breakdown(result), False),
                (
                    "Quality / Latency Pareto",
                    plot_score_latency_pareto(result),
                    False,
                ),
            ],
        ),
        (
            "Energy",
            [
                (
                    "Energy Consumption",
                    plot_energy_consumption(result),
                    False,
                ),
                ("CO₂ Emissions", plot_emissions(result), False),
                ("Energy by Run", plot_mean_energy_by_run(result), False),
                ("Score vs Energy", plot_score_vs_energy(result), False),
            ],
        ),
        (
            "Workflow behavior",
            [
                (
                    "Exact Trace Match",
                    plot_trace_exact_match(result),
                    False,
                ),
                (
                    "Workflow Transitions",
                    plot_trace_transitions(result),
                    True,
                ),
                ("Agent Error Rate by Run", plot_error_rate(result), False),
            ],
        ),
    ]

    section_html: list[str] = []
    for section_name, plots in sections:
        cards: list[str] = []
        for name, fig, _ in plots:
            if fig is None:
                continue
            # Keep the Plotly figure title as the card's only chart heading.
            fig.update_layout(autosize=True)
            div = fig.to_html(
                full_html=False,
                include_plotlyjs=False,
                config={"displayModeBar": False, "responsive": True},
            )
            cards.append(
                f'      <div class="card">'
                f'<button class="expand-button" type="button" aria-label="Expand" title="Expand">'
                f'<svg aria-hidden="true" viewBox="0 0 16 16" width="16" height="16" '
                f'fill="none" stroke="currentColor" stroke-width="1.5">'
                f'<path d="M2 6V2h4M10 2h4v4M14 10v4h-4M6 14H2v-4"/>'
                f"</svg></button>{div}</div>"
            )
        if cards:
            section_html.append(
                f'  <details class="section" open>'
                f'<summary class="section-title">{section_name}</summary>'
                f'<div class="grid">{"\n".join(cards)}</div></details>'
            )

    html = _DASHBOARD_TEMPLATE.format(
        title=title,
        runtime=meta["total_runtime"],
        dataset_line=dataset_line,
        dataset_size_line=dataset_size_line,
        run_summary=run_summary,
        sections="\n".join(section_html),
    )
    return html


# ── Main entry point ─────────────────────────────────────────────────────


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
