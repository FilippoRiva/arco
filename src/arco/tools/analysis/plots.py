"""Plotly figures for benchmark analysis."""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

if TYPE_CHECKING:
    from .benchmark_result import BenchmarkResult

_COLOR_SEQ = ["#0984e3", "#00b894", "#e17055", "#6c5ce7", "#fdcb6e", "#d63031"]


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
