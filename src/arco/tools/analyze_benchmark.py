import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
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
    analysis_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    @classmethod
    def load(cls, benchmark_dir: str) -> BenchmarkResult:
        bdir = Path(benchmark_dir)

        with open(bdir / "bench_metadata.json") as f:
            metadata = json.load(f)

        run_names = _load_run_names(bdir, metadata)
        runs_dir = bdir / "runs"
        runs: dict[str, pd.DataFrame] = {}
        states: dict[str, dict[str, State]] = {}
        analysis_records: list[dict] = []
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
                file_run_name = csv_path.stem
                run_name = run_names.get(file_run_name, file_run_name)
                df = pd.read_csv(csv_path)
                required_columns = {"entry_id", "run_id"}
                missing_columns = required_columns.difference(df.columns)
                if missing_columns:
                    raise ValueError(
                        f"Unsupported benchmark CSV {csv_path}; missing columns: "
                        f"{', '.join(sorted(missing_columns))}"
                    )
                if "state" not in df.columns and "execution_trace" not in df.columns:
                    raise ValueError(
                        f"Unsupported benchmark CSV {csv_path}; expected a serialized "
                        "state or execution_trace column"
                    )

                run_states: dict[str, State] = {}
                for _, row in df.iterrows():
                    state_data = _load_json_value(row.get("state"), default={})
                    if state_data:
                        state = State.from_dict(state_data)
                        run_states[str(row["run_id"])] = state
                        answers = [answer.to_dict() for answer in state.answers]
                        state_prompt = state.prompt
                        global_profile = state_data.get("global_profiling_data", {})
                    else:
                        state_prompt = row.get("prompt")
                        global_profile = {}
                        legacy_trace = _load_json_value(
                            row.get("execution_trace"), default={"answers": []}
                        )
                        answers = legacy_trace.get("answers", [])

                    changes = _load_json_value(row.get("changes"), default={})
                    entry_id = _as_int(row["entry_id"])
                    run_id = str(row["run_id"])
                    for trace_index, answer in enumerate(answers):
                        analysis_records.append(
                            _analysis_record(
                                answer=answer,
                                run_name=run_name,
                                entry_id=entry_id,
                                run_id=run_id,
                                run_fingerprint=row.get("run_fingerprint"),
                                prompt=state_prompt,
                                trace_index=trace_index,
                                changes=changes,
                                global_profile=global_profile,
                            )
                        )

                if run_states:
                    states[run_name] = run_states

        analysis_df = pd.DataFrame(analysis_records)
        analysis_dir = bdir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = analysis_dir / "benchmark.parquet"
        analysis_df.to_parquet(parquet_path, index=False)
        # Use the unified Parquet dataset as the canonical analysis source.
        analysis_df = pd.read_parquet(parquet_path)
        runs = _runs_from_analysis(analysis_df)
        summary_df = _summary_from_analysis(analysis_df)

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
            summary_df=summary_df,
            runs=runs,
            states=states,
            dataset=dataset,
            analysis_df=analysis_df,
        )


def _load_json_value(value, *, default):
    """Parse CSV JSON cells while tolerating already-decoded and null values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _as_int(value) -> int:
    return int(value)


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


def _trace_answer(answer: dict) -> dict:
    """Normalize serialized State answers and legacy trace answers."""
    profile = answer.get("profiling_data") or {}
    ground_truth = answer.get("gt_evaluation") or answer.get("evaluation_gt")
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("score")
    if ground_truth is None:
        ground_truth = answer.get("score")

    return {
        "agent_type": str(answer.get("agent_type", answer.get("agent_id", ""))),
        "evaluation_gt": ground_truth,
        "perplexity": answer.get("perplexity"),
        "total_time": answer.get("total_time", profile.get("total_time")),
        "llm_time": answer.get("llm_time", profile.get("llm_time")),
        "energy_consumed_kwh": answer.get(
            "energy_consumed_kwh", profile.get("energy_consumed_kwh")
        ),
        "cpu_energy_kwh": answer.get("cpu_energy_kwh", profile.get("cpu_energy_kwh")),
        "gpu_energy_kwh": answer.get("gpu_energy_kwh", profile.get("gpu_energy_kwh")),
        "ram_energy_kwh": answer.get("ram_energy_kwh", profile.get("ram_energy_kwh")),
        "emissions_kg_co2": answer.get(
            "emissions_kg_co2", profile.get("emissions_kg_co2")
        ),
        "error": answer.get("error"),
    }


def _analysis_record(
    *,
    answer: dict,
    run_name: str,
    entry_id: int,
    run_id: str,
    run_fingerprint,
    prompt: str | None,
    trace_index: int,
    changes: dict,
    global_profile: dict,
) -> dict:
    """Create a flat, typed row suitable for downstream analysis in Parquet."""
    normalized = _trace_answer(answer)
    evaluation = answer.get("gt_evaluation") or {}
    best_of_n_evaluation = answer.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        evaluation = {}
    if not isinstance(best_of_n_evaluation, dict):
        best_of_n_evaluation = {}
    return {
        "run": run_name,
        "entry_id": entry_id,
        "run_id": run_id,
        "run_fingerprint": (
            None if pd.isna(run_fingerprint) else str(run_fingerprint)
        ),
        "prompt": prompt,
        "trace_index": trace_index,
        "agent": normalized["agent_type"],
        "message": answer.get("message"),
        "score": normalized["evaluation_gt"],
        "evaluation_success": evaluation.get("success"),
        "best_of_n_score": best_of_n_evaluation.get("score"),
        "perplexity": normalized["perplexity"],
        "total_time": normalized["total_time"],
        "llm_time": normalized["llm_time"],
        "energy_consumed_kwh": normalized["energy_consumed_kwh"],
        "cpu_energy_kwh": normalized["cpu_energy_kwh"],
        "gpu_energy_kwh": normalized["gpu_energy_kwh"],
        "ram_energy_kwh": normalized["ram_energy_kwh"],
        "emissions_kg_co2": normalized["emissions_kg_co2"],
        "error": normalized["error"],
        "thinking": answer.get("thinking"),
        "generation_params_json": json.dumps(
            answer.get("generation_params"), ensure_ascii=False, default=str
        ),
        "logprobs_json": json.dumps(
            answer.get("logprobs", []), ensure_ascii=False, default=str
        ),
        "answer_json": json.dumps(answer, ensure_ascii=False, default=str),
        "global_total_time": global_profile.get("total_time"),
        "global_llm_time": global_profile.get("llm_time"),
        "agent_output_json": json.dumps(
            answer.get("agent_output", {}), ensure_ascii=False, default=str
        ),
        "agent_config_json": json.dumps(
            answer.get("agent_config", {}), ensure_ascii=False, default=str
        ),
        "changes_json": json.dumps(changes, ensure_ascii=False, sort_keys=True),
    }


def _runs_from_analysis(analysis_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build the compatibility run views from the unified analysis table."""
    runs: dict[str, pd.DataFrame] = {}
    if analysis_df.empty:
        return runs

    for run_name, run_df in analysis_df.groupby("run", sort=False):
        entries = []
        for entry_id, entry_df in run_df.groupby("entry_id", sort=False):
            entry_df = entry_df.sort_values("trace_index")
            answers = [
                {
                    "agent_type": row.agent,
                    "evaluation_gt": row.score,
                    "perplexity": row.perplexity,
                    "total_time": row.total_time,
                    "llm_time": row.llm_time,
                    "energy_consumed_kwh": row.energy_consumed_kwh,
                    "cpu_energy_kwh": row.cpu_energy_kwh,
                    "gpu_energy_kwh": row.gpu_energy_kwh,
                    "ram_energy_kwh": row.ram_energy_kwh,
                    "emissions_kg_co2": row.emissions_kg_co2,
                    "error": row.error,
                }
                for row in entry_df.itertuples(index=False)
            ]
            entries.append(
                {
                    "entry_id": entry_id,
                    "run_id": entry_df.iloc[0]["run_id"],
                    "trace": {"answers": answers},
                }
            )
        runs[str(run_name)] = pd.DataFrame(entries)
    return runs


def _summary_from_analysis(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the unified Parquet data into the former summary view."""
    summary_rows = []
    if analysis_df.empty:
        return pd.DataFrame(summary_rows, columns=["name", "metrics_by_agent"])

    metrics = {
        "evaluation_gt": "score",
        "perplexity": "perplexity",
        "total_time": "total_time",
        "llm_time": "llm_time",
    }
    for run_name, run_df in analysis_df.groupby("run", sort=False):
        metrics_by_agent: dict[str, dict[str, float]] = defaultdict(dict)
        for agent, agent_df in run_df.groupby("agent", sort=False):
            for summary_metric, column in metrics.items():
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
    grouped_traces = result.analysis_df.groupby(["run", "entry_id"], sort=False)
    for (run_name, raw_entry_id), trace_df in grouped_traces:
        entry_id = int(raw_entry_id)
        entry = entries_by_id.get(entry_id)
        if entry is None:
            continue

        actual_agents = trace_df.sort_values("trace_index")["agent"].astype(str).tolist()
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
    html_dir = out / "html"
    png_dir = out / "png"
    html_dir.mkdir(exist_ok=True)
    png_dir.mkdir(exist_ok=True)

    # Migrate plots generated by older analyzer versions into their new folders.
    for path in out.glob("*.html"):
        if path.name != "dashboard.html":
            path.replace(html_dir / path.name)
    for path in out.glob("*.png"):
        path.replace(png_dir / path.name)
    return out


def _flatten_traces(analysis_df: pd.DataFrame) -> list[dict]:
    """Return plot-ready rows from the canonical unified analysis table."""
    if analysis_df.empty:
        return []
    plot_df = analysis_df.rename(
        columns={
            "entry_id": "entry",
            "perplexity": "ppl",
            "energy_consumed_kwh": "energy",
            "emissions_kg_co2": "emissions",
        }
    )
    return plot_df[
        [
            "run",
            "entry",
            "agent",
            "score",
            "ppl",
            "total_time",
            "llm_time",
            "energy",
            "cpu_energy_kwh",
            "gpu_energy_kwh",
            "ram_energy_kwh",
            "emissions",
            "error",
        ]
    ].to_dict(orient="records")


_COLOR_SEQ = ["#0984e3", "#00b894", "#e17055", "#6c5ce7", "#fdcb6e", "#d63031"]


def _save_fig(fig: go.Figure, path: Path) -> None:
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


def plot_per_agent_scores(
    result: BenchmarkResult, save: bool = True
) -> go.Figure | None:
    records = _flatten_traces(result.analysis_df)
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
    records = _flatten_traces(result.analysis_df)
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
    records = _flatten_traces(result.analysis_df)
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
    records = _flatten_traces(result.analysis_df)
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
    records = _flatten_traces(result.analysis_df)
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
    records = _flatten_traces(result.analysis_df)
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
        entry_rows = result.analysis_df[
            result.analysis_df["entry_id"] == entry.id
        ]
        for _, trace_df in entry_rows.groupby("run", sort=False):
            actual = (
                trace_df.sort_values("trace_index")["agent"].astype(str).tolist()
            )
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
    analysis_df = result.analysis_df.dropna(subset=["score"])
    if analysis_df["run"].nunique() < 2:
        return None

    plot_df = (
        analysis_df.groupby(["run", "agent"], as_index=False)["score"]
        .mean()
    )

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
    records = _flatten_traces(result.analysis_df)
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
    records = _flatten_traces(result.analysis_df)
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
    records = _flatten_traces(result.analysis_df)
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
    records = _flatten_traces(result.analysis_df)
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
    for (run_name, entry_id), trace_df in result.analysis_df.groupby(
        ["run", "entry_id"], sort=False
    ):
        expected = expected_by_id.get(int(entry_id))
        if expected is None:
            continue
        actual = trace_df.sort_values("trace_index")["agent"].astype(str).tolist()
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
    for _, trace_df in result.analysis_df.groupby(
        ["run", "entry_id"], sort=False
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
    if save:
        _save_fig(fig, _output_dir(result) / "trace_transitions.html")
    return fig


def plot_error_rate(result: BenchmarkResult, save: bool = True) -> go.Figure | None:
    records = _flatten_traces(result.analysis_df)
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
    console.print(
        f"\n[bold cyan]Generated plots[/bold cyan]  "
        f"[dim]{out_dir / 'html'} · {out_dir / 'png'}[/dim]"
    )

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
            png_path = out_dir / "png" / f"{name}.png"
            if png_path.exists():
                console.print(f"  [green]✓[/green] html/{name}.html + png/{name}.png")
            else:
                console.print(
                    f"  [green]✓[/green] html/{name}.html "
                    "[dim](PNG skipped: image renderer unavailable)[/dim]"
                )
        else:
            console.print(f"  [dim]– html/{name}.html (skipped: no data)[/dim]")

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
