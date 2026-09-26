"""HTML dashboard assembly for benchmark analysis."""

from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from .benchmark_result import BenchmarkResult, _load_json_dict
from .plots import (
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

_AGENT_LABEL_COLORS = ["#3f6b7a", "#5e7651", "#8a6744", "#725e83"]


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
