"""HTML dashboard assembly for benchmark analysis."""

from __future__ import annotations

import json
import math
import os
import re
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
  .output-visualizer-link {{ margin: -8px 0 20px; font-size: 0.9rem; }}
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
  <nav class="output-visualizer-link"><a href="Output%20Visualizer.html">Open Output Visualizer →</a></nav>
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


_OUTPUT_VISUALIZER_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Output Visualizer — __TITLE__</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; padding: 24px; background: #f5f6fa; color: #2d3436;
         font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
  h1 { font-size: 1.55rem; margin: 0 0 6px; }
  h2 { font-size: 1.1rem; margin: 0 0 12px; }
  h3 { font-size: .98rem; margin: 0 0 8px; }
  a { color: #0876bd; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .topline { display: flex; justify-content: space-between; align-items: baseline;
             gap: 16px; margin-bottom: 20px; }
  .controls { display: grid; grid-template-columns: minmax(190px, 1fr) minmax(130px, .65fr) minmax(0, 2.5fr);
              align-items: start; gap: 16px; padding: 16px; background: #fff;
              border-radius: 8px; margin-bottom: 16px; }
  label { display: grid; gap: 5px; color: #636e72; font-size: .83rem; font-weight: 600; }
  select { width: 100%; min-width: 0; padding: 8px 10px; border: 1px solid #cbd2d7;
           border-radius: 5px; background: white; color: #2d3436; }
  .control-prompt { min-width: 0; padding-left: 16px; border-left: 1px solid #e5e9ec; }
  .control-prompt-title { color: #636e72; font-size: .78rem; font-weight: 600; margin-bottom: 6px; }
  .control-prompt .prompt-text { max-height: 150px; overflow: auto; }
  .control-prompt .meta { margin: 8px 0 0; }
  .prompt-text { white-space: pre-wrap; line-height: 1.5; }
  .meta { color: #636e72; font-size: .85rem; margin: 0 0 10px; }
  .comparison-headings, .step-columns { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; }
  .comparison-headings { padding: 0 14px; align-items: end; }
  .comparison-headings h2 { margin-bottom: 8px; }
  .trace-summary { margin: 4px 0 10px; }
  .trace-step { background: #fff; border-radius: 8px; margin-bottom: 12px;
                box-shadow: 0 1px 4px rgba(0,0,0,.06); overflow: hidden; }
  .trace-step-heading { display: flex; align-items: center; justify-content: space-between;
                        flex-wrap: wrap; gap: 8px; padding: 10px 14px;
                        background: #f9fafb; border-bottom: 1px solid #e5e9ec; }
  .step-columns { gap: 0; }
  .step-side { min-width: 0; padding: 14px; }
  .step-side + .step-side { border-left: 1px solid #e5e9ec; }
  .step-side-title { display: flex; align-items: center; justify-content: space-between;
                     gap: 8px; font-weight: 700; color: #345b6a; margin-bottom: 8px; }
  .error-badge { color: #b42318; background: #fef3f2; border-radius: 4px;
                 padding: 2px 6px; font-size: .7rem; }
  .trace-status { display: inline-flex; align-items: center; gap: 7px;
                  font-size: .8rem; font-weight: 600; }
  .trace-dot { width: 10px; height: 10px; display: inline-block; border-radius: 50%; flex: none; }
  .trace-status.match .trace-dot { background: #18864b; }
  .trace-status.mismatch .trace-dot { background: #d92d20; }
  .trace-status.match { color: #146c3a; }
  .trace-status.mismatch { color: #b42318; }
  .agent { font-weight: 700; color: #345b6a; }
  .error { color: #b42318; white-space: pre-wrap; }
  .message { white-space: pre-wrap; line-height: 1.45; margin: 6px 0; }
  .json-label { color: #636e72; font-size: .78rem; font-weight: 600; margin: 8px 0 4px; }
  pre { margin: 0; padding: 10px; background: #f5f6fa; border-radius: 5px;
        white-space: pre-wrap; overflow-wrap: anywhere; font-size: .8rem; line-height: 1.4;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .json-key { color: #7c3aed; font-weight: 600; }
  .json-string { color: #067647; }
  .json-number { color: #b54708; }
  .json-boolean { color: #175cd3; font-weight: 600; }
  .json-null { color: #667085; font-style: italic; }
  details { margin-top: 14px; background: #fff; border-radius: 8px; padding: 14px; }
  summary { cursor: pointer; color: #485460; font-weight: 600; }
  .empty { color: #636e72; font-style: italic; }
  @media (max-width: 850px) {
    body { padding: 14px; }
    .controls { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
    .control-prompt { grid-column: 1 / -1; padding: 12px 0 0; border-left: 0;
                      border-top: 1px solid #e5e9ec; }
    .comparison-headings, .step-columns { grid-template-columns: 1fr; gap: 0; }
    .comparison-headings h2:nth-child(2) { display: none; }
    .step-side + .step-side { border-left: 0; border-top: 1px solid #e5e9ec; }
  }
  @media (max-width: 520px) {
    .controls { grid-template-columns: 1fr; }
    .control-prompt { grid-column: auto; }
  }
</style>
</head>
<body>
  <div class="topline">
    <div><h1>Output Visualizer</h1><div class="meta">Benchmark: __TITLE__</div></div>
    <a href="dashboard.html">← Back to benchmark dashboard</a>
  </div>
  <div class="controls">
    <label>Run<select id="run-select"></select></label>
    <label>Entry ID<select id="entry-select"></select></label>
    <div class="control-prompt">
      <div class="control-prompt-title">Prompt</div>
      <div id="prompt" class="prompt-text"></div>
      <div id="entry-meta" class="meta"></div>
    </div>
  </div>
  <div class="comparison-headings">
    <div><h2>Selected run state and outputs</h2><div id="run-meta" class="meta"></div></div>
    <h2>Expected ground-truth trace</h2>
  </div>
  <div id="trace-summary" class="trace-summary"></div>
  <div id="trace-comparison"></div>
  <div id="state-details"></div>
  <script id="comparison-data" type="application/json">__DATA__</script>
<script>
  const data = JSON.parse(document.getElementById('comparison-data').textContent);
  const runSelect = document.getElementById('run-select');
  const entrySelect = document.getElementById('entry-select');
  const promptNode = document.getElementById('prompt');
  const entryMetaNode = document.getElementById('entry-meta');
  const runMetaNode = document.getElementById('run-meta');
  const traceSummaryNode = document.getElementById('trace-summary');
  const traceComparisonNode = document.getElementById('trace-comparison');
  const stateDetailsNode = document.getElementById('state-details');

  function node(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }
  function addJson(parent, label, value) {
    parent.appendChild(node('div', 'json-label', label));
    const source = JSON.stringify(value ?? null, null, 2);
    const pre = node('pre', 'json-code');
    const tokenPattern = /"(?:\\\\.|[^"\\\\])*"(?=\\s*:)|"(?:\\\\.|[^"\\\\])*"|-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?|\\b(?:true|false|null)\\b/g;
    let previousIndex = 0;
    for (const match of source.matchAll(tokenPattern)) {
      const token = match[0];
      const tokenIndex = match.index;
      pre.appendChild(document.createTextNode(source.slice(previousIndex, tokenIndex)));
      let className = 'json-number';
      if (token.startsWith('"')) {
        className = /^\\s*:/.test(source.slice(tokenIndex + token.length))
          ? 'json-key'
          : 'json-string';
      } else if (token === 'true' || token === 'false') {
        className = 'json-boolean';
      } else if (token === 'null') {
        className = 'json-null';
      }
      pre.appendChild(node('span', className, token));
      previousIndex = tokenIndex + token.length;
    }
    pre.appendChild(document.createTextNode(source.slice(previousIndex)));
    parent.appendChild(pre);
  }
  function fillSelect(select, values, emptyLabel) {
    select.replaceChildren();
    if (!values.length) {
      const option = node('option', '', emptyLabel);
      option.value = '';
      select.appendChild(option);
      select.disabled = true;
      return;
    }
    select.disabled = false;
    for (const value of values) {
      const option = node('option', '', value);
      option.value = value;
      select.appendChild(option);
    }
  }
  function traceMatches(state, entry) {
    if (!state || !entry) return null;
    const expected = Array.isArray(entry.trace) ? entry.trace : [];
    const answers = Array.isArray(state.answers) ? state.answers : [];
    return expected.length === answers.length
      && expected.every((step, index) => step?.agent_type === answers[index]?.agent_id);
  }
  function refreshEntryOptions() {
    const entryIds = Object.keys(data.entries);
    const previousEntry = entrySelect.value;
    if (!entryIds.length) {
      fillSelect(entrySelect, [], 'No dataset entries available');
      return;
    }
    entrySelect.replaceChildren();
    entrySelect.disabled = false;
    for (const entryId of entryIds) {
      const entry = data.entries[entryId];
      const state = (data.runs[runSelect.value] || {})[entryId];
      const matches = traceMatches(state, entry);
      const option = node('option', '', `${entry.id}${matches === false ? ' 🔴' : ''}`);
      option.value = entryId;
      option.title = matches === false
        ? 'Off the expected agent trace'
        : matches === true
          ? 'Matches the expected agent trace'
          : 'No run state available';
      entrySelect.appendChild(option);
    }
    if (entryIds.includes(previousEntry)) entrySelect.value = previousEntry;
  }
  function renderAnswer(answer) {
    const side = node('div', 'step-side actual-side');
    if (!answer) {
      side.appendChild(node('p', 'empty', 'No answer at this trace position.'));
      return side;
    }
    const title = node('div', 'step-side-title');
    title.appendChild(node('span', '', `Actual · ${answer.agent_id || 'Unknown agent'}`));
    if (answer.error) {
      const errorBadge = node('span', 'error-badge', 'Error');
      errorBadge.title = answer.error;
      title.appendChild(errorBadge);
    }
    side.appendChild(title);
    addJson(side, 'Agent output', answer.agent_output);
    addJson(side, 'Ground-truth evaluation', answer.gt_evaluation);
    side.appendChild(node('div', 'json-label', 'Agent message'));
    side.appendChild(node('div', 'message', answer.message || 'No agent message.'));
    addJson(side, 'Profiling', answer.profiling_data || {});
    return side;
  }

  function renderExpected(expected) {
    const side = node('div', 'step-side expected-side');
    if (!expected) {
      side.appendChild(node('p', 'empty', 'No ground-truth step at this position.'));
      return side;
    }
    side.appendChild(node('div', 'step-side-title', `Expected · ${expected.agent_type || 'Unknown agent'}`));
    addJson(side, 'Expected data', expected.data);
    return side;
  }

  function render() {
    const runName = runSelect.value;
    const entryId = entrySelect.value;
    const entry = data.entries[entryId];
    const state = (data.runs[runName] || {})[entryId];
    promptNode.textContent = (entry && entry.prompt) || (state && state.prompt) || '—';
    entryMetaNode.textContent = entry
      ? `Entry ${entry.id} · Difficulty ${entry.difficulty}`
      : `Entry ${entryId || '—'} · Ground-truth dataset entry unavailable`;
    runMetaNode.textContent = state
      ? `Run ID: ${state.run_id || '—'}`
      : 'No completed run state for this selection';

    const expectedSteps = entry && Array.isArray(entry.trace) ? entry.trace : [];
    const answers = state && Array.isArray(state.answers) ? state.answers : [];
    const stepCount = Math.max(expectedSteps.length, answers.length);
    let matchedSteps = 0;
    traceComparisonNode.replaceChildren();
    traceSummaryNode.replaceChildren();
    stateDetailsNode.replaceChildren();

    if (stepCount === 0) {
      traceSummaryNode.appendChild(node('p', 'empty', 'There are no trace steps to compare.'));
    }
    for (let index = 0; index < stepCount; index += 1) {
      const expected = expectedSteps[index];
      const answer = answers[index];
      const matched = Boolean(expected && answer && expected.agent_type === answer.agent_id);
      if (matched) matchedSteps += 1;

      let statusText;
      let statusClass;
      if (matched) {
        statusText = 'Agent matches expected trace';
        statusClass = 'match';
      } else if (expected && answer) {
        statusText = `Agent mismatch · expected ${expected.agent_type}, got ${answer.agent_id}`;
        statusClass = 'mismatch';
      } else if (expected) {
        statusText = `Missing answer · expected ${expected.agent_type}`;
        statusClass = 'mismatch';
      } else {
        statusText = `Unexpected answer · ${answer.agent_id}`;
        statusClass = 'mismatch';
      }

      const step = node('article', 'trace-step');
      const heading = node('div', 'trace-step-heading');
      heading.appendChild(node('strong', '', `Step ${index + 1}`));
      const status = node('span', `trace-status ${statusClass}`);
      status.appendChild(node('span', 'trace-dot'));
      status.appendChild(node('span', '', statusText));
      heading.appendChild(status);
      step.appendChild(heading);
      const columns = node('div', 'step-columns');
      columns.appendChild(renderAnswer(answer));
      columns.appendChild(renderExpected(expected));
      step.appendChild(columns);
      traceComparisonNode.appendChild(step);
    }

    if (stepCount > 0) {
      const exactMatch = traceMatches(state, entry) === true;
      const summary = node('span', `trace-status ${exactMatch ? 'match' : 'mismatch'}`);
      summary.appendChild(node('span', 'trace-dot'));
      summary.appendChild(node(
        'span',
        '',
        exactMatch
          ? `Exact agent trace · ${matchedSteps} steps`
          : `Trace differs · ${matchedSteps}/${expectedSteps.length} expected steps aligned; ${answers.length} actual steps`,
      ));
      traceSummaryNode.appendChild(summary);
    }

    if (state) {
      const fullState = document.createElement('details');
      fullState.appendChild(node('summary', '', 'Full serialized run state'));
      addJson(fullState, 'State JSON', state);
      stateDetailsNode.appendChild(fullState);
    }
  }

  fillSelect(runSelect, Object.keys(data.runs), 'No runs available');
  refreshEntryOptions();
  runSelect.addEventListener('change', () => {
    refreshEntryOptions();
    render();
  });
  entrySelect.addEventListener('change', render);
  render();
</script>
</body>
</html>"""


def _output_visualizer_data(result: BenchmarkResult) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    if result.benchmark_dataset is not None:
        for entry in result.benchmark_dataset.entries:
            entries[str(entry.id)] = {
                "id": entry.id,
                "prompt": entry.prompt,
                "difficulty": entry.difficulty,
                "trace": entry.trace.to_dict(),
            }

    runs: dict[str, dict[str, Any]] = {str(run_name): {} for run_name in result._states}
    identities = result.answer_analysis_df
    required = {"run_name", "entry_id", "run_id"}
    if not identities.empty and required.issubset(identities.columns):
        records = (
            identities[["run_name", "entry_id", "run_id"]]
            .dropna()
            .drop_duplicates()
            .to_dict("records")
        )
        for record in records:
            run_name = str(record["run_name"])
            entry_id = str(int(record["entry_id"]))
            run_id = str(record["run_id"])
            state = result._states.get(run_name, {}).get(run_id)
            if state is not None:
                runs.setdefault(run_name, {})[entry_id] = state.to_dict()

    return {"entries": entries, "runs": runs}


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(nested) for nested in value]
    return value


def _write_output_visualizer(result: BenchmarkResult, title: str) -> Path:
    analysis_dir = result.benchmark_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        _json_safe(_output_visualizer_data(result)),
        ensure_ascii=False,
        default=str,
        allow_nan=False,
    ).replace("<", "\\u003c")
    replacements = {"TITLE": escape(title), "DATA": data}
    page = re.sub(
        r"__(TITLE|DATA)__",
        lambda match: replacements[match.group(1)],
        _OUTPUT_VISUALIZER_TEMPLATE,
    )
    path = analysis_dir / "Output Visualizer.html"
    path.write_text(page, encoding="utf-8")
    return path


def build_dashboard(result: BenchmarkResult) -> str:
    meta = result.metadata
    title = Path(meta["benchmark_run"]).name
    _write_output_visualizer(result, title)
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
