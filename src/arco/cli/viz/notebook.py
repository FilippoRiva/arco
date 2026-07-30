"""Notebook-compatible workflow visualizer.

Replaces the Rich-based terminal display with ``IPython.display`` primitives
that work in Jupyter notebooks.

Usage in a notebook cell::

    from arco.cli.viz.notebook import display_workflow_notebook

    state = display_workflow_notebook(executor.stream())
    # or with verbose metrics:
    state = display_workflow_notebook(executor.stream(), verbose=True)
"""

from __future__ import annotations

import io
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arco.core import Answer


# ── helpers ──────────────────────────────────────────────────────────────


def _badge(name: str) -> str:
    colors = {
        "planner": "#6c5ce7",
        "orchestrator": "#6c5ce7",
        "retriever": "#0984e3",
        "analyzer": "#00b894",
        "visualizer": "#e17055",
    }
    bg = colors.get(name.lower(), "#636e72")
    return (
        f"<span style='background:{bg};color:#fff;padding:1px 10px;"
        f"border-radius:4px;font-weight:600;font-size:0.9em'>{name}</span>"
    )


def _answer_html(answer: Answer, verbose: bool) -> str:
    conf = answer.agent_config
    parts = [answer.message]

    if verbose:
        bits = []
        if conf.n > 1:
            # idx = {"temperature": 0, "top_p": 1, "top_k": 2}[conf.bon_parameter]
            # vals = [p[idx] for p in conf.get_candidate_params()]
            bits.append(f"Best-of-{conf.n} ({conf.bon_parameter})")
        else:
            t = conf.get_candidate_params()[0][0]
            bits.append(f"Temp={t:.3f}")

        if answer.evaluation is not None:
            bits.append(f"Eval={answer.evaluation.score:.2f}")
        if answer.gt_evaluation is not None:
            bits.append(f"GT={answer.gt_evaluation.score:.2f}")
        if answer.perplexity is not None:
            bits.append(f"PPL={answer.perplexity:.4f}")
        if conf.cot_n > 1:
            bits.append(f"CoT={conf.cot_n}")

        if bits:
            meta = " · ".join(bits)
            parts.append(
                f"<div style='color:#636e72;font-size:0.85em;margin-top:2px'>{meta}</div>"
            )

    if answer.error:
        parts.append(
            f"<div style='color:#d63031;margin-top:4px'>⚠ {answer.error}</div>"
        )

    body = "".join(parts)
    return (
        f"<div style='margin:6px 0;padding:8px 12px;border-left:4px solid #dfe6e9;"
        f"background:#2d3436;border-radius:0 6px 6px 0;color:#dfe6e9'>"
        f"<div style='margin-bottom:4px'>{_badge(answer.agent_id)}</div>"
        f"{body}</div>"
    )


def _render_chart(retriever_answer: Answer, visualizer_answer: Answer) -> None:
    from IPython.display import Image as IPyImage
    from IPython.display import display as ipy_display

    df = retriever_answer.agent_output.get("data_df")
    chart_config = visualizer_answer.agent_output.get("chart_config")
    code = visualizer_answer.agent_output.get("code")
    if df is None or not chart_config or not code:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    buf = io.BytesIO()
    namespace = {
        "data_df": df,
        "config": chart_config,
        "plt": plt,
        "pd": pd,
        "np": np,
        "buf": buf,
    }
    modified = code.replace(
        "plt.show()",
        "plt.savefig(buf, format='png', dpi=100, bbox_inches='tight'); plt.close()",
    )
    try:
        exec(modified, namespace)  # noqa: S102
    except Exception:  # noqa: BLE001
        return

    buf.seek(0)
    ipy_display(IPyImage(buf.read()))


# ── public API ───────────────────────────────────────────────────────────


def display_workflow_notebook(
    events: Generator[dict[str, Any]], verbose: bool = False
) -> Any:
    """Run a workflow inside a notebook with inline progress and charts.

    Parameters
    ----------
    events:
        The generator yielded by ``WorkflowExecutor.stream()``.
    verbose:
        Show detailed metrics per agent step.

    Returns
    -------
    The final :class:`~arco.core.State` or ``None`` on failure.
    """
    from IPython.display import HTML, clear_output
    from IPython.display import display as ipy_display

    last_state: Any = None
    last_retriever_answer: Any = None
    last_visualizer_answer: Any = None
    answers_rendered: list[str] = []
    progress: list[str] = []

    for update in events:
        event_type = update["event"]

        if event_type == "check_connection":
            models = update.get("models", [])
            progress.append(f"⏳ Checking {len(models)} model(s)…")
        elif event_type == "started":
            progress.append(f"🚀 Run `{update.get('run_id', '')}`")
        elif event_type == "node_started":
            progress.append(f"▶ **{update['node']}**")
        elif event_type == "node_finished":
            last_state = update["state"]
            answer: Answer = last_state.get_last_answer()
            agent = answer
            progress.append(f"✅ **{agent}**")

            # Render answer HTML
            answers_rendered.append(_answer_html(answer, verbose=verbose))

            # Hold on to retriever + visualizer answers for chart rendering
            if agent.lower() == "retriever":
                last_retriever_answer = answer
            elif agent.lower() == "visualizer":
                last_visualizer_answer = answer
        elif event_type == "completed":
            t = update["state"].global_profiling_data.total_time
            ts = f"{t:.2f}s" if t is not None else "?"
            progress.append(f"✅ Completed — total time {ts}")
        elif event_type == "error":
            progress.append(f"❌ {update.get('message', '?')}")

    # ── Render final output ──
    clear_output(wait=True)

    # 1) Answer panels
    for html in answers_rendered:
        ipy_display(HTML(html))

    # 2) Chart (render after its answer panel)
    if last_retriever_answer is not None and last_visualizer_answer is not None:
        _render_chart(last_retriever_answer, last_visualizer_answer)

    # 3) Progress summary
    if progress:
        md = "\n\n".join(p for p in progress if p != "---")
        from IPython.display import Markdown

        ipy_display(Markdown(md))

    return last_state
