from pathlib import Path

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.pretty import Pretty
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from arco.core.answer import Answer


def _artifact_link(path: str) -> Text:
    """Render a local artifact path as a terminal-clickable link."""
    artifact = Path(str(path)).expanduser()
    uri = artifact.resolve().as_uri()
    return Text(str(artifact), style=f"underline cyan link {uri}")


def _verbose_output_value(key: str, value) -> object:
    """Make large structured agent outputs readable in the verbose view."""
    if key == "data_df" and hasattr(value, "shape") and hasattr(value, "columns"):
        rows, columns = value.shape
        return f"<DataFrame: {rows} rows × {columns} columns: {list(value.columns)}>"
    if isinstance(value, str) and len(value) > 1600:
        return value[:1600] + "\n… <truncated; see stored output>"
    return value


def _verbose_output(output: dict) -> dict:
    return {key: _verbose_output_value(key, value) for key, value in output.items()}


def _sampling_label(answer: Answer) -> str:
    """Return the sampling settings actually used for this answer.

    Best-of-N candidates store their concrete values because their shared
    ``AgentConfig`` only contains the configured range, not the value selected
    for an individual candidate. Older persisted answers fall back to the
    first configured candidate and are explicitly labelled as configured.
    """
    params = answer.generation_params
    configured = False
    if params is None:
        params = answer.agent_config.get_candidate_params()[0]
        configured = answer.agent_config.n > 1
    temperature, top_p, top_k = params
    sampling = f"temperature={temperature}"
    if top_p is not None:
        sampling += f", top_p={top_p}"
    if top_k is not None:
        sampling += f", top_k={top_k}"
    if configured:
        sampling += " (configured)"
    return sampling


def _verbose_metrics_table(answer: Answer) -> Table:
    config = answer.agent_config
    sampling = _sampling_label(answer)

    evaluation = (
        f"{answer.evaluation.score:.3f}" if answer.evaluation is not None else "-"
    )
    gt_evaluation = (
        f"{answer.gt_evaluation.score:.3f}" if answer.gt_evaluation is not None else "-"
    )
    perplexity = f"{answer.perplexity:.3f}" if answer.perplexity is not None else "-"
    profiling = answer.profiling_data

    table = Table.grid(padding=(0, 2), expand=True)
    table.add_column(style="dim", no_wrap=True)
    table.add_column()
    table.add_column(style="dim", no_wrap=True)
    table.add_column()
    table.add_row("Provider", str(config.provider), "Model", str(config.model))
    table.add_row("Sampling", sampling, "Best-of-N", str(config.n))
    table.add_row(
        "Options",
        (
            f"reasoning={config.enable_reasoning}, "
            f"effort={config.reasoning_effort or 'default'}, "
            f"summary={config.reasoning_summary or 'default'}, "
            f"logprobs={config.enable_logprobs}"
        ),
        "Refinement",
        str(config.iterative_refinement_n),
    )
    table.add_row("Evaluation", evaluation, "Ground truth", gt_evaluation)
    table.add_row("Perplexity", perplexity, "Budget", answer.budget_controller_choice)
    table.add_row(
        "Time",
        f"{profiling.total_time:.2f}s" if profiling.total_time is not None else "-",
        "LLM time",
        f"{profiling.llm_time:.2f}s" if profiling.llm_time is not None else "-",
    )
    table.add_row(
        "Energy",
        f"{profiling.energy_consumed_kwh:.6f} kWh"
        if profiling.energy_consumed_kwh is not None
        else "-",
        "CO₂",
        f"{profiling.emissions_kg_co2:.6f} kg"
        if profiling.emissions_kg_co2 is not None
        else "-",
    )
    return table


def _verbose_token_summary(answer: Answer) -> Text:
    parts = []
    if answer.input_token_count is not None:
        parts.append(f"Input {answer.input_token_count} tokens")
    if answer.output_token_count is not None:
        parts.append(f"Output {answer.output_token_count} tokens")
        llm_time = answer.profiling_data.llm_time
        if llm_time is not None and llm_time > 0:
            tokens_per_second = answer.output_token_count / llm_time
            parts.append(f"{tokens_per_second:.1f} output tokens/s")
    total_tokens = answer.total_token_count
    if (
        total_tokens is None
        and answer.input_token_count is not None
        and answer.output_token_count is not None
    ):
        total_tokens = answer.input_token_count + answer.output_token_count
    if total_tokens is not None:
        parts.append(f"Total {total_tokens} tokens")
    if answer.cache_read_token_count is not None:
        parts.append(f"Cache read {answer.cache_read_token_count} tokens")
    if answer.cache_creation_token_count is not None:
        parts.append(f"Cache creation {answer.cache_creation_token_count} tokens")
    if answer.reasoning_token_count is not None:
        parts.append(f"Reasoning {answer.reasoning_token_count} tokens")
    if answer.logprobs:
        numeric = [float(logprob) for _, logprob in answer.logprobs]
        average = sum(numeric) / len(numeric)
        parts.append(f"Average logprob {average:.4f}")
    if answer.perplexity is not None:
        parts.append(f"Perplexity {answer.perplexity:.3f}")
    if not parts:
        return Text("No token usage or log-probability data returned.", style="dim")
    return Text("  ·  ".join(parts))


def render_answer_verbose(
    answer: Answer,
    *,
    include_message: bool = True,
    include_agent: bool = True,
) -> RenderableType:
    """Render a readable, information-dense answer card for ``--verbose``.

    ``include_message=False`` is used when the answer was already displayed
    through token streaming.
    """
    sections = [
        Markdown(answer.message or "No summary returned.")
        if include_message
        else Text("Output streamed above.", style="dim"),
        Rule("Run details", style="dim"),
        _verbose_metrics_table(answer),
        Rule("Agent output", style="dim"),
        Pretty(
            _verbose_output(answer.agent_output),
            max_depth=4,
            max_length=6,
            indent_size=2,
        ),
        Rule("Token statistics", style="dim"),
        _verbose_token_summary(answer),
    ]

    image_path = answer.agent_output.get("image_path")
    if image_path:
        sections.extend(
            [
                Rule("Stored visualization", style="dim"),
                _artifact_link(image_path),
            ]
        )

    if answer.thinking:
        sections.extend(
            [
                Rule("Reasoning summary", style="magenta"),
                Text(answer.thinking, style="magenta"),
            ]
        )

    if answer.error:
        sections.extend(
            [
                Rule("Error", style="red"),
                Text(answer.error, style="red"),
            ]
        )

    if answer.discarded_bon_answers:
        discarded = Table.grid(padding=(0, 2), expand=True)
        discarded.add_column(style="dim", no_wrap=True)
        discarded.add_column(ratio=4)
        discarded.add_column(style="dim", no_wrap=True)
        discarded.add_column(ratio=2)
        for index, candidate in enumerate(answer.discarded_bon_answers, start=1):
            evaluation = (
                f"{candidate.evaluation.score:.3f}"
                if candidate.evaluation is not None
                else "-"
            )
            summary = candidate.message or "No summary returned."
            if candidate.error:
                summary += f" ({candidate.error})"
            discarded.add_row(
                f"Discarded {index}",
                summary,
                f"Evaluation: {evaluation}",
                f"Sampling: {_sampling_label(candidate)}",
            )
        sections.extend([Rule("Discarded candidates", style="dim"), discarded])

    return Group(
        Text("\n"),
        Panel(
            Group(*sections),
            title=(f"[bold cyan]{answer.agent_id}[/bold cyan]" if include_agent else None),
            subtitle_align="right",
            border_style="red" if answer.error else "cyan",
            padding=(1, 2),
            expand=True,
        ),
    )


def render_streamed_output(content: str) -> RenderableType:
    """Render completed token output so it remains visible after Live exits."""
    return Markdown(content or "No output returned.")


def render_answer(
    answer: Answer,
    *,
    include_message: bool = True,
    include_agent: bool = True,
) -> RenderableType:
    """Render the normal run transcript without wrapping it in a panel.

    ``include_message=False`` prevents an answer from being displayed twice
    after its content has already been streamed token by token.
    """
    renderables: list[RenderableType] = []
    if include_agent:
        if answer.error:
            icon = Text("✗ ", style="bold red")
        else:
            icon = Text("✓ ", style="bold green")
        renderables.append(icon + Text(str(answer.agent_id), style="bold cyan"))

    if include_message and answer.message:
        renderables.extend([Text("  "), Markdown(answer.message)])

    image_path = answer.agent_output.get("image_path")
    if image_path:
        renderables.extend(
            [Text("\n  image: ", style="dim") + _artifact_link(image_path)]
        )
    if answer.error:
        renderables.append(Text(f"`\n  ({answer.error})", style="red"))

    return Group(*renderables)
