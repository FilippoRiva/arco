import math

from langchain_community.chat_models import perplexity
from rich.console import Group
from rich.panel import Panel
from rich.pretty import Pretty
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from arco.core import Answer


def _format_answer_subtitle(answer: Answer) -> str:
    conf = answer.agent_config
    subtitle_elements = []
    if conf.n > 1:
        idx = {"temperature": 0, "top_p": 1, "top_k": 2}[conf.bon_parameter]
        varying_vals = [p[idx] for p in conf.get_candidate_params()]
        subtitle_elements.append(f"Best-of-{conf.n} on {conf.bon_parameter}({varying_vals})")
        if answer.evaluation and answer.evaluation.success:
            subtitle_elements.append(f"Eval : {round(answer.evaluation.score, 3)}")
        else:
            subtitle_elements.append("Eval : none")
    else:
        subtitle_elements.append(f"Temp: {round(answer.agent_config.get_candidate_params()[0][0], 3)}")
    if answer.gt_evaluation and answer.gt_evaluation.success:
        subtitle_elements.append(f"GT-Eval : {round(answer.gt_evaluation.score, 3)}")
    if conf.cot_n > 1: subtitle_elements.append(f"CoT : {conf.cot_n}")
    return "[dim]" + ", ".join(subtitle_elements) + "[/dim]"


def _render_discarded_answer_panel(answer: Answer) -> Panel:
    return Panel(
        renderable=answer.message,
        title=f"[dim cyan]Discarded[/dim cyan]",
        subtitle=_format_answer_subtitle(answer),
        subtitle_align="right",
        border_style="dim",
        expand=True,
    )


def render_answer(answer: Answer, verbose: bool) -> Panel:
    # Build the main panel
    group_elements = [answer.message]
    if verbose and answer.agent_config:
        ## Mandatory Subpanels
        # generation_and_eval info
        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan")
        table.add_column(justify="right")
        rows = [
            ("Ground-Truth eval", f"{answer.gt_evaluation.score:.2f}" if answer.gt_evaluation is not None else "-"),
            ("Best-of-N eval", f"{answer.evaluation.score:.2f}" if answer.evaluation is not None else "-"),
            ("Perplexity", f"{answer.perplexity:.4f}" if answer.perplexity is not None else "-"),
            ("BC choice", f"{answer.budget_controller_choice}" if answer.perplexity is not None else "-")
        ]
        for k, v in rows:
            table.add_row(k, v)
        generation_and_eval_info = Panel(
            table,
            title="[dim]Generation and Evaluation Info[/dim]",
            title_align="left",
            border_style="dim",
            expand=False
        )

        # profiling_data
        p = answer.profiling_data
        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan")
        table.add_column(justify="right")
        rows = [
            ("Total time", f"{p.total_time:.2f} s" if p.total_time is not None else "-"),
            ("LLM time", f"{p.llm_time:.2f} s" if p.llm_time is not None else "-"),
            ("Energy", f"{p.energy_consumed_kwh:.6f} kWh" if p.energy_consumed_kwh is not None else "-"),
            ("CPU", f"{p.cpu_energy_kwh:.6f} kWh" if p.cpu_energy_kwh is not None else "-"),
            ("GPU", f"{p.gpu_energy_kwh:.6f} kWh" if p.gpu_energy_kwh is not None else "-"),
            ("RAM", f"{p.ram_energy_kwh:.6f} kWh" if p.ram_energy_kwh is not None else "-"),
            ("CO₂", f"{p.emissions_kg_co2:.6f} kg" if p.emissions_kg_co2 is not None else "-"),
        ]
        for k, v in rows:
            table.add_row(k, v)
        profiling_subpanel = Panel(
            table,
            title="[dim]Profiling[/dim]",
            title_align="left",
            border_style="dim",
            expand=False
        )

        # perplexity
        perplexity_text = Text()

        if answer.logprobs:
            for token, logprob in answer.logprobs:
                try:
                    token_ppl = math.exp(-logprob)
                except OverflowError:
                    token_ppl = float('inf')
                if token_ppl < 1.2:
                    style = "bold green"
                elif token_ppl < 5.0:
                    style = "yellow"
                else:
                    style = "bold red"
                perplexity_text.append(token, style=style)
        else:
            perplexity_text = "-"

        perplexity_subpanel = Panel(
            perplexity_text,
            title="[dim]Token Perplexity Analysis[/dim]",
            title_align="left",
            subtitle="[bold green]■ <1.2 (High)[/bold green] [yellow]■ <5.0 (Mid)[/yellow] [bold red]■ ≥5.0 (Low Confidence)[/bold red]",
            subtitle_align="right",
            border_style="dim",
            expand=False
        )

        # config panel
        config_subpanel = Panel(
            Pretty(answer.agent_config, max_length=2, max_depth=2, indent_size=2),
            title="[dim]Config[/dim]",
            title_align="left",
            border_style="dim",
            expand=False
        )

        group_elements += [
            generation_and_eval_info,
            profiling_subpanel,
            perplexity_subpanel,
            config_subpanel,
        ]

        ## Optional subpanels
        # error messages
        if answer.error:
            error_subpanel = Panel(
                answer.error,
                title="[red]Error Message[/red]",
                title_align="left",
                border_style="red",
                expand=False
            )
            group_elements += [error_subpanel]
        content = Group(*group_elements)

        # Create a subpanel for discarded answers
        if answer.agent_config.n > 1:
            group_elements.append(Panel(
                Group(_render_discarded_answer_panel(discarded_answer)
                      for discarded_answer in (answer.discarded_bon_answers if answer.discarded_bon_answers else [])),
                title="[dim]Discarded Answer[/dim]",
                title_align="left",
                border_style="dim",
                expand=False
            ))
    else:
        content = answer.message

    return Group(
        Rule(title=f"[bold cyan]{answer.agent_id.value}[/bold cyan]", style="cyan"),
        content
    )


def render_answer_compact(answer: Answer) -> Panel:
    metrics = []

    if answer.evaluation and answer.evaluation.success:
        metrics.append(
            f"[cyan]Eval[/cyan] {answer.evaluation.score:.3f}"
        )
    if answer.gt_evaluation and answer.gt_evaluation.success:
        metrics.append(
            f"[green]GT[/green] {answer.gt_evaluation.score:.3f}"
        )
    if getattr(answer, "perplexity", None) is not None:
        metrics.append(
            f"[yellow]PPL[/yellow] {answer.perplexity:.2f}"
        )
    if answer.agent_config.cot_n > 1:
        metrics.append(
            f"[magenta]CoT[/magenta] {answer.agent_config.cot_n}"
        )
    content = " • ".join(metrics)
    if not content:
        content = "[dim]Completed[/dim]"
    return Panel(
        content,
        title=f"[bold cyan]{answer.agent_id.value}[/bold cyan]",
        border_style="cyan",
        expand=False,
    )


def render_energy_impact_panel(energy_dict: dict[str, Any]) -> Panel:
    """Pretty prints CodeCarbon metrics from a structured energy_dict."""
    if not energy_dict:
        return Panel("No energy dict")

    ed = energy_dict
    # Extract metrics using your exact dictionary keys
    emissions = ed.get("emissions_kg_co2", 0.0)
    total_energy = ed.get("energy_consumed_kwh", 0.0)
    duration = ed.get("duration_sec", 0.0)

    cpu_power = ed.get("cpu_power_w", 0.0)
    cpu_energy = ed.get("cpu_energy_kwh", 0.0)

    gpu_power = ed.get("gpu_power_w", 0.0)
    gpu_energy = ed.get("gpu_energy_kwh", 0.0)

    ram_energy = ed.get("ram_energy_kwh", 0.0)

    # Build a grid for formatting
    grid = Table.grid(expand=True)
    grid.add_column(style="bold green", width=22)
    grid.add_column(style="cyan")

    grid.add_row("🌱 Carbon Footprint:", f"{emissions:.6f} kg CO₂eq")
    grid.add_row("⚡ Total Energy:", f"{total_energy:.6f} kWh")
    grid.add_row("⏱️ Duration:", f"{duration:.2f} seconds")
    grid.add_row("", "")  # Spacer
    grid.add_row("💻 CPU Core:", f"{cpu_power:.2f} W  ({cpu_energy:.6f} kWh)")
    if gpu_power > 0 or gpu_energy > 0:
        grid.add_row("🎮 GPU Core:", f"{gpu_power:.2f} W  ({gpu_energy:.6f} kWh)")
    grid.add_row("💾 RAM Overhead:", f"Pooled  ({ram_energy:.6f} kWh)")

    # Render unified layout
    return Panel(grid,
                 title="[bold green]📊 Codecarbon Report Summary[/bold green]",
                 border_style="green",
                 padding=(1, 2),
                 expand=False
                 )
