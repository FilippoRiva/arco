from typing import TYPE_CHECKING

from rich.spinner import Spinner

if TYPE_CHECKING:
    from arco.core import Answer

# load singleton console
from arco.cli.console import console

from rich import box
from rich.status import Status
from rich.table import Table
from rich.live import Live
from rich.console import Group
from rich.panel import Panel
from rich.pretty import Pretty
from rich.rule import Rule

import time


class StatusDisplay:
    def __init__(self):
        self.status = ""
        self.init_time = time.time()
        self.node_start_time = None
        self.spinner = Spinner("dots")
        self.stopped = False

    def stop(self):
        self.stopped = True

    def set(self, status, start_time=None):
        self.status = status
        self.node_start_time = start_time

    def __rich__(self):
        if self.stopped:
            return ""

        text = f"[yellow]{self.status}[/yellow]"

        if self.node_start_time is not None:
            elapsed = time.time() - self.node_start_time
            text += f" [dim]Node time : {elapsed:.1f}s[/dim] "

        text += f"[dim]Total time : {time.time() - self.init_time:.1f}s[/dim]"

        self.spinner.update(text=text),
        return Panel(
            self.spinner,
            border_style="cyan",
        )


def _generate_answer_subtitle(answer: Answer) -> str:
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
    if conf.cot_n > 1: subtitle_elements.append(f"CoT : {conf.cot_n}")
    return "[dim]" + ", ".join(subtitle_elements) + "[/dim]"


def _generate_discarded_answer_panel(answer: Answer) -> Panel:
    return Panel(
        renderable=answer.message,
        title=f"[dim cyan]Discarded[/dim cyan]",
        subtitle=_generate_answer_subtitle(answer),
        subtitle_align="right",
        border_style="dim",
        expand=True,
    )


def generate_answer_panel(answer: Answer, verbose: bool) -> Panel:
    # Build the main panel
    group_elements = [answer.message]
    if verbose and answer.agent_config:
        # Create a subpanel for the configs
        config_subpanel = Panel(
            Pretty(answer.agent_config, max_length=2, max_depth=2, indent_size=2),
            title="[dim]Config[/dim]",
            title_align="left",
            border_style="dim"
        )
        if answer.agent_config.n > 1:
            discarded_bon_subpanel = [
                _generate_discarded_answer_panel(discarded_answer) for discarded_answer in
                (answer.discarded_bon_answers if answer.discarded_bon_answers else [])
            ]
            group_elements += discarded_bon_subpanel

        if answer.__getattribute__(
                'perplexity') is not None:
            arco_subpanel = Panel(
                f"Final perplexity: {answer.perplexity}\nBudget controller choice: {answer.budget_controller_choice}",
                title="[dim]ARCO evaluations[/dim]",
                title_align="left",
                border_style="dim"
            )
            group_elements += [arco_subpanel]

        if answer.error:
            error_subpanel = Panel(
                answer.error,
                title="[red]Error Message[/red]",
                title_align="left",
                border_style="red"
            )
            group_elements += [error_subpanel]
        group_elements += [config_subpanel]
        content = Group(*group_elements)

    else:
        content = answer.message
    return Panel(
        renderable=content,
        title=f"[bold cyan]{answer.agent_id.value}[/bold cyan]",
        subtitle=_generate_answer_subtitle(answer),
        subtitle_align="right",
        border_style="green",
        expand=True,
    )


def _energy_impact_panel(energy_dict: dict[str, Any]) -> Panel:
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
                 padding=(1, 2)
                 )


def agent_events_visualizer(events: Generator[str, Any], verbose=False) -> State | None:
    """
    Visualizes generated updates produced by the agent.run() generator.
    :param events: The events generated by the agent run
    :param verbose: Whether if we want verbose visualization or not
    :return: The final resulting state
    """
    status = StatusDisplay()

    with Live(status, refresh_per_second=8, screen=False) as live:
        energy_dict: dict = {}
        for update in events:
            event_type = update["event"]
            if event_type == "started":
                live.console.print(Panel(f"[bold cyan]Agent Run Started[/bold cyan]\n[dim]ID: {update["run_id"]}[/dim]",
                                         border_style="blue"))
                status.set("Started the graph execution")
            elif event_type == "cache":
                value = update["value"]
                if value == "search":
                    status.set("Searching cached runs")
                elif value == "hit":
                    live.console.print(Panel(f"[dim]Cache hit[/dim]",
                                             border_style="dim"))
                elif value == "miss":
                    live.console.print(Panel(f"[dim]Cache miss[/dim]",
                                             border_style="dim"))
                elif value == "store":
                    status.set("Storing results to cache")
                elif value == "store_completed":
                    live.console.print(Panel(f"[dim]Successfully stored in cache[/dim]",
                                             border_style="dim"))
            elif event_type == "node_started":
                start_time = time.time()
                status.set(f"{update['node']} is running", start_time)
            elif event_type == "node_finished":
                last_state = update["state"]
                last_answer: Answer = last_state.get_last_answer()
                status.set(f"{last_answer.agent_id.value} ended its run")
                live.console.print(generate_answer_panel(answer=last_answer, verbose=verbose))
            elif event_type == "codecarbon":
                energy_dict = update["energy_dict"]
            elif event_type == "completed":
                status.stop()
                live.console.print(Panel(
                    f"[bold cyan]Agent Run Completed[/bold cyan]\n[dim]Total run time : {update["state"].profiling_metrics['total_run_time_sec']}s[/dim]",
                    border_style="blue"))
            elif event_type == "error":
                live.console.print(Panel(f"[bold red]Error[/bold red]\n[dim]{update["message"]}[/dim]",
                                         border_style="red"))

    if energy_dict:
        console.print(_energy_impact_panel(energy_dict))

    from arco.core import AgentType
    vis_answer: Answer | None = last_state.get_last_answer(AgentType.VISUALIZER)
    ret_answer: Answer | None = last_state.get_last_answer(AgentType.RETRIEVER)
    if vis_answer and vis_answer.code and "plt" in vis_answer.code and ret_answer and ret_answer.data_df is not None:
        try:
            visualize_chart(ret_answer.data_df, vis_answer.chart_config, vis_answer.code)
        except Exception as e:
            console.print(f"[bold red]Error[/bold red]: couldn't execute visualization code\n{e}")
            console.print(Rule(title="[bold red]Execution Failed[/bold red]", style="red"))
            return

    console.print(Rule(title="[bold green]Execution Finished[/bold green]"))
    return


def print_config_table(params_list):
    """Helper to render a consistent Rich tables."""
    table = Table(box=box.ROUNDED)
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Current Value", style="white")
    for key, value in params_list:
        table.add_row(key, str(value))
    console.print(table)


def visualize_chart(df, chart_config, code):
    with Status("Waiting for image to render", spinner="dots", refresh_per_second=8) as status:
        import matplotlib
        for backend in ("TkAgg", "Qt5Agg", "GTK3Agg"):
            try:
                matplotlib.use(backend)
                import matplotlib.pyplot as plt
                break
            except ImportError:
                plt = None

        if plt is None:
            print(
                "No interactive backend available. Install one of: python3-tkinter (system), PyQt5 (pip), or PyGObject (pip)")
            return

        namespace = {
            "data_df": df,
            "config": chart_config,
            "plt": plt,
            "pd": __import__("pandas"),
            "np": __import__("numpy"),
        }
        status.update("Waiting for visualization window to close")
        exec(code, namespace)  # noqa: S102
