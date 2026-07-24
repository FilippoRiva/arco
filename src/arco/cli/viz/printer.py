from collections import defaultdict
from statistics import mean
from typing import TYPE_CHECKING, Any

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from arco.cli.console import console
from arco.core import Config

if TYPE_CHECKING:
    from arco.data import BenchmarkSummary
    from arco.workflows import Workflow


def print_benchmark_header(name: str, description: str, changes: dict[str, Any]) -> None:
    """Print a rich panel summarizing the benchmark run's name, description, and changes."""
    header = Text(f"Benchmark run : {name}", style="bold cyan", justify="center")

    body = Table.grid(padding=(0, 1))
    body.add_column(justify="right", style="bold dim")
    body.add_column()
    body.add_row("Description:", description or "[dim italic]none provided[/dim italic]")

    if changes:
        changes_table = Table(
            title="Overrides", show_header=True, header_style="bold magenta",
            box=box.SIMPLE_HEAVY, expand=False
        )
        changes_table.add_column("Agent", style="bold yellow")
        changes_table.add_column("Parameter", style="cyan")
        changes_table.add_column("Value", style="green")

        for agent_name, params in changes.items():
            if isinstance(params, dict):
                first = True
                for param, value in params.items():
                    changes_table.add_row(
                        agent_name if first else "",
                        param,
                        str(value)
                    )
                    first = False
            else:
                changes_table.add_row(agent_name, "-", str(params))
    else:
        changes_table = Text("No overrides — running with defaults.", style="dim italic")

    body.add_row("Changes:", changes_table)

    console.print(Panel(
        body,
        title=header,
        border_style="blue",
        box=box.ROUNDED,
        padding=(1, 2)
    ))


PROFILE_FIELDS = [
    "total_time",
    "llm_time",
    "energy_consumed_kwh",
    "cpu_energy_kwh",
    "gpu_energy_kwh",
    "ram_energy_kwh",
    "emissions_kg_co2",
]


def _avg(values: list[float | None]) -> float:
    values = [v for v in values if v is not None]
    return mean(values) if values else 0.0


def print_benchmark_summary(summary: BenchmarkSummary):
    grouped = defaultdict(
        lambda: {
            "ppl": [],
            "score": [],
            **{field: [] for field in PROFILE_FIELDS},
        }
    )

    # Aggregate
    for agent, ppl, score, profiling in zip(
            summary.agents,
            summary.ppls,
            summary.scores,
            summary.profiling_datas,
    ):
        g = grouped[agent]
        g["ppl"].append(ppl)
        g["score"].append(score)

        for field in PROFILE_FIELDS:
            g[field].append(getattr(profiling, field))

    # Table
    table = Table(title="Evaluation Summary")

    table.add_column("Agent")
    table.add_column("#", justify="right")
    table.add_column("Avg PPL", justify="right")
    table.add_column("Avg Score", justify="right")
    table.add_column("Time (s)", justify="right")
    table.add_column("LLM (s)", justify="right")
    table.add_column("Energy (Wh)", justify="right")
    table.add_column("CPU (Wh)", justify="right")
    table.add_column("GPU (Wh)", justify="right")
    table.add_column("RAM (Wh)", justify="right")
    table.add_column("CO₂ (gCO₂)", justify="right")

    for agent, values in grouped.items():
        table.add_row(
            agent.value,
            str(len(values["ppl"])),
            f"{_avg(values['ppl']):.2f}",
            f"{_avg(values['score']):.2f}",
            f"{_avg(values['total_time']):.2f}",
            f"{_avg(values['llm_time']):.2f}",
            f"{_avg(values['energy_consumed_kwh']) * 1000:.3f}",
            f"{_avg(values['cpu_energy_kwh']) * 1000:.3f}",
            f"{_avg(values['gpu_energy_kwh']) * 1000:.3f}",
            f"{_avg(values['ram_energy_kwh']) * 1000:.3f}",
            f"{_avg(values['emissions_kg_co2']) * 1000:.4f}",
        )

    console.print(table)

    # Timeline
    timeline = []

    for agent, ppl, score in zip(summary.agents, summary.ppls, summary.scores):
        color = (
            "green" if score >= 0.9 else
            "yellow" if score >= 0.7 else
            "red"
        )

        t = Text()
        t.append("█", style=color)
        t.append(f" {agent.value if len(agent.value) < 15 else agent.value[:5] + "..."}({score:.2f})")
        timeline.append(t)

    console.print(
        Panel(
            Text(" → ").join(timeline),
            title="Trace Summary",
        )
    )

    console.print(
        f"Completion: [bold cyan]{summary.completion_percentage:.1%}[/]"
    )


def print_config_table(config: Config, verbose: bool | None = None):
    """Helper to render a consistent Rich tables."""
    configs_to_show = {f.name: getattr(config, f.name) for f in config.__dataclass_fields__.values()}
    configs_to_show.pop("agent_configs")

    # Visualize run configuration
    params_list = [
        *[(key, value) for key, value in configs_to_show.items()],
    ]
    if verbose is not None:
        params_list.append(("verbose", verbose))
    table = Table(box=box.ROUNDED)
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Current Value", style="white")
    for key, value in params_list:
        table.add_row(key, str(value))
    console.print(table)


def print_workflow_graph(workflow: Workflow):
    console.print(Panel(
        str(workflow),
        title="Selected Workflow",
        title_align="center",
        expand=False
    ))
