from collections import defaultdict
from statistics import mean
from typing import TYPE_CHECKING, Any

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from arco.cli.console import console
from arco.core import Config

if TYPE_CHECKING:
    from arco.core import Workflow
    from arco.data import BenchmarkSummary


def print_benchmark_header(
    name: str, description: str, changes: dict[str, Any]
) -> None:
    """Print one compact, borderless benchmark run summary."""
    console.print()
    console.print(f"[bold cyan]▶ {name}[/bold cyan]")
    if description:
        console.print(f"  [dim]{description}[/dim]")

    if not changes:
        console.print("  [dim]Default configuration[/dim]")
        return

    changes_table = Table(box=None, padding=(0, 1), expand=False)
    changes_table.add_column("Agent", style="yellow", no_wrap=True)
    changes_table.add_column("Parameter", style="cyan", no_wrap=True)
    changes_table.add_column("Value", style="green")
    for agent_name, params in changes.items():
        if isinstance(params, dict):
            first = True
            for param, value in params.items():
                changes_table.add_row(
                    agent_name if first else "", param, str(value)
                )
                first = False
        else:
            changes_table.add_row(agent_name, "-", str(params))
    console.print(changes_table)


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
            agent,
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
        color = "green" if score >= 0.9 else "yellow" if score >= 0.7 else "red"

        t = Text()
        t.append("█", style=color)
        t.append(f" {agent if len(agent) < 15 else agent[:5] + '...'}({score:.2f})")
        timeline.append(t)

    console.print(
        Panel(
            Text(" → ").join(timeline),
            title="Trace Summary",
        )
    )

    console.print(f"Completion: [bold cyan]{summary.completion_percentage:.1%}[/]")


def _config_table(config: Config, verbose: bool | None = None) -> Table:
    """Build the global configuration table without printing it."""
    configs_to_show = {
        f.name: getattr(config, f.name) for f in config.__dataclass_fields__.values()
    }
    configs_to_show.pop("agent_configs")

    table = Table(box=None, padding=(0, 1), expand=False)
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Current Value", style="white")
    for key, value in configs_to_show.items():
        table.add_row(key, str(value))
    if verbose is not None:
        table.add_row("verbose", str(verbose))
    return table


def print_config_table(config: Config, verbose: bool | None = None):
    """Print the global configuration as a borderless table."""
    console.print(_config_table(config, verbose=verbose))


def print_workflow_graph(workflow: Workflow):
    """Print the selected workflow without wrapping it in a panel."""
    console.print(Text("Selected workflow", style="bold cyan"))
    console.print(Text(str(workflow), style="cyan"))


def print_run_overview(
    config: Config, workflow: Workflow, verbose: bool | None = None
):
    """Print configuration and workflow side by side.

    This is deliberately borderless so the overview reads as one piece of
    information instead of two nested boxes.
    """
    config_view = Group(
        Text("Global parameters", style="bold cyan"),
        _config_table(config, verbose=verbose),
    )
    workflow_view = Group(
        Text("Selected workflow", style="bold cyan"),
        Text(str(workflow), style="cyan"),
    )

    overview = Table.grid(padding=(0, 4), expand=True)
    overview.add_column(ratio=1)
    overview.add_column(ratio=1)
    overview.add_row(config_view, workflow_view)
    console.print(overview)
