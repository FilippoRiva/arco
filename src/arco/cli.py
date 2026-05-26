#!/usr/bin/env python3
"""Run the DataAgent from a YAML configuration file.

Usage:
    python run_agent.py                           # uses config/run_config.yaml
    python run_agent.py config/my_config.yaml     # custom config path
"""
import argparse
import os
import sys
import warnings
from argparse import Namespace
from dataclasses import replace
from typing import Any, cast
from typing import TYPE_CHECKING
# Suppress all general UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress the specific LangChain warning
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

from arco.workflow import SalesDataWorkflow
from arco.core import State, EmpoweredAnswer

from arco.data import RunCache
from rich import box
from rich.prompt import Prompt
from rich.rule import Rule
from rich.status import Status
from rich.table import Table
from rich.live import Live
from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.pretty import Pretty
from rich.spinner import Spinner
from rich.text import Text

if TYPE_CHECKING:
    from arco.core import Answer

from arco.core import ArcoConfig, AgentType

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Initialize console for visualization
console = Console()

# ---------------------------------------------------------------------------
# Interactive run-parameter configuration
# ---------------------------------------------------------------------------

# Parameters the user can override, with (key, type, description).
_GLOBAL_PARAMS = [
    ("prompt", str, "Natural language query"),
    ("visualization_goal", str, "Chart description (empty to skip)"),
    ("run_id", str, "Run ID (empty = auto-generate)"),
    #("save_dir", str, "Output directory"),
    #("enable_codecarbon", bool, "Enable CodeCarbon tracking"),
    #("model", str, "LLM model name"),
    #("provider", str, "LLM provider: openai | ollama"),
    #("ollama_url", str, "Ollama server URL (ignored for openai)"),
]


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
    else :
        subtitle_elements.append(f"Temp: {round(answer.agent_config.get_candidate_params()[0][0], 3)}")
    if conf.cot_n > 1: subtitle_elements.append(f"CoT : {conf.cot_n}")
    return "[dim]" + ", ".join(subtitle_elements) + "[/dim]"


def _generate_discarded_answer_panel(answer) -> Panel:
    return Panel(
        renderable=answer.message,
        title=f"[dim cyan]Discarded[/dim cyan]",
        subtitle=_generate_answer_subtitle(answer),
        subtitle_align="right",
        border_style="dim",
        expand=True,
    )


def _generate_answer_panel(answer: Answer, verbose) -> Panel:
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

        if answer.__getattribute__('perplexity') is not None: #if it has perplexity empower() has applied its evaluation to the answer
            answer = cast(EmpoweredAnswer, answer)
            arco_subpanel = Panel(
                f"Final perplexity: {answer.perplexity}\nBudget controller choice: {answer.budget_controller_choice}",
                title="[dim]ARCO evaluations[/dim]",
                title_align = "left",
                border_style = "dim"
            )
            group_elements += [arco_subpanel]
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


def _generate_display(message: str, current_state: State | None, verbose: bool):
    """Creates a Renderable rich object to show the overall workflow execution"""

    spinner = Spinner("line", style="bold white")
    status = Text(message, style="bold magenta")
    header = Columns([spinner, status])

    if current_state:
        panels = [_generate_answer_panel(answer, verbose) for answer in current_state.answers]
        last_ans = current_state.get_last_answer()
        if last_ans and last_ans.agent_id is not AgentType.VISUALIZER:
            renderables = [header, *panels]
        else:
            renderables = [
                *panels]
    else:
        renderables = [header]

    return Group(*renderables)


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


def agent_rich_run(agent: SalesDataWorkflow, verbose=False) -> State | None:
    with Live(refresh_per_second=8) as live:
        current_state: State | None = None
        energy_dict: dict = {}
        final_result: State | None = None
        for update in agent.run():
            event_type = update["event"]
            if event_type == "started":
                console.print(Panel(f"[bold cyan]Agent Run Started[/bold cyan]\n[dim]ID: {update["run_id"]}[/dim]",
                                    border_style="blue"))
                live.update(_generate_display("Started the graph execution", None, verbose=verbose))
            elif event_type == "cache":
                value = update["value"]
                if value == "search":
                    live.update(
                        _generate_display(f"Searching cached runs", current_state=current_state, verbose=verbose))
                elif value == "hit":
                    console.print(Panel(f"[dim]Cache hit[/dim]",
                                        border_style="dim"))
                elif value == "miss":
                    console.print(Panel(f"[dim]Cache miss[/dim]",
                                        border_style="dim"))
                elif value == "store":
                    live.update(
                        _generate_display(f"Storing results to cache", current_state=current_state, verbose=verbose))
                elif value == "store_completed":
                    console.print(Panel(f"[dim]Successfully stored in cache[/dim]",
                                        border_style="dim"))
            elif event_type == "node_started":
                live.update(
                    _generate_display(f"{update['node']} is running", current_state=current_state, verbose=verbose))
            elif event_type == "node_finished":
                current_state = update["state"]
                live.update(_generate_display(
                    f"{current_state.get_last_answer().agent_id.value} ended its run",
                    current_state=current_state,
                    verbose=verbose)
                )
            elif event_type == "codecarbon":
                energy_dict = update["energy_dict"]
            elif event_type == "completed":
                final_result = update["state"]
                console.print(Panel(
                    f"[bold cyan]Agent Run Completed[/bold cyan]\n[dim]Total run time : {final_result.profiling_metrics['total_run_time_sec']}s[/dim]",
                    border_style="blue"))
            elif event_type == "error":
                console.print(Panel(f"[bold red]Error[/bold red]\n[dim]{update["message"]}[/dim]",
                                    border_style="red"))

    if energy_dict:
        console.print(_energy_impact_panel(energy_dict))
    return final_result


def _interactive_configure(config: ArcoConfig) -> ArcoConfig:
    """Show YAML defaults and let the user override them one by one.

    Returns the (possibly modified) config
    """

    def _prompt_value(name, current_value, param_type):
        """Prompt for a single value; return current_value on empty input."""
        while True:
            raw = input(f"  {name} [{current_value}]: ").strip()
            if not raw:
                return current_value
            try:
                if param_type is bool:
                    if raw.lower() in ("true", "1", "yes", "y"):
                        return True
                    if raw.lower() in ("false", "0", "no", "n"):
                        return False
                    raise ValueError
                if param_type is str:
                    if raw.lower() in ("none", "null", ""):
                        return None
                    return raw
                return param_type(raw)
            except (ValueError, TypeError):
                print(f"    Invalid value for {name} (expected {param_type.__name__}). Try again.")

    if not sys.stdin.isatty():
        return config

    agent_display = [(key, getattr(config, key)) for key, _, _ in _GLOBAL_PARAMS]
    _print_config_table(agent_display)

    choice = Prompt.ask("Accept [bold cyan]Global Settings[/bold cyan]? [Y/n]: ").strip().lower()
    if choice in ("n", "no"):
        choices = {}
        for key, ptype, _ in _GLOBAL_PARAMS:
            current = getattr(config, key)
            new_val = _prompt_value(key, current, ptype)
            choices.update({key: new_val})
        return replace(config, **choices)

    return config


def _print_config_table(params_list):
    """Helper to render a consistent Rich tables."""
    table = Table(box=box.ROUNDED)
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Current Value", style="white")
    for key, value in params_list:
        table.add_row(key, str(value))
    console.print(table)


def _visualize_chart(df, chart_config, code):
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


def cache_script(args: Namespace):
    save_dir = args.save_dir
    cache = RunCache(save_dir)
    if args.clear:
        count = cache.clear_cache()
        console.print(f"[bold cyan]Cache cleared[/bold cyan] : {count} runs deleted")
        return

    if args.delete:
        if cache.delete_run(args.delete):
            console.print(f"[bold cyan]Run deleted[/bold cyan] : run_id={args.delete}")
        else:
            console.print(f"[bold red]Failed to delete[/bold red] : run_id={args.delete} not found in cache")
        return

    if args.runs:
        runs = cache.list_runs()  # Assumed to be a list of dicts: [{}, {}, {}]
        console.print(f"[bold magenta]Available Runs[/bold magenta]({len(runs)}):")
        console.print(runs)
    if args.stats:
        stats = cache.get_cache_stats()  # Assumed to be a flat or nested dict
        console.print(f"[bold magenta]Cache Statistics[/bold magenta]:")
        console.print(stats)
    else:  # defaults to stats
        stats = cache.get_cache_stats()  # Assumed to be a flat or nested dict
        console.print(f"[bold magenta]Cache Statistics[/bold magenta]:")
        console.print(stats)


def run(args: Namespace):
    ## Initialization
    # Load config from YAML
    if not os.path.isfile(args.config):
        console.print(f"[bold red]Error[/bold red]: config file not found at [bold cyan]{args.config}[/bold cyan]")
        sys.exit(1)
    console.print(f"Loading configuration from: [bold cyan]{args.config}[/bold cyan]")
    config = ArcoConfig.from_yaml(args.config)

    # Interactive YAML config change if interactive mode is selected
    if args.interactive:
        config = _interactive_configure(config)

    configs_to_show = {f.name: getattr(config, f.name) for f in config.__dataclass_fields__.values()}
    configs_to_show.pop("schema")
    configs_to_show.pop("agent_configs")

    # Visualize run configuration
    run_config_data = [
        *[(key, value) for key, value in configs_to_show.items()],
        ("verbose", args.verbose),
        ("interactive", args.interactive)
    ]
    _print_config_table(run_config_data)
    console.print(Rule(title="[bold green]Running the Agent[/bold green]"))

    ## Run the agent
    agent = SalesDataWorkflow(
        config=config
    )

    result = agent_rich_run(agent, verbose=args.verbose)  # runs and visualize agents outputs

    ## Final output and visualization
    if not result:
        console.print(Rule(title="[bold red]Execution Failure[/bold red]", style="red"))
        return

    vis_answer: Answer | None = result.get_last_answer(AgentType.VISUALIZER)
    ret_answer: Answer | None = result.get_last_answer(AgentType.RETRIEVER)
    if vis_answer and vis_answer.code and "plt" in vis_answer.code and ret_answer and ret_answer.data_df is not None:
        try:
            _visualize_chart(ret_answer.data_df, vis_answer.chart_config, vis_answer.code)
        except Exception as e:
            console.print(f"[bold red]Error[/bold red]: couldn't execute visualization code\n{e}")
            console.print(Rule(title="[bold red]Execution Failed[/bold red]", style="red"))
            return

    console.print(Rule(title="[bold green]Execution Success[/bold green]"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Parse script arguments
    parser = argparse.ArgumentParser(
        description=(
            "Runs the agent from a configuration file. Optionally invokes the cache management system if the corresponding subcommand is invoked\n"
        )
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/minimal.yaml",
        help="Path to config YAML"
    )
    parser.add_argument(
        "--interactive", "-i",
        action='store_true',
        help="Whether if an interactive run is needed",
    )
    parser.add_argument(
        "--verbose", "-v",
        action='store_true',
        help="Whether if the agent's configuration and other metrics should be shown after each execution"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 3. Define the 'cache-mode' Subcommand Parser
    cache_parser = subparsers.add_parser(
        "cache",
        help="Invoke the cache management system. See 'arco-cli cache -h' for more information."
    )

    # Move arguments specific to cache-mode under this new parser
    cache_parser.add_argument(
        "--save-dir", "-d",
        type=str,
        default="output",
        help="Directory of the cache to analyze"
    )
    cache_parser.add_argument(
        "--clear",
        action="store_true",
        help="If set it entirely clears the cached runs"
    )
    cache_parser.add_argument(
        "--delete",
        type=str,
        help="run_id of the run to delete"
    )
    cache_parser.add_argument(
        "--runs", "-r",
        action="store_true",
        help="Shows the list of cached runs"
    )
    cache_parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="Prints the cache statistics"
    )

    args = parser.parse_args()

    # Runs the script
    try:
        if args.command == "cache":
            cache_script(args)
        else:
            run(args)
    except KeyboardInterrupt:
        console.print("[bold red]Stopped[/bold red]: Keyboard Interrupt detected")


if __name__ == "__main__":
    main()
