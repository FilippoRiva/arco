import collections
import os
import random
import sys
from argparse import ArgumentParser, Namespace
from functools import partial

import pandas as pd
from rich.rule import Rule

from arco.cli import viz
from arco.cli.console import console
from arco.core import ArcoConfig, AgentType
from arco.core.state import ProfilingData
from arco.workflow import SalesDataWorkflow

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Script Parser Registration
# ---------------------------------------------------------------------------
def register(subparsers: ArgumentParser) -> ArgumentParser:
    parser = subparsers.add_parser("benchmark", help="Runs a given set of configurations against a GT dataset")
    parser.add_argument("--dataset", "-d", required=True, help="Path to benchmark dataset JSON")
    parser.add_argument("--config", "-c", required=True, help="Path to benchmark_config.yaml")
    parser.add_argument("--save-dir", default="./output/benchmarks", help="Output directory")
    parser.add_argument("--id", type=str, default=None, help="ID of this benchmark")
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                        help="Whether if all the agent output should be shown")
    return parser


# ---------------------------------------------------------------------------
# Script Handler
# ---------------------------------------------------------------------------
def handle(args: Namespace, parser: ArgumentParser) -> None:
    df_list = []
    list_of_run_configs = ArcoConfig.from_benchmark_yaml(args.config, args.dataset)
    for run_config_dict in list_of_run_configs:
        df_list.append(
            run_benchmark(
                **run_config_dict,
                save_dir=args.save_dir,
                benchmark_id=args.id,
                verbose=args.verbose
            )
        )

    if len(df_list) > 1:
        # Aggregate results to compute the needed statistics
        pass


def generate_benchmark_id():
    prefixes = [
        "measured", "scored", "ranked", "evaluated", "tested",
        "validated", "benchmarked", "profiled", "timed", "calibrated",
        "optimized", "compared", "sampled", "analyzed", "monitored",
        "stress", "load", "latency", "throughput", "accuracy"
    ]

    nouns = [
        "benchmark", "suite", "trial", "run", "dataset",
        "metric", "baseline", "profile", "report", "score",
        "evaluation", "assessment", "experiment", "scenario",
        "workload", "sample", "result", "measurement", "test", "index"
    ]

    number = random.randint(100, 999)
    return f"{random.choice(prefixes)}-{random.choice(nouns)}-{number}"


def _set_prefix_keys(d, prefix):
    return {f"{prefix}_{k}": v for k, v in d.items()}


def run_benchmark(
        name: str,
        description: str,
        configs: list[ArcoConfig],
        difficulties: list[int],
        changes: dict[str, Any],
        *,
        save_dir: str = "./output/benchmarks",
        benchmark_id: str | None = None,
        verbose: bool = False
) -> pd.DataFrame:
    """Run benchmark against a unified GT dataset.

    Args:
        name: the name of this benchmark
        description: Description of the purpose for this benchmark
        configs: The list of configs to run during this benchmark
        difficulties: Difficulties associated to each config
        changes: Changes associated to each config
        save_dir: Directory to save results CSV.
        benchmark_id: id of this run
        verbose: set to true if verbose visualization is needed

    Returns:
        DataFrame with per-test-case information.
    """
    if benchmark_id is None:
        benchmark_id = generate_benchmark_id()

    viz.print_benchmark_header(name, description, changes)
    results = []
    agents_executed = []

    if verbose:
        visualization_logic = partial(viz.agent_events_visualizer, verbose=True, show_plot=False)
    else:
        visualization_logic = viz.compact_agent_events_visualizer

    for idx, config in enumerate(configs):
        # Run agent
        console.print(Rule(f"[bold blue]Test Case {idx + 1}/{len(configs)}"))
        agent = SalesDataWorkflow(config=config)
        result = visualization_logic(agent.stream())

        ## Handle Profiling Data
        global_profiling_data: ProfilingData = result.global_profiling_data
        agents_profiling_datas: dict[AgentType, ProfilingData] = result.agents_profiling_data

        # Global Level Profiling
        global_profiling_dict = _set_prefix_keys(global_profiling_data.get_energy_dict(), "global")

        # Agent Level Profiling
        agents_profiling_dict = collections.defaultdict(int)
        for agent_type in AgentType:
            answer = result.get_last_answer(agent_type)
            if answer is None:
                continue
            if agent_type not in agents_executed:
                agents_executed.append(agent_type)
            agents_profiling_dict.update(_set_prefix_keys(agents_profiling_datas[agent_type].get_energy_dict(),
                                                          agent_type.value + "_cumulative"))

        # Answer Level Profiling
        answer_profiling_dict = collections.defaultdict(int)
        agent_answer_count = collections.defaultdict(int)
        for answer in result.answers:
            agent_type: AgentType = answer.agent_id
            agent_answer_count[agent_type] += 1
            answer_energy_dict = answer.profiling_data.get_energy_dict()
            answer_dict = ({
                "message": answer.message,
                "evaluation": answer.evaluation.score if answer.evaluation else None,
                "evaluation_gt": answer.gt_evaluation.score if answer.gt_evaluation else None,
                "perplexity": answer.perplexity if answer.perplexity else None,
                **answer_energy_dict
            })
            answer_profiling_dict.update(
                _set_prefix_keys(answer_dict, agent_type.value + "_" + str(agent_answer_count[agent_type])))

        # Store results into a df row
        row = {
            "benchmark_id": benchmark_id,
            "test_case_id": idx,
            "prompt": config.prompt,
            "difficulty": difficulties[idx],
            **answer_profiling_dict,
            **agents_profiling_dict,
            **global_profiling_dict
        }

        results.append(row)

    # Build results DataFrame
    df = pd.DataFrame(results)

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, benchmark_id + name.replace(" ", "_") + ".csv")
    df.to_csv(out_path, index=False)
    console.print(f"\nResults saved to [cyan]{out_path}[/cyan]")

    return df
