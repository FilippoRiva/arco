import collections
import json
import os
import random
import sys
from argparse import ArgumentParser, Namespace

import pandas as pd
from arco.cli import viz
from arco.cli.console import console
from rich.rule import Rule
from arco.core import ArcoConfig, AgentType, Answer
from arco.core.state import ProfilingData
from arco.workflow import SalesDataWorkflow

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Script Parser Registration
# ---------------------------------------------------------------------------
def register(subparsers: ArgumentParser) -> ArgumentParser:
    parser = subparsers.add_parser("bench", help="Run DataAgent benchmark against GT dataset")
    parser.add_argument("dataset", help="Path to benchmark dataset JSON")
    parser.add_argument("--config", "-c", required=True, help="Path to run_config.yaml")
    parser.add_argument("--save-dir", default="./output/benchmarks", help="Output directory")
    parser.add_argument("--max-prompts", type=int, default=None, help="Maximum number of prompts")
    parser.add_argument("--id", type=str, default=None, help="ID of this benchmark")
    return parser


# ---------------------------------------------------------------------------
# Script Handler
# ---------------------------------------------------------------------------
def handle(args: Namespace, parser: ArgumentParser) -> None:
    run_benchmark(
        args.dataset,
        config_path=args.config,
        save_dir=args.save_dir,
        max_prompts=args.max_prompts,
        benchmark_id=args.id
    )


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


def load_benchmark_dataset(path: str) -> List[Dict]:
    """Load and validate a unified GT dataset JSON."""
    with open(path) as f:
        entries = json.load(f)
    if not entries:
        raise ValueError(f"Empty dataset: {path}")
    first = entries[0]
    if "prompt" not in first:
        raise ValueError("Dataset entries must have a 'prompt' field")
    if "gt_data" not in first and "gt_chart_config" not in first:
        raise ValueError("Dataset entries must have 'gt_data' and/or 'gt_chart_config'")
    return entries

def _set_prefix_keys(d, prefix):
    return {f"{prefix}_{k}": v for k, v in d.items()}

def run_benchmark(
        dataset_path: str,
        *,
        config: Optional[ArcoConfig] = None,
        config_path: Optional[str] = None,
        save_dir: str = "./output/benchmarks",
        max_prompts: Optional[int] = None,
        benchmark_id: str | None = None
) -> pd.DataFrame:
    """Run benchmark against a unified GT dataset.

    Args:
        dataset_path: Path to the benchmark JSON file.
        config: Pre-built AgentConfig to use directly.
        config_path: Path to a config.yaml for base AgentConfig. Overrides the provided config
        save_dir: Directory to save results CSV.
        max_prompts: Max number of prompts from the dataset to run
        benchmark_id: id of this run

    Returns:
        DataFrame with per-test-case information.
    """
    if benchmark_id is None:
        benchmark_id = generate_benchmark_id()

    entries = load_benchmark_dataset(dataset_path)
    if max_prompts is not None:
        entries = entries[:max_prompts]

    if config is None and config_path is None:
        raise Exception("No config provided")

    if config_path is not None:
        config = ArcoConfig.from_yaml(config_path)

    results = []
    agents_executed = []

    viz.print_config_table(config)

    for idx, entry in enumerate(entries):
        # Setup config for this execution (prompt, visualization goal and gt evaluation
        prompt = entry["prompt"]
        visualization_goal = entry.get("visualization_goal")
        gt_dict = {key: value for key, value in entry.items() if key.startswith("gt_")}
        config = config.update_prompt(prompt, visualization_goal)
        config.set_gt(gt_dict)

        # Run agent
        console.print(Rule(f"[bold blue]Test Case {idx+1}/{len(entries)}"))
        agent = SalesDataWorkflow(config=config)
        result = viz.compact_agent_events_visualizer(agent.stream())

        ## Handle Profiling Data
        global_profiling_data : ProfilingData = result.global_profiling_data
        agents_profiling_datas : dict[AgentType, ProfilingData] = result.agents_profiling_data

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
            agents_profiling_dict.update(_set_prefix_keys(agents_profiling_datas[agent_type].get_energy_dict(), agent_type.value + "_cumulative"))

        # Answer Level Profiling
        answer_profiling_dict = collections.defaultdict(int)
        agent_answer_count = collections.defaultdict(int)
        for answer in result.answers:
            agent_type : AgentType = answer.agent_id
            agent_answer_count[agent_type] += 1
            answer_energy_dict = answer.profiling_data.get_energy_dict()
            answer_dict= ({
                "message" : answer.message,
                "evaluation": answer.evaluation.score if answer.evaluation else None,
                "evaluation_gt": answer.gt_evaluation.score if answer.gt_evaluation else None,
                "perplexity": answer.perplexity if answer.perplexity else None,
                **answer_energy_dict
            })
            answer_profiling_dict.update(_set_prefix_keys(answer_dict, agent_type.value + "_" +str(agent_answer_count[agent_type])))

        # Store results into a df row
        row = {
            "benchmark_id": benchmark_id,
            "test_case_id": idx,
            "prompt": prompt,
            "difficulty": entry.get("difficulty"),
            **answer_profiling_dict,
            **agents_profiling_dict,
            **global_profiling_dict
        }

        results.append(row)

    # Build results DataFrame
    df = pd.DataFrame(results)

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, benchmark_id+".csv")
    df.to_csv(out_path, index=False)
    console.print(f"\nResults saved to [cyan]{out_path}[/cyan]")

    return df
