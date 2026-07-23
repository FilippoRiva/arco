from typing import TYPE_CHECKING

from arco.cli.viz import display, printer
from arco.core import evaluator
from arco.data import BenchmarkSummary
from arco.data.benchmark_dataset import BenchmarkDataset
from arco.workflows.workflow import WorkflowFactory

if TYPE_CHECKING:
    from arco.core import State
    from argparse import ArgumentParser, Namespace


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
    # Dependencies
    from arco.cli.console import console

    status = console.status("[bold cyan]Loading pre-benchmark dependencies[/bold cyan]")
    status.start()
    from pathlib import Path
    import os
    import pandas as pd
    import time
    import json
    console.print("[green]✓[/green] Pre-benchmark dependencies loaded")
    status.stop()

    start_time = time.time()
    workflow, default_config = WorkflowFactory.get_from_config(args.config)
    benchmark_dataset = BenchmarkDataset.from_json(args.dataset)

    run_config_to_result_list: list[tuple[dict, pd.DataFrame]] = []
    with console.status("[bold cyan]Processing run configurations[/bold cyan]"):
        list_of_run_configs = default_config.generate_benchmark_configs(args.config)
        console.print("[green]✓[/green] Run configurations loaded")

    benchmark_id = args.id or Path(args.config).stem
    benchmark_save_folder = Path(args.save_dir) / benchmark_id
    os.makedirs(benchmark_save_folder, exist_ok=True)
    runs_folder = benchmark_save_folder / 'runs'

    for run_config_dict in list_of_run_configs:
        run_name = run_config_dict['name'].replace(" ", "_")
        run_csv_name = run_name + ".csv"
        if (runs_folder / run_name / run_csv_name).exists():
            console.print(
                f"[yellow]Benchmark already exists: {runs_folder / run_name / run_csv_name}. Skipping.[/yellow]")
            result_df = pd.read_csv(runs_folder / run_name / run_csv_name)
        else:
            result_df, resulting_states = run_benchmark(
                **run_config_dict,
                benchmark_dataset=benchmark_dataset,
                workflow=workflow,
                benchmark_id=benchmark_id,
                verbose=args.verbose
            )
            # Save results
            os.makedirs(runs_folder / run_name, exist_ok=True)
            result_df.to_csv(runs_folder / run_name / run_csv_name, index=False)
            for result in resulting_states:
                result.save(runs_folder / run_name)
            console.print(f"\nResults saved to [cyan]{runs_folder / run_name / run_csv_name}[/cyan]")

        run_config_to_result_list.append((run_config_dict, result_df))

    aggregated_df = aggregate_results(run_config_to_result_list)
    aggregated_df.to_csv(benchmark_save_folder / 'summary.csv', index=False)
    bench_metadata = {
        "benchmark_run": args.config,
        "total_runtime": time.time() - start_time,
    }
    with open(benchmark_save_folder / 'bench_metadata.json', 'w') as f:
        json.dump(bench_metadata, f)


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
        workflow: Workflow,
        name: str,
        description: str,
        config: ArcoConfig,
        changes: dict[str, Any],
        benchmark_dataset: BenchmarkDataset,
        *,
        benchmark_id: str | None = None,
        verbose: bool = False
) -> tuple[pd.DataFrame, list[State]]:
    """Run benchmark against a unified GT dataset.

    Args:
        workflow: the workflow tested on this benchmark.
        name: the name of this benchmark
        description: Description of the purpose for this benchmark
        config: The config to run during this benchmark
        changes: Changes associated to each config
        benchmark_dataset: The Dataset to be used by the benchmark
        benchmark_id: id of this run
        verbose: set to true if verbose visualization is needed

    Returns:
        DataFrame with per-test-case information.
    """
    # Dependencies loaded dynamically
    from arco.cli.console import console
    from functools import partial
    import pandas as pd
    from rich.rule import Rule
    from arco.workflows.workflow_executor import WorkflowExecutor
    import json

    if benchmark_id is None:
        benchmark_id = generate_benchmark_id()

    printer.print_benchmark_header(name, description, changes)

    df_rows: list[dict] = []
    resulting_states: list[State] = []

    if verbose:
        visualization_logic = partial(display.display_workflow, verbose=True, show_plot=False)
    else:
        visualization_logic = display.display_workflow_compact

    for entry in benchmark_dataset:
        # Run agent
        console.print(Rule(f"[bold blue]Test Case {entry.id + 1}/{len(benchmark_dataset)}"))
        config = config.update_prompt(entry.prompt)
        executor = WorkflowExecutor(workflow=workflow, config=config)
        resulting_state: State = visualization_logic(executor.stream())
        with console.status("[bold cyan]Evaluating the run[/bold cyan]"):
            evaluation_summary: BenchmarkSummary = (
                evaluator.evaluate_state(resulting_state, entry, workflow.get_evaluators(),
                                         config.default_provider_judge, config.default_model_judge))
        printer.print_benchmark_summary(evaluation_summary)

        # Answer Level Profiling
        execution_trace = {"answers": []}
        for answer in resulting_state.answers:
            answer_energy_dict = answer.profiling_data.as_dict()
            answer_dict = ({
                "agent_type": answer.agent_id.value,
                "message": answer.message,
                "evaluation_gt": answer.gt_evaluation.score if answer.gt_evaluation else None,
                "perplexity": answer.perplexity if answer.perplexity else None,
                **answer_energy_dict
            })
            execution_trace['answers'].append(answer_dict)

        # Store results into a df row
        row = {
            "entry_id": entry.id,
            "run_id": resulting_state.run_id,
            "execution_trace": json.dumps(execution_trace),
        }

        df_rows.append(row)
        resulting_states.append(resulting_state)

    # Build results DataFrame
    df = pd.DataFrame(df_rows)

    return df, resulting_states


def aggregate_results(run_config_to_result_list: list[tuple[dict, pd.DataFrame]]) -> pd.DataFrame:
    import json
    import collections
    import pandas as pd

    to_aggregate = [
        "evaluation_gt",
        "perplexity",
        "total_time",
        "llm_time",
        "cpu_energy_kwh",
        "ram_energy_kwh",
        "emissions_kg_co2",
    ]

    run_summaries = []

    for run_config, result_df in run_config_to_result_list:
        name = run_config['name']
        description = run_config['description']
        changes = run_config['changes']

        traces = result_df["execution_trace"].apply(json.loads)

        # agent -> metric -> list of values
        agents_summary_stats = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )

        for trace in traces:
            for answer in trace["answers"]:
                agent = answer["agent_type"]

                for metric in to_aggregate:
                    value = answer.get(metric)

                    # Ignore missing values
                    if value is not None:
                        agents_summary_stats[agent][metric].append(value)

        # Compute averages
        for agent, metrics in agents_summary_stats.items():
            for metric, values in metrics.items():
                metrics[metric] = sum(values) / len(values)

        run_summaries.append({
            "name": name,
            "description": description,
            "changes": changes,
            "metrics_by_agent": json.dumps(agents_summary_stats),
        })

    return pd.DataFrame(run_summaries)
