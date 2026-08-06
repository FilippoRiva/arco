import collections
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from arco import workflows
from arco.core import (
    Config,
    State,
    Workflow,
    WorkflowFactory,
    evaluate_state_with_benchmark_entry,
)
from arco.data import BenchmarkDataset
from arco.logs import initialize as init_logging


def benchmark_from_config(
    config_path: str,
    dataset_path: str,
    id: str | None,
    save_dir: str,
    logging_level: str | None,
    run_visualization_logic: Callable,
):
    workflows.load_library_workflows()
    start_time = time.time()
    default_config = Config.from_yaml(config_path)
    workflow = WorkflowFactory.get(config=default_config)
    benchmark_dataset = BenchmarkDataset.from_json(dataset_path)

    benchmark_id = id or Path(config_path).stem
    benchmark_save_folder = Path(save_dir) / benchmark_id
    runs_folder = benchmark_save_folder / "runs"
    os.makedirs(benchmark_save_folder, exist_ok=True)

    init_logging(benchmark_id, log_dir=benchmark_save_folder, level=logging_level)

    # Load run configurations
    list_of_run_configs = default_config.generate_benchmark_configs(config_path)
    yield {"event": "run_configs_loaded", "configs": list_of_run_configs}

    # Run each configuration
    run_config_to_result_list: list[tuple[dict, pd.DataFrame]] = []
    for run_config_dict in list_of_run_configs:
        run_name = run_config_dict["name"].replace(" ", "_")
        run_csv_name = run_name + ".csv"
        if (runs_folder / run_name / run_csv_name).exists():  # loads it if available
            result_df = pd.read_csv(runs_folder / run_name / run_csv_name)
            yield {
                "event": "benchmark_already_exists",
                "path": runs_folder / run_name / run_csv_name,
            }
        else:
            result_df, resulting_states = None, None
            for event in benchmark(
                **run_config_dict,
                benchmark_dataset=benchmark_dataset,
                workflow=workflow,
                visualization_logic=run_visualization_logic,
            ):
                if event["event"] == "result":
                    result_df, resulting_states = event["result"]
                else:
                    yield event
            os.makedirs(runs_folder / run_name, exist_ok=True)
            result_df.to_csv(runs_folder / run_name / run_csv_name, index=False)
            for result in resulting_states:
                result.save(runs_folder / run_name)
            yield {
                "event": "benchmark_run_save",
                "path": runs_folder / run_name / run_csv_name,
            }

        run_config_to_result_list.append((run_config_dict, result_df))

    aggregated_df = aggregate_results(run_config_to_result_list)
    aggregated_df.to_csv(benchmark_save_folder / "summary.csv", index=False)
    bench_metadata = {
        "benchmark_run": config_path,
        "dataset_path": dataset_path,
        "total_runtime": time.time() - start_time,
    }
    with open(benchmark_save_folder / "bench_metadata.json", "w") as f:
        json.dump(bench_metadata, f)
    yield {
        "event": "aggregated_summary_save",
        "path": benchmark_save_folder / "summary.csv",
    }


def _set_prefix_keys(d, prefix):
    return {f"{prefix}_{k}": v for k, v in d.items()}


def benchmark(
    workflow: Workflow,
    name: str,
    description: str,
    config: Config,
    changes: dict[str, str | float | int],
    benchmark_dataset: BenchmarkDataset,
    visualization_logic: Callable,
):
    yield {
        "event": "benchmark_start",
        "name": name,
        "description": description,
        "changes": changes,
    }

    df_rows: list[dict] = []
    resulting_states: list[State] = []

    for entry in benchmark_dataset:
        # Run agent
        yield {
            "event": "test_case_start",
            "iteration": entry.id + 1,
            "max_iteration": len(benchmark_dataset),
        }

        config: Config = config.update_prompt(entry.prompt)
        config: Config = config.set(run_id=name + str(entry.id))
        resulting_state: State = visualization_logic(workflow.stream(config=config))

        yield {"event": "test_case_evaluation_start"}
        benchmark_summary, updated_state = evaluate_state_with_benchmark_entry(
            resulting_state,
            entry,
            workflow.get_evaluators(),
            config.default_provider_judge,
            config.default_model_judge,
        )
        yield {"event": "test_case_evaluation_stop"}

        yield {"event": "test_case_stop", "evaluation_summary": benchmark_summary}

        # Build execution trace
        execution_trace = {"answers": []}
        for answer in updated_state.answers:
            answer_energy_dict = answer.profiling_data.as_dict()
            answer_dict = {
                "agent_type": answer.agent_id,
                "message": answer.message,
                "evaluation_gt": answer.gt_evaluation.score
                if answer.gt_evaluation
                else None,
                "perplexity": answer.perplexity if answer.perplexity else None,
                **answer_energy_dict,
            }
            execution_trace["answers"].append(answer_dict)

        # Store results into a df row
        row = {
            "entry_id": entry.id,
            "run_id": updated_state.run_id,
            "execution_trace": json.dumps(execution_trace),
        }

        df_rows.append(row)
        resulting_states.append(updated_state)

    # Build results DataFrame
    df = pd.DataFrame(df_rows)

    yield {"event": "result", "result": (df, resulting_states)}


def aggregate_results(
    run_config_to_result_list: list[tuple[dict, pd.DataFrame]],
) -> pd.DataFrame:
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
        name = run_config["name"]
        description = run_config["description"]
        changes = run_config["changes"]

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

        run_summaries.append(
            {
                "name": name,
                "description": description,
                "changes": changes,
                "metrics_by_agent": json.dumps(agents_summary_stats),
            }
        )

    return pd.DataFrame(run_summaries)
